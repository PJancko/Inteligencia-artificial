import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Configuración del modelo y entrenamiento
learning_rate = 0.0001
batch_size = 32
epochs = 200
hidden_size = 20

# Función para cargar y preparar los datos
def prepare_data(file_path):
    data = pd.read_csv(file_path)
    # Eliminar columnas de tipo 'object' (que generalmente incluyen fechas o strings)
    data = data.select_dtypes(exclude=['object'])

    X = data.drop(['phishing', 'url'], axis=1, errors='ignore')
    y = data['phishing'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader, scaler, X_train.shape[1]

def create_model(input_size, hidden_size):
    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),  # Capa oculta
        nn.Sigmoid(),                          # Función de activación
        nn.Linear(hidden_size, 2)           # Capa de salida (dos clases: phishing o no)
    )
    return model


# Función para entrenar el modelo
def train_model(model, train_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)


    model.to(device)
    train_losses = []
    train_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels) # calcula la funcion de costo o perdida entre la predicciones y las etiquetas reales
            loss.backward() # calcula los gradientes
            optimizer.step() # ajusta los pesos con los gradientes calculados

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

    return train_losses, train_accuracies

# Función para evaluar el modelo
def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'\nTest Accuracy: {accuracy:.2f}%')
    return accuracy

# Funciones auxiliares para guardar resultados
def save_training_plots(losses, accuracies):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f'resultados_modelo_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(save_dir, 'training_loss.png'))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(accuracies)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.savefig(os.path.join(save_dir, 'training_accuracy.png'))
    plt.close()

    return save_dir

def save_model(model, scaler, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler
    }, os.path.join(save_dir, 'model.pth'))

# Programa principal
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Preparar datos
    train_loader, test_loader, scaler, input_size = prepare_data("phishing_dataset.csv")

    # Crear modelo
    model = create_model(input_size, hidden_size)

    # Entrenar modelo
    train_losses, train_accuracies = train_model(model, train_loader, device)

    # Evaluar modelo
    accuracy = evaluate_model(model, test_loader, device)

    # Guardar resultados
    save_dir = save_training_plots(train_losses, train_accuracies)
    save_model(model, scaler, save_dir)

