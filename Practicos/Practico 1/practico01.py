import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
estaturas = np.random.uniform(1.4, 2.0, 100)
pesos = []

for i in estaturas:
    peso_min = 18.5 * (i ** 2)
    peso_max = 24.9 * (i ** 2)
    peso = np.random.uniform(peso_min, peso_max)
    pesos.append(peso)

data = pd.DataFrame({
    'Estatura (m)': estaturas,
    'Peso (kg)': pesos
})

x = data['Estatura (m)']
y = data['Peso (kg)']
m = np.sum((x - np.mean(x)) * (y - np.mean(y))) / np.sum((x - np.mean(x)) ** 2)
b = np.mean(y) - m * np.mean(x)

y_line = m * x + b

plt.scatter(data['Estatura (m)'], data['Peso (kg)'], color='blue', label='Datos')
plt.plot(x, y_line, color='red', label='Línea ajustada')
plt.title('Estatura vs Peso con Línea Ajustada')
plt.xlabel('Estatura (m)')
plt.ylabel('Peso (kg)')
plt.legend()
plt.show()