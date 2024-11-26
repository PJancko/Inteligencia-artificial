import random
import pandas as pd
import numpy as np
import datetime

# Existing lists remain the same
phishing_keywords = [
    'secure', 'account', 'banking', 'login', 'verify', 'update', 'service',
    'confirm', 'user', 'validate', 'payment', 'signin', 'security', 'password',
    'billing', 'authenticate', 'wallet', 'recover', 'help', 'support',
    'reset', 'verify', 'unusual', 'activity', 'suspended', 'locked'
]

legitimate_keywords = [
    'home', 'about', 'contact', 'services', 'products', 'blog', 'news',
    'shop', 'store', 'cart', 'checkout', 'profile', 'settings', 'dashboard',
    'privacy', 'terms', 'faq', 'careers', 'investors', 'partners',
    'docs', 'api', 'status', 'pricing', 'features', 'enterprise'
]

legitimate_tlds = ['.com', '.org', '.net', '.edu', '.gov', '.io', 
    '.co', '.us', '.uk', '.ca', '.au', '.de', '.fr'
]

suspicious_tlds = ['.xyz', '.info', '.online', '.site', '.tk', '.ml',
    '.ga', '.cf', '.gq', '.buzz', '.top', '.work', '.loan'
]

legitimate_domains = [
    'amazon', 'google', 'microsoft', 'apple', 'facebook', 'twitter', 
    'github', 'linkedin', 'youtube', 'netflix', 'spotify', 'dropbox',
    'slack', 'zoom', 'adobe', 'paypal', 'walmart', 'ebay', 'instagram'
]
        
suspicious_patterns = [
    '@', 'login', 'signin', 'verify', 'account', 'secure', 'update',
    'password', 'credential', 'confirm', 'billing', 'payment'
]

# Function to generate URLs with mixed characteristics
def generate_url(is_phishing):
    if is_phishing:
        # With 30% chance, use a legitimate domain for phishing
        if random.random() < 0.3:
            base = random.choice(legitimate_domains)
        else:
            base = random.choice(phishing_keywords)
        # With 70% chance, use a suspicious TLD
        if random.random() < 0.7:
            tld = random.choice(suspicious_tlds)
        else:
            tld = random.choice(legitimate_tlds)
    else:
        # With 10% chance, use a phishing keyword for legitimate
        if random.random() < 0.1:
            base = random.choice(phishing_keywords)
        else:
            base = random.choice(legitimate_domains)
        # With 10% chance, use a suspicious TLD for legitimate
        if random.random() < 0.1:
            tld = random.choice(suspicious_tlds)
        else:
            tld = random.choice(legitimate_tlds)
    
    domain = base + tld
    
    # Randomly decide the complexity of the path and parameters
    path = '/'.join(random.choices(phishing_keywords + legitimate_keywords, k=random.randint(0, 4)))
    params = '&'.join([f"{random.choice(['id', 'session', 'token', 'user'])}={random.randint(1000, 9999)}"
                       for _ in range(random.randint(0, 4))])
    url = f"https://{domain}"
    if path:
        url += f"/{path}"
    if params:
        url += f"?{params}"
    return url

# Function to generate security features with overlapping distributions
def generate_security_features(is_phishing):
    if is_phishing:
        ssl_valid_days = random.randint(1, 365)
        domain_age_days = random.randint(1, 1000)
        qty_nameservers = random.randint(1, 5)
        qty_mx_servers = random.randint(0, 5)
        qty_redirects = random.randint(0, 10)
        ssl_issuer_trusted = random.choices([0, 1], weights=[0.5, 0.5], k=1)[0]
    else:
        ssl_valid_days = random.randint(30, 730)
        domain_age_days = random.randint(30, 5000)
        qty_nameservers = random.randint(1, 5)
        qty_mx_servers = random.randint(1, 5)
        qty_redirects = random.randint(0, 5)
        ssl_issuer_trusted = random.choices([0, 1], weights=[0.3, 0.7], k=1)[0]
    
    # Add some noise by occasionally swapping values
    if random.random() < 0.1:
        ssl_valid_days, domain_age_days = domain_age_days, ssl_valid_days
    return {
        'ssl_valid_days': ssl_valid_days,
        'domain_age_days': domain_age_days,
        'qty_nameservers': qty_nameservers,
        'qty_mx_servers': qty_mx_servers,
        'qty_redirects': qty_redirects,
        'ssl_issuer_trusted': ssl_issuer_trusted,
    }

# Create dataset
data = []
n_samples = 100000
for _ in range(n_samples):
    is_phishing = random.random() < 0.5
    url = generate_url(is_phishing)
    
    # URL features
    url_features = {
        'url_length': len(url),
        'qty_dot': url.count('.'),
        'qty_hyphen': url.count('-'),
        'qty_slash': url.count('/'),
        'qty_questionmark': url.count('?'),
        'qty_equal': url.count('='),
        'qty_at': url.count('@'),
        'qty_numeric': sum(c.isdigit() for c in url),
    }
    
    # Security features with overlapping distributions
    security_features = generate_security_features(is_phishing)
    
    # Combine into a row
    row = {
        'url': url,
        **url_features,
        **security_features,
        'phishing': int(is_phishing)
    }
    data.append(row)

# Convert to DataFrame
dataset = pd.DataFrame(data)

# Save the dataset to a CSV file
dataset.to_csv('phishing_dataset.csv', index=False)

# Display the first few rows
print(dataset.head())