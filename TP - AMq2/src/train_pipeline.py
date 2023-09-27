import subprocess

# subprocess.run(['Python', 'feature_engineering.py', '../data/BigMart.csv', '../data/BigMart_Final.csv'])

subprocess.run(['Python', 'train.py','../data/BigMart_Final.csv', '../data/model.pkl'])