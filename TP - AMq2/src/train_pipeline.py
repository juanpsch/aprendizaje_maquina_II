'''Module for training'''
import subprocess

subprocess.run(['Python', 'feature_engineering.py',
                '../data/BigMart.csv',
                '../data/BigMart_Final.csv'], check=False)

subprocess.run(['Python', 'train.py',
                '../data/BigMart_Final.csv',
                '../data/model.pkl'], check=False)
