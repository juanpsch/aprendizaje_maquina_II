'''Module for inference'''
import subprocess

subprocess.run(['Python', 'feature_engineering.py',
                '../data/BigMart.csv',
                '../data/input_Inference_Final.csv'], check=False)

subprocess.run(['Python', 'predict.py',
                '../data/input_Inference_Final.csv',
                '../data/output_Inference_Final.csv',
                '../data/model.pkl'], check=False)
