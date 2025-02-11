"""
train.py

DESCRIPCIÓN:
AUTOR: Juan Pablo Schamun
FECHA: September 2023
"""

# Imports

import argparse
import pickle
import pandas as pd

# Importando librerías para el modelo
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn import metrics
from sklearn.linear_model import LinearRegression


class ModelTrainingPipeline():

    '''
    Class for training pipeline
    
    '''

    def __init__(self, input_path, model_path):
        self.input_path = input_path
        self.model_path = model_path

    def read_data(self) -> pd.DataFrame:
        """
        Reads file with training data

        :return pandas_df: The desired DataLake table as a DataFrame
        :rtype: pd.DataFrame
        """

        pandas_df = pd.read_csv(self.input_path)

        return pandas_df


    def model_training(self, df: pd.DataFrame):
        """
        Trains the model from Data coming from Feature Engineering
        
        """

        # Eliminación de variables que no contribuyen a la predicción por ser muy específicas
        dataset = df.drop(columns=['Item_Identifier', 'Outlet_Identifier']).copy()

        # División del dataset de train y test
        df_train = dataset.loc[df['Set'] == 'train']
        df_test = dataset.loc[df['Set'] == 'test']

        # Eliminando columnas sin datos
        df_train = df_train.drop(['Set'], axis=1)
        df_test = df_test.drop(['Item_Outlet_Sales','Set'], axis=1)

        seed = 28
        model = LinearRegression()

        # División de dataset de entrenaimento y validación
        X = df_train.drop(columns='Item_Outlet_Sales')
        x_train, x_val, y_train, y_val = \
            train_test_split(X, df_train['Item_Outlet_Sales'], test_size = 0.3, random_state=seed)

        # Entrenamiento del modelo
        model.fit(x_train,y_train)

        # Predicción del modelo ajustado para el conjunto de validación
        pred = model.predict(x_val)

        # Cálculo de los errores cuadráticos medios y Coeficiente de Determinación (R^2)
        mse_train = metrics.mean_squared_error(y_train, model.predict(x_train))
        R2_train = model.score(x_train, y_train)
        # print('Métricas del Modelo:')
        # print('ENTRENAMIENTO: RMSE: {:.2f} - R2: {:.4f}'.format(mse_train**0.5, R2_train))

        mse_val = metrics.mean_squared_error(y_val, pred)
        R2_val = model.score(x_val, y_val)
        # print('VALIDACIÓN: RMSE: {:.2f} - R2: {:.4f}'.format(mse_val**0.5, R2_val))

        # print('\nCoeficientes del Modelo:')
        # Constante del modelo
        # print('Intersección: {:.2f}'.format(model.intercept_))

        # Coeficientes del modelo
        coef = pd.DataFrame(x_train.columns, columns=['features'])
        coef['Coeficiente Estimados'] = model.coef_
        # print(coef, '\n')
        # coef.sort_values(by='Coeficiente Estimados').set_index('features')\
        #   .plot(kind='bar', title='Importancia de las variables', figsize=(12, 6))       


        return model

    def model_dump(self, model_trained) -> None:
        """
        Saves de model in a pickle file
        
        """
 
        pickle.dump(model_trained, open(self.model_path, 'wb'))

        return None

    def run(self):

        '''
        Run methods of the class secuencially
        '''

        df = self.read_data()
        model_trained = self.model_training(df)
        self.model_dump(model_trained)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help='input file from Feature Engineering')
    parser.add_argument("output", type=str, help='output pickle file of the model')

    args = parser.parse_args()

    ModelTrainingPipeline(input_path = args.input, model_path = args.output).run()
