"""
predict.py

COMPLETAR DOCSTRING

DESCRIPCIÓN: Class for Making Predictions
AUTOR: Juan Pablo Schamun
FECHA: Septiembre 2023
"""

# Imports

import argparse
import pickle
import pandas as pd

class MakePredictionPipeline():

    '''
    Class for Making Predictions
    '''

    def __init__(self, input_path, output_path, model_path: str = None):
        self.input_path = input_path
        self.output_path = output_path
        self.model_path = model_path


    def load_data(self) -> pd.DataFrame:
        """
        Loads data for prediction (Raw)
        """
        data = pd.read_csv(self.input_path)

        return data

    def load_model(self) -> None:
        """
        Loads model already trained
        """    
        self.model = pickle.load(open(self.model_path, "rb"))

        return None


    def make_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts output from loaded data
        """

        # Eliminación de variables que no contribuyen a la predicción por ser muy específicas
        dataset = df.drop(columns=['Item_Identifier', 'Outlet_Identifier']).copy()

        # División del dataset de train y test
        df_train = dataset.loc[df['Set'] == 'train']
        df_test = dataset.loc[df['Set'] == 'test']

        # Eliminando columnas sin datos
        df_train = df_train.drop(['Set'], axis=1)
        df_test = df_test.drop(['Item_Outlet_Sales','Set'], axis=1)

        # Predecir   
        df_test['pred_Sales'] = self.model.predict(df_test)

        return df_test


    def write_predictions(self, predicted_data: pd.DataFrame) -> None:
        """
        writes output to file
        """

        predicted_data.to_csv(self.output_path)

        return None


    def run(self):

        '''
        Run methods of the class secuencially
        '''

        data = self.load_data()
        self.load_model()
        df_preds = self.make_predictions(data)
        self.write_predictions(df_preds)


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help='input file with data for predictions')
    parser.add_argument("output", type=str, help='output file of predicted data')
    parser.add_argument("model", type=str, help='input pickle file of the model')

    args = parser.parse_args()

    # spark = Spark()

    pipeline = MakePredictionPipeline(input_path = args.input,
                                      output_path = args.output,
                                      model_path = args.model)
    pipeline.run()
