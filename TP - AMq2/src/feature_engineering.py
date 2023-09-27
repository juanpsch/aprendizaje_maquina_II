"""
feature_engineering.py

COMPLETAR DOCSTRING

DESCRIPCIÓN:
AUTOR:
FECHA:
"""

# Imports
import pandas as pd
import argparse

class FeatureEngineeringPipeline(object):

    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def read_data(self) -> pd.DataFrame:
        """
        COMPLETAR DOCSTRING 
        
        :return pandas_df: The desired DataLake table as a DataFrame
        :rtype: pd.DataFrame
        """
            
        pandas_df = pd.read_csv(self.input_path)
        
        return pandas_df

    
    def data_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        COMPLETAR DOCSTRING
       
        """
        
        df_transformed = df
        
        # FEATURE ENGINEERING: para los años de establecimiento
        df_transformed['Outlet_Establishment_Year'] = 2020 - df_transformed['Outlet_Establishment_Year']
        
        # LIMPIEZA: unificando etiquetas para 'Item_Fat_Content' 
        df_transformed['Item_Fat_Content'] = df_transformed['Item_Fat_Content'].replace({'low fat':  'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'})

        # LIMPIEZA: faltantes en el peso de los productos
        productos = list(df_transformed[df_transformed['Item_Weight'].isnull()]['Item_Identifier'].unique())
        for producto in productos:
            moda = (df_transformed[df_transformed['Item_Identifier']==producto][['Item_Weight']]).mode().iloc[0,0]
            df_transformed.loc[df_transformed['Item_Identifier'] == producto, 'Item_Weight'] = moda

        # LIMPIEZA: faltantes en el tamaño de las tiendas
        outlets = list(df_transformed[df_transformed['Outlet_Size'].isnull()]['Outlet_Identifier'].unique())
        for outlet in outlets:
            df_transformed.loc[df_transformed['Outlet_Identifier'] == outlet,'Outlet_Size'] =  'Small'

        # FEATURE ENGINEERING: asignación de nueva categoría para 'Item_Fat_Content'
        df_transformed.loc[df_transformed['Item_Type'] == 'Household', 'Item_Fat_Content'] = 'NA'
        df_transformed.loc[df_transformed['Item_Type'] == 'Health and Hygiene', 'Item_Fat_Content'] = 'NA'
        df_transformed.loc[df_transformed['Item_Type'] == 'Hard Drinks', 'Item_Fat_Content'] = 'NA'
        df_transformed.loc[df_transformed['Item_Type'] == 'Soft Drinks', 'Item_Fat_Content'] = 'NA'
        df_transformed.loc[df_transformed['Item_Type'] == 'Fruits and Vegetables', 'Item_Fat_Content'] = 'NA'        

        # FEATURES ENGINEERING: creando categorías para 'Item_Type'
        df_transformed['Item_Type'] = df_transformed['Item_Type'].replace({'Others': 'Non perishable', 'Health and Hygiene': 'Non perishable', 'Household': 'Non perishable',
        'Seafood': 'Meats', 'Meat': 'Meats', 'Baking Goods': 'Processed Foods', 'Frozen Foods': 'Processed Foods', 'Canned': 'Processed Foods', 'Snack Foods': 'Processed Foods',
        'Breads': 'Starchy Foods', 'Breakfast': 'Starchy Foods', 'Soft Drinks': 'Drinks', 'Hard Drinks': 'Drinks', 'Dairy': 'Drinks'})

        # FEATURES ENGINEERING: asignación de nueva categorías para 'Item_Fat_Content'
        df_transformed.loc[df_transformed['Item_Type'] == 'Non perishable', 'Item_Fat_Content'] = 'NA'

        # FEATURE ENGINEERING: codificación los niveles de precios de los productos
        df_transformed['Item_MRP'] = pd.qcut(df_transformed['Item_MRP'], 4, labels = [1, 2, 3, 4])
        
        # FEATURE ENGINEERING: codificación de variables ordinales
        df_transformed = df_transformed.drop(columns=['Item_Type', 'Item_Fat_Content'])
        df_transformed['Outlet_Size'] = df_transformed['Outlet_Size'].replace({'High': 2, 'Medium': 1, 'Small': 0})
        # Estas categorias se ordenaron asumiendo la categoria 2 como más lejos
        df_transformed['Outlet_Location_Type'] = df_transformed['Outlet_Location_Type'].replace({'Tier 1': 2, 'Tier 2': 1, 'Tier 3': 0})
        
        # FEATURE ENGINEERING: codificación de variables nominales
        df_transformed = pd.get_dummies(df_transformed, columns=['Outlet_Type'])
        
        return df_transformed

    def write_prepared_data(self, transformed_dataframe: pd.DataFrame) -> None:
        """
        COMPLETAR DOCSTRING
        
        """
        
        transformed_dataframe.to_csv(self.output_path, index=False)
        
        return None

    def run(self):
    
        df = self.read_data()
        df_transformed = self.data_transformation(df)
        self.write_prepared_data(df_transformed)

  
if __name__ == "__main__":

    
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help='input file raw')
    parser.add_argument("output", type=str, help='output file wiht feature engineering')
    args = parser.parse_args()

    FeatureEngineeringPipeline(input_path = args.input, output_path = args.output).run()

    