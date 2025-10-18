import pandas as pd
import numpy as np


class Dataset:
    """
    Dataset class for loading the dataset. Additionally, adds derived features to the dataset.
    """

    df: pd.DataFrame
    target_col: str

    def __init__(self, df: pd.DataFrame, target_col: str = 'Class'):
        self.df = df
        self.target_col = target_col

    @property
    def X(self) -> np.ndarray:
        return self.df.drop(columns=[self.target_col]).values
    
    @property
    def y(self) -> np.ndarray:
        return self.df[self.target_col].values

    @classmethod
    def load_dataset(cls, input_path: str):

        df = pd.read_excel(input_path)
        
        return cls(df = df)

    def add_derived_features(self) -> None:

        self.df['PAR'] = self.df['Perimeter'] / self.df['Area']

        self.df['NSD'] = (self.df['MajorAxisLength'] - self.df['MinorAxisLength']) / (self.df['MajorAxisLength'] + self.df['MinorAxisLength'])

        self.df['CUR'] = self.df['Area'] / (self.df['ConvexArea'] - self.df['Area'])

        self.df['ASI'] = (self.df['MajorAxisLength'] - self.df['MinorAxisLength']) / (self.df['MajorAxisLength'] + self.df['MinorAxisLength']) * (self.df['Solidity'] * self.df['Eccentricity'])

        self.df['CAER'] = np.pi * self.df['EquivDiameter'] ** 2 / (4 * self.df['Area'])

        self.df['CBI'] = self.df['Compactness'] / self.df['Solidity']

        self.df['FII'] = self.df['AspectRation'] / self.df['roundness']

        self.df['PEF'] = (self.df['MajorAxisLength'] - self.df['MinorAxisLength']) ** 2 / (self.df['Area'])

        self.df['RSC'] =  (self.df['Perimeter'] ** 2) / self.df['ConvexArea']

        self.df['ARV'] = (self.df['MajorAxisLength'] / self.df['MinorAxisLength'] - 1) ** 2

        self.df['APB'] = self.df['Area'] / (self.df['MajorAxisLength'] + self.df['MinorAxisLength'] + self.df['Perimeter'])

        self.df['RFI'] = np.sqrt(self.df['Area']) / (self.df['MajorAxisLength'] - self.df['MinorAxisLength'])

        self.df['ELI'] = (self.df['MajorAxisLength'] ** 2 - self.df['MinorAxisLength'] **2) / (self.df['MajorAxisLength'] ** 2 + self.df['MinorAxisLength'] **2)