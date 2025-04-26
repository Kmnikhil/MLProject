import os
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    processor_obj_file_path = os.path.join('my_files',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformation_object(self):
        """
        This function is responsible for data transformation
        """

        try:
            numerical_columns = ['reading_score', 'writing_score']
            Categorical_Columns = ['gender', 'race_ethnicity', 'parental_level_of_education',
                                    'lunch', 'test_preparation_course']
            
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ("scaler",StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("encoding",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {Categorical_Columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline",num_pipeline,numerical_columns),
                    ("categorical_pipeline",cat_pipeline,Categorical_Columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
    
    def data_transformation_initiate(self, train_data_path, test_data_path):
        try:
            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)

            logging.info("Read train and test data")

            processor_obj = self.get_data_transformation_object()

            target_feature_name = "math_score"

            in_fea_train = train_data.drop(columns=[target_feature_name], axis=1)
            tar_fea_train = train_data[target_feature_name]

            in_fea_test = test_data.drop(columns=[target_feature_name], axis=1)
            tar_fea_test = test_data[target_feature_name]

            logging.info("Applying preprocessing on training and testing data")

            transformed_in_fea_train = processor_obj.fit_transform(in_fea_train)
            transformed_in_fea_test = processor_obj.transform(in_fea_test)

            transformed_train_data = np.c_[transformed_in_fea_train,np.array(tar_fea_train)]
            transformed_test_data = np.c_[transformed_in_fea_test,np.array(tar_fea_test)]

            save_object(
                file_path=self.data_transformation_config.processor_obj_file_path,
                obj=processor_obj
            )

            logging.info("Saved the preprocessor")
            return (
                    transformed_train_data,
                    transformed_test_data,
                    self.data_transformation_config.processor_obj_file_path
                    )
        
        except Exception as e:
            raise CustomException(e,sys)
            