import os
import sys
import dill
import numpy as np
import pandas as pd
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]
            
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)
            
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score
        
        return report
    
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)


def check_current_champion():
    """Quick check of your current champion model"""
    
    # Import here to avoid circular imports
    from src.components.model_trainer import ModelTrainer
    
    trainer = ModelTrainer()
    
    # Check if model exists
    model_path = trainer.model_trainer_config.trained_model_file_path
    report_path = trainer.model_trainer_config.model_report_file_path
    
    if os.path.exists(model_path):
        try:
            # Use your existing load_object function
            model = load_object(model_path)
            
            print(f"Current Champion: {type(model).__name__}")
            print(f"Model Path: {model_path}")
            print(f"Model Size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
            
            # Show performance report if available
            if os.path.exists(report_path):
                print(f"\nPerformance Report:")
                with open(report_path, 'r') as f:
                    print(f.read())
            else:
                print("\nNo detailed performance report found")
                
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print("No trained model found. Run training first!")


if __name__ == "__main__":
    check_current_champion()
