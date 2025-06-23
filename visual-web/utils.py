"""
utils.py

This file contains utility functions for:
1. Loading the dataset from a CSV file
2. Preprocessing the data:
   - Splitting into training and test sets
   - Standardizing features using StandardScaler
3. Training the pytrch model using binary vross entropy(BCE) loss and the adam optimizer
"""




import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os




def load_diabetes_data(filepath='../dataset/diabetes.csv'):
    """
    Loads the diabetes dataset from a CSV file

    Args:
        filepath (str): Path to the CSV file containing the dataset
                        Default is 'diabetes.csv'

    Returns:
        pd.DataFrame or None: The loaded dataset as a Pandas DataFrame, or None if loading fails.

    Notes:
        If the file does not exist or an error occurs while reading, 
        a message is printed and None is returned.
    """
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        print(f"File {filepath} Not found!")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None





def preprocess_data(df, test_size=0.2, random_state=42):
    """
    Preprocesses the diabetes dataset to prepare it for model training.

    This function performs the following steps:
    1. Selects the relevant feature columns and target labels.
    2. Splits the dataset into training and testing sets using stratified sampling.
    3. Normalizes the feature values using standard scaling (zero mean, unit variance).

    Args:
        df (pd.DataFrame): The original diabetes dataset as a pandas DataFrame.
        test_size (float): The proportion of the dataset to be used for testing. Default is 0.2 (20%).
        random_state (int): A random seed for reproducibility of the train-test split.

    Returns:
        Tuple:
            X_train (np.ndarray): Scaled training features.
            X_test (np.ndarray): Scaled testing features.
            y_train (np.ndarray): Training labels.
            y_test (np.ndarray): Testing labels.
            scaler (StandardScaler): The fitted StandardScaler object used for normalization.
    """

    # Define the list of feature columns that will be used for prediction
    feature_columns = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]

    # Extract feature values (X) and target labels (y) from the DataFrame
    X = df[feature_columns].values
    y = df['Outcome'].values  # 'Outcome' is the binary target (0 or 1)

    # split the data into training and testing sets ysing stratified sampling
    # Stratify =y ensures that both sets maintain the same class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Init a StandardScaler to normqlize the feature data
    scaler = StandardScaler()

    # Fit the scaler on the training data and transform both training and test sets
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Return the processed datasets and the fitted scaler
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler





def save_scaler(scaler, filepath='scaler.pkl'):
    """
    Saves a fitted standardscaler object to a file using pickle format

    This allows the same scaling transformation to be reused later
    (during prediction or deployment), ensuring that input data
    is transformed consistently with the training phase.

    Args:
        scaler (standardscaler): The fitted scaler to be saved
        filepath (str): Path where the scaler will be stored
                        Defaults to 'scaler.pkl'
    """
    #Open the file in binary write mode
    with open(filepath, 'wb') as f:
        # serialize the scaler object and save it to the specified file
        pickle.dump(scaler, f)





def load_scaler(filepath='scaler.pkl'):
    """
    Loads a previously saved standardscaler object from a pickle (.pkl) file

    This fynction is typically used during inference, allowing us to apply the
    same feature scaling that was used during training. Without this step,
    model predictions may become inaccurate due to inconsistent input formats.

    Args:
        filepath (str): The path to the pickle file that contains teh fitted scaler
                        default is 'scaler.pkl'

    Returns:
        standardscaler or None: Returns the loaded scaler object if successful
                                If the file doesn't exist, returns None and prints an error
    """
    try:
        # attempt to open the pickle file in read-binary mode
        with open(filepath, 'rb') as f:
            scaler = pickle.load(f)  # Load the scaler object
        return scaler

    except FileNotFoundError:
        # Handle the case where the file doesnnt exist
        print(f"File {filepath} not found!")
        return None





def prepare_input_data(input_dict, scaler):
    """
    Preprocessing data innnnnnn
    Returns:
        Normalized input array
    """
    # The order of features must be the same as during training
    feature_order = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    # convert to array
    input_array = np.array([input_dict[feature] for feature in feature_order])
    
    # Reshape for scaler (2d array)
    input_array = input_array.reshape(1, -1)
    
    # normalization with array
    input_scaled = scaler.transform(input_array)
    
    return input_scaled.flatten()  # Return 1d array




def get_dataset_summary(df):
    """
    returns a summary of the dataset, including the number of sample,
    number of feature, class distribution (diabetes vs nondiabetes),
    and descriptive statstics for each feature.

    This function is useful for quick explration and understanding
    of the dataset before training a model.

    Args:
        df (pd.Dataframe): The diabetes dataset in the form of a pandas DaaFrame

    Returns:
        dict or None: A dictionary containing summary statistics, or None if the input is invalid
    """
    if df is None:
        # If the input DataFrame is none, there nothing to summarize
        return None

    summary = {
        'total_samples': len(df),                                       # Total number of rows/samples in the dataset
        'total_features': len(df.columns) - 1,                          # Number of features (excluding the target column 'Outcome')
        'diabetes_cases': df['Outcome'].sum(),                          # Count of positive cases ()utcome = 1)
        'non_diabetes_cases': len(df) - df['Outcome'].sum(),            # Count of negative cases (Outcome = 0)
        'diabetes_percentage': (df['Outcome'].sum() / len(df)) * 100,   # Proportion of positive cases in percent
        'feature_stats': df.describe()                                  # Descriptive statistics (mean, std, min, etc) for all numeric features
    }

    return summary





def validate_input_data(input_dict):
    """
    Validates user provided input data before making predictions

    this function ensures that:
    - All required fields are present in the input dictionary
    - Each field has a value within an expected and safe range

    Input data is expected to represent a single patient medicql information
    used for diabetes prediction. This validation step is important for
    ensuring model safety, input integrity, and avoiding runtime errors.

    Args:
        input_dict (dict): A dictionary containing user input, where each key 
                           corresponds to a required feature name.

    Returns:
        tuple: (bool, str) 
            - True and a success message if input is valid
            - False and an error message if input is invalid
    """

    # Define the required input fields based on model training features
    required_fields = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                       'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

    # first, check that all expected fields are present in the input
    for field in required_fields:
        if field not in input_dict:
            # Mising field detected
            return False, f"Field {field} is missing"

    # Next, define acceptable value ranges for each input field
    # These ranges are based on medical logic or dataset bounds
    validations = {
        'Pregnancies': (0, 20),                # Typically a small number
        'Glucose': (0, 300),                   # mg/dL, upper bound safety
        'BloodPressure': (0, 200),             # mmHg
        'SkinThickness': (0, 100),             # mm
        'Insulin': (0, 1000),                  # mu U/ml
        'BMI': (0, 70),                        # kg/m^2
        'DiabetesPedigreeFunction': (0, 3),    # Score based on family hist0ry
        'Age': (18, 120)                       # Adult range assumed
    }

    # check if each value falls within the acceptable range
    for field, (min_val, max_val) in validations.items():
        value = input_dict[field]
        if not (min_val <= value <= max_val):
            return False, f"{field} should be between {min_val} and {max_val}"

    # If all checks pass, return success
    return True, "Valid input"




def format_prediction_result(probability, threshold=0.5):
    """
    Formats the model output probability into a structured prediction result.

    This function takes the raw probability produced by the model (usually from
    a sigmoid activation function) and applies a decision threshold (default is 0.5)
    to determine the predicted class (0 or 1).

    It also provides humqn readable output including:
    - the predicted probability
    - the final binary classification (0 or 1)
    - a status message in Bahasa Indonesia ("Positive for Diabetes" or "Negative for Diabetes")
    - and the model confidence in that prediction

    Args:
        probability (float): The predicted probability output by the model
        threshold (float, optional): The classification threshold. Defaults to 0.5

    Returns:
        dict: A dictionary containing:
            - 'probability' (float): the original probability
            - 'prediction' (int): The binary prediction (1 = diabetes, 0 = not diabetes)
            - 'status' (str): Human readable prediction status in Bahasa Indonesia
            - 'confidence' (float): Confidence score for the predicted class
    """
    # Ensure the input is treated as a float (in case it a tensor or numpy type)
    prob_float = float(probability)

    # Convert probability into binary prediction using the given threshold
    prediction = 1 if prob_float > threshold else 0

    # Construct a dictionary containing prediction details
    result = {
        'probability': prob_float,  # hhe raw model probability
        'prediction': prediction,   # Binary classification result
        'status': "Positive for Diabetes" if prediction == 1 else "Negative for Diabetes",  # In Bahasa
        'confidence': prob_float if prediction == 1 else (1 - prob_float)  # Confidence of chosen class
    }

    return result





def check_model_files():
    """
    Checks whether both the trained model file and scaler file exist
    in the current working directory.

    This function is usefull to verify that the model and preprocessing
    objects are saved before attempting to load or use them for predictions.

    Returns:
        Tuple[bool, bool]: A tuple indicating the existencce of the files:
            - first value is true if 'diabetes_model.pth' exists
            - Second value is rtrue if 'scaler.pkl' exists
    """
    # Vheck if the saved model file exists
    model_exists = os.path.exists('diabetes_model.pth')

    # check if the saved scaler file exists
    scaler_exists = os.path.exists('scaler.pkl')

    # Return both results as a tuple (True/False)
    return model_exists, scaler_exists





def print_model_evaluation(y_true, y_pred, y_prob=None):
    """
    Displays evaluation metrics for a binary classification model.

    This function prints the confusion matrix and classification report 
    to help understand the model prformance on test data. 
    If prediction probabilities are provided, it also calculates and shows the AUC score.

    Args:
        y_true (array-like): True labels (0 or 1)
        y_pred (array-like): Predicted labels (0 or 1)
        y_prob (array-like, optional): Predicted probabilities for the positive class (1)
    """
    print("=== Model Evaluation ===")

    # show confusion matrix: counts of TP, TN, FP, FN
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    #detailed classification metrics: precision, recall, F1 score, etc
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['No Diabetes', 'Diabetes']))

    # If probability scores are provided, compute and display AUC (Area Under the ROC Curve)
    if y_prob is not None:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_true, y_prob)
        print(f"\nAUC Score: {auc:.4f}")





def get_feature_descriptions():
    """
    Returns a human readable description for each input feature used in the diabetes dataset

    This is useful for displaying tooltips, labels, or explanations in user interfaces,
    especially when collecting input from nontechnical user.
    """
    descriptions = {
        'Pregnancies': 'Number of times pregnant',
        'Glucose': 'Plasma glucose concentration (mg/dL)',
        'BloodPressure': 'Diastolic blood pressure (mm Hg)',
        'SkinThickness': 'Triceps skinfold thickness (mm)',
        'Insulin': '2 Hour serum insulin (μU/mL)',
        'BMI': 'Body Mass Index (kg/m²)',
        'DiabetesPedigreeFunction': 'Diabetes pedigree function (genetic influence)',
        'Age': 'Age (in years)'
    }

    return descriptions