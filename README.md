# -Auto-MPG---Miles-Per-Gallons---by-using-ML-Gradient-Descent-with-Python-NumPy-Pandas

This project applies a machine learning regression model (trained using Stochastic Gradient Descent) to the classic Auto MPG dataset to predict a vehicle's miles per gallon (MPG) based on its characteristics (cylinders, horsepower, weight, etc.).


  1. Data Source :-
    - Dataset: Auto MPG (often used for regression analysis).
    - Target Variable (Y): mpg (Miles Per Gallon).
    - Key Features (X): cylinders, displacement, horsepower, weight, acceleration, model-year.


  2. Project Challenges & Solutions (The Debugging Journey):-
   - This project required several critical data cleaning steps to resolve issues that prevented the model from training.
   - This section documents the key steps taken to ensure the data was clean and consistent.


      Problem Encountered	               Error Type(s)	                              Permanent Solution
      Missing Values (80 NaNs)	         ValueError: Input contains NaN	              Replaced all non-numeric '?' characters                                                                                       in the horsepower column with np.nan,                                                                                         then used mean imputation to fill                                                                                             missing  values in key columns.

      Inconsistent Data Types	           ValueError: could not convert                Used pd.to_numeric(...,errors='coerce')                                                      string to float                  to ensure all feature columns were                                                                                            strictly float type before imputation.

      Ambiguous Conditional Logic	       ValueError: The truth value of               Corrected the NaN-check logic                                                                a Series is ambiguous            inside the imputation loop from if                                                                                            df.isnull().any(): to the column-                                                                                             specific check: if                                                                                                            df[col].isnull().any():

      Rerunning Code/Mismatches          KeyError: 'cylinders' not in index,          Implemented a strict pipeline: Clean                                             ValueError: Found array with 0 sample(s)      -> Encode -> Split -> Predict,                                                  ValueError: inconsistent samples [80, 318]    ensuring the entire process is run                                                                                            sequentially on the original data only                                                                                        once.
                                                                                                       
      Array Mismatch	                     AttributeError: 'DataFrame' object has no  	Replaced deprecated/incorrect array                                                               attribute 'ravel'           reshaping with the robust NumPy                                                                                               method: y_test.to_numpy().flatten()                                                                                           and y_pred.flatten().
     
   3. Technical Implementation:-
                                                                                         
      (i) Data Cleaning & Preprocessing:
        - Handling Missing Data: Replaced '?' with NaN and imputed all null values in the feature columns using the column                                     mean.
        - Categorical Encoding: Converted categorical features (cylinders and model-year) using One-Hot Encoding via                                          pd.get_dummies(), utilizing drop_first=True to avoid multicollinearity.

      (ii) Model Training:
          - Splitting: Data was split into 80% for training (X_train, y_train) and 20% for testing (X_test, y_test) using                            train_test_split(..., random_state=42).
          - Fitting: The model was trained using only the clean, encoded X_train and y_train arrays. 


  4. Results:-
     After implementing all the robust cleaning and engineering steps, the model was successfully trained and evaluated.

     Metric	                         Value
     Final Test Sample Size	         80 vehicles (20% of dataset)
     
