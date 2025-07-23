# titanic_survival_prediction
Titanic Survival Prediction using Logistic Regression
This project aims to predict the survival of passengers on the Titanic using a Logistic Regression machine learning model. The notebook covers data loading, cleaning, preprocessing, model training, evaluation, and visualization of key insights.

Table of Contents
Objective

Techniques Used

Dataset

Installation

Usage

Project Structure

Data Cleaning and Preprocessing

Model Training and Evaluation

Results and Final Thoughts

Objective
The main objective of this project is to build a machine learning model to predict which passengers survived the Titanic disaster. Logistic Regression is used for this binary classification task.

Techniques Used
Data Loading and Inspection: Using pandas to load and get a preliminary understanding of the dataset.

Missing Data Identification: Checking for null values in each column to plan for data imputation.

Data Cleaning and Imputation: Handling missing values by filling numerical columns with medians and categorical columns with modes, and dropping irrelevant columns.

Categorical Variable Encoding: Converting non-numerical features into a numerical format suitable for machine learning models using mapping.

Feature and Target Split: Separating the dataset into independent variables (features) and the dependent variable (target).

Data Splitting: Dividing the data into training and testing sets to evaluate model performance on unseen data.

Logistic Regression: Implementing and training a Logistic Regression model, a common algorithm for binary classification.

Model Evaluation: Assessing the model's performance using metrics such as accuracy, confusion matrix, and classification report.

Feature Importance Visualization: Understanding the contribution of each feature to the model's predictions.

Dataset
The dataset used is titanic.csv, which contains information about Titanic passengers, including their survival status, passenger class, name, sex, age, number of siblings/spouses aboard, number of parents/children aboard, ticket number, fare, cabin number, and port of embarkation.

Installation
To run this notebook, you need Python and the following libraries. You can install them using pip:

Bash

pip install pandas numpy scikit-learn matplotlib seaborn
Usage
Clone the repository:

Bash

git clone https://github.com/your-username/titanic-survival-prediction.git
cd titanic-survival-prediction
Place the dataset:
Ensure the titanic.csv file is located in the root directory of the cloned repository.

Open and run the Jupyter Notebook:
Launch Jupyter Lab or Jupyter Notebook and open titanic_survival_prediction.ipynb. Run all cells sequentially.

Bash

jupyter notebook titanic_survival_prediction.ipynb
The notebook will execute all steps from data loading to model evaluation and visualization.

Project Structure
titanic_survival_prediction.ipynb: The main Jupyter Notebook containing all the code for the project.

titanic.csv: The dataset used for the prediction task.

README.md: This README file providing an overview of the project.

Data Cleaning and Preprocessing
The data cleaning and preprocessing steps are crucial for preparing the dataset for model training:

Loading the Dataset: The titanic.csv file is loaded into a pandas DataFrame.

Identifying Missing Data: The isnull().sum() method is used to count null values in each column, revealing missing data in 'Age', 'Cabin', and 'Embarked'.

Handling Missing 'Age' Values: Missing 'Age' values are filled using the median of the 'Age' column.

Dropping 'Cabin' Column: The 'Cabin' column is dropped due to a high number of missing values.

Dropping Irrelevant Columns: 'Name' and 'Ticket' columns are removed as they are not considered useful for the prediction model. The 'PassengerId' column is also dropped.

Handling Missing 'Embarked' Values: Missing 'Embarked' values are filled using the mode (most frequent value) of the 'Embarked' column.

Encoding Categorical Variables:

'Embarked' values ('S', 'C', 'Q') are mapped to numerical representations (0, 1, 2 respectively).

'Sex' values ('male', 'female') are mapped to numerical representations (0, 1 respectively).

Model Training and Evaluation
Feature and Target Split:

X (features) is created by dropping the 'Survived' column from the DataFrame.

y (target) is assigned the 'Survived' column.

Data Splitting: The dataset is split into training (80%) and testing (20%) sets using train_test_split with random_state=42 for reproducibility.

Training the Model: A LogisticRegression model is initialized with max_iter=1000 to ensure convergence and trained using the training data (X_train, y_train).

Model Evaluation:

Predictions (y_pred) are made on the test set (X_test).

The model's accuracy is calculated using accuracy_score.

A confusion matrix is generated using confusion_matrix.

A classification report, including precision, recall, and f1-score, is generated using classification_report.

The confusion matrix is visualized as a heatmap for better interpretation.

Results and Final Thoughts
The Logistic Regression model achieved an accuracy of approximately 79.89% in predicting passenger survival on the Titanic.

Key insights from the model include:

'Sex' was identified as the most influential factor, with females having a higher survival probability.

'Fare' and 'Pclass' (passenger class) also had a significant impact on survival predictions.

Logistic Regression served as an excellent starting point for this classification problem. For potential future improvements, exploring other models such as Random Forest or Support Vector Machines (SVM) could be beneficial.
