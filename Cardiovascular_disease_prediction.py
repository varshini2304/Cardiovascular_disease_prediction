import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


data=pd.read_csv('cardio_train.csv')
print(data)

print(data.head())

# Load the dataset
data = pd.read_csv('cardio_train.csv', delimiter=';')

# Data Pre-processing
# Check for missing values
print("Missing values:\n", data.isnull().sum())

# Convert categorical variables using one-hot encoding
data = pd.get_dummies(data, columns=['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active'])

# Standardize numerical features
scaler = StandardScaler()
data[['age', 'height', 'weight', 'ap_hi', 'ap_lo']] = scaler.fit_transform(data[['age', 'height', 'weight', 'ap_hi', 'ap_lo']])

# Data Analysis and Visualizations
# Plot age distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['age'], bins=30, kde=True, color='blue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Correlation Matrix
correlation_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Machine Learning Techniques
# Split the dataset
X = data.drop(['id', 'cardio'], axis=1)
y = data['cardio']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Support Vector Machines
svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)
print("Support Vector Machines Accuracy:", svm_accuracy)

# K-Nearest Neighbor
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_pred)
print("K-Nearest Neighbor Accuracy:", knn_accuracy)

# Decision Trees
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)
print("Decision Trees Accuracy:", dt_accuracy)

# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
print("Logistic Regression Accuracy:", lr_accuracy)

# Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print("Random Forest Accuracy:", rf_accuracy)

data = pd.read_csv('cardio_train.csv', delimiter=';')

# Check for missing values
print(data.isnull().sum())

# Encode categorical variables if needed (here, 'gender' is encoded assuming 1 for male and 2 for female)
data['gender'] = data['gender'].map({1: 'male', 2: 'female'})

# Split the dataset into features (X) and target variable (y)
X = data.drop(['id', 'cardio'], axis=1)
y = data['cardio']

# Step 2: Exploratory Data Analysis (EDA)
# Descriptive statistics
print(X.describe())

# Visualize the distribution of the target variable
sns.countplot(x='cardio', data=data)
plt.title('Distribution of Cardiovascular Disease')
plt.show()

# Explore numerical features
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
for feature in numerical_features:
    sns.histplot(X[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.show()

# Explore the relationship between categorical features and the target variable
categorical_features = X.select_dtypes(include=['object']).columns
for feature in categorical_features:
    sns.countplot(x=feature, hue='cardio', data=data)
    plt.title(f'Relationship between {feature} and Cardiovascular Disease')
    plt.show()

# Step 3: Correlation Matrix
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Step 4: Model Building and Evaluation
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the numerical features
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# Implement various machine learning models
models = {
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{name} Accuracy: {accuracy:.4f}')

# Step 5: Visualization of Model Performance
model_names = list(models.keys())
accuracies = [accuracy_score(y_test, models[name].predict(X_test)) for name in model_names]

plt.bar(model_names, accuracies, color=['blue', 'orange', 'green', 'red', 'purple'])
plt.title('Model Performance Comparison')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.show()

data = pd.read_csv('cardio_train.csv', delimiter=';')

# One-hot encode categorical variables
data = pd.get_dummies(data, columns=['gender'], drop_first=True)

# Split the dataset into features (X) and target variable (y)
X = data.drop(['id', 'cardio'], axis=1)
y = data['cardio']

# Standardize the numerical features
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])

# Train the Random Forest model on the entire dataset
random_forest_model = RandomForestClassifier()
random_forest_model.fit(X, y)

# Save the model for future use
import joblib
joblib.dump(random_forest_model, 'heart_disease_model.pkl')

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv('cardio_train.csv', delimiter=';')

# Encode categorical variables (if any)
data = pd.get_dummies(data, columns=['gender'], drop_first=True)

# Split the dataset into features (X) and target variable (y)
X = data.drop(['id', 'cardio'], axis=1)
y = data['cardio']

# Standardize the numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
random_forest_model = RandomForestClassifier()
random_forest_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = random_forest_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the evaluation metrics
print(f'Accuracy: {accuracy:.4f}')
print('\nConfusion Matrix:')
print(conf_matrix)
print('\nClassification Report:')
print(classification_rep)

# Save the trained model for future use
import joblib
joblib.dump(random_forest_model, 'heart_disease_model.pkl')