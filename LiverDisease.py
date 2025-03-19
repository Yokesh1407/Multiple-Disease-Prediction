# -*- coding: utf-8 -*-

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from scipy.stats import shapiro


#df = pd.read_csv('/Users/yokesh/Documents/Datascience/Project/Mulitple disease/indian_liver_patient - indian_liver_patient.csv')



def EDA_data(df):
    df.isnull().sum()
    #EDA
    #filling Null values

    #If data is normally distributed → Mean
    #If data has outliers or skewness → Median
    #If time-series data → Forward/Backward Fill
    #If a trend is present → Interpolation
    
    df.describe()

    #we have outliers so we are using median since the data is numerical
    df['Albumin_and_Globulin_Ratio'].fillna(df['Albumin_and_Globulin_Ratio'].median(),inplace = True)
    df['Gender']=df['Gender'].astype("category") #changing object to category

    df.info()

    df.columns

    return df

def Distribution(df):

    column = "Albumin_and_Globulin_Ratio"
    stat, p = shapiro(df[column].dropna())
    st.write("### SHAPIRO")
    st.write(f"Shapiro-Wilk Test: p-value = {p}")

    if p > 0.05:
        st.write("Likely Normally Distributed (Fail to reject H0)")
    else:
        st.write("Not Normally Distributed (Reject H0)")


    #checking outlier or skewness
    from scipy.stats import skew

    column = "Albumin_and_Globulin_Ratio"  # Change to your target column
    skewness_value = skew(df[column].dropna())
    st.write("### SKEWNESS")
    st.write(f"Skewness of {column}: {skewness_value}")
    
    if skewness_value > 1:
        st.write("Highly Positively Skewed (Right-skewed)")
    elif skewness_value < -1:
        st.write("Highly Negatively Skewed (Left-skewed)")
    elif -1 <= skewness_value <= 1:
        st.write("Approximately Symmetrical")

    st.write("### IQR")
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    st.write(f"Number of Outliers in {column}: {outliers.shape[0]}")
    #For skewness: Use histograms, skewness formula, and Q-Q plots.
    #For outliers: Use IQR method, boxplots, and z-score analysis.


    return df


def plots(df):
    column = "Albumin_and_Globulin_Ratio"

    st.write("### ProbPlot for Classification")
    plt.figure(figsize=(6,4))
    stats.probplot(df[column].dropna(), dist="norm", plot=plt)
    plt.title(f"Q-Q Plot of {column}")
    st.pyplot(plt)

    st.write("### Histogram for Classification")
    plt.figure(figsize=(6,4))
    sns.histplot(df[column], kde=True, bins=30)
    plt.title(f"Histogram of {column}")
    st.pyplot(plt)

    st.write("### Boxplot for Classification")
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[column])
    plt.title(f"Boxplot of {column}")
    st.pyplot(plt)


    class_counts = df["Dataset"].value_counts()

    # Print the class distribution
    print("Class Distribution:\n", class_counts)

    # Plot the class distribution
    plt.figure(figsize=(6, 4))
    sns.barplot(x=class_counts.index, y=class_counts.values, palette="viridis")
    plt.xlabel("Class Label")
    plt.ylabel("Count")
    plt.title("Class Distribution (Balanced vs. Imbalanced)")
    plt.xticks(ticks=[0, 1], labels=["Class 0", "Class 1"])
    st.pyplot(plt)

    #univariate Analysis
    #Plot univariate or bivariate histograms to show distributions of datasets.
    #checking normal distribution or not
    #visualizing skewness
    
    #column = "Albumin_and_Globulin_Ratio"  # Change to your target column
    #sns.histplot(df[column], kde=True, bins=30)
    #plt.title(f"Histogram of {column}")
    #plt.show()

    #Change to your target column
    st.write("### Histogram for Dataset")
    plt.figure(figsize=(6, 4))
    sns.histplot(df['Dataset'], kde=True, bins=30)
    plt.title(f"Histogram of {'Dataset'}")
    st.pyplot(plt)

    #st.write("### Histogram for Dataset")
    #plt.figure(figsize=(6, 4))
    #sns.histplot(df['Dataset'], kde=True, bins=30)
    #plt.title(f"Histogram of {'Dataset'}")
    #st.pylot(plt)
    
    st.write("### Scatterplot")
    plt.figure(figsize=(6, 4))
    sns.scatterplot(df)
    st.pyplot(plt)
    

    ##Categorical
    st.write("### Countplot for Gender")
    plt.figure(figsize=(6, 4))
    sns.countplot(df['Gender'])
    st.pyplot(plt)

    #Categorical
    #Draw a patch representing a KDE and add observations or box plot statistics.
    st.write("### ViolinPlot for Gender")
    plt.figure(figsize=(6, 4))
    sns.violinplot(df["Gender"])
    st.pyplot(plt)


    st.write("### Barplot for Gender")
    plt.figure(figsize=(6, 4))
    sns.barplot(data=df,x='Gender',y='Total_Bilirubin')
    st.pyplot(plt)

    #sns.boxplot(x="Gender", y="Total_Bilirubin", data=df)
    #plt.title("Boxplot of Total Bilirubin by Gender")
    #plt.show()

    #lmplot in Seaborn is used to plot a linear regression model between two numerical variables
    st.write("### LMPLOT for Gender")
    plt.figure(figsize=(6, 4))
    sns.lmplot(x="Total_Bilirubin", y="Direct_Bilirubin", hue="Gender", data=df)
    plt.title("Regression Plot by Gender")
    st.pyplot(plt)

    #regplot is used to visualize the relationship between two numerical variables with a regression line. It's similar to lmplot, but it works directly within matplotlib subplots, making it more flexible for customization.
    st.write("### REPLOT for Gender")
    plt.figure(figsize=(6, 4))
    sns.regplot(x="Total_Bilirubin", y="Direct_Bilirubin", data=df)
    plt.title("Regression Plot")
    st.pyplot(plt)

    return df


def Encoded_data(df):
    #ENCODING THE CATEGORICAL COLUMN
    label_encoder = LabelEncoder()
    df["Gender"] = label_encoder.fit_transform(df["Gender"])


    #One-Hot Encoding Gender
    #df = pd.get_dummies(df, columns=["Gender"], prefix=["Gender"])
    #df = pd.get_dummies(df, columns=["Gender"], drop_first=True)
    #df = df.rename(columns={'Gender_Male': 'Gender'})

    return df

st.title("Liver Disease Classification")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Liver Disease Classification")
    st.write(df.head())
    st.write(df.shape)

    df = EDA_data(df)
    st.write("### PreProcessed Data")
    st.write(df)

    df = Distribution(df)
    st.write("### Distribution Data")
    st.write(df)

    plots(df)
    

    df=Encoded_data(df)
    st.write("### Encoded Data")
    st.write(df)


    df.to_csv("liver_output.csv", index = False)

    #After Encoding split target and feature.

    feature = df.drop(columns =['Dataset'])
    st.write("### Feature Column")
    st.table(feature.head())
    target = df['Dataset']
    st.write("### Target Column")
    st.table(target.head())

    #SMOTE may also refer to Synthetic Minority Oversampling Technique, a statistical technique used in machine learning to balance imbalanced datasets
    st.title("SMOTE TECHNIQUE")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(feature,target)
    st.write("Using SMOTE TECHNIQUE for balancing Data")

    X_train, X_test, y_train, y_test = train_test_split(X_train_resampled,y_train_resampled,test_size=0.2,random_state=42)

    #test_size=0.2 (20% test, 80% train), test_size=0.1 (90% train, 10% test),test_size=0.3 (70% train, 30% test)
    #random_state=42

    #Feature Scaling:(UNSUPERVISED LEARNING)
    #Use Standardization (StandardScaler()) for most ML models(When dealing with normally distributed data (SVM, Logistic Regression).
    #Use Min-Max Scaling (MinMaxScaler()) for Neural Networks(When data is bounded (Neural Networks, KNN)).
    #No need for scaling if using tree-based models like Decision Trees or Random Forest.

    #scaler= StandardScaler()
    #X_train_scaled = scaler.fit_transform(X_train)
    #X_test_scaled = scaler.transform(X_test)

    #since data is imbalanced using smote to balance

    #print("After SMOTE Class Distribution:\n", y_train_resampled.value_counts())

    #target column is classifier which is binary( logistic regression)
    st.title("Machine learning Model")
    Model = LogisticRegression()
    st.write(Model)

    Model.fit(X_train,y_train)

    predict = Model.predict(X_test)

    accuracy = accuracy_score(y_test, predict)
    st.write(f"Accuracy: {accuracy:.4f}")
    

    cm = confusion_matrix(y_test, predict) # 52 - TN , 73 - TP ,23- FP, 19-FN
    st.write("Confusion Matrix:")
    st.write(cm)

    report = classification_report(y_test, predict)
    st.write("Classification Report:")
    st.write(report)
    st.write(f"Since accuracy is underfitting {accuracy:.4f} in LogisticRegression we are using next model")

    #Accuracy: Percentage of correctly classified samples out of all samples.
    #For example, an accuracy of 0.85 means 85% of the test samples were correctly classified.

    #Confusion Matrix: Shows True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).
    #Helps to understand the types of errors your model is making.

    #Classification Report: Provides detailed metrics including:

    #Precision: How many of the positive predictions were correct.

    #Recall: How many actual positives were identified correctly.

    #F1-Score: Harmonic mean of Precision and Recall.


    model = RandomForestClassifier(n_estimators=100, random_state=42)
    st.write(model)
    model.fit(X_train, y_train)
    y_pred_rf = model.predict(X_test)

    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    st.write(f"Random Forest Accuracy: {accuracy_rf:.4f}")

    y_train.unique()

    st.write("### Future Prediction")
    new_data = pd.DataFrame({
        'Age': [38], 'Gender': ["Male"], 'Total_Bilirubin': [1.8],
        'Direct_Bilirubin': [0.8], 'Alkaline_Phosphotase': [342], 'Alamine_Aminotransferase': [168],
        'Aspartate_Aminotransferase': [441], 'Total_Protiens': [7.6], 'Albumin': [4.4],
        'Albumin_and_Globulin_Ratio': [1.3], 'Dataset': [1]
    })

    new_data = new_data.drop(columns=['Dataset'], errors='ignore')


    # Ensure categorical features are correctly encoded (if necessary)
    # Example: If 'Gender' needs encoding
    new_data['Gender'] = new_data['Gender'].map({'Male': 0, 'Female': 1})  # Adjust based on training data

    future_prediction = model.predict(new_data)

    st.write("Future Prediction:", future_prediction[0])