# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st
from scipy.stats import kurtosis,skew
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from scipy.stats import kurtosis, skew

#df = pd.read_csv('/Users/yokesh/Documents/Datascience/Project/Mulitple disease/parkinsons - parkinsons.csv')

#df.head()

#df.info()

#df["status"].value_counts()

#df.shape

#df.columns

def EDA_Data(df):
    df.isnull().sum()

    df['name'].value_counts() #

    df["name"]= df["name"].astype("category") #if we are not using

    df.info()

    #analysis

    df.describe()

    return df


def plots(df):

    st.write("### DistPlot for MDVP:Fo(Hz)")
    plt.figure(figsize=(6,4))
    sns.displot(df["MDVP:Fo(Hz)"])
    st.pyplot(plt)

    st.write("### Histplot for spread2")
    plt.figure(figsize=(6,4))
    sns.histplot(df['spread2'])
    st.pyplot(plt)

    st.write("### Boxplot for Classification")
    plt.figure(figsize=(6,4))
    sns.boxplot(df)
    st.pyplot(plt)


    st.write("### Histplot for status")
    plt.figure(figsize=(6,4))
    sns.histplot(df["status"], kde=True, bins=30)
    plt.title(f"Histogram of status")
    plt.show()
    st.pyplot(plt)

    st.write("### Boxplot for MDVP:Flo(Hz)")
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df["MDVP:Flo(Hz)"])
    st.pyplot(plt)

    st.write("### Boxplot for MDVP:Fhi(Hz)")
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df["MDVP:Fhi(Hz)"])
    st.pyplot(plt)

    st.write("### Boxplot for Status")
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df["status"])
    st.pyplot(plt)

    numerical_df = df.select_dtypes(include=['int64', 'float64'])

    st.write("### Heatmap for classification")
    plt.figure(figsize=(6,4))
    sns.heatmap(numerical_df.corr(), cmap="coolwarm", annot=False)
    st.pyplot(plt)

    st.write("### Scatterplot for classification")
    plt.figure(figsize=(6,4))
    sns.scatterplot(df)
    st.pyplot(plt)


    #st.write("### KDEplot of status & MDVP:Flo(Hz)")
    #plt.figure(figsize=(6,4))
    #sns.kdeplot(x=df["status"],y=df["MDVP:Flo(Hz)"], data=df, palette="coolwarm")
    #st.pyplot(plt)

    #st.write("### KDEplot for status & MDVP:Flo(Hz)")
    #plt.figure(figsize=(6,4))
    #sns.kdeplot(x="status", y="MDVP:Fo(Hz)", data=df, palette="coolwarm")
    #st.pyplot(plt)

    #st.write("### LMplot of MDVP:Fo(Hz) & status")
    #plt.figure(figsize=(6,4))
    #sns.lmplot(x="status", y="MDVP:Fo(Hz)", data=df)
    #plt.title("Regression Plot")
    #plt.show()
    #st.pyplot(plt)

        
    #sns.boxplot(x="status", y="MDVP:Fo(Hz)")
    #plt.figure(figsize=(6,4))
    #plt.title("Boxplot of MDVP:Fo(Hz) by Status")
    #plt.xlabel("Status (0 = Healthy, 1 = Parkinson’s)")
    #plt.ylabel("MDVP:Fo(Hz)")
    #plt.show()
    #st.pyplot(plt)


    # Print the class distribution
    class_counts = df["status"].value_counts()
    st.write("Class Distribution:\n", class_counts)

    plt.figure(figsize=(6, 4))
    sns.barplot(x=class_counts.index, y=class_counts.values, palette="viridis")
    plt.xlabel("Class Label")
    plt.ylabel("Count")
    plt.title("Class Distribution (Balanced vs. Imbalanced)")
    plt.xticks(ticks=[0, 1], labels=["Class 0", "Class 1"])
    plt.show()
    st.pyplot(plt)


    #categorical_df

    return df


def Distribution(df):
#checking outlier or skewness

    numerical_df = df.select_dtypes(include=['int64', 'float64'])
    categorical_df = df.select_dtypes(include=['category'])

    st.write("### Skewness of Target column")
    column = "status"  # Change to your target column
    skewness_value = skew(df[column].dropna())
    st.write(f"Skewness of {column}: {skewness_value}")

    if skewness_value > 1:
        st.write("Highly Positively Skewed (Right-skewed)")
    elif skewness_value < -1:
        st.write("Highly Negatively Skewed (Left-skewed)")
    elif -1 <= skewness_value <= 1:
        st.write("Approximately Symmetrical")

    st.write("### Spearmanr Correlation")
    from scipy.stats import spearmanr
    stat, p = spearmanr(df['status'],df['MDVP:Fo(Hz)'])

    st.write('stat=%.3f,p=%5f' %(stat,p))
    if p>0.05:
        st.write("Independent samples")
    else:
        st.write("Dependent samples")


#p (p-value) tells us whether the correlation is statistically significant.

#If p > 0.05, we fail to reject the null hypothesis, meaning the variables are independent.

#If p ≤ 0.05, we reject the null hypothesis, meaning the variables are dependent (statistically significant relationship).
    # Compute skewness & kurtosis for each column
    skewness = skew(numerical_df)
    kurt = kurtosis(numerical_df)

    # Print results
    st.write("Skewness:\n", skewness)
    st.write("Kurtosis:\n", kurt)

    # Plot histograms for all numeric columns
    plt.style.use('ggplot')
    numerical_df.hist(figsize=(12, 8), bins=30, edgecolor='black')
    plt.suptitle("Feature Distributions")
    plt.show()
    st.pyplot(plt)

    return df

#Selects only numerical columns.
#Computes skewness (measure of asymmetry) and kurtosis (measure of tail heaviness).
#Plots histograms for better visualization.


st.title("Parkinsons Classification")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("###  Parkinsons Classification")
    st.write(df.head())
    st.write(df.shape)

    df = EDA_Data(df)
    st.write("### Preprocessed Data")
    st.write(df)

    df = Distribution(df)
    st.write("### Distribution Data")
    st.write(df)

    plots(df)

    #Recursive Feature Elimination (RFE)
    X = df.drop(columns=['status','name'])
    y = df['status']
    
    st.write("### Recursive Feature Elimination(RFE)")
    # Using RandomForestClassifier as the estimator
    model = RandomForestClassifier()
    rfe = RFE(model, n_features_to_select=10)  # Select top 10 features
    X_selected = rfe.fit_transform(X, y)

    #Recursive Feature Elimination (RFE) is a feature selection technique used in machine learning. 
    # It helps identify the most important features by recursively eliminating the least important ones.

    # Get selected feature names
    selected_features = X.columns[rfe.support_]
    st.write("Selected Features:", selected_features)

    #rfe.support_ returns a boolean mask indicating which features were selected (True) and which were eliminated (False).
    #X.columns[rfe.support_] extracts the names of the selected features.

    st.write("### Feature Importance")
    # Train a random forest model
    model = RandomForestClassifier()
    model.fit(X, y)

    # Get feature importance
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    st.write("Feature Importance:\n", importances)

    # Select top N important features
    top_features = importances.head(10).index
    df_selected = df[top_features].columns

    st.write("Feature Selected columns: ", df_selected)

    feature = df.drop(columns=['status','name'])
    st.write("### Feature Column")
    st.write(feature.head())
    st.write("### Target Column")
    target = df['status']
    st.write(target.head())

    # Assuming feature matrix and target variable
    #since data is imbalanced using smote to balance

    #SMOTE 
    st.write("### SMOTE Technique")
    smote = SMOTE(random_state=42)  # Initialize SMOTE
    st.write(smote)
    X_resampled, y_resampled = smote.fit_resample(feature,target)

    X_train,X_test,y_train,y_test = train_test_split(X_resampled,y_resampled,test_size=0.2,random_state=42)


    model = LogisticRegression()
    st.write("### Machine Learning Model")
    st.write(model)
    model.fit(X_train, y_train)

    predict = model.predict(X_test)

    accuracy = accuracy_score(y_test, predict) #Accuracy = (TP + TN) / (TP + TN + FP + FN)
    st.write(f"Accuracy: {accuracy:.4f}")

    report = classification_report(y_test, predict)
    st.write("Classification Report:")
    st.write(report)

    #future prediction

    # Assuming 'feature' contains the original features used for training
    # and you have new data in a variable called 'new_data'
    # 'new_data' should have the same 22 features as 'feature'

    # Create a sample DataFrame for new_data (replace with your actual data)
    # Ensure new_data has the same columns as your original feature data (22 features)

    st.write("### Future Prediction")
    new_data = pd.DataFrame({
        'MDVP:Fo(Hz)': [237.226], 'MDVP:Fhi(Hz)': [247.326], 'MDVP:Flo(Hz)': [225.227],
        'MDVP:Jitter(%)': [0.00298], 'MDVP:Jitter(Abs)': [0.00001], 'MDVP:RAP': [0.00169],
        'MDVP:PPQ': [0.00182], 'Jitter:DDP': [0.00507], 'MDVP:Shimmer': [0.01752],
        'MDVP:Shimmer(dB)': [0.164], 'Shimmer:APQ3': [0.01035], 'Shimmer:APQ5': [0.01024],
        'MDVP:APQ': [0.01133], 'Shimmer:DDA': [0.03104], 'NHR': [0.0074], 'HNR': [22.736],
        'status': [0], 'RPDE': [0.305062], 'DFA': [0.654172], 'spread1': [-7.31055],
        'spread2': [0.098648], 'D2': [2.416838], 'PPE': [0.095032]
    })

    #Remove the target variable (status) if present in your new_data:
    new_data = new_data.drop(columns=['status'], errors='ignore')


    future_prediction = model.predict(new_data)

    st.write("Future Prediction:", future_prediction[0])

    #future prediction

    # Assuming 'feature' contains the original features used for training
    # and you have new data in a variable called 'new_data'
    # 'new_data' should have the same 22 features as 'feature'

    future_prediction = model.predict(new_data)

    st.write("Future Prediction:", future_prediction[0])