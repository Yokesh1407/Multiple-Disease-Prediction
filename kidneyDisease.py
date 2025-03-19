# -*- coding: utf-8 -*-


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

    
def EDA_data(df):
    df.info()

    df.describe()

    df.isnull().sum() #finding null values

    #replacing the "?" data from dataset
    df['pcv'].replace("?",pd.NA,inplace=True)
    df['wc'].replace("?",pd.NA,inplace=True)
    df['rc'].replace("?",pd.NA,inplace=True)

    df.isnull().sum()


    #converting object datatypes to numerical
    df['pcv'] = pd.to_numeric(df['pcv'], errors='coerce')
    df['wc']  = pd.to_numeric(df['wc'], errors='coerce')
    df['rc']  = pd.to_numeric(df['rc'], errors='coerce')

    #we have outliers so we are using median since the data is numerical
    #age,bp,su,bgr,bu,sc,sod,pot,hemo has outliers , #sg,al,pcv,wc,rc has no outlier

    df['age'].fillna(df['age'].median(),inplace = True)
    df['bp'].fillna(df['bp'].median(),inplace= True)
    df['sg'].fillna(df['sg'].mean(),inplace= True) #no outlier
    df['al'].fillna(df['al'].mean(),inplace= True) #no outlier
    df['su'].fillna(df['su'].median(),inplace= True)
    df['bgr'].fillna(df['bgr'].median(),inplace= True)
    df['bu'].fillna(df['bu'].median(),inplace= True)
    df['sc'].fillna(df['sc'].median(),inplace= True)
    df['sod'].fillna(df['sod'].median(),inplace= True)
    df['pot'].fillna(df['pot'].median(),inplace= True)
    df['hemo'].fillna(df['hemo'].median(),inplace= True)
    df['pcv'].fillna(df['pcv'].mean(),inplace=True) #no outlier
    df['wc'].fillna(df['wc'].mean(),inplace= True) #no outlier
    df['rc'].fillna(df['rc'].mean(),inplace= True) #no outlier

    #changing the incorrect datatypes
    df['age'] = df['age'].astype('int')
    df['bp'] = df['bp'].astype('int')
    df['al'] = df['al'].astype('int')
    df['su'] = df['su'].astype('int')
    df['bgr'] = df['bgr'].astype('int')
    df['bu'] = df['bu'].astype('int')
    df['pcv'] = df['pcv'].astype('int')
    df['wc']=df['wc'].astype('int')
    df['rc']=df['rc'].astype('float')

    df.info()

    df['cad'].value_counts() #checking data's in categorical column

    #filling categorical data's using mode
    #mode() returns a Series (a type of pandas DataFrame), not a single value. This is because there can technically be multiple modes in a dataset. By indexing with [0], you are selecting the first mode in the dataset

    df['rbc'].fillna(df['rbc'].mode()[0],inplace=True) #By indexing with [0], you are selecting the first mode in the dataset.
    df['pc'].fillna(df['pc'].mode()[0],inplace = True)
    df['pcc'].fillna(df['pcc'].mode()[0],inplace = True)
    df['ba'].fillna(df['ba'].mode()[0],inplace = True)
    df['htn'].fillna(df['htn'].mode()[0],inplace = True)
    df['dm'].fillna(df['dm'].mode()[0],inplace = True)
    df['cad'].fillna(df['cad'].mode()[0],inplace = True)
    df['appet'].fillna(df['appet'].mode()[0],inplace = True)
    df['pe'].fillna(df['pe'].mode()[0],inplace = True)
    df['ane'].fillna(df['ane'].mode()[0],inplace = True)

    df.isnull().sum()

    #converting object into category datatypes
    df['rbc']=df['rbc'].astype('category')
    df['pc']=df['pc'].astype('category')
    df['pcc']=df['pcc'].astype('category')
    df['ba']=df['ba'].astype('category')
    df['htn']=df['htn'].astype('category')
    df['dm']=df['dm'].astype('category')
    df['cad']=df['cad'].astype('category')
    df['appet']=df['appet'].astype('category')
    df['pe']=df['pe'].astype('category')
    df['ane']=df['ane'].astype('category')
    df['classification']=df['classification'].astype('category')

    df.info()

    return df
    
def plots(df):
    st.subheader("Data Visualizations")

    st.write("### Histogram")
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x='classification', kde=False, color="blue")
    st.pyplot(plt)

    class_counts = df["classification"].value_counts()


    st.write("### Boxplot for Age")
    plt.figure(figsize=(6, 4))
    sns.boxplot(df['age']) #age,bp,su,bgr,bu,sc,sod,pot,hemo has outliers , #sg,al,pcv,wc,rc has no outlier
    #Replace with Mode	Simple, when a dominant category exists
    st.pyplot(plt)
    
    st.write("### Boxplot for Classification")
    plt.figure(figsize=(6, 4))
    #finding target column balance / imbalance
    sns.barplot(df['classification'])
    st.pyplot(plt)


    st.write("### Violinplot for Classification")
    plt.figure(figsize=(6, 4))
    sns.violinplot(df['classification'])
    st.pyplot(plt)

    st.write("### Boxplot for Classification")
    plt.figure(figsize=(6, 4))  
    sns.boxplot(df['classification'])
    st.pyplot(plt)


    st.write("### Scatterplot for Classification")
    plt.figure(figsize=(6, 4))
    sns.scatterplot(df['classification'])
    st.pyplot(plt)

    #corr = df.select_dtypes('int64','float64').corr() #finding correlation

    sns.barplot(df)

# Print the class distribution
    print("Class Distribution:\n", class_counts)

# Plot the class distribution
    plt.figure(figsize=(6, 4))
    sns.barplot(x=class_counts.index, y=class_counts.values, palette="viridis")
    plt.xlabel("Class Label")
    plt.ylabel("Count")
    plt.title("Class Distribution (Balanced vs. Imbalanced)")
    plt.show()
    st.pyplot(plt)

    st.write("### Correlation Heatmap")
    numeric_df = df.select_dtypes(include=['int64','float64'])  # Select only numeric columns
    corr_matrix = numeric_df.corr()

    if corr_matrix.isnull().values.any():
        st.write("Correlation matrix contains NaN values. Please check the dataset.")
    else:
        plt.figure(figsize=(15, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        st.pyplot(plt)


    #st.write("### Correlation Heatmap")
    #plt.figure(figsize=(15, 6))
    #sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    #st.pyplot(plt)

    return df

#encoding categorical columns
def encoding_process(df):
    label_encoder = LabelEncoder()
    df['rbc'] = label_encoder.fit_transform(df['rbc'])
    df['pc'] = label_encoder.fit_transform(df['pc'])
    df['pcc'] = label_encoder.fit_transform(df['pcc'])
    df['ba'] = label_encoder.fit_transform(df['ba'])
    df['htn'] = label_encoder.fit_transform(df['htn'])
    df['dm'] = label_encoder.fit_transform(df['dm'])
    df['cad']=label_encoder.fit_transform(df['cad'])
    df['appet']=label_encoder.fit_transform(df['appet'])
    df['pe']=label_encoder.fit_transform(df['pe'])
    df['ane']=label_encoder.fit_transform(df['ane'])

    return df

st.title("Kidney Disease Classification")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Kidney Disease Classification")
    st.write(df.head())
    st.write(df.shape)

    df = EDA_data(df)
    st.write("### PreProcessed Data")
    st.write(df)

    df = encoding_process(df)
    st.write("### Enocoded_Data")
    st.write(df)
    
    plots(df)

    df.head()

    df.to_csv("kidney_output.csv", index = False)

#selecting feature and target

    feature = df.drop(columns = ['classification'])
    st.title("Feature Column")
    st.table(feature.head())
    target = df['classification']
    st.title("Target Column")
    st.table(target.head())

#SMOTE (Synthetic Minority Over-sampling Technique) is a technique used to handle class imbalance in datasets
#by generating synthetic samples for the minority class.

#smote = SMOTE(sampling_strategy='auto',random_state=42)
#X_train_resampled, y_train_resampled = smote.fit_resample(feature,target)

    X_train, X_test, y_train, y_test = train_test_split(feature,target,test_size=0.2,random_state=42)

#scaler= StandardScaler()
#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.transform(X_test)

    
    #pca = PCA(n_components=10)  # Keep only 10 important features
    #X_train_pca = pca.fit_transform(X_train)
    #X_test_pca = pca.transform(X_test)

    st.title("Machine learning Model")
    model = LogisticRegression()  #since the target column is binary so we are using logistic regression.
    st.write(model)

    model.fit(X_train,y_train)

    predict = model.predict(X_test)

    accuracy = accuracy_score(y_test,predict)
    st.write(f"Accuracy_Score: {accuracy:.4f}")

    cm = confusion_matrix(y_test, predict) # 46 - TN , 54 - TP ,0- FP, 0-FN
    st.write("Confusion Matrix:")
    st.write(cm)

    report = classification_report(y_test, predict)
    st.write("Classification Report:")
    st.write(report)

    st.write("### Future Prediction")
    new_data = pd.DataFrame({
        'id': [11], 'age': [63], 'bp': [70],
        'sg': [1.01], 'al': [3], 'su': [0],
        'rbc': [0], 'pc': [0], 'pcc': [1],
        'ba': [0], 'bgr': [380] ,'bu': [60],'sc':[2.7],'sod':[131],'pot':[4.2],'hemo':[10.8],'pcv':[32],
        'wc':[4500],'rc':[3.8],'htn':[1],'dm':[1],'cad':[0],'appet':[1],'pe':[1],'ane':[0],'classification':["ckd"]
    })

    new_data = new_data.drop(columns=['classification'], errors='ignore')


    # Ensure categorical features are correctly encoded (if necessary)


    future_prediction = model.predict(new_data)

    st.write("Future Prediction:", future_prediction[0])
