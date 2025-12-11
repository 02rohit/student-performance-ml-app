import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

st.title("Student Performance ML App")

df = pd.read_csv("../Datasets/student_performance.csv")
st.dataframe(df.head())

le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

columns = df.columns
target_cols = ['final_grade', 'pass_fail', 'admission_status']

target_col = st.selectbox("Select Target Variable", target_cols)

x_features_col = []
for col in columns:
    if col not in target_cols:
        x_features_col.append(col)

x_features_cols = st.multiselect('Select multiple X features', x_features_col)

xfeatures = df[x_features_cols]
ytarget = df[[target_col]]

if target_col == 'final_grade':
    st.write('Linear Regression will be appiled...')
    scaling_required = False
else:
    algo_name = st.selectbox("Select Algorithm",['KNN', 'Decision Tree', 'Logistic Regression', 'Random Forest'])

is_btn_click = st.button('Train Model')

if is_btn_click:
    if len(x_features_cols) > 0:

        # Model Selection
        if target_col == 'final_grade':
            model = LinearRegression()
        else:
            if algo_name == 'KNN':
                model = KNeighborsClassifier(n_neighbors=5)
                scaling_required = True
            elif algo_name == 'Logistic Regression':
                model = LogisticRegression()
                scaling_required = True
            elif algo_name == 'Decision Tree':
                model = DecisionTreeClassifier()
                scaling_required = False
            elif algo_name == 'Random Forest':
                model = RandomForestClassifier(n_estimators=100)
                scaling_required = False

        # Train-test split
        xtrain, xtest, ytrain, ytest = train_test_split( xfeatures, ytarget, train_size=0.8, random_state=5)

        if scaling_required:
            scaler = StandardScaler()
            xtrain = scaler.fit_transform(xtrain)
            xtest = scaler.transform(xtest)
        
        # Train model
        model.fit(xtrain, ytrain)
        st.write("Model Trained..")

        # Prediction
        y_pred_test = model.predict(xtest)
        y_pred_train = model.predict(xtrain)

        # Evaluation
        if target_col == 'final_grade':  
            acc_train = round(model.score(xtrain, ytrain) * 100)
            acc_test = round(model.score(xtest, ytest) * 100)

            st.success(f"Train Accuracy : {acc_train}")
            st.success(f"Test Accuracy : {acc_test}")

        else:
            acc_train = round(accuracy_score(ytrain, y_pred_train) * 100)
            acc_test = round(accuracy_score(ytest, y_pred_test) * 100)

            st.success(f"Train Accuracy : {acc_train}")
            st.success(f"Test Accuracy : {acc_test}")

    else:
        st.error("Please Select atleast one X-Feature")
