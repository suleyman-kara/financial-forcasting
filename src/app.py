import streamlit as st
import pandas as pd
from skopt.space import Real, Integer

st.title("Financial Forecasting")

st.subheader("Load Data")
with st.container(border=True):
    train_file = st.file_uploader("Upload train.csv")
    stores_file = st.file_uploader("Upload stores.csv")
    features_file = st.file_uploader("Upload features.csv")
    
st.subheader("Data Preprocessing")
with st.container(border=True):
    store_choice = st.checkbox("Shrink all stores into one", value=True)
    store_id = None
    if not store_choice:
        store_id = st.number_input("Please enter the store number", min_value=1, max_value=45, value=1)
    
    dept_choice = st.checkbox("Shrink all departments into one", value=True)
    dept_id = None
    if not dept_choice:
        dept_id = st.number_input("Please enter the department number", min_value=1, max_value=98, value=1)
    
    features_list = [
        "Year", "Month", "Day", "Week of Year", "Quarter of Year",
        "Is Pre Holiday", "Sales Lag 1 Week", "Sales Lag 1 Year", "Sales Mean Last Month"
    ]
    selected_features = st.multiselect("Please select the features", features_list, default=features_list)
    split_months = st.number_input("Please enter split point for test data (month)", min_value=1, max_value=24, value=1)

st.subheader("Model Hyperparameters")
with st.container(border=True):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n_estimators = st.number_input("n_estimators", value=100)
    with col2:
        learning_rate = st.number_input("learning_rate", value=0.1)
    with col3:
        random_state = st.number_input("random_state", value=42)
    with col4:
        n_jobs = st.number_input("n_jobs", value=-1)

st.subheader("Feature Selection")
with st.container(border=True):
    col1, col2 = st.columns(2)
    algorithms = ["MRMR", "RFE", "SFS"]
    with col1:
        col11, col12 = st.columns(2)
        with col11:
            f_upper = st.number_input("Feature upper limit", value=12)
        with col12:
            f_lower = st.number_input("Feature lower limit", value=6)
    with col2:
        selected_algos = st.multiselect("Please select the algorithms", algorithms, default=algorithms)

st.subheader("Training & Validation")
with st.container(border=True):
    methods = ["RMSE", "MAPE"]
    st.multiselect("Please select validation methods", methods, default=methods)
    show_df = st.checkbox("Create data frame with results", value=True)
    show_charts = st.checkbox("Create individual chart for each feature selection algorithm", value=True)

st.subheader("Bayesian Optimisation")
with st.container(border=True):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        min_n, max_n = st.slider("n_estimator", 0, 300, (100, 250))
    with col2:
        min_d, max_d = st.slider("max_depth", 0, 10, (3, 8))
    with col3:
        min_lr, max_lr = st.slider("learning_rate", 0.0, 1.0, (0.01, 0.30))
    with col4:
        min_sub, max_sub = st.slider("subsample", 0.0, 3.0, (0.5, 1.0))

if st.button("Run Pipeline"):
    if train_file and stores_file and features_file:
        try:
            df_train = pd.read_csv(train_file)
            df_stores = pd.read_csv(stores_file)
            df_features = pd.read_csv(features_file)
            
            base_params = {
                "n_estimators": int(n_estimators),
                "learning_rate": float(learning_rate),
                "random_state": int(random_state),
                "n_jobs": int(n_jobs)
            }
            
            bayes_spaces = {
                'n_estimators': Integer(int(min_n), int(max_n)),
                'max_depth': Integer(int(min_d), int(max_d)),
                'learning_rate': Real(float(min_lr), float(max_lr), prior='uniform'),
                'subsample': Real(float(min_sub), min(float(max_sub), 1.0), prior='uniform')
            }
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.error("Please upload all files.")