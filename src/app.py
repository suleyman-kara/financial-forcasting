import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

from mrmr import mrmr_regression
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Integer

st.set_page_config(layout="wide")

# State initialization
if "config" not in st.session_state:
    st.session_state.config = {
        "train_df": None,
        "test_df": None,
        "feature_list": [],
        "feature_count": 0,
        "selected_target": None,
        "feature_sel_algo_list": ["MRMR", "RFE", "SFS"],
        "feature_eng_list": ["Year", "Month", "Week of Year", "Quarter of Year", "Target Lag 1", "Target Mean Last Month"],
        "tuning_method": "Manual",
        "xgboost": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6, "random_state": 42, "n_jobs": -1},
        "bo": {"n_estimators": (50, 200), "max_depth": (3, 8), "learning_rate": (0.01, 0.30), "subsample": (0.5, 1.0), "n_iter": 10},
        "gs": {"n_estimators": [50, 100, 150], "max_depth": [3, 6, 9], "learning_rate": [0.01, 0.1, 0.2]},
        "results": None,
        "best_per_algo": None,
        "best_predictions": {},
        "test_data_processed": None
    }

# Helper functions
def apply_feature_engineering(df, target_col, selected_eng_features):
    df = df.copy()
    
    date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
    if date_cols:
        date_col = date_cols[0]
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=date_col).reset_index(drop=True)
        
        if "Year" in selected_eng_features:
            df['Year'] = df[date_col].dt.year
        if "Month" in selected_eng_features:
            df['Month'] = df[date_col].dt.month
        if "Week of Year" in selected_eng_features:
            df['Week_of_Year'] = df[date_col].dt.isocalendar().week.astype(int)
        if "Quarter of Year" in selected_eng_features:
            df['Quarter_of_Year'] = df[date_col].dt.quarter

    if target_col in df.columns:
        if "Target Lag 1" in selected_eng_features:
            df['Target_Lag_1'] = df[target_col].shift(1).fillna(0)
        if "Target Mean Last Month" in selected_eng_features:
            df['Target_Mean_Last_Month'] = df[target_col].shift(1).rolling(window=4, min_periods=1).mean().fillna(0).round(2)
            
    return df

def get_optimized_model(X_train, y_train):
    cfg = st.session_state.config
    method = cfg["tuning_method"]
    base_params = {"random_state": 42, "n_jobs": -1}

    if method == "Manual":
        xgb_p = cfg["xgboost"]
        return xgb.XGBRegressor(
            n_estimators=int(xgb_p["n_estimators"]),
            learning_rate=float(xgb_p["learning_rate"]),
            max_depth=int(xgb_p["max_depth"]),
            **base_params
        )

    elif method == "Grid Search":
        gs_p = cfg["gs"]
        grid = {
            'n_estimators': gs_p["n_estimators"],
            'max_depth': gs_p["max_depth"],
            'learning_rate': gs_p["learning_rate"]
        }
        model = xgb.XGBRegressor(**base_params)
        gs_search = GridSearchCV(estimator=model, param_grid=grid, scoring='neg_mean_squared_error', cv=3)
        gs_search.fit(X_train, y_train)
        st.info(f"GridSearch Best Params: {gs_search.best_params_}")
        return gs_search.best_estimator_

    elif method == "Bayesian Optimization":
        bo_p = cfg["bo"]
        search_spaces = {
            'n_estimators': Integer(bo_p["n_estimators"][0], bo_p["n_estimators"][1]),
            'max_depth': Integer(bo_p["max_depth"][0], bo_p["max_depth"][1]),
            'learning_rate': Real(bo_p["learning_rate"][0], bo_p["learning_rate"][1], prior='uniform'),
            'subsample': Real(bo_p["subsample"][0], bo_p["subsample"][1], prior='uniform')
        }
        model = xgb.XGBRegressor(**base_params)
        bayes_search = BayesSearchCV(
            estimator=model,
            search_spaces=search_spaces,
            n_iter=int(bo_p["n_iter"]),
            cv=3,
            scoring='neg_mean_squared_error',
            random_state=42
        )
        bayes_search.fit(X_train, y_train)
        st.info(f"BayesSearchCV Best Params: {bayes_search.best_params_}")
        return bayes_search.best_estimator_

def run_pipeline():
    cfg = st.session_state.config
    
    # Data preprocessing
    df = cfg["train_df"].copy()
    target_col = cfg["selected_target"]
    
    df = apply_feature_engineering(df, target_col, cfg["approved_feature_eng_list"])
    
    if cfg["test_df"] is not None:
        test_df = apply_feature_engineering(cfg["test_df"], target_col, cfg["approved_feature_eng_list"])
    else:
        split_pt = int(len(df) * 0.9)
        test_df = df.iloc[split_pt:].reset_index(drop=True)
        df = df.iloc[:split_pt].reset_index(drop=True)
        
    cfg["test_data_processed"] = test_df
    
    y_train = df[target_col]
    X_train = df.drop(columns=[target_col] + [c for c in df.columns if 'date' in c.lower()], errors='ignore').select_dtypes(include=[np.number])
    
    y_test = test_df[target_col]
    X_test = test_df.drop(columns=[target_col] + [c for c in test_df.columns if 'date' in c.lower()], errors='ignore').select_dtypes(include=[np.number])

    # Model instantiation
    main_model = get_optimized_model(X_train, y_train)
    
    # Feature limits
    f_min = cfg.get("feature_lower_limit", 1)
    f_max = min(cfg.get("feature_upper_limit", len(X_train.columns)), len(X_train.columns))
    f_min = max(1, min(f_min, f_max))

    # Feature selection algorithms
    methods = {}
    selected_algos = cfg["approved_feature_sel_algo_list"]
    
    if "MRMR" in selected_algos:
        methods["MRMR"] = mrmr_regression(X=X_train, y=y_train, K=f_max, show_progress=False)
    if "RFE" in selected_algos:
        rfe_sel = RFE(estimator=main_model, n_features_to_select=1).fit(X_train, y_train)
        rfe_ranking = sorted(zip(rfe_sel.ranking_, X_train.columns))
        methods["RFE"] = [col for rank, col in rfe_ranking]
    if "SFS" in selected_algos:
        sfs_sel = SequentialFeatureSelector(estimator=main_model, n_features_to_select=f_max, direction='forward').fit(X_train, y_train)
        methods["SFS"] = X_train.columns[sfs_sel.get_support()].tolist()

    # Evaluation loop
    all_results = []
    best_predictions = {}
    best_mape_tracker = {method: float('inf') for method in methods.keys()}
    
    for k in range(f_min, f_max + 1):
        for name, full_col_list in methods.items():
            active_cols = full_col_list[:k]
            
            main_model.fit(X_train[active_cols], y_train)
            preds = main_model.predict(X_test[active_cols])
            
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            mape = mean_absolute_percentage_error(y_test, preds) * 100
            
            all_results.append({"Method": name, "K": k, "RMSE": rmse, "MAPE (%)": mape})
            
            if mape < best_mape_tracker[name]:
                best_mape_tracker[name] = mape
                best_predictions[name] = preds

    # Baseline evaluation
    main_model.fit(X_train, y_train)
    preds_all = main_model.predict(X_test)
    rmse_all = np.sqrt(mean_squared_error(y_test, preds_all))
    mape_all = mean_absolute_percentage_error(y_test, preds_all) * 100
    
    best_predictions["All Features"] = preds_all
    all_results.append({"Method": "All Features", "K": len(X_train.columns), "RMSE": rmse_all, "MAPE (%)": mape_all})
    
    exp_df = pd.DataFrame(all_results).round(2)
    best_per_algo = exp_df.loc[exp_df.groupby("Method")["MAPE (%)"].idxmin()].sort_values(by="MAPE (%)").reset_index(drop=True)
    
    cfg["results"] = exp_df
    cfg["best_per_algo"] = best_per_algo
    cfg["best_predictions"] = best_predictions

# UI Components
def load_data():
    st.header("Load Data")
    col1, col2 = st.columns(2)
    with col1:
        train_file = st.file_uploader("Upload train data", type=["csv", "xlsx"])
        if train_file:
            st.session_state.config["train_df"] = pd.read_csv(train_file) if train_file.name.endswith(".csv") else pd.read_excel(train_file)
            st.session_state.config["feature_list"] = list(st.session_state.config["train_df"].columns)
            st.session_state.config["feature_count"] = len(st.session_state.config["feature_list"])
    with col2:
        test_file = st.file_uploader("Upload test data", type=["csv", "xlsx"])
        if test_file:
            st.session_state.config["test_df"] = pd.read_csv(test_file) if test_file.name.endswith(".csv") else pd.read_excel(test_file)

    if st.session_state.config["feature_list"]:
        st.session_state.config["selected_target"] = st.selectbox("Please select target", st.session_state.config["feature_list"])

def feature_selection():
    st.header("Feature Selection")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("Feature count: ", st.session_state.config["feature_count"])
    with col2:
        st.session_state.config["feature_lower_limit"] = st.number_input("Feature lower limit", value=2, step=1)
    with col3:
        st.session_state.config["feature_upper_limit"] = st.number_input("Feature upper limit", value=max(2, st.session_state.config["feature_count"]), step=1)
    
    st.session_state.config["approved_feature_sel_algo_list"] = st.multiselect(
        "Feature Selection Algorithms", st.session_state.config["feature_sel_algo_list"], default=st.session_state.config["feature_sel_algo_list"]
    )
    st.session_state.config["approved_feature_eng_list"] = st.multiselect(
        "Feature Engineering", st.session_state.config["feature_eng_list"], default=st.session_state.config["feature_eng_list"]
    )

def model_tuning_settings():
    st.header("Model & Hyperparameter Tuning")
    
    method = st.radio("Tuning Method", ["Manual", "Grid Search", "Bayesian Optimization"], horizontal=True)
    st.session_state.config["tuning_method"] = method

    if method == "Manual":
        col1, col2, col3 = st.columns(3)
        xgb_c = st.session_state.config["xgboost"]
        with col1:
            xgb_c["n_estimators"] = st.number_input("n_estimators", value=xgb_c["n_estimators"], step=10)
        with col2:
            xgb_c["learning_rate"] = st.number_input("learning_rate", value=xgb_c["learning_rate"], step=0.01)
        with col3:
            xgb_c["max_depth"] = st.number_input("max_depth", value=xgb_c["max_depth"], step=1)

    elif method == "Grid Search":
        gs = st.session_state.config["gs"]
        col1, col2, col3 = st.columns(3)
        with col1:
            n_est = st.text_input("n_estimators (comma separated)", value="50, 100, 150")
            gs["n_estimators"] = [int(x.strip()) for x in n_est.split(",") if x.strip().isdigit()]
        with col2:
            depths = st.text_input("max_depth (comma separated)", value="3, 6, 9")
            gs["max_depth"] = [int(x.strip()) for x in depths.split(",") if x.strip().isdigit()]
        with col3:
            lrs = st.text_input("learning_rate (comma separated)", value="0.01, 0.1, 0.2")
            gs["learning_rate"] = [float(x.strip()) for x in lrs.split(",") if x.strip()]

    elif method == "Bayesian Optimization":
        bo = st.session_state.config["bo"]
        col1, col2 = st.columns(2)
        with col1:
            bo["n_estimators"] = st.slider("n_estimators range", 10, 500, bo["n_estimators"])
            bo["max_depth"] = st.slider("max_depth range", 1, 15, bo["max_depth"])
        with col2:
            bo["learning_rate"] = st.slider("learning_rate range", 0.001, 0.5, bo["learning_rate"])
            bo["subsample"] = st.slider("subsample range", 0.1, 1.0, bo["subsample"])
            bo["n_iter"] = st.number_input("Search Iterations", value=bo["n_iter"], min_value=1, step=5)

def results_view():
    st.header("Results & Actions")
    st.session_state.config["show_table"] = st.checkbox("Show results as a table", value=True)
    st.session_state.config["show_individual_charts"] = st.checkbox("Show individual charts")

# App layout
st.title("Financial Forecasting Pipeline")
col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        load_data()
    with st.container(border=True):
        model_tuning_settings()

with col2:
    with st.container(border=True):
        feature_selection()
    with st.container(border=True):
        results_view()
        if st.button("Run Pipeline", type="primary", use_container_width=True):
            if st.session_state.config["train_df"] is None:
                st.error("Please upload train data first!")
            else:
                with st.spinner("Processing features and training models..."):
                    run_pipeline()
                    st.success("Execution Completed!")

# Output rendering
if st.session_state.config["results"] is not None:
    st.header("📊 Model Results")
    
    if st.session_state.config.get("show_table"):
        st.subheader("Best Performance Per Algorithm")
        st.dataframe(st.session_state.config["best_per_algo"], use_container_width=True)
        
        with st.expander("Show All Iterations"):
            st.dataframe(st.session_state.config["results"], use_container_width=True)

    if st.session_state.config.get("show_individual_charts"):
        st.subheader("Prediction Comparison")
        preds_dict = st.session_state.config["best_predictions"]
        
        fig, ax = plt.subplots(figsize=(10, 4))
        for m_name, p_vals in preds_dict.items():
            ax.plot(p_vals, label=m_name, linestyle='--')
        ax.legend()
        st.pyplot(fig)