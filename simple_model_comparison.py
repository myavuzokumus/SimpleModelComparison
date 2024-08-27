import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


# Missing data insertion function
def introduce_missing_data(df, missing_fraction=0.05):
    df_missing = df.copy()
    np.random.seed(0)
    for column in df_missing.columns:
        df_missing.loc[df_missing.sample(frac=missing_fraction).index, column] = np.nan
    return df_missing

# Function to fill in missing data
def impute_missing_data(df, strategy='mean', n_neighbors=4):
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if strategy == 'mean':
        imputer = SimpleImputer(strategy='mean')
    elif strategy == 'median':
        imputer = SimpleImputer(strategy='median')
    elif strategy == 'most_frequent':
        imputer = SimpleImputer(strategy='most_frequent')
    elif strategy == 'knn':
        imputer = KNNImputer(n_neighbors=n_neighbors)
    elif strategy == 'random_forest':
        imputer = IterativeImputer(estimator=RandomForestRegressor(), random_state=0)
    elif strategy == 'expectation–maximization-EM':
        imputer = IterativeImputer(max_iter=10, random_state=0)
    elif strategy == 'gradient_boosting':
        imputer = IterativeImputer(estimator=GradientBoostingRegressor(), random_state=0)
    elif strategy == 'linear_regression':
        imputer = IterativeImputer(estimator=LinearRegression(), random_state=0)
    df_imputed = df.copy()
    try:
        df_imputed[numeric_columns] = imputer.fit_transform(df[numeric_columns])
    except ValueError as e:
        st.error(f"Imputation error: {e}")
    return df_imputed

# Metric calculation function
def calculate_metrics(original, imputed):
    try:
        # Calculate RMSE and MAD while ignoring NaN values
        rmse = root_mean_squared_error(original, imputed)
        mad = mean_absolute_error(original, imputed)
    except ValueError as e:
        st.error(f"Metric calculation error: {e}")
        rmse, mad = None, None
    return rmse, mad

# Class column auto-detection function
def detect_class_column(df):
    for column in df.columns:
        if df[column].nunique() < 0.1 * len(df):
            return column
    return None

# Function to remove other classification columns
def remove_other_class_columns(df, class_column):
    for column in df.columns:
        if column != class_column and df[column].nunique() < 0.1 * len(df):
            df.drop(columns=[column], inplace=True)
    return df

# Main Function
def main():
    st.set_page_config(page_title="Simple Model Comparison", page_icon=":shark:", initial_sidebar_state="expanded")
    st.title("Simple Model Comparison")

    # About section
    st.sidebar.title("About")
    st.sidebar.info(
        """
        This application allows users to upload datasets, handle missing data, and compare different imputation strategies.
        Developed by: Mustafa Yavuz Okumuş
        """
    )

    # Ask the user to upload the dataset or enter a URL
    data_source = st.radio("Select Data Source", ("Upload File", "Enter URL"))

    if data_source == "Upload File":
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "txt"])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith("csv") or uploaded_file.name.endswith("txt"):
                    delimiter = st.text_input("Enter the delimiter for the file", value="," if uploaded_file.name.endswith("csv") else "\t")
                    has_header = st.checkbox("Does the file have column names?", value=True)
                    df = pd.read_csv(uploaded_file, delimiter=delimiter, header=0 if has_header else None)
                elif uploaded_file.name.endswith("xlsx"):
                    df = pd.read_excel(uploaded_file)
                    has_header = True  # xlsx dosyaları için varsayılan olarak sütun isimleri olduğunu varsayıyoruz
            except Exception as e:
                st.error(f"File upload error: {e}")
                return
    else:
        url = st.text_input("Enter URL")
        if url:
            has_header = st.checkbox("Does the file have column names?", value=True)
            try:
                df = pd.read_csv(url, header=0 if has_header else None)
            except Exception as e:
                st.error("Invalid URL or unable to fetch data. Please check the URL and try again.")
                return

    if 'df' in locals():

        # Filling strategies
        if df.isnull().values.any():
            missing_action = st.radio("There are missing values in the dataset. What would you like to do?", ("Drop missing value rows", "Fill missing values"))
            if missing_action == "Fill missing values":
                strategy = st.selectbox("Select an imputation strategy", ["mean", "median", "most_frequent", "random_forest", "expectation–maximization-EM", "gradient_boosting", "linear_regression"])
                df = impute_missing_data(df, strategy=strategy)

        # Check if the dataset has column names
        if has_header or not df.columns.equals(pd.RangeIndex(start=0, stop=df.shape[1], step=1)):
            st.write("The dataset has column names:")
        else:
            st.write("The dataset does not have column names.")
            column_names = st.text_input("Enter column names separated by commas")
            if column_names:
                df.columns = column_names.split(',')
                st.write("Updated Data:")

        st.write(df.head(15))

        # Auto-detect class column
        class_column = detect_class_column(df)
        class_column = st.selectbox("Select Class Column", df.columns, index=df.columns.get_loc(class_column) if class_column else 0)

        # Remove other classification columns
        df = remove_other_class_columns(df, class_column)

        # Descriptive statistics
        st.write("Descriptive Statistics:")
        st.write(df.describe())

        df.dropna(inplace=True)

        # Visualise distributions by class
        st.write("Pairplot by Class:")
        try:
            sns.pairplot(df, hue=class_column)
            st.pyplot(plt)
        except Exception as e:
            st.error(f"Visualization error: {e}")

        # Intentionally creating missing data
        missing_fraction = st.slider("Select Missing Data Fraction", 0.01, 0.50, 0.05)
        df_missing = introduce_missing_data(df.drop(columns=[class_column]), missing_fraction=missing_fraction)
        st.write("Missing Data Information:")
        st.write(df_missing.isnull().sum())  # Eksik veri sayısı
        st.write("Rows with Missing Data:")
        st.write(df_missing[df_missing.isnull().any(axis=1)].index)  # Eksik verilerin olduğu satırlar

        # Select numeric columns
        df_numeric = df.drop(columns=[class_column])

        for column in df.columns:
            try:
                df[column] = pd.to_numeric(df[column], errors='coerce')
            except ValueError:
                st.warning(f"Column {column} could not be converted to numeric and will be skipped.")

        # Filling strategies
        strategies = ['mean', 'median', 'most_frequent', 'random_forest', 'expectation–maximization-EM', 'gradient_boosting', 'linear_regression']
        metrics_data = []
        with st.spinner("Metrics loading..."):
            for strategy in strategies:
                df_imputed = impute_missing_data(df_missing, strategy=strategy)
                metrics = calculate_metrics(df_numeric, df_imputed)
                if metrics[0] is not None and metrics[1] is not None:
                    metrics_data.append({
                        "Strategy": strategy.capitalize(),
                        "RMSE": metrics[0],
                        "MAD": metrics[1]
                    })

        # Display metrics in a formatted table
        if metrics_data:
            st.markdown("<h3 style='text-align: center;'>Imputation Metrics</h3>", unsafe_allow_html=True)
            metrics_df = pd.DataFrame(metrics_data).sort_values(by=["RMSE", "MAD"], ascending=True)
            st.table(metrics_df)

if __name__ == "__main__":
    main()