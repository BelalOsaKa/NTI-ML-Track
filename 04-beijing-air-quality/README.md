# Beijing PM2.5 Air Quality
Modelling hourly PM2.5 pollution in Beijing (2010–2014) from weather data in the UCI PRSA dataset. The notebook builds a preprocessing pipeline (imputation, encoding, scaling with `ColumnTransformer`), a Linear Regression model for the pollution level, and a Logistic Regression classifier for high-pollution hours, with class imbalance handled through imbalanced-learn.

**Data:** `PRSA_data_2010.1.1-2014.12.31.csv` · **Tools:** pandas, scikit-learn, imbalanced-learn, seaborn
