import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR # Mới
from sklearn.ensemble import RandomForestRegressor # Mới
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import Holt # Mới
from prophet import Prophet
import warnings
from io import BytesIO
import base64
from collections import deque

warnings.filterwarnings("ignore")

class DataProcessor:
    def __init__(self, filepath="World GDP Dataset.csv"):
        self.filepath = filepath
        self.df_long = None
        self.country_data = None 
        self.all_countries = []

    def load_and_transform(self):
        try:
            df = pd.read_csv(self.filepath)
        except FileNotFoundError:
            print(f"Lỗi: Không tìm thấy tệp {self.filepath}")
            return None
        
        country_col_name = df.columns[0]
        self.df_long = df.melt(id_vars=[country_col_name], var_name="Year", value_name="GDP")
        self.df_long.columns = ["Country", "Year", "GDP"]
        self.df_long.dropna(subset=["GDP"], inplace=True)
        try:
            self.df_long["Year"] = pd.to_numeric(self.df_long["Year"], errors='coerce')
        except Exception as e: 
            print(f"Lỗi chuyển đổi cột 'Year' sang số: {e}.")
            self.df_long["Year"] = pd.to_numeric(self.df_long["Year"], errors='coerce')

        self.df_long.dropna(subset=["Year"], inplace=True)
        self.df_long["Year"] = self.df_long["Year"].astype(int)
        
        self.all_countries = sorted(self.df_long["Country"].unique().tolist())
        return self.df_long

    def get_all_countries(self):
        if not self.all_countries:
            if self.load_and_transform() is None: return []
        return self.all_countries
    
    def filter_country(self, country_name):
        if self.df_long is None:
            if self.load_and_transform() is None: return None
        
        self.country_data = self.df_long[self.df_long["Country"] == country_name].sort_values("Year").copy()
        if self.country_data.empty:
            print(f"Không có dữ liệu cho quốc gia: {country_name}")
            return None
        
        min_gdp_1980_series = self.country_data.loc[(self.country_data["Year"] == 1980) & (self.country_data["GDP"] > 0), "GDP"]
        if not min_gdp_1980_series.empty:
            min_gdp_1980 = min_gdp_1980_series.min()
            self.country_data.loc[self.country_data["GDP"] == 0, "GDP"] = min_gdp_1980
        else:
            min_positive_gdp = self.country_data.loc[self.country_data["GDP"] > 0, "GDP"].min()
            if pd.notna(min_positive_gdp):
                 self.country_data.loc[self.country_data["GDP"] == 0, "GDP"] = min_positive_gdp
            else:
                self.country_data.loc[self.country_data["GDP"] == 0, "GDP"] = 1e-6 

        self.country_data.dropna(subset=['GDP'], inplace=True)
        if self.country_data.empty:
            print(f"Không còn dữ liệu cho quốc gia: {country_name} sau khi xử lý.")
            return None
        return self.country_data

    def create_features_on_df(self, input_df, lags_config=[1, 2, 3]):
        if input_df is None or input_df.empty: return None
        df_to_feature = input_df.copy()
        for lag in lags_config:
            df_to_feature[f"GDP_lag{lag}"] = df_to_feature["GDP"].shift(lag)
        df_to_feature["GDP_growth"] = df_to_feature["GDP"].pct_change()
        df_to_feature.dropna(inplace=True)
        return df_to_feature

    def split_data(self, df_with_features, train_end_year, test_start_year, test_end_year, 
                     features_cols, target_col="GDP"):
        if df_with_features is None or df_with_features.empty: return None, None, None, None, None, None
        train_df = df_with_features[df_with_features["Year"] <= train_end_year].copy()
        test_df = pd.DataFrame() 
        if test_start_year <= test_end_year:
            test_df = df_with_features[(df_with_features["Year"] >= test_start_year) & 
                                       (df_with_features["Year"] <= test_end_year)].copy()
        if train_df.empty : return None, None, None, None, None, None
        X_train = train_df[features_cols].values
        y_train = train_df[target_col].values
        X_test = test_df[features_cols].values if not test_df.empty else np.array([])
        y_test = test_df[target_col].values if not test_df.empty else np.array([])
        return X_train, y_train, X_test, y_test, train_df, test_df


class GDPForecaster:
    def __init__(self, model_name="LinearRegression", country_name="Vietnam", data_filepath="World GDP Dataset.csv", **model_params):
        self.model_name = model_name
        self.country_name = country_name
        self.model_params = model_params 
        self.model = None
        self.data_processor = DataProcessor(filepath=data_filepath)
        self.X_train, self.y_train, self.X_test, self.y_test = None, None, None, None
        self.train_df, self.test_df = None, None 
        self.df_featured_full = None
        self.y_pred_train, self.y_pred_test = None, None
        
        lags_conf = self.model_params.get('lags', [1, 2, 3])
        if model_name not in ["ARIMA", "Prophet", "Holt"]:
            self.regression_features = ["Year"] + [f"GDP_lag{l}" for l in lags_conf] + ["GDP_growth"]
        else:
            self.regression_features = [] 
        self.prophet_regressors = [f"GDP_lag{l}" for l in lags_conf] + ["GDP_growth"]
        self._initialize_model()

    def _initialize_model(self):
        specific_params = self.model_params.get('model_specific_params', {})
        if self.model_name == "LinearRegression": self.model = LinearRegression(**specific_params)
        elif self.model_name == "Ridge": self.model = Ridge(**specific_params)
        elif self.model_name == "XGBoost":
            xgb_defaults = {'objective': 'reg:squarederror', 'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'random_state': 42}; xgb_defaults.update(specific_params)
            self.model = xgb.XGBRegressor(**xgb_defaults)
        elif self.model_name == "SVR": 
            svr_defaults = {'kernel': 'rbf', 'C': 1.0, 'epsilon': 0.1}; svr_defaults.update(specific_params)
            self.model = SVR(**svr_defaults)
        elif self.model_name == "RandomForest": 
            rf_defaults = {'n_estimators': 100, 'random_state': 42, 'max_depth': None}; rf_defaults.update(specific_params)
            self.model = RandomForestRegressor(**rf_defaults)
        elif self.model_name == "ARIMA": self.model = ARIMA(endog=np.array([]), order=specific_params.get('order', (1,1,1)))
        elif self.model_name == "Prophet":
            prophet_defaults = {'seasonality_mode': 'additive'}; prophet_defaults.update(specific_params)
            self.model = Prophet(**prophet_defaults)
        elif self.model_name == "Holt":
            self.model_holt_params = {'exponential': False, 'damped_trend': False}; self.model_holt_params.update(specific_params)
            self.model = None 
        else: raise ValueError(f"Mô hình {self.model_name} không được hỗ trợ.")

    def prepare_data(self, train_end_year=2020, test_start_year=2021, test_end_year=2023, lags_config=[1,2,3]):
        self.data_processor.load_and_transform()
        country_raw_data = self.data_processor.filter_country(self.country_name)
        if country_raw_data is None or country_raw_data.empty: return False
        
        if self.model_name not in ["ARIMA", "Holt"]:
            self.df_featured_full = self.data_processor.create_features_on_df(self.data_processor.country_data, lags_config=lags_config)
            if self.df_featured_full is None or self.df_featured_full.empty:
                if self.model_name not in ["Prophet"]: 
                     print(f"Lỗi prepare_data: Không thể tạo features cho {self.model_name}")
                     return False
        
        if self.model_name == "Prophet":
            prophet_df_base = country_raw_data[['Year', 'GDP']].copy(); prophet_df_base.rename(columns={'Year': 'ds', 'GDP': 'y'}, inplace=True)
            prophet_df_base['ds'] = pd.to_datetime(prophet_df_base['ds'].astype(str) + '-01-01')
            prophet_df_merged = prophet_df_base
            if self.df_featured_full is not None and not self.df_featured_full.empty:
                df_featured_for_prophet = self.df_featured_full.copy(); df_featured_for_prophet['ds'] = pd.to_datetime(df_featured_for_prophet['Year'].astype(str) + '-01-01')
                existing_regressors = [reg for reg in self.prophet_regressors if reg in df_featured_for_prophet.columns]
                if existing_regressors: prophet_df_merged = pd.merge(prophet_df_base, df_featured_for_prophet[['ds'] + existing_regressors], on='ds', how='left')
            prophet_df_merged.dropna(subset=['y'], inplace=True)
            if 'existing_regressors' in locals():
                for reg in existing_regressors:
                    if reg in prophet_df_merged.columns: prophet_df_merged[reg] = prophet_df_merged[reg].fillna(0)
            self.train_df = prophet_df_merged[prophet_df_merged['ds'].dt.year <= train_end_year].copy()
            self.test_df = prophet_df_merged[(prophet_df_merged['ds'].dt.year >= test_start_year) & (prophet_df_merged['ds'].dt.year <= test_end_year)].copy() if test_start_year <= test_end_year else pd.DataFrame()
            if self.train_df.empty: return False
            self.y_train = self.train_df['y'].values; self.y_test = self.test_df['y'].values if not self.test_df.empty else np.array([])
            return True

        if self.model_name in ["ARIMA", "Holt"]:
            train_data_ts = country_raw_data[country_raw_data['Year'] <= train_end_year].copy()
            if train_data_ts.empty: return False
            self.y_train = train_data_ts['GDP'].values; self.y_test = np.array([])
            if test_start_year <= test_end_year:
                test_data_ts = country_raw_data[(country_raw_data['Year'] >= test_start_year) & (country_raw_data['Year'] <= test_end_year)].copy()
                if not test_data_ts.empty: self.y_test = test_data_ts['GDP'].values
            self.train_df = train_data_ts[['Year', 'GDP']]
            self.test_df = country_raw_data[(country_raw_data['Year'] >= test_start_year) & (country_raw_data['Year'] <= test_end_year)][['Year', 'GDP']] if test_start_year <= test_end_year else pd.DataFrame(columns=['Year', 'GDP'])
            return True

        if self.df_featured_full is None or self.df_featured_full.empty: return False
        self.X_train, self.y_train, self.X_test, self.y_test, self.train_df, self.test_df = \
            self.data_processor.split_data(self.df_featured_full, train_end_year, test_start_year, test_end_year,
                                           features_cols=self.regression_features, target_col="GDP")
        return not (self.X_train is None or self.X_train.shape[0] == 0)

    def train(self):
        try:
            if self.model_name == "Prophet":
                if self.train_df is None or self.train_df.empty: return False
                specific_params = self.model_params.get('model_specific_params', {}); prophet_defaults = {'seasonality_mode': 'additive'}; prophet_defaults.update(specific_params)
                current_model = Prophet(**prophet_defaults)
                existing_regressors = [reg for reg in self.prophet_regressors if reg in self.train_df.columns]
                for regressor in existing_regressors: current_model.add_regressor(regressor)
                current_model.fit(self.train_df[['ds', 'y'] + existing_regressors]); self.model = current_model
            elif self.model_name == "ARIMA":
                if self.y_train is None or len(self.y_train) == 0: return False
                order = self.model_params.get('model_specific_params', {}).get('order', (1,1,1))
                if len(self.y_train) < sum(order) + 1: return False
                self.model = ARIMA(endog=self.y_train, order=order).fit()
            elif self.model_name == "Holt":
                if self.y_train is None or len(self.y_train) < 2: return False
                holt_params = self.model_holt_params 
                self.model = Holt(self.y_train, exponential=holt_params.get('exponential', False), damped_trend=holt_params.get('damped_trend', False)).fit()
            else: 
                if self.X_train is None or self.y_train is None or self.X_train.shape[0] == 0: return False
                self.model.fit(self.X_train, self.y_train)
            return True
        except Exception as e: print(f"Lỗi train {self.model_name}: {e}"); return False

    def predict(self):
        try:
            self.y_pred_train = np.array([])
            if self.train_df is not None and not self.train_df.empty and self.model is not None:
                if self.model_name == "Prophet":
                    existing_reg = [r for r in self.prophet_regressors if r in self.train_df.columns]
                    self.y_pred_train = self.model.predict(self.train_df[['ds'] + existing_reg])['yhat'].values
                elif self.model_name == "ARIMA" and self.y_train is not None and len(self.y_train)>0:
                    self.y_pred_train = self.model.predict(start=0, end=len(self.y_train)-1)
                elif self.model_name == "Holt": self.y_pred_train = self.model.fittedvalues
                elif self.X_train is not None and self.X_train.shape[0] > 0 :
                    self.y_pred_train = self.model.predict(self.X_train)
            self.y_pred_test = np.array([])
            if self.test_df is not None and not self.test_df.empty and self.y_test is not None and len(self.y_test) > 0:
                if self.model_name == "Prophet":
                    existing_reg = [r for r in self.prophet_regressors if r in self.test_df.columns]
                    self.y_pred_test = self.model.predict(self.test_df[['ds'] + existing_reg])['yhat'].values
                elif self.model_name == "ARIMA":
                    start, end = len(self.y_train), len(self.y_train) + len(self.y_test) - 1
                    if start <= end: self.y_pred_test = self.model.predict(start=start, end=end)
                elif self.model_name == "Holt": self.y_pred_test = self.model.forecast(steps=len(self.y_test))
                elif self.X_test is not None and self.X_test.shape[0] > 0:
                    self.y_pred_test = self.model.predict(self.X_test)
            return True
        except Exception as e: print(f"Lỗi predict (test) {self.model_name}: {e}"); return False

    def _get_gdp_history_for_future_deque(self, up_to_year, lags_config):
        source_df_for_hist = self.df_featured_full
        if source_df_for_hist is None or source_df_for_hist.empty:
            if self.data_processor.country_data is None or self.data_processor.country_data.empty: return None
            source_df_for_hist = self.data_processor.country_data # Dùng dữ liệu thô nếu chưa có features
        hist_data = source_df_for_hist[source_df_for_hist['Year'] <= up_to_year]
        if hist_data.empty: return None
        num_lags_needed = max(lags_config) if lags_config else 3
        recent_gdps = hist_data['GDP'].tail(num_lags_needed + 1).tolist()
        while len(recent_gdps) < (num_lags_needed + 1): recent_gdps.insert(0, np.nan)
        return deque(recent_gdps, maxlen=num_lags_needed + 1)

    def predict_future_gdp(self, num_future_years, actual_data_end_year):
        if self.model is None: return None, None
        lags_config = self.model_params.get('lags', [1, 2, 3])
        gdp_history_deque = self._get_gdp_history_for_future_deque(actual_data_end_year, lags_config)
        if gdp_history_deque is None: return None, None
        future_preds, future_years = [], []
        start_pred_year = actual_data_end_year + 1

        if self.model_name in ["ARIMA", "Holt"]:
            if hasattr(self.model, 'forecast'):
                last_train_yr = self.train_df['Year'].max() if self.train_df is not None and not self.train_df.empty else actual_data_end_year
                total_steps = (start_pred_year - 1 - last_train_yr) + num_future_years
                if total_steps <=0: return [],[]
                all_preds = self.model.forecast(steps=total_steps)
                for i in range(num_future_years):
                    target_yr = start_pred_year + i; idx = target_yr - (last_train_yr + 1)
                    if 0 <= idx < len(all_preds): future_years.append(target_yr); future_preds.append(all_preds[idx])
                return future_years, future_preds
            else: return None, None

        for i in range(num_future_years):
            next_year = start_pred_year + i; future_years.append(next_year)
            lags_vals = {}
            for lag_num in lags_config: 
                lags_vals[f'GDP_lag{lag_num}'] = gdp_history_deque[-lag_num] if len(gdp_history_deque) >= lag_num and pd.notna(gdp_history_deque[-lag_num]) else 0
            lag1 = lags_vals.get(f'GDP_lag{lags_config[0] if lags_config else 1}',0)
            lag2 = lags_vals.get(f'GDP_lag{lags_config[1] if len(lags_config)>1 else 2}',0)
            growth = (lag1 - lag2) / lag2 if lag2!=0 and pd.notna(lag1) and pd.notna(lag2) else 0
            feat_dict = {'Year': next_year, 'GDP_growth': growth, **lags_vals}
            final_feat_vec = [feat_dict.get(fn, 0) for fn in self.regression_features]
            pred_gdp = np.nan
            if self.model_name == "Prophet":
                df_p = pd.DataFrame({'ds': [pd.to_datetime(f"{next_year}-01-01")]})
                for reg in self.prophet_regressors: df_p[reg] = feat_dict.get(reg, 0)
                valid_cols = ['ds'] + ([r for r in df_p.columns if r in self.model.regressors.keys()] if hasattr(self.model,'regressors') and self.model.regressors else [])
                try: pred_gdp = self.model.predict(df_p[valid_cols])['yhat'].iloc[0]
                except Exception as e: print(f"Err Prophet future {next_year}: {e}")
            else: # Linear, Ridge, XGBoost, SVR, RandomForest
                try: pred_gdp = self.model.predict(np.array([final_feat_vec]))[0]
                except Exception as e: print(f"Err Reg future {next_year}: {e}")
            future_preds.append(pred_gdp)
            gdp_history_deque.append(pred_gdp if pd.notna(pred_gdp) else 0)
        return future_years, future_preds

    def evaluate(self):
        metrics = {"train": {"r2": "N/A", "mse": "N/A", "mae": "N/A", "mape": "N/A"},
                   "test": {"r2": "N/A", "mse": "N/A", "mae": "N/A", "mape": "N/A", "details": []}}
        def calculate_metrics_safe(y_true, y_pred):
            if y_true is None or y_pred is None or len(y_true) == 0 or len(y_pred) == 0 or len(y_true) != len(y_pred): return "N/A","N/A","N/A","N/A"
            valid_idx = np.isfinite(y_true) & np.isfinite(y_pred)
            y_t, y_p = y_true[valid_idx], y_pred[valid_idx]
            if len(y_t) < 2: r2 = "N/A"
            else: r2 = f"{r2_score(y_t, y_p):.4f}"
            if len(y_t) == 0: mse, mae, mape = "N/A","N/A","N/A"
            else:
                mse = f"{mean_squared_error(y_t, y_p):.4f}"; mae = f"{mean_absolute_error(y_t, y_p):.4f}"
                y_t_mape, y_p_mape = y_t[y_t != 0], y_p[y_t != 0]
                if len(y_t_mape) == 0: mape = "N/A (all true for MAPE are zero)"
                else: mape = f"{np.mean(np.abs((y_t_mape - y_p_mape) / y_t_mape)) * 100:.2f}%"
            return r2, mse, mae, mape
        r2_tr,mse_tr,mae_tr,mape_tr = calculate_metrics_safe(self.y_train, self.y_pred_train)
        metrics["train"] = {"r2":r2_tr,"mse":mse_tr,"mae":mae_tr,"mape":mape_tr}
        if self.y_test is not None and len(self.y_test) > 0:
            r2_te,mse_te,mae_te,mape_te = calculate_metrics_safe(self.y_test, self.y_pred_test)
            metrics["test"] = {"r2":r2_te,"mse":mse_te,"mae":mae_te,"mape":mape_te, "details":[]}
            if self.test_df is not None and not self.test_df.empty and len(self.y_test) == len(self.y_pred_test):
                yrs_series = self.test_df['Year'] if self.model_name not in ['Prophet'] else self.test_df['ds'].dt.year
                for i, yr_val in enumerate(yrs_series.iloc[:len(self.y_test)]):
                    metrics["test"]["details"].append({"year":int(yr_val), "actual":f"{self.y_test[i]:.2f}" if pd.notna(self.y_test[i]) else "N/A", "predicted":f"{self.y_pred_test[i]:.2f}" if pd.notna(self.y_pred_test[i]) else "N/A"})
        return metrics

    def plot_forecast_to_base64(self, future_years=None, future_predictions=None):
        fig, ax = plt.subplots(figsize=(12, 6))
        # Plot train
        if self.train_df is not None and self.y_train is not None and self.y_pred_train is not None and len(self.y_train) > 0:
            train_yrs = self.train_df['Year'] if self.model_name not in ['Prophet'] else self.train_df['ds'].dt.year
            if len(train_yrs) == len(self.y_train): ax.plot(train_yrs, self.y_train, label="Train GDP (Actual)", marker='o')
            if len(train_yrs) == len(self.y_pred_train): ax.plot(train_yrs, self.y_pred_train, label=f"Train Pred ({self.model_name})", linestyle='--')
        # Plot test
        if self.test_df is not None and self.y_test is not None and self.y_pred_test is not None and len(self.y_test) > 0:
            test_yrs = self.test_df['Year'] if self.model_name not in ['Prophet'] else self.test_df['ds'].dt.year
            if len(test_yrs) == len(self.y_test): ax.plot(test_yrs, self.y_test, label="Test GDP (Actual)", marker='s')
            if len(test_yrs) == len(self.y_pred_test): ax.plot(test_yrs, self.y_pred_test, label=f"Test Pred ({self.model_name})", linestyle='--', marker='x')
        # Plot future
        if future_years and future_predictions and len(future_years) == len(future_predictions):
            ax.plot(future_years, future_predictions, label=f"Future Forecast ({self.model_name})", linestyle=':', marker='^', color='green')
        ax.set_title(f"{self.country_name} GDP Forecast: {self.model_name}"); ax.set_xlabel("Year"); ax.set_ylabel("GDP")
        ax.legend(); ax.grid(True); plt.tight_layout()
        img_io=BytesIO(); plt.savefig(img_io,format='png',bbox_inches='tight'); img_io.seek(0)
        img_b64=base64.b64encode(img_io.getvalue()).decode('utf-8'); plt.close(fig)
        return f"data:image/png;base64,{img_b64}"

    def get_model_parameters_dict(self):
        params_dict = {"type": self.model_name, "details": {}}
        if self.model is None: params_dict["details"]["Error"] = "Model not trained."; return params_dict
        try:
            if self.model_name in ["LinearRegression", "Ridge", "SVR", "RandomForest", "XGBoost"]:
                is_sklearn_reg = self.model_name in ["LinearRegression", "Ridge", "SVR", "RandomForest"]
                if is_sklearn_reg and hasattr(self.model, 'intercept_') and self.model_name != "RandomForest" and (self.model_name != "SVR" or self.model.kernel == 'linear'): 
                    params_dict["details"]["Intercept"] = f"{self.model.intercept_[0] if isinstance(self.model.intercept_, np.ndarray) else self.model.intercept_:.4f}"
                coeffs_or_imps = None; p_name = ""
                if hasattr(self.model, 'coef_') and (self.model_name != "SVR" or self.model.kernel == 'linear'): 
                    coeffs_or_imps = self.model.coef_.flatten() # Flatten for SVR linear
                    p_name = "Coefficient"
                elif hasattr(self.model, 'feature_importances_'): 
                    coeffs_or_imps = self.model.feature_importances_
                    p_name = "Importance"
                
                f_names = self.regression_features
                if hasattr(self.model, 'feature_names_in_'): f_names = self.model.feature_names_in_
                elif self.X_train is not None and coeffs_or_imps is not None and self.X_train.shape[1] == len(coeffs_or_imps): f_names = self.regression_features[:self.X_train.shape[1]]
                
                if coeffs_or_imps is not None:
                    for feat, val in zip(f_names, coeffs_or_imps): params_dict["details"][f"{p_name} for {feat}"] = f"{val:.4f}"
                if self.model_name == "SVR": params_dict["details"].update({"Kernel":self.model.kernel, "C":self.model.C, "Epsilon":self.model.epsilon})
                if self.model_name == "RandomForest": params_dict["details"].update({"n_estimators":self.model.n_estimators, "max_depth":self.model.max_depth if self.model.max_depth else "None"})
            elif self.model_name == "ARIMA":
                if hasattr(self.model,'model') and hasattr(self.model.model,'order'): params_dict["details"]["Order (p,d,q)"] = str(self.model.model.order)
                if hasattr(self.model,'aic'): params_dict["details"]["AIC"] = f"{self.model.aic:.2f}"
            elif self.model_name == "Prophet":
                params_dict["details"].update({"Growth":getattr(self.model,'growth','N/A'), "Seasonality":getattr(self.model,'seasonality_mode','N/A')})
            elif self.model_name == "Holt":
                if self.model: params_dict["details"].update({
                    "Alpha":f"{self.model.params.get('smoothing_level','N/A'):.4f}", "Beta":f"{self.model.params.get('smoothing_trend','N/A'):.4f}",
                    "Phi":f"{self.model.params.get('damping_trend','N/A'):.4f}", "Exponential":str(self.model_holt_params.get('exponential',False)),
                    "Damped":str(self.model_holt_params.get('damped_trend',False))})
        except Exception as e: params_dict["details"]["Error"] = f"Param error: {str(e)}"
        return params_dict
            
    def run_workflow_for_web(self, train_end_year=2020, test_start_year=2021, test_end_year=2023, 
                             lags_config=[1,2,3], num_future_forecast_years=0):
        self.model_params['lags'] = lags_config 
        if self.model_name not in ["ARIMA", "Prophet", "Holt"]: self.regression_features = ["Year"] + [f"GDP_lag{l}" for l in lags_config] + ["GDP_growth"]
        else: self.regression_features = [] 
        self.prophet_regressors = [f"GDP_lag{l}" for l in lags_config] + ["GDP_growth"]
        results = {"error_message":None,"metrics":None,"plot_base64":None,"model_params_info":None,"test_years_range":{},"future_predictions_data":None}
        if not self.prepare_data(train_end_year,test_start_year,test_end_year,lags_config): results["error_message"]=(f"Data prep error."); return results
        if not self.train(): results["error_message"]=f"Train error {self.model_name}."; results["model_params_info"]=self.get_model_parameters_dict(); return results
        if not self.predict(): results["error_message"]=(results.get("error_message","") + f" Test predict error {self.model_name}.").strip()
        results["metrics"]=self.evaluate(); results["model_params_info"]=self.get_model_parameters_dict()
        act_test_start,act_test_end = test_start_year,test_end_year
        if self.test_df is not None and not self.test_df.empty:
            col='Year' if self.model_name not in ['Prophet'] else 'ds'
            try:
                min_yr,max_yr=self.test_df[col].min(),self.test_df[col].max()
                if self.model_name=='Prophet':min_yr,max_yr=min_yr.year,max_yr.year
                act_test_start,act_test_end=min_yr,max_yr
            except:pass
        results["test_years_range"]={"start":act_test_start,"end":act_test_end}
        if act_test_start>act_test_end:results["test_years_range"]["message"]="No historical test set."
        fut_yrs,fut_preds=None,None
        if num_future_forecast_years>0:
            actual_data_ends=train_end_year
            if self.test_df is not None and not self.test_df.empty and test_start_year<=test_end_year: actual_data_ends=act_test_end
            print(f"WORKFLOW: Future base {actual_data_ends} for {num_future_forecast_years} yrs.")
            fut_yrs,fut_preds=self.predict_future_gdp(num_future_forecast_years,actual_data_ends)
            if fut_yrs and fut_preds: results["future_predictions_data"]=[{"year":int(y),"predicted_gdp":f"{p:.2f}" if pd.notna(p) else "N/A"} for y,p in zip(fut_yrs,fut_preds)]
        results["plot_base64"]=self.plot_forecast_to_base64(fut_yrs,fut_preds)
        return results