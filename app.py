# app.py
from flask import Flask, render_template, request, jsonify
from forecasting_logic import GDPForecaster, DataProcessor 
import traceback

app = Flask(__name__)

try:
    data_processor_global = DataProcessor()
    ALL_COUNTRIES = data_processor_global.get_all_countries()
    if not ALL_COUNTRIES:
        print("CẢNH BÁO: Không tải được danh sách quốc gia khi khởi động.")
        ALL_COUNTRIES = ["Vietnam", "United States", "Germany", "Japan", "China"] 
except Exception as e:
    print(f"LỖI NGHIÊM TRỌNG khi khởi tạo DataProcessor: {e}")
    ALL_COUNTRIES = ["Vietnam"] 

SUPPORTED_MODELS = ["LinearRegression", "Ridge", "XGBoost", "SVR", "RandomForest", 
                    "ARIMA", "Prophet", "Holt"]
DEFAULT_TRAIN_END_YEAR = 2020
DEFAULT_TEST_START_YEAR = 2021
DEFAULT_TEST_END_YEAR = 2023
DEFAULT_NUM_FUTURE_YEARS = 0
DEFAULT_LAGS = "1,2,3" 


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html',
                           countries=ALL_COUNTRIES,
                           models=SUPPORTED_MODELS,
                           default_train_end=DEFAULT_TRAIN_END_YEAR,
                           default_test_start=DEFAULT_TEST_START_YEAR,
                           default_test_end=DEFAULT_TEST_END_YEAR,
                           default_num_future_years=DEFAULT_NUM_FUTURE_YEARS,
                           default_lags=DEFAULT_LAGS)

@app.route('/forecast', methods=['POST'])
def forecast():
    form_data = {k: request.form.get(k) for k in request.form}
    form_data_for_rerender = {
        "selected_country": form_data.get('country'),
        "selected_model": form_data.get('model_name'),
        "train_end_year_val": form_data.get('train_end_year', str(DEFAULT_TRAIN_END_YEAR)),
        "test_start_year_val": form_data.get('test_start_year', str(DEFAULT_TEST_START_YEAR)),
        "test_end_year_val": form_data.get('test_end_year', str(DEFAULT_TEST_END_YEAR)),
        "num_future_years_val": form_data.get('num_future_years', str(DEFAULT_NUM_FUTURE_YEARS)),
        "lags_val": form_data.get('lags_config', DEFAULT_LAGS)
    }
    try:
        country = form_data.get('country')
        model_name = form_data.get('model_name')
        train_end_year = int(form_data_for_rerender["train_end_year_val"])
        test_start_year = int(form_data_for_rerender["test_start_year_val"])
        test_end_year = int(form_data_for_rerender["test_end_year_val"])
        num_future_years = int(form_data_for_rerender["num_future_years_val"])
        lags_str = form_data_for_rerender["lags_val"]
        lags_config = [int(lag.strip()) for lag in lags_str.split(',') if lag.strip().isdigit()]
        if not lags_config: lags_config = [1,2,3]

        if not country or not model_name:
            return render_template('index.html',error_message="Vui lòng chọn quốc gia và mô hình.",**form_data_for_rerender,countries=ALL_COUNTRIES,models=SUPPORTED_MODELS,default_lags=DEFAULT_LAGS)
        if test_start_year <= test_end_year and train_end_year >= test_start_year :
                 return render_template('index.html',error_message="Năm kết thúc huấn luyện phải nhỏ hơn năm bắt đầu kiểm tra.",**form_data_for_rerender,countries=ALL_COUNTRIES,models=SUPPORTED_MODELS,default_lags=DEFAULT_LAGS)

        model_specific_params = {}; model_params_for_display = {"lags": ", ".join(map(str,lags_config))}
        def get_float(key, default): return float(form_data.get(key, default)) if form_data.get(key, default).replace('.', '', 1).isdigit() else default
        def get_int(key, default): return int(form_data.get(key, default)) if form_data.get(key, default).isdigit() else default

        if model_name == "Ridge": model_specific_params['alpha'] = get_float('ridge_alpha', 1.0)
        elif model_name == "XGBoost":
            model_specific_params['n_estimators'] = get_int('xgb_n_estimators', 100)
            model_specific_params['learning_rate'] = get_float('xgb_learning_rate', 0.1)
        elif model_name == "SVR":
            model_specific_params['kernel'] = form_data.get('svr_kernel', 'rbf')
            model_specific_params['C'] = get_float('svr_c', 1.0)
            model_specific_params['epsilon'] = get_float('svr_epsilon', 0.1)
        elif model_name == "RandomForest":
            model_specific_params['n_estimators'] = get_int('rf_n_estimators', 100)
            max_depth_str = form_data.get('rf_max_depth', '')
            model_specific_params['max_depth'] = int(max_depth_str) if max_depth_str and max_depth_str.isdigit() else None
        elif model_name == "ARIMA": model_specific_params['order'] = (get_int('arima_p',1),get_int('arima_d',1),get_int('arima_q',1))
        elif model_name == "Prophet": model_specific_params['seasonality_mode'] = form_data.get('prophet_seasonality_mode', 'additive')
        elif model_name == "Holt":
            model_specific_params['exponential'] = form_data.get('holt_exponential') == 'true'
            model_specific_params['damped_trend'] = form_data.get('holt_damped_trend') == 'true'
        model_params_for_display.update(model_specific_params)
        if model_name == "ARIMA": model_params_for_display['order'] = str(model_specific_params['order']) # For display

        forecaster_init_params = {'lags': lags_config, 'model_specific_params': model_specific_params}
        forecaster = GDPForecaster(model_name=model_name, country_name=country, **forecaster_init_params)
        results = forecaster.run_workflow_for_web(train_end_year, test_start_year, test_end_year, lags_config, num_future_years)
        return render_template('results.html', country=country, model_name=model_name, results=results,
                               train_end_year=train_end_year, test_start_year=test_start_year, test_end_year=test_end_year, 
                               num_future_years_val=num_future_years, model_specific_params_display=model_params_for_display)
    except ValueError as ve:
        return render_template('index.html',error_message=f"Dữ liệu nhập không hợp lệ: {ve}.",**form_data_for_rerender,countries=ALL_COUNTRIES,models=SUPPORTED_MODELS,default_lags=DEFAULT_LAGS)
    except Exception as e:
        print(f"Lỗi /forecast: {e}\n{traceback.format_exc()}")
        return render_template('index.html',error_message=f"Lỗi xử lý: {str(e)}.",**form_data_for_rerender,countries=ALL_COUNTRIES,models=SUPPORTED_MODELS,default_lags=DEFAULT_LAGS)

if __name__ == '__main__':
    app.run(debug=True)