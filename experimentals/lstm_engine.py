'''
LSTM ENGINE V1.1

Clase que integra las funcionalidades y servicios de LSTM
Tipo de Modelo soportado: Multivariado - Multistep
Los hiperparámetros están por defecto a un look forward de 180 días (6 meses), a 31 días de predicción
Es decir, que el modelo se le presentan secuencias de 6 meses para predecir el siguiente mes
'''

import pandas as pd
import os
import shutil
import json
from datetime import datetime, timedelta
import matplotlib.pylab as plt
import numpy as np
import joblib
import pickle

from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Input, Reshape
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

class Lstm_engine: 
    def __init__(self, 
                pasos=180,
                training_percentage=0.9,
                n_predictions = 31,
                epochs = 30,
                neurons = 100,
                batch_size = 15,
                isfloatdata = True,
                ):
        self.PASOS = pasos # No. DE OBSERVACIONES EN EL TIEMPO PARA LA DATA Y PARA ALGORITMO DE 'SLIDDING WINDOW'
        self.TRAINING_PERCENTAGE = training_percentage  #PORCENTAJE DE DATOS A TOMAR PARA ENTRENAMIENTO
        self.N_PREDICTIONS = n_predictions  #NUMERO DE PREDICCIONES A REALIZAR
        self.EPOCHS = epochs #EPOCAS DE ENTRENAMIENTO DEL MODELO
        self.NEURONS = neurons #Mismo que el de pasos
        self.BATCH_SIZE = batch_size #TAMANIO DE LAS MUESTRAS DE ENTRENAMIENTO
        self.ISFLOATDATA = isfloatdata #VARIABLE QUE CONVIERTE LOS DATOS A PUNTO FLOTANTE

    def set_index_datetime(self,data):
        """
        Elimina la columna numérica original y setea la nueva columna como index
        """
        if str(type(data) == "<class 'pandas.core.frame.DataFrame'>"):
            for column in data.columns: 
                try: 
                    pd.to_datetime(data[column])
                    data.set_index(column,inplace=True)
                    return data
                except Exception as e:  
                    pass
        else: 
            return 0
    
    def split_sequences(self,df, input_steps, output_steps, features, target):
        """
        Divide la secuencia multivariante en muestras para LSTM, permitiendo ventanas de entrada y salida de diferente tamaño.
        """
        X, y = [], []
        datos = df[features].values
        target_vals = df[target].values
        for i in range(len(datos) - input_steps - output_steps + 1):
            X.append(datos[i:i+input_steps])
            y.append(target_vals[i+input_steps:i+input_steps+output_steps])
        return np.array(X), np.array(y)
    
    def saveMetadata(self,metadata,path, customNameMetadata=None, showMetadata=False):
        """
        Permite guardar la metadata en formato json, unicamente pasando la misma metadata
        y el directorio padre.
        """

        if customNameMetadata != None: 
            metadata_path = path+'/'+customNameMetadata+'.json'
        else:
            metadata_path = path+'/metadata.json'
        
        if showMetadata:
            print(type(metadata))
            print(metadata)

        with open(metadata_path, 'w') as file: 
            json.dump(metadata,file, indent=4)
    
    def tratamiento_datos(self,data, dictionary, show_nan=False, show_data_output=False):
        """
        Realiza una mejora y garantiza la secuencia temporal de los datos a manejar para LSTM
        """

        val = data.copy()
        val = val.reset_index()

        GROUP_ID = dictionary['FORECAST_ID_COLUMN_ENCODED']
        target_col = dictionary['TARGET_COLUMN']
        new_col = dictionary['COLUMN_ADD']

        series_modified = []

        for i, group in val.groupby(GROUP_ID):
            producK_ID = group[GROUP_ID].iloc[0]
            productID = group['ProductKey'].iloc[0]
            date_ini = group['Date'].min()
            date_last = group['Date'].max()
            group['Date'] = pd.to_datetime(group['Date'], errors='coerce')
            group = group.sort_values('Date')
            group = group[(group['Date']>= pd.Timestamp(date_ini)) & (group['Date']<=pd.Timestamp(date_last))]
            group_index = group.set_index('Date')
            days_corrected = pd.date_range(start=date_ini, end=date_last - pd.Timedelta(days=1), freq='D')
            group_reindex = group_index.reindex(days_corrected)

            if show_nan:
                print(group_reindex)

            group_reindex = group_reindex.reset_index()
            group_reindex['YEAR'] = group_reindex['index'].dt.year
            group_reindex['MONTH'] = group_reindex['index'].dt.month
            group_reindex['Day'] = group_reindex['index'].dt.day
            group_reindex['DayOfWeek'] = group_reindex['index'].dt.dayofweek
            group_reindex['IsWeekend'] = group_reindex['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

            group_reindex = group_reindex.set_index('index')
            group_reindex[target_col] = group_reindex[target_col].fillna(0)
            group_reindex[target_col] = group_reindex[target_col].astype(int)
            group_reindex['UnitPrice'] = group_reindex['UnitPrice'].fillna(0.0)
            group_reindex['UnitPriceDiscountPct'] = group_reindex['UnitPriceDiscountPct'].fillna(0.0)
            group_reindex['DiscountAmount'] = group_reindex['DiscountAmount'].fillna(0)
            group_reindex['ProductStandardCost'] = group_reindex['ProductStandardCost'].fillna(0.0)
            group_reindex['SalesAmount'] = group_reindex['SalesAmount'].fillna(0.0)
            group_reindex['ProductKey'] = productID
            group_reindex[GROUP_ID] = producK_ID
            group_reindex[new_col] = group_reindex[target_col].apply(lambda x: 1 if x > 0 else 0)

            if show_data_output:
                print(group_reindex)

            series_modified.append(group_reindex)
        data_modified = pd.concat(series_modified)
        return data_modified

    def create_x_y_train(self,data, new_col, var_columns, var_features, target_col, parameters, show_paremeters=False, steps=0, n_predictions=0, n_validationData=None):
        """
        Método que divide la data en entrenamiento y validación
        Los escaladores son: 1 para la serie inicial, contemplando 12 columnas, 
        uno para x con n_features, y otro para y con target
        generamos un escalador para el tratamiento de los datos a algoritmo SLDW
        """

        if show_paremeters:
            print(var_columns)
            print(var_features)
        
        temp = data.copy()
        values_y = data.copy()
        deleted = [new_col, 'IsWeekend']
        only_scaled = [x for x in var_columns if x not in deleted]
        total_rows = len(data)

        if n_validationData != None:
            n_train_rows = int(total_rows-n_validationData)
        else:
            n_train_rows = int(total_rows * self.TRAINING_PERCENTAGE)

        df_train = data.iloc[:n_train_rows].copy()
        df_val = data.iloc[n_train_rows - steps - n_predictions + 1:].copy()  # Incluye pasos previos para ventana de validación
        parameters['ID_FORECAST_ENC_VALUE'] = str(data['ProductKey_encoded'].iloc[0])
        df = df_val.copy()
        df = df.drop(columns=['ProductKey_encoded','IsSelled'])
        del_col = parameters['forecast_id_col']
        df_train = df_train.drop(columns=[del_col], axis=1)
        df_val = df_val.drop(columns=[del_col], axis=1)
        parameters['DATOS_ENTRENAMIENTO'] = len(df_train)
        parameters['LAST_DATE'] = df_train.index.max().date()
        parameters['DATOS_VALIDACION'] = len(df_val)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler_y = MinMaxScaler(feature_range=(-1, 1))
        df_train[only_scaled] = scaler.fit_transform(df_train[only_scaled])
        scaler_y.fit_transform(values_y[[target_col]])
        df_val[only_scaled] = scaler.fit_transform(df_val[only_scaled])
        X_train, y_train = self.split_sequences(df_train, steps, n_predictions, var_features, target_col)
        X_val, y_val = self.split_sequences(df_val, steps, n_predictions, var_features, target_col)

        if show_paremeters:
            print("Columns :",df_train.columns)
            print("Columns :",df_val.columns)
            print("Num. Train Days: ",n_train_rows)
            print(f"Tensors: (X_train): {X_train.shape}, (y_train): {y_train.shape}")
            print(f"Tensors:  (X_val): {X_val.shape}, (y_val): {y_val.shape}")

        return X_train, y_train, X_val, y_val, scaler_y, scaler, only_scaled, df, parameters

    def train_model(self,x_train, y_train, x_val, y_val, model, model_path, scaler_y, parameters):
        """
        Función de entrenamiento de modelos
        """
        early_stop = EarlyStopping(monitor='val_mae', patience=5, restore_best_weights=True)
        history = model.fit(
            x_train, y_train,
            epochs= self.EPOCHS,
            batch_size=self.BATCH_SIZE,
            validation_data=(x_val, y_val),
            verbose=2,
            shuffle=False
        )

        results = model.predict(x_val)
        path = model_path+"/accuracy/"
        os.makedirs(path, exist_ok=True)
        datetim_e_path = datetime.now().strftime('%H-%M-%S_%d-%m-%Y')
        path = path+'Training/'+datetim_e_path
        os.makedirs(path,exist_ok=True)

        if results.ndim == 2 and results.shape[1] > 1:
            results_plot = results[:, 0]
            y_val_plot = y_val[:, 0]
        else:
            results_plot = results.flatten()
            y_val_plot = y_val.flatten()

        results_plot = scaler_y.inverse_transform(results_plot.reshape(-1, 1)).flatten()
        y_val_plot = scaler_y.inverse_transform(y_val_plot.reshape(-1, 1)).flatten()
        results_plot = results_plot.astype(int)
        y_val_plot = y_val_plot.astype(int)

        ##### MODIFICADOR DEL TIPO DE GRÁFICO Y COLORIMETRÍA
        plt.rcParams['figure.facecolor'] = '#001f3f'       
        plt.rcParams['axes.facecolor'] = '#001f3f'    
        plt.rcParams['text.color'] = 'white'
        plt.rcParams['axes.labelcolor'] = 'white'
        plt.rcParams['xtick.color'] = 'white'
        plt.rcParams['ytick.color'] = 'white'
        plt.rcParams['axes.edgecolor'] = 'white'

        plt.figure(figsize=(10, 5))
        plt.plot(y_val_plot, label='Valores Reales', color='g', linewidth=2)
        plt.plot(results_plot, label='Valores Pronosticados', color='orange', linewidth=2)
        plt.xlabel('Índice')
        plt.ylabel('Valores')
        plt.title('Comparación: Valores reales vs Pronósticos')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.figtext(0.01, 0.01, "Realizado el: "+datetime.now().strftime('%H:%M:%S %d-%m-%Y'), fontsize=10, color="gray")
        plt.figtext(0.50, 0.01, "Gestión de Innovación en Tecnología Informática S.C.P. | Grupo Consultores®", fontsize=10, color="gray")
        plt.savefig(path+"/performance_validation_1.jpg",dpi=300)
        plt.close()

        # 1
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
        plt.plot(history.history['val_loss'], label='Pérdida de Validación')
        plt.xlabel('Épocas')
        plt.ylabel('Pérdida')
        plt.title('Pérdidas en Entrenamiento y Validación')
        plt.legend()
        plt.figtext(0.01, 0.01, "Realizado el: "+datetime.now().strftime('%H:%M:%S %d-%m-%Y'), fontsize=10, color="gray")
        plt.figtext(0.60, 0.01, "Gestión de Innovación en Tecnología Informática S.C.P. | Grupo Consultores®", fontsize=10, color="gray")
        plt.savefig(path+"/performance_validation_2.jpg",dpi=300)
        plt.close()

        # 2
        if 'mse' in history.history:
            plt.figure(figsize=(10, 5))
            plt.plot(history.history['mse'], label='MSE de Entrenamiento')
            if 'val_mse' in history.history:
                plt.plot(history.history['val_mse'], label='MSE de Validación')
            plt.xlabel('Épocas')
            plt.ylabel('MSE')
            plt.title('MSE de Entrenamiento y Validación')
            plt.legend()
            plt.figtext(0.01, 0.01, "Realizado el: "+datetime.now().strftime('%H:%M:%S %d-%m-%Y'), fontsize=10, color="gray")
            plt.figtext(0.60, 0.01, "Gestión de Innovación en Tecnología Informática S.C.P. | Grupo Consultores®", fontsize=10, color="gray")
            plt.savefig(path+"/performance_validation_3.jpg",dpi=300)
            plt.close()

        # Preparamos la data a guardar en metadata
        datetim_e = datetime.now().strftime('%H:%M:%S %d-%m-%Y')
        optimizer_model = model.optimizer.get_config()
        model_summary = json.loads(model.to_json())

        metadata = [
            {
                "MODEL_REFERENCY":{
                    "CATEGORY": parameters['category_id'],
                    "ID": parameters['ID_model'],
                    "MODEL_NAME": parameters['MODEL_NAME'],
                    "TECHNOLOGY": "LSTM"
                }
            },
            {
                "GENERAL_INFO": {
                    "DATOS_ENTRENAMIENTO": parameters['data_size'],
                    "DATOS_VALIDACION":parameters['DATOS_VALIDACION'],
                    "EPOCH": len(history.history['loss']),
                    "FECHA_ENTRENAMIENTO": datetim_e,
                    "FECHA_MODIFICACION": datetim_e,
                    "ID_ENCODED": parameters['ID_FORECAST_ENC_VALUE'],
                    "STEPS": str(self.PASOS),
                    "PREDICTIONS_DEFAULT": str(self.N_PREDICTIONS),
                    "TECHNOLOGY":"LSTM"
                }
            },
            {
                "MODEL_ARCHITECTURE": model_summary
            },
            {
                "OPTIMIZER_CONFIG": optimizer_model
            }
        ]
        self.saveMetadata(metadata,model_path)
        return model


    def create_model(self,n_features):
        model = Sequential()
        model.add(Input(shape=(self.PASOS, n_features)))
        model.add(LSTM(self.NEURONS, return_sequences=True))
        model.add(LSTM(int(self.NEURONS/2)))
        model.add(Dense(64, activation ='tanh'))
        model.add(Dense(32, activation ='tanh'))
        model.add(Dense(self.N_PREDICTIONS, activation ='linear'))
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        model.add(Reshape((self.N_PREDICTIONS, 1)))
        model.summary()
        return model
    
    def loadUtils(self, model_path,model_metadata, category_metadata):
        """
        Función que carga lo necesario para reentrenamiento de modelos

        * MODELO ENTRENADO (.keras)
        * CSV
        * (X) SCALER
        * (Y) SCALER
        * METADATA DEL MODELO
        """

        enable = False
        root_dir = os.path.dirname(model_path)
        util_dir = root_dir+'/utils/'
        df = pd.read_csv(util_dir+'LSLDWINDOW.csv')
        Scaler_x = self.reloadScaler(util_dir+'Scaler_x.pkl')
        Scaler_y = self.reloadScaler(util_dir+'Scaler_y.pkl')
        only_scaled = self.reloadArray(util_dir+'only_scaled.pkl')
        model = self.loadModel(model_path)

        dictionary = category_metadata
        features = dictionary['FEATURES']
        n_features = dictionary['N_FEATURES']
        steps = int(model_metadata['GENERAL_INFO']['STEPS'])
        p_default = int(model_metadata['GENERAL_INFO']['PREDICTIONS_DEFAULT'])

        # VALIDACION DE ARCHIVOS CARGADOS CORRECTAMENTE
        if (
            df is not None and not df.empty and
            Scaler_x is not None and
            Scaler_y is not None and
            only_scaled is not None and
            model is not None
        ):
            print("////// SE HAN CARGADO TODOS LOS ARCHIVOS NECESARIOS CORRECTAMENTE //////")
            print("Scaler x: ",type(Scaler_x))
            print("Scaler y: ",type(Scaler_y))
            print("Only_scaled: ",type(only_scaled))
            enable = True
        else:
            print("////// NO TODOS LOS ARCHIVOS SE CARGARON CORRECTAMENTE //////")

        if enable:
            df = df.rename(columns={"index": "Date"})
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = self.set_index_datetime(df)
            id_product = df[dictionary['FORECAST_ID_COLUMN']].iloc[0]
            df = df.sort_values([dictionary['FORECAST_ID_COLUMN'], 'Date'])
            df[dictionary['FORECAST_ID_COLUMN_ENCODED']] = model_metadata['GENERAL_INFO']['ID_ENCODED']
            data_trat = self.tratamiento_datos(df, category_metadata)            
            return model, data_trat, features, n_features, Scaler_y, steps, Scaler_x, p_default, root_dir,id_product

    
    def reloadScaler(self,path):
        """
        Función que permite recuperar escaladores usando Joblib
        """
        load_scaler = joblib.load(path)
        return load_scaler
    
    def reloadArray(self,path):
        """
        Función que permite recuperar array usando Pickle
        """
        with open(path, "rb") as f:
            load_scaler = pickle.load(f)
        return load_scaler
    
    def loadModel(self, model_path):
        """
        Función que permite recuperar modelos LSTM
        """
        model = load_model(model_path)
        model.summary()
        return model

    def predictions(self, model, df, features, n_features, scaler_y, id_product, root_dir, steps=180, predictions=31, scaler_x=None, n_steps_model=None):
        """
        Función principal de predicciones para los modelos entrenados
        
        * Soportado: Multivariado - Multistep
        """
        data_trat = df.copy()
        if n_steps_model is None:
            try:
                dummy_input = np.zeros((1, steps, n_features))
                pred_shape = model.predict(dummy_input, verbose=0).shape
                if len(pred_shape) == 3:
                    n_steps_model = pred_shape[1]
                else:
                    n_steps_model = pred_shape[-1]
            except Exception:
                n_steps_model = predictions 
        total_steps = predictions
        preds = []
        temp_for_pred = df[features].tail(steps).copy()
        last_date = df.index.max().date()

        if scaler_x is not None:
            temp_for_pred[features] = scaler_x.fit_transform(temp_for_pred[features])

        for _ in range(0, total_steps, n_steps_model):
            x_input = temp_for_pred.to_numpy().reshape((1, steps, n_features))
            pred = model.predict(x_input, verbose=0)

            if pred.ndim == 3:
                pred_2d = pred.reshape(pred.shape[0], pred.shape[1])
            else:
                pred_2d = pred

            y_pred_inv = scaler_y.inverse_transform(pred_2d)
            n_to_take = min(n_steps_model, total_steps - len(preds))
            preds.extend(y_pred_inv.flatten()[:n_to_take].tolist())

            if len(preds) >= total_steps:
                break

            temp_for_pred = pd.concat(
                [temp_for_pred,
                 pd.DataFrame(
                    np.zeros((n_to_take, n_features)),
                    columns=features
                 )],
                ignore_index=True
            )

            target_col = features[-1]
            temp_for_pred.iloc[-n_to_take:, temp_for_pred.columns.get_loc(target_col)] = y_pred_inv.flatten()[:n_to_take]
            temp_for_pred = temp_for_pred.tail(steps).reset_index(drop=True)

            if scaler_x is not None:
                temp_for_pred[features] = scaler_x.fit_transform(temp_for_pred[features])

        predicciones = preds[:total_steps]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=len(predicciones), freq='D')
        
        pred_df = pd.DataFrame({
            'Fecha': future_dates,
            'Prediccion': predicciones
        })

        # GRAFICO DE PRONOSTICOS
        historico_df = data_trat[['OrderQuantity']].copy()
        historico_df = historico_df.reset_index().rename(columns={'index': 'Fecha'})
        historico_df = historico_df[['Fecha', 'OrderQuantity']]

        date_c = datetime.now().strftime('%H_%M_%S_%d-%m-%Y')
        plt_path = root_dir+f'/accuracy/Predictions/{date_c}'

        os.makedirs(plt_path,exist_ok=True)

        ##### MODIFICADOR DEL TIPO DE GRÁFICO Y COLORIMETRÍA
        plt.rcParams['figure.facecolor'] = '#001f3f'
        plt.rcParams['axes.facecolor'] = '#001f3f'
        plt.rcParams['text.color'] = 'white'
        plt.rcParams['axes.labelcolor'] = 'white'
        plt.rcParams['xtick.color'] = 'white'
        plt.rcParams['ytick.color'] = 'white'
        plt.rcParams['axes.edgecolor'] = 'white'

        # GRAFICO
        plt.figure(figsize=(12, 6))
        plt.plot(historico_df['Fecha'], historico_df['OrderQuantity'], label='Histórico Ventas (OrderQuantity)', color='#40E0D0', linewidth=2)
        plt.plot(pred_df['Fecha'], pred_df['Prediccion'], marker='o', color='orange', label='Pronóstico', linewidth=2)
        plt.xlabel('Periodo')
        plt.ylabel('Ventas')
        plt.title(f'HISTÓRICO DE VENTAS VS PRONÓSTICO DE LA DEMANDA \n Producto {id_product}')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.figtext(0.01, 0.01, "Realizado el: "+datetime.now().strftime('%H:%M:%S %d-%m-%Y'), fontsize=10, color="gray")
        plt.figtext(0.50, 0.01, "Gestión de Innovación en Tecnología Informática S.C.P. | Grupo Consultores®", fontsize=10, color="gray")
        plt.savefig(plt_path+f'/predictions_{date_c}.jpg',dpi=300)
        plt.show()
        plt.close()

    
    def retrainingModel(self, df, productKeyId, dictionary, new_data, epochs=None):
        """
        Realiza el reentrenamiento de un modelo LSTM existente con nuevos datos.

        Args:
            df (pd.DataFrame): DataFrame original usado para entrenar el modelo.
            productKeyId (str or int): ID del producto cuyo modelo se va a reentrenar.
            dictionary (dict): Diccionario de configuración de columnas y parámetros.
            new_data (pd.DataFrame): Nuevos datos para reentrenar el modelo.
            epochs (int, optional): Número de épocas para el reentrenamiento. Si es None, usa self.EPOCHS.

        Returns:
            model: El modelo reentrenado.
        """

        # Rutas y nombres
        category_id = 'CAT' + str(dictionary['CATEGORY_ID'])
        model_dir = os.path.join(dictionary['MODEL_PATH_DIR'], f'models/{productKeyId}/')
        model_name = f'mod{productKeyId}-{category_id}'
        model_path = os.path.join(model_dir, f'{model_name}.keras')
        utils_path = os.path.join(model_dir, 'utils')
        model = self.loadModel(model_path)
        scaler_x = self.reloadScaler(os.path.join(utils_path, 'Scaler_x.pkl'))
        scaler_y = self.reloadScaler(os.path.join(utils_path, 'Scaler_y.pkl'))

        with open(os.path.join(utils_path, 'only_scaled.pkl'), 'rb') as f:
            only_scaled = pickle.load(f)

        combined_df = pd.concat([df, new_data], ignore_index=False)
        combined_df = combined_df.sort_values([dictionary['FORECAST_ID_COLUMN'], 'Date'])
        data_trat = self.tratamiento_datos(combined_df, dictionary)

        # Columnas y features
        all_columns_l = dictionary['ALL_COLUMNS'].copy()
        all_columns_l.remove(dictionary['FORECAST_ID_COLUMN'])
        features = dictionary['FEATURES']
        target_col = dictionary['TARGET_COLUMN']
        new_col = dictionary['COLUMN_ADD']

        # Parámetros para entrenamiento
        parameters = {
            'category_id': category_id,
            'ID_model': str(productKeyId),
            'forecast_id_col': dictionary['FORECAST_ID_COLUMN'],
            'MODEL_NAME': model_name,
            'data_size': str(len(data_trat))
        }

        # Entrenamiento y validación
        x_train, y_train, x_val, y_val, scaler_y, scaler, only_scaled, df_val, parameters = self.create_x_y_train(
            data_trat, new_col, all_columns_l, features, target_col, parameters, False, self.PASOS, self.N_PREDICTIONS
        )

        # Reentrenar el modelo
        history = model.fit(
            x_train, y_train,
            epochs=epochs if epochs is not None else self.EPOCHS,
            batch_size=self.BATCH_SIZE,
            validation_data=(x_val, y_val),
            verbose=2,
            shuffle=False
        )

        # Guardado del modelo
        model.save(model_path)

        # Actualizacion de escaladores y datos
        joblib.dump(scaler, os.path.join(utils_path, 'Scaler_x.pkl'))
        joblib.dump(scaler_y, os.path.join(utils_path, 'Scaler_y.pkl'))
        with open(os.path.join(utils_path, 'only_scaled.pkl'), 'wb') as f:
            pickle.dump(only_scaled, f)
        df_val.to_csv(os.path.join(utils_path, 'LSLDWINDOW.csv'), index=True)

        return model


    def createModels(self, df_lst, productKeyIds, dictionary):
        """
        Función que crea los modelos conforme a cada producto
        Para este caso, se toman cada uno de los productos en una columna de productkey, se crean y se van entrenando cada uno
        de los modelos
        """

        col_gby = dictionary['FORECAST_ID_COLUMN']
        n_products = len(productKeyIds)
        category_id = 'CAT'+dictionary['CATEGORY_ID']
        new_col = dictionary['COLUMN_ADD']
        features = dictionary['FEATURES']
        target_col = dictionary['TARGET_COLUMN']

        seriesGB = df_lst.groupby(col_gby)
        all_columns_l = dictionary['ALL_COLUMNS'].copy()
        all_columns_l.remove(col_gby)

        path = dictionary['MODEL_PATH_DIR']+'/'+category_id

        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
        models_dir = {}
        for pk in productKeyIds:
            model_name = f'MOD{pk}_CAT{category_id}'

            parameters = {
            'category_id': category_id,
            'ID_model': str(pk)
            }

            dirmodelpath = path+f'/models/{pk}/'
            os.makedirs(dirmodelpath, exist_ok=True)
            model = self.create_model(dictionary['N_FEATURES'])
            pkSeries = seriesGB.get_group(str(pk)).copy()
            parameters['forecast_id_col'] = col_gby
            tt = int(len(pkSeries))
            parameters['data_size'] = str(tt)
            model_name = f'mod{pk}-{category_id}'
            model_path = dirmodelpath+f'/{model_name}.keras'
            parameters['MODEL_NAME'] = model_name
            x_train, y_train, x_val, y_val, scaler_y, scaler,  only_scaled, df_val, parameters = self.create_x_y_train(pkSeries, new_col, all_columns_l, features, target_col, parameters, True,self.PASOS,self.N_PREDICTIONS)
            model = self.train_model(x_train,y_train,x_val,y_val, model, dirmodelpath, scaler_y, parameters)
            model.save(model_path)
            utils_path = dirmodelpath+'/utils'
            os.makedirs(utils_path, exist_ok=True)
            name = utils_path+f'/LSLDWINDOW.csv'
            df_val.to_csv(name,index=True)

            # METADATA
            metadata = [
                {
                    "NAME":"LSLDWINDOW.csv",
                    "LAST_MODIFIED": datetime.now().strftime("%H_%M_%S-%d-%m-%Y"),
                    "LAST_REGISTERED_DATE": str(df_val.index.max()),
                    "HISTORIC_DATA_SIZE": str(len(df_val))
                }
            ]
            self.saveMetadata(metadata,utils_path)

            # GUARDAR ESCALADORES
            name = utils_path+"/Scaler_x.pkl"
            joblib.dump(scaler,name)
            name = utils_path+"/Scaler_y.pkl"
            joblib.dump(scaler_y, name)
            name = utils_path+"/only_scaled.pkl"

            with open(name, 'wb') as f:
                pickle.dump(only_scaled, f)
            models_dir[int(pk)] = model_path
            parameters = {}
            if pk == 234:
                break

        return models_dir
    
    #Core o función con flujo de secuencias establecido por defecto
    def main(self,category_id,data, data_Product):
        dataProduct = data_Product.copy()
        prepData = data.copy()
        copyprepData = prepData
        copyprepData = self.set_index_datetime(copyprepData)

        #----------- SECCION DE PARÁMETROS GENERALES ----------
        dirmodels_name = './models/'+datetime.now().strftime('%Y-%m-%d')

        dictionary = {
            "FORECAST_ID_COLUMN": "ProductKey",
            "FORECAST_ID_COLUMN_ENCODED": "ProductKey_encoded",
            "TARGET_COLUMN": "OrderQuantity",
            "COLUMN_ADD": "IsSelled",
            "CATEGORY_ID": category_id
        }
        #----------- / ----------

        if not os.path.exists(dirmodels_name):
            os.makedirs(dirmodels_name, exist_ok=True)
        path_model = dirmodels_name+f'/CAT_{category_id}-'+datetime.now().strftime('%H_%M_%S %d-%m-%Y')
        os.makedirs(path_model, exist_ok=True)
        dictionary['MODEL_PATH_DIR'] = path_model

        df = prepData.sort_values([dictionary['FORECAST_ID_COLUMN'], 'Date'])
        dfg = df.copy()
        productKeyIds = dfg[dictionary['FORECAST_ID_COLUMN']].unique()
        df[dictionary['FORECAST_ID_COLUMN']] = df[dictionary['FORECAST_ID_COLUMN']].astype(str)
        le = LabelEncoder()
        df[dictionary['FORECAST_ID_COLUMN_ENCODED']] = le.fit_transform(df[dictionary['FORECAST_ID_COLUMN']])
        features = list(df.columns.copy())
        dictionary['FEATURES'] = features
        features.append(dictionary['COLUMN_ADD'])
        all_columns = features.copy()
        dictionary['ALL_COLUMNS'] = all_columns
        features.remove(dictionary['TARGET_COLUMN'])
        features.remove(dictionary['FORECAST_ID_COLUMN'])
        n_features = int(len(features))
        dictionary['N_FEATURES'] = n_features
        dataProduct = dataProduct[dataProduct[dictionary['FORECAST_ID_COLUMN']].isin(productKeyIds)]
        data_trat = self.tratamiento_datos(df, dictionary)
        models_dir = self.createModels(data_trat,productKeyIds,dictionary)

        # METADATA
        metadata = [
            {
                "CATEGORY_ID": str(category_id),
                "Cantidad de modelos": str(len(models_dir))
            },
            models_dir,
            dictionary
        ]
        self.saveMetadata(metadata,path_model,'metadata_CAT')

