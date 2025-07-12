'''
LSTM ENGINE V1.1

Clase que integra las funcionalidades y servicios de LSTM

Tipo de Modelo soportado: Multivariado - Multistep

Los hiperparámetros están por defecto a un look forward de 180 días (6 meses), a 31 días de predicción

Es decir, que el modelo se le presentan secuencias de 6 meses para predecir el siguiente mes
'''

import pyodbc
import pandas as pd
import os
import shutil
import json
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.pylab as plt
import numpy as np
import joblib
import pickle
from datetime import date, datetime, timedelta

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, Input, Reshape
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import copy

#Configuracion de paraametros





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
        # self.sql_server= sql_server_config
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
            # data.sort_values('fecha', inplace=True)
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

        #obtener la fecha inicial
        for i, group in val.groupby(GROUP_ID):

            #obtenemos el productKeyActual
            producK_ID = group[GROUP_ID].iloc[0]

            #obtenemos el productkey
            productID = group['ProductKey'].iloc[0]

            #obtener la fecha inicial
            date_ini = group['Date'].min()

            #obtener la fecha final
            date_last = group['Date'].max()

            #Comvertimos la columna 'Date' a datetime
            group['Date'] = pd.to_datetime(group['Date'], errors='coerce')

            #ordenamos la columna
            group = group.sort_values('Date')
            
            #filtramos y obtenemos los valores únicamente en el rango establecido
            group = group[(group['Date']>= pd.Timestamp(date_ini)) & (group['Date']<=pd.Timestamp(date_last))]

            #seteamos como index
            group_index = group.set_index('Date')

            #Creamos el rango que debería tener lo ya establecido
            days_corrected = pd.date_range(start=date_ini, end=date_last - pd.Timedelta(days=1), freq='D')

            #Reindexamos para poder garantizar continuidad (rellenamos los días faltantes)
            group_reindex = group_index.reindex(days_corrected)

            if show_nan:
                #Comprobamos, debe haber NaN o NA
                print(group_reindex)

            #PASO 2: RELLENAR EL RESTO DE COLUMNAS FALTANTES
            group_reindex = group_reindex.reset_index()

            #Rellenar columnas relacionado a la fecha
            group_reindex['YEAR'] = group_reindex['index'].dt.year
            group_reindex['MONTH'] = group_reindex['index'].dt.month
            group_reindex['Day'] = group_reindex['index'].dt.day
            group_reindex['DayOfWeek'] = group_reindex['index'].dt.dayofweek
            group_reindex['IsWeekend'] = group_reindex['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

            #setear index
            group_reindex = group_reindex.set_index('index')

            #COLUMNA TARGET
            group_reindex[target_col] = group_reindex[target_col].fillna(0)

            #Asegurarnos de que es de tipo int
            group_reindex[target_col] = group_reindex[target_col].astype(int)

            #============================ PARTES PARA RELLENAR PRECIO UNITARIO Y DE ENVIO
            # Obtenemos el precio unitario
            # precioUNitario = dataProduct[dataProduct['ProductKey'] == producK_ID]['Price'].values[0]

            # #obtenemos el envio
            # precioEnvio = dataProduct[dataProduct['ProductKey'] == producK_ID]['StandardCost'].values[0]
            #============================ FIN

            #PRECIO UNITARIO
            group_reindex['UnitPrice'] = group_reindex['UnitPrice'].fillna(0.0)

            #DESCUENTO UNITARIO
            group_reindex['UnitPriceDiscountPct'] = group_reindex['UnitPriceDiscountPct'].fillna(0.0)

            #DESCUENTO ACUMULADO
            group_reindex['DiscountAmount'] = group_reindex['DiscountAmount'].fillna(0)

            #PRECIO DE ENVÍO
            group_reindex['ProductStandardCost'] = group_reindex['ProductStandardCost'].fillna(0.0)

            #VENTAS
            group_reindex['SalesAmount'] = group_reindex['SalesAmount'].fillna(0.0)

            group_reindex['ProductKey'] = productID

            #AGREGAR NUEVAMENTE EL PRODUCTKEY_ENCODED
            group_reindex[GROUP_ID] = producK_ID

            #INCLUIMOS UNA NUEVA COLUMNA QUE INDIQUE SI HAY VENTAS
            group_reindex[new_col] = group_reindex[target_col].apply(lambda x: 1 if x > 0 else 0)

            #CONTROLAR LA SALIDA DE LA DATA TRATADA
            if show_data_output:
                print(group_reindex)

            # Agregar la serie modificada a la lista
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

        #creamos una copia de la data
        if show_paremeters:
            print(var_columns)
            print(var_features)
        
        #Creamos una copia de la data
        # values = data.copy()
        temp = data.copy()
        values_y = data.copy()

        #eliminamos la columnas de la isSelled y IsWeekend
        deleted = [new_col, 'IsWeekend']

        #eliminamos de las columnas
        only_scaled = [x for x in var_columns if x not in deleted]

        # Determinar el índice de corte para train/val
        total_rows = len(data)

        #una variable en la que si una variable de validación es diferente de none
        #entonces, se tiene que 

        #El numero total de elementos menos 
        if n_validationData != None:
            n_train_rows = int(total_rows-n_validationData)
        else:
            n_train_rows = int(total_rows * self.TRAINING_PERCENTAGE)
        # n_val_rows = total_rows - n_train_rows

        # Particionar el DataFrame original
        df_train = data.iloc[:n_train_rows].copy()
        df_val = data.iloc[n_train_rows - steps - n_predictions + 1:].copy()  # Incluye pasos previos para ventana de validación

        #Obtendremos el ProdutKeyEncoded
        parameters['ID_FORECAST_ENC_VALUE'] = str(data['ProductKey_encoded'].iloc[0])

        #generamos una copia de la data
        df = df_val.copy()
        df = df.drop(columns=['ProductKey_encoded','IsSelled'])

        #Eliminar la columna de product key en validacion y entrenamiento
        del_col = parameters['forecast_id_col']
        df_train = df_train.drop(columns=[del_col], axis=1)
        df_val = df_val.drop(columns=[del_col], axis=1)

        #añadir el total de elementos de entrenamiento al modelo
        parameters['DATOS_ENTRENAMIENTO'] = len(df_train)
        parameters['LAST_DATE'] = df_train.index.max().date()
        parameters['DATOS_VALIDACION'] = len(df_val)

        scaler = MinMaxScaler(feature_range=(-1, 1))

        #generamos un escalador para las feature target 'Y'
        scaler_y = MinMaxScaler(feature_range=(-1, 1))

        # Escalar train
        df_train[only_scaled] = scaler.fit_transform(df_train[only_scaled])

        #Escalador que servirá para las predicciones
        scaler_y.fit_transform(values_y[[target_col]])

        # Escalar val usando el mismo scaler
        df_val[only_scaled] = scaler.fit_transform(df_val[only_scaled])

        # Crear ventanas para train
        X_train, y_train = self.split_sequences(df_train, steps, n_predictions, var_features, target_col)
        
        # Crear ventanas para val
        X_val, y_val = self.split_sequences(df_val, steps, n_predictions, var_features, target_col)

        ####Visualizacion de parámetros#####
        if show_paremeters:
            print("Columns :",df_train.columns)
            print("Columns :",df_val.columns)
        ####################################

        ####Visualizacion de parámetros#####
        if show_paremeters:
            # print("Data length: ",len(values)," (values)")
            print("Num. Train Days: ",n_train_rows)
        ####################################

        ####Visualizacion de parámetros#####
        if show_paremeters:
            print(f"Tensors: (X_train): {X_train.shape}, (y_train): {y_train.shape}")
            print(f"Tensors:  (X_val): {X_val.shape}, (y_val): {y_val.shape}")
        ####################################

        return X_train, y_train, X_val, y_val, scaler_y, scaler, only_scaled, df, parameters

    def train_model(self,x_train, y_train, x_val, y_val, model, model_path, scaler_y, parameters):
        early_stop = EarlyStopping(monitor='val_mae', patience=5, restore_best_weights=True)
        history = model.fit(
            x_train, y_train,
            epochs= self.EPOCHS,
            batch_size=self.BATCH_SIZE,
            validation_data=(x_val, y_val),
            verbose=2,
            shuffle=False
        )

        # Predicciones sobre el set de validación
        results = model.predict(x_val)

        #Creamos la carpeta principal accuracy
        path = model_path+"/accuracy/"
        os.makedirs(path, exist_ok=True)

        #definimos el modo
        datetim_e_path = datetime.now().strftime('%H-%M-%S_%d-%m-%Y')
        path = path+'Training/'+datetim_e_path

        #dependiendo del caso, creamos la carpeta de accuracy
        os.makedirs(path,exist_ok=True)

        # Si la predicción es multistep (2D), tomar solo la primera columna para comparar con y_val
        if results.ndim == 2 and results.shape[1] > 1:
            results_plot = results[:, 0]
            y_val_plot = y_val[:, 0]
        else:
            results_plot = results.flatten()
            y_val_plot = y_val.flatten()

        # Si tienes el scaler_y, desescala
        results_plot = scaler_y.inverse_transform(results_plot.reshape(-1, 1)).flatten()
        y_val_plot = scaler_y.inverse_transform(y_val_plot.reshape(-1, 1)).flatten()

        # Convertir a entero
        results_plot = results_plot.astype(int)
        y_val_plot = y_val_plot.astype(int)

        ##### MODIFICADOR DEL TIPO DE GRÁFICO Y COLORIMETRÍA
        plt.rcParams['figure.facecolor'] = '#001f3f'  # Fondo total (figura)      
        plt.rcParams['axes.facecolor'] = '#001f3f'    # Fondo del área del gráfico

        # Opcional: cambiar color del texto para que sea visible
        plt.rcParams['text.color'] = 'white'
        plt.rcParams['axes.labelcolor'] = 'white'
        plt.rcParams['xtick.color'] = 'white'
        plt.rcParams['ytick.color'] = 'white'
        plt.rcParams['axes.edgecolor'] = 'white'
        ### / ######

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

        # Gráfico de pérdidas (loss) de entrenamiento y validación
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

        # Gráfico de MSE de entrenamiento y validación (si está disponible)
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
        # Obtener el resumen del modelo en formato JSON y formatearlo correctamente
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

        #guardar la metadata
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
        Función que se encarga de recargar el modelo o reconstruirlo como estuvo originalmente
        """

        enable = False

        # Obtener el directorio padre 
        root_dir = os.path.dirname(model_path)

        # Utils dir
        util_dir = root_dir+'/utils/'

        # Cargar el CSV como dataframe
        df = pd.read_csv(util_dir+'LSLDWINDOW.csv')

        # Cargar los escaladores
        Scaler_x = self.reloadScaler(util_dir+'Scaler_x.pkl')
        Scaler_y = self.reloadScaler(util_dir+'Scaler_y.pkl')
        only_scaled = self.reloadArray(util_dir+'only_scaled.pkl')

        #Cargar el modelo
        model = self.loadModel(model_path)

        #renombrar 
        dictionary = category_metadata

        features = dictionary['FEATURES']
        n_features = dictionary['N_FEATURES']
        steps = int(model_metadata['GENERAL_INFO']['STEPS'])
        p_default = int(model_metadata['GENERAL_INFO']['PREDICTIONS_DEFAULT'])

        # Validar que todos los objetos se hayan cargado correctamente
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
            #Cambiar el nomnbre del index
            df = df.rename(columns={"index": "Date"})

            # Convertir a tipo dataframe la columna Date
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

            df = self.set_index_datetime(df)

            #obtenemos el id del producto
            id_product = df[dictionary['FORECAST_ID_COLUMN']].iloc[0]

            #ordenamos la data por product key
            df = df.sort_values([dictionary['FORECAST_ID_COLUMN'], 'Date'])
            
            #Codificamos el productKey a labelEncoder
            df[dictionary['FORECAST_ID_COLUMN_ENCODED']] = model_metadata['GENERAL_INFO']['ID_ENCODED']

            # Hacemos tratamiento de datos
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
        Función que permite recuperar array usando pickle
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
        Realiza predicciones multistep, permitiendo encadenar predicciones si se requieren más pasos que los que el modelo predice por llamada.
        """
        data_trat = df.copy()
        # Determinar cuántos pasos predice el modelo por llamada
        if n_steps_model is None:
            try:
                dummy_input = np.zeros((1, steps, n_features))
                pred_shape = model.predict(dummy_input, verbose=0).shape
                if len(pred_shape) == 3:
                    n_steps_model = pred_shape[1]
                else:
                    n_steps_model = pred_shape[-1]
            except Exception:
                n_steps_model = predictions  # fallback
        total_steps = predictions
        preds = []

        # Usar las últimas 'steps' filas como ventana inicial
        temp_for_pred = df[features].tail(steps).copy()

        last_date = df.index.max().date()

        # Escalar los datos de entrada si se proporciona scaler_x
        if scaler_x is not None:
            temp_for_pred[features] = scaler_x.fit_transform(temp_for_pred[features])

        for _ in range(0, total_steps, n_steps_model):
            # Preparar input
            x_input = temp_for_pred.to_numpy().reshape((1, steps, n_features))
            pred = model.predict(x_input, verbose=0)
            # Si pred es 3D, reducir a 2D
            if pred.ndim == 3:
                pred_2d = pred.reshape(pred.shape[0], pred.shape[1])
            else:
                pred_2d = pred
            # Inversa de la escala
            y_pred_inv = scaler_y.inverse_transform(pred_2d)
            # Tomar solo los pasos requeridos en esta iteración
            n_to_take = min(n_steps_model, total_steps - len(preds))
            preds.extend(y_pred_inv.flatten()[:n_to_take].tolist())
            # Para la siguiente iteración, agregar las nuevas predicciones al final de la ventana
            # y quitar del inicio para mantener el tamaño de la ventana
            if len(preds) >= total_steps:
                break
            # Crear un nuevo DataFrame para la ventana siguiente
            # Rellenar las columnas de features, solo la columna target se actualiza con la predicción
            # Suponemos que la columna target es la última de features
            temp_for_pred = pd.concat(
                [temp_for_pred,
                 pd.DataFrame(
                    np.zeros((n_to_take, n_features)),
                    columns=features
                 )],
                ignore_index=True
            )
            # Insertar las predicciones en la columna target
            target_col = features[-1]
            temp_for_pred.iloc[-n_to_take:, temp_for_pred.columns.get_loc(target_col)] = y_pred_inv.flatten()[:n_to_take]
            # Mantener solo las últimas 'steps' filas
            temp_for_pred = temp_for_pred.tail(steps).reset_index(drop=True)
            # Si scaler_x existe, volver a escalar la ventana (excepto la columna target, que ya está en escala original)
            if scaler_x is not None:
                temp_for_pred[features] = scaler_x.fit_transform(temp_for_pred[features])

        # return preds[:total_steps], last_date
        predicciones = preds[:total_steps]
        # Generar rango de fechas futuras a partir de last_date
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=len(predicciones), freq='D')
        
        # Crear DataFrame para visualizar fechas y predicciones juntas
        pred_df = pd.DataFrame({
            'Fecha': future_dates,
            'Prediccion': predicciones
        })

        # Obtener histórico de ventas reales
        historico_df = data_trat[['OrderQuantity']].copy()
        historico_df = historico_df.reset_index().rename(columns={'index': 'Fecha'})
        historico_df = historico_df[['Fecha', 'OrderQuantity']]

        #Obtener el path para guardar el las predicciones
        date_c = datetime.now().strftime('%H_%M_%S_%d-%m-%Y')
        plt_path = root_dir+f'/accuracy/Predictions/{date_c}'

        os.makedirs(plt_path,exist_ok=True)

        ##### MODIFICADOR DEL TIPO DE GRÁFICO Y COLORIMETRÍA
        plt.rcParams['figure.facecolor'] = '#001f3f'  # Fondo total (figura)      
        plt.rcParams['axes.facecolor'] = '#001f3f'    # Fondo del área del gráfico

        # Opcional: cambiar color del texto para que sea visible
        plt.rcParams['text.color'] = 'white'
        plt.rcParams['axes.labelcolor'] = 'white'
        plt.rcParams['xtick.color'] = 'white'
        plt.rcParams['ytick.color'] = 'white'
        plt.rcParams['axes.edgecolor'] = 'white'
        ### / ######

        # Graficar histórico y predicciones
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

    def retrainingModel_2(self, model, df, features, n_features, Scaler_y, steps, Scaler_x, p_default, root_dir, id_product, CAT_metadata, epochs=None):
        """
        Realiza el reentrenamiento de un modelo LSTM existente con datos nuevos.

        Args:
            model: El modelo LSTM ya cargado.
            df (pd.DataFrame): DataFrame con los datos completos para reentrenar.
            features (list): Lista de columnas/features a usar.
            n_features (int): Número de features.
            Scaler_y: Escalador para la variable objetivo.
            steps (int): Número de pasos de entrada.
            Scaler_x: Escalador para las features.
            p_default (int): Número de predicciones a realizar.
            root_dir (str): Directorio raíz donde guardar el modelo y utilidades.
            id_product: ID del producto.
            CAT_metadata (dict): Diccionario de configuración de la categoría.
            epochs (int, optional): Número de épocas para el reentrenamiento.

        Returns:
            model: El modelo reentrenado.
        """
        # Parámetros para entrenamiento
        parameters = {
            'category_id': CAT_metadata.get('CATEGORY_ID', ''),
            'ID_model': str(id_product),
            'forecast_id_col': CAT_metadata.get('FORECAST_ID_COLUMN', ''),
            'MODEL_NAME': f'mod{id_product}-CAT{CAT_metadata.get("CATEGORY_ID", "")}',
            'data_size': str(len(df))
        }

        # Definir columnas
        new_col = CAT_metadata.get('COLUMN_ADD', '')
        all_columns_l = CAT_metadata.get('ALL_COLUMNS', features + [new_col])
        if parameters['forecast_id_col'] in all_columns_l:
            all_columns_l = [x for x in all_columns_l if x != parameters['forecast_id_col']]
        target_col = CAT_metadata.get('TARGET_COLUMN', '')

        # Crear sets de entrenamiento y validación
        x_train, y_train, x_val, y_val, scaler_y, scaler, only_scaled, df_val, parameters = self.create_x_y_train(
            df, new_col, all_columns_l, features, target_col, parameters, False, steps, p_default
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

        # Guardar el modelo reentrenado
        model_dir = os.path.join(root_dir)
        model_name = f'mod{id_product}-CAT{CAT_metadata.get("CATEGORY_ID", "")}.keras'
        model_path = os.path.join(model_dir, model_name)
        utils_path = os.path.join(model_dir, 'utils')
        os.makedirs(utils_path, exist_ok=True)
        model.save(model_path)

        # Ruta del archivo de metadata en el root_dir
        metadata_path = os.path.join(root_dir, 'metadata.json')

        # Leer la metadata existente si el archivo existe
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = []

        # Modificar partes de la metadata según sea necesario
        # Ejemplo: actualizar la fecha de modificación y el tamaño de datos
        for entry in metadata:
            if isinstance(entry, dict) and "GENERAL_INFO" in entry:
                entry["GENERAL_INFO"]["FECHA_MODIFICACION"] = datetime.now().strftime('%H:%M:%S %d-%m-%Y')
                entry["GENERAL_INFO"]["DATOS_ENTRENAMIENTO"] = str(len(df))
                entry["GENERAL_INFO"]["DATOS_VALIDACION"] = str(len(df_val))

        # Guardar la metadata modificada
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        # Actualizar los escaladores y datos
        # joblib.dump(scaler, os.path.join(utils_path, 'Scaler_x.pkl'))
        # joblib.dump(scaler_y, os.path.join(utils_path, 'Scaler_y.pkl'))
        # with open(os.path.join(utils_path, 'only_scaled.pkl'), 'wb') as f:
        #     pickle.dump(only_scaled, f)
        # df_val.to_csv(os.path.join(utils_path, 'LSLDWINDOW.csv'), index=True)

        return model

        

    
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

        # Definir rutas y nombres
        category_id = 'CAT' + str(dictionary['CATEGORY_ID'])
        model_dir = os.path.join(dictionary['MODEL_PATH_DIR'], f'models/{productKeyId}/')
        model_name = f'mod{productKeyId}-{category_id}'
        model_path = os.path.join(model_dir, f'{model_name}.keras')
        utils_path = os.path.join(model_dir, 'utils')

        # Cargar modelo existente
        model = self.loadModel(model_path)

        # Cargar escaladores y columnas
        scaler_x = self.reloadScaler(os.path.join(utils_path, 'Scaler_x.pkl'))
        scaler_y = self.reloadScaler(os.path.join(utils_path, 'Scaler_y.pkl'))
        with open(os.path.join(utils_path, 'only_scaled.pkl'), 'rb') as f:
            only_scaled = pickle.load(f)

        # Preparar datos combinando df y new_data
        combined_df = pd.concat([df, new_data], ignore_index=False)
        combined_df = combined_df.sort_values([dictionary['FORECAST_ID_COLUMN'], 'Date'])

        # Hacer tratamiento de datos
        data_trat = self.tratamiento_datos(combined_df, dictionary)

        # Obtener columnas y features
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

        # Crear sets de entrenamiento y validación
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

        # Guardar el modelo reentrenado
        model.save(model_path)

        # Actualizar los escaladores y datos
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

        #Agrupamos por ProductKey
        seriesGB = df_lst.groupby(col_gby)
        all_columns_l = dictionary['ALL_COLUMNS'].copy()
        all_columns_l.remove(col_gby)

        #obtener el directorio actual y crear dentro de la carpeta models 
        path = dictionary['MODEL_PATH_DIR']+'/'+category_id

        # ----------- CREACION DE CARRPETAS -------------
        #Creamos la carpeta padre
        if os.path.exists(path):
            shutil.rmtree(path)

        os.makedirs(path, exist_ok=True)
        # ----------- / -------------

        models_dir = {}

        #Por cada producto
        for pk in productKeyIds:

            #generamos un nombre para el modelo
            model_name = f'MOD{pk}_CAT{category_id}'

            parameters = {
            'category_id': category_id,
            'ID_model': str(pk)
            }

            #generamos un nombre para el modelo
            dirmodelpath = path+f'/models/{pk}/'
            os.makedirs(dirmodelpath, exist_ok=True)

            # Creamos un modelo LSTM
            model = self.create_model(dictionary['N_FEATURES'])

            # Obtener data de acuerdo al product key
            pkSeries = seriesGB.get_group(str(pk)).copy()

            # Aniadir la columna de productkey
            parameters['forecast_id_col'] = col_gby

            # Obtenemos el total de la data
            tt = int(len(pkSeries))

            parameters['data_size'] = str(tt)

            #nombre para el modelo
            model_name = f'mod{pk}-{category_id}'
            model_path = dirmodelpath+f'/{model_name}.keras'

            #guardamos en parametros
            parameters['MODEL_NAME'] = model_name

            # Pasamos por create x_y_train
            x_train, y_train, x_val, y_val, scaler_y, scaler,  only_scaled, df_val, parameters = self.create_x_y_train(pkSeries, new_col, all_columns_l, features, target_col, parameters, True,self.PASOS,self.N_PREDICTIONS)

            # Entrenamos el modelo y obtenemos su rendimiento y acuracy
            model = self.train_model(x_train,y_train,x_val,y_val, model, dirmodelpath, scaler_y, parameters)

            # Guardamos el modelo
            model.save(model_path)

            # Creamos una carpeta donde guardar los escaladores
            utils_path = dirmodelpath+'/utils'
            os.makedirs(utils_path, exist_ok=True)

            # Guardamos el dataframe
            name = utils_path+f'/LSLDWINDOW.csv'
            df_val.to_csv(name,index=True)

            # Generamos una metadata para el csv
            metadata = [
                {
                    "NAME":"LSLDWINDOW.csv",
                    "LAST_MODIFIED": datetime.now().strftime("%H_%M_%S-%d-%m-%Y"),
                    "LAST_REGISTERED_DATE": str(df_val.index.max()),
                    "HISTORIC_DATA_SIZE": str(len(df_val))
                }
            ]

            self.saveMetadata(metadata,utils_path)

            #Guardamos el scaler x
            name = utils_path+"/Scaler_x.pkl"
            joblib.dump(scaler,name)

            #guardamos el scaler y
            name = utils_path+"/Scaler_y.pkl"
            joblib.dump(scaler_y, name)

            #Guardamos only_scaled
            name = utils_path+"/only_scaled.pkl"
            with open(name, 'wb') as f:
                pickle.dump(only_scaled, f)

            # regustramos la ubicación del modelo
            models_dir[int(pk)] = model_path

            #eliminamos todo lo que habia en parametros
            parameters = {}

            if pk == 234:
                break

        return models_dir
    
    #Core o función con flujo de secuencias establecido por defecto
    def main(self,category_id,data, data_Product):
        dataProduct = data_Product.copy()
        prepData = data.copy()
        category_id_1 = category_id
        print(category_id_1)
        
        # historyData = pd.read_sql_query(self.query2, cursor)
        copyprepData = prepData
        copyprepData = self.set_index_datetime(copyprepData)
        # historyData = self.set_index_datetime(historyData)

        #----------- SECCION DE PARÁMETROS GENERALES ----------
        # Obtenemos el path del modelo
        dirmodels_name = './models/'+datetime.now().strftime('%Y-%m-%d')

        # Identificar la columna que identifique a cada serie temporal
        # forecast_id_col = 'ProductKey'

        # forecast_id_col_enc = 'ProductKey_encoded'

        # # Columna que deseamos predecir
        # target_col = 'OrderQuantity'

        # # Generamos una columna para detectar si hay ventas o no
        # isSelledCol = 'IsSelled'

        dictionary = {
            "FORECAST_ID_COLUMN": "ProductKey",
            "FORECAST_ID_COLUMN_ENCODED": "ProductKey_encoded",
            "TARGET_COLUMN": "OrderQuantity",
            "COLUMN_ADD": "IsSelled",
            "CATEGORY_ID": category_id
        }
        #----------- / ----------

        #Creamos la carpeta
        if not os.path.exists(dirmodels_name):
            os.makedirs(dirmodels_name, exist_ok=True)

        path_model = dirmodels_name+f'/CAT_{category_id}-'+datetime.now().strftime('%H_%M_%S %d-%m-%Y')
        os.makedirs(path_model, exist_ok=True)
        dictionary['MODEL_PATH_DIR'] = path_model

        #ordenamos la data por product key
        df = prepData.sort_values([dictionary['FORECAST_ID_COLUMN'], 'Date'])

        #Creamos la copia del dataframe
        dfg = df.copy()

        #obtenemos la lista de los elementos
        productKeyIds = dfg[dictionary['FORECAST_ID_COLUMN']].unique()

        #convertir ProductKey a String
        df[dictionary['FORECAST_ID_COLUMN']] = df[dictionary['FORECAST_ID_COLUMN']].astype(str)

        #label encoder
        le = LabelEncoder()

        #Codificamos el productKey a labelEncoder
        df[dictionary['FORECAST_ID_COLUMN_ENCODED']] = le.fit_transform(df[dictionary['FORECAST_ID_COLUMN']])

        #generamos una lista de las 'x_features'
        features = list(df.columns.copy())
        dictionary['FEATURES'] = features

        #agregamos la nueva columna
        features.append(dictionary['COLUMN_ADD'])

        #obtenemos una copia de todas las columnas
        all_columns = features.copy()
        dictionary['ALL_COLUMNS'] = all_columns

        #eliminamos la columna target
        features.remove(dictionary['TARGET_COLUMN'])

        #Eliminamos la columna del productkey
        features.remove(dictionary['FORECAST_ID_COLUMN'])

        # #Preparar la columna de dia
        # first_day = copyprepData.index.min() + timedelta(days=1)
        # last_day = copyprepData.index.max() + timedelta(days=1)
        # future_days = [last_day + timedelta(days=i) for i in range(self.N_PREDICTIONS)]
        # for i in range(len(future_days)):
        #     future_days[i] = str(future_days[i])[:10]
        
        # #De momento solamente será la fecha
        # future_data = pd.DataFrame(future_days, columns=['fecha'])

        #obtenemos el total de features con las que se hará el entrenamiento del modelo
        n_features = int(len(features))
        dictionary['N_FEATURES'] = n_features

        # Filtrar productos que están en la serie temporal
        dataProduct = dataProduct[dataProduct[dictionary['FORECAST_ID_COLUMN']].isin(productKeyIds)]

        #Haremos tratamiento de datos
        data_trat = self.tratamiento_datos(df, dictionary)

        #Hacer creacion de los modelos y guardar 
        models_dir = self.createModels(data_trat,productKeyIds,dictionary)

        # #eliminamos 
        # del dictionary['MODEL_PATH_DIR']
        # del dictionary['CATEGORY_ID']

        #Guardar como metadata la direccion de los modelos pertenecientes a esa categoría
        metadata = [
            {
                "CATEGORY_ID": str(category_id),
                "Cantidad de modelos": str(len(models_dir))
            },
            models_dir,
            dictionary
        ]

        #Guardamos la información sobre las direcciones de los modelos
        self.saveMetadata(metadata,path_model,'metadata_CAT')

