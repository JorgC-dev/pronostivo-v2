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
from datetime import date, datetime, timedelta

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, Input, Reshape
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Configuracion de paraametros





class lstm_engine: 
    def __init__(self, sql_server_config, query,query_products,
                pasos=180,
                training_percentage=0.7,
                n_predictions = 31,
                epochs = 30,
                neurons = 100,
                batch_size = 15,
                isfloatdata = True,
                ):
        self.sql_server= sql_server_config
        self.query = query
        self.query_products = query_products
        self.PASOS = pasos # No. DE OBSERVACIONES EN EL TIEMPO PARA LA DATA Y PARA ALGORITMO DE 'SLIDDING WINDOW'
        self.TRAINING_PERCENTAGE = training_percentage  #PORCENTAJE DE DATOS A TOMAR PARA ENTRENAMIENTO
        self.N_PREDICTIONS = n_predictions  #NUMERO DE PREDICCIONES A REALIZAR
        self.EPOCHS = epochs #EPOCAS DE ENTRENAMIENTO DEL MODELO
        self.NEURONS = neurons #Mismo que el de pasos
        self.BATCH_SIZE = batch_size #TAMANIO DE LAS MUESTRAS DE ENTRENAMIENTO
        self.ISFLOATDATA = isfloatdata #VARIABLE QUE CONVIERTE LOS DATOS A PUNTO FLOTANTE

    def get_sqlconnection(self):
        """
        Método para realizar conexión hacia el servidor de la base de datos de donde extraer la data
        """
        status = "inicializando...."
        try: 
            connection = pyodbc.connect(self.sql_server_config)
            status = "Conexion establecida satisfactoriamente"
        except Exception as e:
            status = "Error al establecer la conexión:"+e
        print(status)
        return connection
    
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
    
    def saveMetadata(self,metadata,path):
        metadata_path = path+'/metadata.json'

        with open(metadata_path, 'w') as file: 
            json.dump(metadata)
    
    def tratamiento_datos(self,data, columns, features, target_col, new_col, productkeyIds,dataProduct, GROUP_ID, show_nan=False, show_data_output=False):
        """
        Realiza una mejora y garantiza la secuencia temporal de los datos a manejar para LSTM
        """

        val = data.copy()
        val = val.reset_index()
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

    def create_x_y_train(self,data, new_col, var_columns, var_features, target_col, parameters, show_paremeters=False, steps=PASOS, n_predictions=N_PREDICTIONS):
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
        values = data.copy()
        temp = data.copy()
        values_y = data.copy()

        #eliminamos la columnas de la isSelled y IsWeekend
        deleted = [new_col, 'IsWeekend']

        #eliminamos de las columnas
        only_scaled = [x for x in var_columns if x not in deleted]

        # Determinar el índice de corte para train/val
        total_rows = len(data)
        n_train_rows = int(total_rows * self.TRAINING_PERCENTAGE)
        # n_val_rows = total_rows - n_train_rows

        # Particionar el DataFrame original
        df_train = data.iloc[:n_train_rows].copy()
        df_val = data.iloc[n_train_rows - steps - n_predictions + 1:].copy()  # Incluye pasos previos para ventana de validación

        #añadir el total de elementos de entrenamiento al modelo
        print(df_train.columns)
        parameters['DATOS_ENTRENAMIENTO'] = len(df_train)
        parameters['LAST_DATE'] = '2021-03-04'

        scaler = MinMaxScaler(feature_range=(-1, 1))

        #generamos un escalador para las feature target 'Y'
        scaler_y = MinMaxScaler(feature_range=(-1, 1))

        if self.ISFLOATDATA:
            # Convertimos las columnas a punto flotante
            values = values.astype('float32')
            temp = temp.astype('float32')

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
            print("Data length: ",len(values)," (values)")
            print("Num. Train Days: ",n_train_rows)
        ####################################

        ####Visualizacion de parámetros#####
        if show_paremeters:
            print(f"Tensors: (X_train): {X_train.shape}, (y_train): {y_train.shape}")
            print(f"Tensors:  (X_val): {X_val.shape}, (y_val): {y_val.shape}")
        ####################################

        return X_train, y_train, X_val, y_val, scaler_y, scaler,only_scaled, parameters

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

        plt.figure(figsize=(12, 6))
        plt.plot(y_val_plot, label='Valores Reales', color='g', linewidth=2)
        plt.plot(results_plot, label='Valores Pronosticados', color='orange', linewidth=2)
        plt.xlabel('Índice')
        plt.ylabel('Valores')
        plt.title('Comparación: Valores reales vs Pronósticos')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.figtext(0.01, 0.01, "Realizado el: "+datetime.now().strftime('%H:%M:%S %d-%m-%Y'), fontsize=10, color="gray")
        plt.figtext(0.60, 0.01, "Gestión de Innovación en Tecnología Informática S.C.P. | Grupo Consultores®", fontsize=10, color="gray")
        plt.savefig(path+"/performance_validation_1.jpg",dpi=300)
        plt.show()

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
        plt.show()

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
            plt.show()

        # Preparamos la data a guardar en metadata
        datetim_e = datetime.now().strftime('%H:%M:%S %d-%m-%Y')
        optimizer_model = model.optimizer.get_config()
        model_summary = model.to_json()

        metadata = [
            {
                "MODEL_REFERENCY":{
                    "CATEGORY": parameters['category_id'],
                    "ID": parameters['ID_model']
                }
            },
            {
                "GENERAl_INFO":{
                    "TOTAL_DE_DATOS": parameters['data_size'],
                    "EPOCH": len(history.history['loss']),
                    "FECHA_ENTRENAMIENTO": datetim_e,
                    "FECHA_MODIFICACION": datetim_e
                }
            },
            {
                "Model Architecture: ": model_summary
            },
            {
                "OPTIMIZER_CONFIG": optimizer_model
            },
        ]

        #guardar la metadata
        self.saveMetadata(metadata,path)

        return model


    def newModelLSTM(self,n_features):
        """
        Función principal que crea la arquitectura y los modelos LSTM
        """
        dropout_rate = 0.2
        model = Sequential()
        model.add(LSTM(self.NEURONS, input_shape=(self.PASOS, n_features), return_sequences= True))
        model.add(LSTM(int(self.NEURONS/2)))
        model.add(Dense(self.NEURONS, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(self.N_PREDICTIONS, activation='linear'))  
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        model.summary()
        return model
    

    def predictions(self, model, df, features, n_features, scaler_y, steps=30):
        """
        Función dedicada a realizar predicciones
        """
        # Seleccionar solo las columnas de features usadas por el modelo
        temp_for_pred = df[features].tail(steps)
        x_input = temp_for_pred.to_numpy().reshape((1, steps, n_features))

        # Realizar la predicción multistep (la salida es de tamaño N_PREDICTIONS)
        pred = model.predict(x_input, verbose=0)  # pred.shape = (1, N_PREDICTIONS) or (1, N_PREDICTIONS, 1)

        # Si pred es 3D, reducir a 2D
        if pred.ndim == 3:
            pred_2d = pred.reshape(pred.shape[0], pred.shape[1])
        else:
            pred_2d = pred
        
        y_pred_inv = scaler_y.inverse_transform(pred_2d)  # y_pred_inv.shape = (1, N_PREDICTIONS)
        
        # Distribuir las predicciones individuales en la lista
        predicciones = y_pred_inv.flatten().tolist()

        return predicciones


    def createModels(self, df_lst, productKeyIds, col_gby, columns, n_features, new_col, features, target_col, category_id, forecast_id_col, dirmodels_name):
        """
        Función que crea los modelos conforme a cada producto
        Para este caso, se toman cada uno de los productos en una columna de productkey, se crean y se van entrenando cada uno
        de los modelos
        """

        n_products = len(productKeyIds)
        category_id = 'CAT'+category_id

        print("Total de modelos a crear: ", n_products)

        #Agrupamos por ProductKey
        seriesGB = df_lst.groupby(col_gby)
        all_columns_l = columns.copy()
        all_columns_l.remove(col_gby)

        #obtener el directorio actual y crear dentro de la carpeta models 
        path = dirmodels_name+'/'+category_id

        # ----------- CREACION DE CARRPETAS -------------
        #Creamos la carpeta padre
        if os.path.exists(path):
            shutil.rmtree(path)

        os.makedirs(path, exist_ok=True)
        # ----------- / -------------

        models_dir = []

        #Por cada producto
        for pk in productKeyIds:

            #generamos un nombre para el modelo
            model_name = f'MOD{pk}_CAT{category_id}'

            parameters = {
            'category_id': category_id,
            'ID_model': pk
            }

            #generamos un nombre para el modelo
            dirmodelpath = path+f'/models/{pk}/'
            os.makedirs(dirmodelpath, exist_ok=True)

            # Creamos un modelo LSTM
            model = self.create_model(n_features)

            # Obtener data de acuerdo al product key
            pkSeries = seriesGB.get_group(str(pk)).copy()

            # Eliminar la columna de productkey
            pkSeries = pkSeries.drop([forecast_id_col], axis=1)

            # Obtenemos el total de la data
            tt = int(len(pkSeries))

            # Pasamos por create x_y_train
            x_train, y_train, x_val, y_val, scaler_y, scaler, only_scaled, parameters = self.create_x_y_train(pkSeries, new_col, all_columns_l, features, target_col, parameters, True)

            # Entrenamos el modelo y obtenemos su rendimiento y acuracy
            model = self.train_model(x_train,y_train,x_val,y_val, model, scaler_y, parameters)

            # Guardamos el modelo
            model_name = dirmodelpath+f'/mod{pk}-cat{category_id}.keras'
            model.save(model_name)

            # regustramos la ubicación del modelo
            models_dir[pk] = model_name

        return models_dir
    
    #Core o función con flujo de secuencias establecido por defecto
    def main(self,category_id):
        with self.get_sqlconnection(self.sql_server) as cursor: 
            prepData = pd.read_sql_query(self.query,cursor)
            historyData = pd.read_sql_query(self.query2, cursor)
            copyprepData = prepData
            copyprepData = self.set_index_datetime(copyprepData)
            historyData = self.set_index_datetime(historyData)

            #----------- SECCION DE PARÁMETROS GENERALES ----------
            # Obtenemos el path del modelo
            dirmodels_name = './models/'+datetime.now().strftime('%Y-%m-%d')

            # Identificar la columna que identifique a cada serie temporal
            forecast_id_col = 'ProductKey'

            forecast_id_col_enc = 'ProductKey_encoded'

            # Columna que deseamos predecir
            target_col = 'OrderQuantity'

            # Generamos una columna para detectar si hay ventas o no
            isSelledCol = 'IsSelled'
            #----------- / ----------

            #Creamos la carpeta
            if not os.path.exists(dirmodels_name):
                os.makedirs(dirmodels_name, exist_ok=True)

            #Creamos una subcarpeta con la fecha y hora

            path_model = dirmodels_name+f'/CAT_{category_id}-'+datetime.now().strftime('%H:%M:%S %d-%m-%Y')
            os.makedirs(path_model, exist_ok=True)

            #ordenamos la data por product key
            df = prepData.sort_values([forecast_id_col, 'Date'])

            #Creamos la copia del dataframe
            dfg = df.copy()

            #obtenemos la lista de los elementos
            productKeyIds = dfg[forecast_id_col].unique()

            #convertir ProductKey a String
            df[forecast_id_col] = df[forecast_id_col].astype(str)

            #label encoder
            le = LabelEncoder()

            #Codificamos el productKey a labelEncoder
            df[forecast_id_col_enc] = le.fit_transform(df[forecast_id_col])

            #generamos una lista de las 'x_features'
            features = list(df.columns.copy())

            #agregamos la nueva columna
            features.append(isSelledCol)

            #obtenemos una copia de todas las columnas
            all_columns = features.copy()

            #eliminamos la columna target
            features.remove(target_col)

            #Eliminamos la columna del productkey
            features.remove(forecast_id_col)

            #Preparar la columna de dia
            first_day = copyprepData.index.min() + timedelta(days=1)
            last_day = copyprepData.index.max() + timedelta(days=1)
            future_days = [last_day + timedelta(days=i) for i in range(self.N_PREDICTIONS)]
            for i in range(len(future_days)):
                future_days[i] = str(future_days[i])[:10]
            
            #De momento solamente será la fecha
            future_data = pd.DataFrame(future_days, columns=['fecha'])

            #obtenemos el total de features con las que se hará el entrenamiento del modelo
            n_features = int(len(features))

            # antes de iniciar, hacer una consulta para obtener mediante productKey, el standar cost y el precio del producto
            with self.get_sqlconnection(self.sql_server) as cursor: 
                dataProduct = pd.read_sql_query(self.query_products,cursor)

            # Filtrar productos que están en la serie temporal
            dataProduct = dataProduct[dataProduct[forecast_id_col].isin(productKeyIds)]

            #Haremos tratamiento de datos
            data_trat = self.tratamiento_datos(df, all_columns,features,target_col,isSelledCol,productKeyIds,dataProduct,forecast_id_col_enc)

            #Hacer creacion de los modelos y guardar 
            models_dir = self.createModels(data_trat,productKeyIds,forecast_id_col,all_columns,n_features,isSelledCol,features,target_col,category_id,forecast_id_col,path_model)

            #Guardar como metadata la direccion de los modelos pertenecientes a esa categoría
            metadata = [
                {
                    "CATEGORY_ID":category_id,
                    "Cantidad de modelos": len(models_dir)
                },
                models_dir
            ]

            #Guardamos la información sobre las direcciones de los modelos
            self.saveMetadata(metadata,path_model)
