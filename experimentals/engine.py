'''
ENGINE V1.1

Para esta versión, se hacen unas modificaciones, y distinciones de versiones pasadas:

EN ESTA VERSIÓN SE ESTABLECEN DOS PUNTOS QUE SON POSIBLES E IMPORTANTES DE RESALTAR: 
- LA VARIABLE "PASOS" PUEDE SER MODIFICADA PARA ADAPTARSE A LA SEGMENTACÍON Y LA PREPARACIÓN DE LOS DATOS
- LA VARIABLE "TRAINING_PERCENTAGE" HACE REFERENCIA AL PORCENTAJE DE DATOS QUE SE USARÁN PARA ENTRENAMIENTO
- LA VARIABLE "NEURONS" HACE REFERENCIA AL NÚMERO DE NEURONAS QUE CONTEMPLA LA CAPA DE LA RED NEURONAL
- LA VARIABLE "PREDICCIONES" SE DISTINGUE DE "PASOS", YA QUE, EL No. DE PASOS ES INDISTINTO AL No. DE PREDICCIONES
    LA CUAL PUEDE ADAPTARSE DESDE 1, 2, 7, 31, 62, etc días.


'''
import pyodbc
import pandas as pd
import os
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.pylab as plt
import numpy as np
# %matplotlib inline
from keras.models import Sequential
#nuevo
from keras.models import load_model
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.callbacks import EarlyStopping
# from keras.optimizers import SGD, RMSprop
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
import json



class engine:
    def __init__(self, sql_serverConfig,query):
        self.query = query
        self.sql_server= sql_serverConfig
        self.PASOS = 31
        self.EPOCHS = 100
        self.NEURONS = 31
        self.TRAINING_PERCENTAGE = 0.8
        print(sql_serverConfig)

    def get_sqlconnection(self, config_sqlServer):
        status = "inicializando...."
        try: 
            connection = pyodbc.connect(config_sqlServer)
            status = "Conexion establecida satisfactoriamente"
        except Exception as e:
            status = "Error al establecer la conexión:"+e
        print(status)
        return connection

    def set_index_datetime(self,data):
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


    def series_to_supervised(self, data, n_in=1, n_out = 1, dropnan = True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        for i in range(n_in,0,-1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1,i)) for j in range(n_vars)]
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var&d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        if dropnan:
            agg.dropna(inplace=True)
        return agg


    def create_x_y_train(self,data):
        values = data.values
        values = values.astype('float32')
        scaler = MinMaxScaler(feature_range=(-1, 1))
        values= values.reshape(-1, 1)
        scaled = scaler.fit_transform(values)
        reframed = self.series_to_supervised(scaled, self.PASOS, 1)
        values = reframed.values
        n_train_days = int(len(values) * self.TRAINING_PERCENTAGE)
        train = values[:n_train_days, :]
        test = values[n_train_days:, :]
        x_train, y_train = train[:, :- 1], train[:, -1]
        x_val, y_val = test[:, :- 1], test[:, -1]
        x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
        x_val = x_val.reshape((x_val.shape[0], 1, x_val.shape[1]))
        print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
        return x_train, y_train, x_val, y_val, scaler, values

    def crear_modeloFF(self):
        model = Sequential()
        model.add(Dense(self.NEURONS, input_shape=(1,self.PASOS),activation='tanh'))
        model.add(Dropout(0.3)) 
        model.add(Flatten())
        model.add(Dense(1, activation='linear'))
        # model.compile(loss='mean_absolute_error',  optimizer=SGD(learning_rate=0.01, momentum=0.9),metrics=['mse', 'mae'])
        model.compile(loss='mean_absolute_error',  optimizer='Adam',metrics=['mse', 'mae'])
        model.summary()
        return model

    def entrenar_modelo(self,x_train, y_train, x_val, y_val, scaler, values, data, model,model_path,mode,id=0):
        EPOCHS = 100
        # early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_val, y_val), batch_size=self.PASOS)#, callbacks=[early_stop])
        results = model.predict(x_val)


        #Creamos la carpeta principal accuracy
        path = model_path+"/accuracy/"
        os.makedirs(path, exist_ok=True)

        #definimos el modo
        datetim_e_path = datetime.now().strftime('%H-%M-%S_%d-%m-%Y')
        if mode == 't':
            path = path+'Training/'+datetim_e_path
        if mode == 'r':
            if id == 0:
                path = path+'Training/'+datetim_e_path
            else:
                path = path+'Training/'+str(id)

        #dependiendo del caso, creamos la carpeta de accuracy
        os.makedirs(path,exist_ok=True)

        #Configuracion de las imagenes
        plt.rcParams['figure.figsize' ] = (16, 9)
        plt.style.use('fast')

        #Validacion 1: 
        plt.scatter(range(len(y_val)),y_val,c='g', label='Valores Reales')
        plt.scatter(range(len(results)),results,c='r', label='Valores Predecidos')
        plt.xlabel('Índice')
        plt.ylabel('Valores(escalados)')
        plt.title('Grafico de dispersión entre Valores reales vs Predecidos ')
        plt.legend()
        plt.figtext(0.01,0.01,"Realizado el: "+datetime.now().strftime('%H:%M:%S %d-%m-%Y'),fontsize=10,color="gray")
        plt.figtext(0.60,0.01,"Gestión de Innovación en Tecnología Informática S.C.P. | Grupo Consultores®", fontsize=10, color="gray")
        plt.savefig(path+"/performance_validation_1.jpg",dpi=300)
        plt.close()

        #Validacion 2: 
        plt.plot(history.history['loss'],label='Pérdida de Entrenamiento')
        plt.plot(history.history['val_loss'],label='Pérdida de Validación')
        plt.xlabel('Épocas')
        plt.ylabel('Pérdida')
        plt.title('Grafico de Pérdidas de Entrenamiento y validación, según el # de Épocas')
        plt.legend()
        plt.figtext(0.01,0.01,"Realizado el: "+datetime.now().strftime('%H:%M:%S %d-%m-%Y'),fontsize=10,color="gray")
        plt.figtext(0.60,0.01,"Gestión de Innovación en Tecnología Informática S.C.P. | Grupo Consultores®", fontsize=10, color="gray")
        plt.savefig(path+"/performance_validation_2.jpg",dpi=300)
        plt.close()

        #Validacion 3: 
        plt.title('Grafico de MSE de acuerdo al # de Épocas')
        plt.plot(history.history['mse'],label='MSE de Entrenamiento')
        plt.xlabel('Épocas')
        plt.ylabel('MSE')
        plt.legend()
        plt.figtext(0.01,0.01,"Realizado el: "+datetime.now().strftime('%H:%M:%S %d-%m-%Y'),fontsize=10,color="gray")
        plt.figtext(0.60,0.01,"Gestión de Innovación en Tecnología Informática S.C.P. | Grupo Consultores®", fontsize=10, color="gray")
        plt.savefig(path+"/performance_validation_3.jpg",dpi=300)
        plt.close()

        #Validacion 4
        datetim_e = datetime.now().strftime('%H:%M:%S %d-%m-%Y')
        #identificamos que modo es
        if mode == 't':
            #buscamos el optimizador
            optimizer_model = model.optimizer.get_config()

            metadata = [
                {
                    "GENERAl_INFO":{
                        "TOTAL_DE_DATOS": str(data.size),
                        "EPOCH": len(history.history['loss']),
                        "FECHA_ENTRENAMIENTO": datetim_e,
                        "FECHA_MODIFICACION": datetim_e
                    }
                },
                {
                    "OPTIMIZER_CONFIG": optimizer_model
                }
            ]

            #Guardamos la metadata
            self.saveMetadata(model_path,metadata)
        if mode == 'r':
            path = model_path+'/metadata.json'
            #Preparamos para guardar
            with open(path,'r') as file: 
                metadata = json.load(file)
            if metadata:
                total_data = metadata[0]["GENERAl_INFO"]["TOTAL_DE_DATOS"]
                total_data = (int(total_data)+data.size)
                metadata[0]["GENERAl_INFO"]["FECHA_MODIFICACION"] = datetim_e
                metadata[0]["GENERAl_INFO"]["TOTAL_DE_DATOS"] = str(total_data)
                
                with open(path,'w') as file: 
                    json.dump(metadata, file, indent=4)

        ultimosDias = data[data.index[int(len(data)* self.TRAINING_PERCENTAGE)]:]
        values = ultimosDias.values
        values = values.astype('float32' )
        values = values.reshape(-1, 1)
        scaled = scaler.fit_transform(values)
        reframed = self.series_to_supervised(scaled, self.PASOS, 1)
        reframed.drop(reframed.columns[[self.PASOS]], axis=1, inplace=True)
        values = reframed.values
        print("Meses registrados: ",len(values))
        x_test = values[len(values)-1:, :]
        print("La cantidad de días son: ",x_test.size)
        
        #Tensor
        x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
        return model, x_test

    def agregarNuevoValor(self,x_test, nuevoValor):
        for i in range(x_test.shape[2]-1):
            x_test[0][0][i] = x_test[0][0][i+1]
        x_test[0][0][x_test. shape[2]-1] = nuevoValor
        return x_test

    def eliminar_anomalias(self,dtaframe):
        dataFrame_anomalias = dtaframe.copy()
        modeloIsolation = IsolationForest(contamination=0.05)
        modeloIsolation.fit(dataFrame_anomalias)
        anomalias = modeloIsolation.predict(dataFrame_anomalias)
        dtaframe['anomalias' ] = anomalias
        dataFrameSinAnomalias = dtaframe[dtaframe['anomalias' ] != -1]
        dataFrameSinAnomalias = dataFrameSinAnomalias.drop('anomalias', axis=1)
        return dataFrameSinAnomalias
    #Funciones nuevas para prediccion===================


    #Reconstruir el modelo
    def reconstrured_modelFunc(self,model_path):
        model = load_model(model_path)
        model.summary()
        return model
    

    #funcion principal para tomar un modelo y hacer predicciones
    def modelPredicFuncion(self,sqlServerConfig, query, pasos, model_path):
        pasos = int(pasos)
        reconstrured_model = self.reconstrured_modelFunc(model_path)
        future_date, future_data, first_day, last_day = self.prepareData(sqlServerConfig,query,pasos,reconstrured_model)
        self.GraphicDataCreate(future_date, future_data, model_path, first_day, last_day,'p')


    def prepareData(self, sqlServerConfig, query, pasos, reconstrured_model):
        with self.get_sqlconnection(sqlServerConfig) as cursor:
            dataPrepare = pd.read_sql_query(query, cursor)
            dataPrepare = self.set_index_datetime(dataPrepare)
            #Modulo que controla si es en días o en meses
            try: 
                #Si se trata de meses
                first_day = datetime.strptime(dataPrepare.index.min(),'%Y-%m') + relativedelta(months=1)
                last_day = datetime.strptime(dataPrepare.index.max(), '%Y-%m' ) + relativedelta(months=1)
                future_days = [last_day + relativedelta(months=i) for i in range(self.PASOS)]
                for i in range(len(future_days)):
                    future_days[i] = str(future_days[i])[:7]
                print("Month_format_detected")
            except Exception as e: 
                #si se trata de dias
                print("Days_format_detected")
                first_day = dataPrepare.index.min() + timedelta(days=1)
                last_day = dataPrepare.index.max() + timedelta(days=1)
                future_days = [last_day + timedelta(days=i) for i in range(self.PASOS)]
                for i in range(len(future_days)):
                    future_days[i] = str(future_days[i])[:10]
            #MD
            future_data = pd.DataFrame(future_days)
            future_data.columns = ['date']
            
            for column in dataPrepare.columns:
                new_data = dataPrepare.filter([column])
                new_data.set_index(dataPrepare.index, inplace=True)
                new_data = self.eliminar_anomalias(new_data)
                x_train, y_train, x_val, y_val, scaler, values = self.create_x_y_train(new_data)
                x_test = self.reorderData(scaler, values,new_data,pasos)
                results = []
                for i in range(pasos):
                    parcial = reconstrured_model.predict(x_test)
                    results.append(parcial[0])
                    x_test = self.agregarNuevoValor(x_test,parcial[0])
                adimen = [x for x in results]
                inverted = scaler.inverse_transform(adimen)
                y_pred = pd.DataFrame(inverted.astype(int))
                future_data[column]= inverted.astype(int)
            future_data = self.set_index_datetime(future_data)
            dataPrepare.index = pd.to_datetime(dataPrepare.index)
            future_data.index = pd.to_datetime(future_data.index)
            return dataPrepare, future_data, first_day, last_day
        
    #Modulo para guardar las gráficas, cuando se hace una predicción    
    def GraphicDataCreate(self,datos, futureData, model_path,first_day, last_day, mode):
        """
        Este módulo puede generar una gráfica y establecer los directorios, de acuerdo 
        al modo en el que esté: e -> ENTRENAMIENTO, r -> RE-ENTRENAMIENTO, p -> predicción 
        """

        #tomamos la ruta del modelo y eliminamos el nombre del modelo
        path = os.path.dirname(model_path)

        if mode == 'e':
            pass
        elif mode == 'r':
            path = path+'/reTrainingModel_dataPredict'
        elif mode == 'p':
            path = path+'/reconstruredModel_dataPredict'
        #ubicamos la carpeta de predicciones de modelos reconstruidos

        #Creamos una carpeta de dato entrenados con el modelo reconstruido
        if not os.path.exists(path):
            os.makedirs(path)
        
        #en el mismo directorio, creamos una carpeta 
        path = path+'/'+datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        os.makedirs(path)
        #Graficar los dataframes
        #considerar almacenar la variable de la columna, ya que, el nombre de la misma puede cambiar
        #Configuracion de las imagenes
        plt.rcParams['figure.figsize' ] = (16, 9)
        plt.style.use('fast')


        for i in range(len(datos.columns)):
            data = datos[datos.columns[i]][:]
            plt.plot(data.index, data,label='Historial {p0} - {p1}'.format(p0=str(first_day.year),p1=str(last_day.year-1)))
            plt.plot(futureData.index, futureData[futureData.columns[i]], label='Predicción {p0}'.format(p0=str(last_day.year)))
            # xtics = data.index.union(futureData.index)[::8]
            # plt.xticks(xtics)
            plt.xlabel('Fecha')
            plt.ylabel('Ventas')
            plt.title('Predicción de la demanda global {p0} para el año del {p1}'.format(p0=datos.columns[i],p1=str(last_day.year)))
            plt.legend()
            plt.figtext(0.01, 0.01, "Realizado el: "+datetime.now().strftime('%H:%M:%S %d-%m-%Y'), fontsize=10, color="gray")
            plt.figtext(0.60, 0.01, "Gestión de Innovación en Tecnología Informática S.C.P | Grupo Consultores®", fontsize=10, color="gray")
            name = path+'/GraphicalPrediction_on_'+str(datos.columns[i])+".jpg"
            plt.savefig(name, dpi=300)
            plt.close()  # Cerrar la figura
#
    def reorderData(self, scaler, values, data, pasos):
        ultimosDias = data[data.index[int(len(data)*0.80)]:]
        values = ultimosDias.values
        values = values.astype('float32')
        values = values.reshape(-1, 1)
        scaled = values
        reframed = self.series_to_supervised(scaled, pasos, 1)
        reframed.drop(reframed.columns[[12]], axis=1, inplace=True)
        values = ultimosDias.values
        values = values.astype('float32' )
        values = values.reshape(-1, 1)
        scaled = scaler.fit_transform(values)
        reframed = self.series_to_supervised(scaled, pasos, 1)
        reframed.drop(reframed.columns[[12]], axis=1, inplace=True)
        values = reframed.values
        x_test = values[len(values)-1:, :]
        x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]) )
        return x_test
    
    def saveMetadata(self, model_path,features):
        #Preparamos un archivo para guardar la metadata
        metaData = model_path+'/metadata.json'
        print(features)
        with open(metaData, 'w') as file:
            json.dump(features,file, indent=4)
    #Funciones nuevas para prediccion===================

    #Funciones nuevas para reentrenamiento================

    def modelRetrainingFunction(self, sqlServerConfig,query, steps, model_path):
        steps = int(steps)
        reconstrured_model = self.reconstrured_modelFunc(model_path)
        self.retrainingModel(model_path, reconstrured_model, sqlServerConfig,query,steps)
        # future_date, future_data, first_day, last_day = self.retrainingModel(model_path, reconstrured_model, sqlServerConfig,query,steps)
        # self.GraphicDataCreate(future_date, future_data, model_path, first_day, last_day,'r')

    def retrainingModel(self, model_path, reconstrured_model,sqlServerConfig,query,pasos):
        with self.get_sqlconnection(sqlServerConfig) as cursor:
                datos = pd.read_sql_query(query, cursor)
                datos = self.set_index_datetime(datos)
                
                #Modulo que controla si es en días o en meses
                try: 
                    #Si se trata de meses
                    first_day = datetime.strptime(datos.index.min(),'%Y-%m') + relativedelta(months=1)
                    last_day = datetime.strptime(datos.index.max(), '%Y-%m' ) + relativedelta(months=1)
                    future_days = [last_day + relativedelta(months=i) for i in range(self.PASOS)]
                    for i in range(len(future_days)):
                        future_days[i] = str(future_days[i])[:7]
                    print("Month_format_detected")
                except Exception as e: 
                    #si se trata de dias
                    print("Days_format_detected")
                    first_day = datos.index.min() + timedelta(days=1)
                    last_day = datos.index.max() + timedelta(days=1)
                    future_days = [last_day + timedelta(days=i) for i in range(self.PASOS)]
                    for i in range(len(future_days)):
                        future_days[i] = str(future_days[i])[:10]
                #MD
                
                future_data = pd.DataFrame(future_days, columns=['fecha'])
                model = reconstrured_model
                model.compile(loss='mean_absolute_error',  optimizer='Adam',metrics=['mse', 'mae'])
                data = []
                for column in datos.columns:
                    data = datos.filter([column])
                    data.set_index(datos.index, inplace=True)
                    data = self.eliminar_anomalias(data)
                    x_train, y_train, x_val, y_val, scaler, values = self.create_x_y_train(data)
                    model_dirname = os.path.dirname(model_path)
                    model, x_test = self.entrenar_modelo(x_train, y_train, x_val, y_val, scaler, values, data, model,model_dirname,'r')
                    # results = []
                    # for i in range(pasos):
                    #     parcial = model.predict(x_test)
                    #     results.append(parcial[0])
                    #     x_test = self.agregarNuevoValor(x_test, parcial[0])
                    # adimen = [x for x in results]
                    # inverted = scaler.inverse_transform(adimen)
                    # future_data[column]= inverted.astype(int)
                #Continuacion para guardar el modelo
                model_name = model_path#+'/model_training-'+datetim_e+'.keras'
                model.save(model_name)

                # future_data = self.set_index_datetime(future_data)

                # datos.index = pd.to_datetime(datos.index)
                # future_data.index = pd.to_datetime(future_data.index)
                # return datos, future_data, first_day, last_day

    #Funciones nuevas para reentrenamiento================

    def main(self):
        '''
        NOTA: CUANDO EL SISTEMA ENTRENA POR PRIMERA VEZ, LA CANTIDAD DE NEURONAS, BACH_SIZE, CANTIDA DE DÍAS, PREDICCIONES DE PRUEBA
        SERÁN POR DEFECTO 31, CUANDO SE HACE LA PPREDICCIÓN, ESTOS PARÁMETROS PUEDEN CAMBIAR
        '''
        #Core
        with self.get_sqlconnection(self.sql_server) as cursor:
            datos = pd.read_sql_query(self.query, cursor)
            datos = self.set_index_datetime(datos)

            dirmodels_name = './models/'+datetime.now().strftime('%Y-%m-%d')
            if not os.path.exists(dirmodels_name):
                os.makedirs(dirmodels_name, exist_ok=True)
            
            #Modulo que controla si es en días o en meses
            try: 
                #si se trata de dias
                print("Days_format_detected")
                first_day = datos.index.min() + timedelta(days=1)
                last_day = datos.index.max() + timedelta(days=1)
                future_days = [last_day + timedelta(days=i) for i in range(self.PASOS)]
                for i in range(len(future_days)):
                    future_days[i] = str(future_days[i])[:10]
            except Exception as e: 
                #Si se trata de meses
                first_day = datetime.strptime(datos.index.min(),'%Y-%m') + relativedelta(months=1)
                last_day = datetime.strptime(datos.index.max(), '%Y-%m' ) + relativedelta(months=1)
                future_days = [last_day + relativedelta(months=i) for i in range(self.PASOS)]
                for i in range(len(future_days)):
                    future_days[i] = str(future_days[i])[:7]
                print("Month_format_detected")
            #MD
            
            future_data = pd.DataFrame(future_days, columns=['fecha'])
            model = self.crear_modeloFF()
            
            # CREAMOS DOS CARPETAS: UNO PARA GUARDAR EL MODELO Y OTRO PARA GUARDAR EL SLD WINDOW
            datetim_e = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            model_path = dirmodels_name+"/model-training"+datetim_e
            #----------------------------
            model_path_copy = model_path
            #----------------------------
            os.makedirs(model_path, exist_ok=True)
            #si hay más de un producto
            fathers_Mdir = model_path+'/artifacts'
            productID = 12
            #---------

            data = []
            for column in datos.columns:
                #----------------------------
                name_column = fathers_Mdir+ f'/{productID}'
                model_path = name_column
                #----------------------------
                
                #nota: se guarda en la misma direccion donde se encuentra la metadata, el modelo entrenado
                data = datos.filter([column])
                data.set_index(datos.index, inplace=True)
                data = self.eliminar_anomalias(data)
                x_train, y_train, x_val, y_val, scaler, values = self.create_x_y_train(data)
                # model, x_test = self.entrenar_modelo(x_train, y_train, x_val, y_val, scaler, values, data, model,model_path,'t')
                #----------------------------
                model, x_test = self.entrenar_modelo(x_train, y_train, x_val, y_val, scaler, values, data, model,model_path,'t',productID)
                #guardar x_test
                L_sld_window = model_path+'/LSLDWINDOW.npy'
                np.save(L_sld_window,x_test)
                #----------------------------
                results = []
                for i in range(self.PASOS):
                    parcial = model.predict(x_test)
                    results.append(parcial[0])
                    x_test = self.agregarNuevoValor(x_test, parcial[0])
                adimen = [x for x in results]
                inverted = scaler.inverse_transform(adimen)
                future_data[column]= inverted.astype(int)

                #----------------------------
                datetim_e = str(productID)
                #----------------------------
                
                #Continuacion para guardar el modelo
                model_name = model_path+'/model_training-'+datetim_e+'.keras'
                model.save(model_name)

            future_data = self.set_index_datetime(future_data)

            datos.index = pd.to_datetime(datos.index)
            future_data.index = pd.to_datetime(future_data.index)

            #----------------------------
            model_path = model_path_copy
            #----------------------------

            #Creamos un directorio para guardar los datos del primer entrenamiento
            # path = model_path+"/trainedModel_dataPredict/"+datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            path = model_path+"/trainedModel_dataPredict/"+str(productID)
            os.makedirs(path)
            
            #Configuracion de las imagenes
            plt.rcParams['figure.figsize' ] = (16, 9)
            plt.style.use('fast')

            #Graficar los dataframes
            for i in range(len(datos.columns)):
                data = datos[datos.columns[i]][:]
                plt.plot(data.index, data,label='Historial {p0} - {p1}'.format(p0=str(first_day.year),p1=str(last_day.year-1)))
                plt.plot(future_data.index, future_data[future_data.columns[i]], label='Predicción {p0}'.format(p0=str(last_day.year)))
                # xtics = data.index.union(future_data.index)[::6]

                plt.xlabel('Fecha')
                plt.ylabel('Ventas')
                plt.title('Predicción de la demanda de {p0} para el año del {p1}'.format(p0=datos.columns[i],p1=str(last_day.year-1)))
                plt.legend()
                plt.figtext(0.01, 0.01, "Realizado el: "+datetime.now().strftime('%H:%M:%S %d-%m-%Y'), fontsize=10, color="gray")
                plt.figtext(0.60, 0.01, "Gestión de Innovación en Tecnología Informática S.C.P | Grupo Consultores®", fontsize=10, color="gray")
                name = path+'/GraphicalPrediction_on_'+str(datos.columns[i])+".jpg"
                plt.savefig(name, dpi=300)
                plt.close()  # Cerrar la figura para liberar memoria



