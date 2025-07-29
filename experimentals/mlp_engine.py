'''
MLP ENGINE V1.1

Para esta versión, se hacen unas modificaciones, y distinciones de versiones pasadas:

EN ESTA VERSIÓN SE ESTABLECEN DOS PUNTOS QUE SON POSIBLES E IMPORTANTES DE RESALTAR: 
- LA VARIABLE "PASOS" PUEDE SER MODIFICADA PARA ADAPTARSE A LA SEGMENTACÍON Y LA PREPARACIÓN DE LOS DATOS
- LA VARIABLE "TRAINING_PERCENTAGE" HACE REFERENCIA AL PORCENTAJE DE DATOS QUE SE USARÁN PARA ENTRENAMIENTO
- LA VARIABLE "NEURONS" HACE REFERENCIA AL NÚMERO DE NEURONAS QUE CONTEMPLA LA CAPA DE LA RED NEURONAL
- LA VARIABLE "PREDICCIONES" SE DISTINGUE DE "PASOS", YA QUE, EL No. DE PASOS ES INDISTINTO AL No. DE PREDICCIONES
    LA CUAL PUEDE ADAPTARSE DESDE 1, 2, 7, 31, 62, etc días.


'''

import pandas as pd
import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.pylab as plt
import numpy as np
from keras.models import Sequential
#nuevo
from keras.models import load_model
from keras.layers import Dense, Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
import json
import joblib

class Mlp_engine:
    def __init__(self,pasos=31, epochs=100, neurons=31, training_percentaje=0.8):
        self.PASOS = pasos
        self.EPOCHS = epochs
        self.NEURONS = neurons
        self.TRAINING_PERCENTAGE = training_percentaje

    def set_index_datetime(self,data):
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
        """
        Funcion para creación de tensores únicamente para entrenamiento de modelos MLP
        """
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
    
    def create_tensors(self, data, scaler, steps,nval_data):
        """
        Funcion para creación de tensores unicamente para reentrenamiento de modelos MLP
        """
        values = data.values
        values = values.astype('float32')
        values= values.reshape(-1, 1)
        scaled = scaler.fit_transform(values)
        reframed = self.series_to_supervised(scaled, steps, 1)
        values = reframed.values
        n_val_days = nval_data
        n_train_days = len(values) - n_val_days
        train = values[:n_train_days, :]
        test = values[n_train_days:, :]
        x_train, y_train = train[:, :- 1], train[:, -1]
        x_val, y_val = test[:, :- 1], test[:, -1]
        x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
        x_val = x_val.reshape((x_val.shape[0], 1, x_val.shape[1]))
        return x_train, y_train, x_val, y_val, scaler, values

    def crear_modeloFF(self):
        model = Sequential()
        model.add(Dense(self.NEURONS, input_shape=(1,self.PASOS),activation='tanh'))
        model.add(Flatten())
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mean_absolute_error',  optimizer='Adam',metrics=['mse', 'mae'])
        model.summary()
        return model

    def entrenar_modelo(self,x_train, y_train, x_val, y_val, scaler, values, data, model,model_path,mode, parameters, epoch=100, id=0):
        """
        Función principal para entrenamiento de modelos MLP
        """
        if  epoch != "":
            EPOCHS = epoch
        else:
            EPOCHS = 100
        history = model.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_val, y_val), batch_size=self.PASOS)
        results = model.predict(x_val)

        path = model_path+"/accuracy/"
        os.makedirs(path, exist_ok=True)

        datetim_e_path = datetime.now().strftime('%H-%M-%S_%d-%m-%Y')
        if mode == 't':
            path = path+'Training/'+datetim_e_path
        if mode == 'r':
            if id == 0:
                path = path+'Training/'+datetim_e_path
            else:
                path = path+'Training/'+str(id)

        os.makedirs(path,exist_ok=True)

        # GRÁFICO DE PRONÓSTICO
        plt.rcParams['figure.figsize' ] = (16, 9)
        plt.style.use('fast')

        #Validacion 1: 
        plt.scatter(range(len(y_val)),y_val,c='g', label='Valores Reales')
        plt.scatter(range(len(results)),results,c='r', label='Pronóstico')
        plt.xlabel('Índice')
        plt.ylabel('Valores(escalados)')
        plt.title('Grafico de dispersión entre Valores reales vs Pronósticos')
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
        
        # MODO
        if mode == 't':
            optimizer_model = model.optimizer.get_config()
            model_summary = json.loads(model.to_json())
            output_neurons = None
            for layer in model.layers[::-1]:
                if isinstance(layer, Dense):
                    output_neurons = layer.units
                    break

            metadata = [
                {
                    "MODEL_REFERENCY": {
                        "ID": parameters['ID'],
                        "MODEL_NAME": parameters['MODEL_NAME'],
                        "TECHNOLOGY": "MLP",
                        "OUTPUT_NEURONS": output_neurons
                    }
                },
                {
                    "GENERAL_INFO": {
                        "DATOS_ENTRENAMIENTO": str(len(x_train)),
                        "DATOS_VALIDACION": str(len(y_val)),
                        "EPOCH": len(history.history['loss']),
                        "FECHA_ENTRENAMIENTO": datetim_e,
                        "FECHA_MODIFICACION": datetim_e,
                        "STEPS": str(self.PASOS),
                        "PREDICTIONS_DEFAULT": output_neurons,
                        "TECHNOLOGY": "MLP"
                    }
                },
                {
                    "MODEL_ARCHITECTURE": model_summary
                },
                {
                    "OPTIMIZER_CONFIG": optimizer_model
                }
            ]
            self.saveMetadata(model_path,metadata)

            #operaciones ´
            ultimosDias = data[data.index[int(len(data)* self.TRAINING_PERCENTAGE)]:]
            values = ultimosDias.values
            values = values.astype('float32')
            values = values.reshape(-1, 1)
            scaled = scaler.fit_transform(values)
            reframed = self.series_to_supervised(scaled, self.PASOS, 1)
            reframed.drop(reframed.columns[[self.PASOS]], axis=1, inplace=True)
            values = reframed.values
            x_test = values[len(values)-1:, :]
            
            #Tensor
            x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
            return model, x_test
        if mode == 'r':
            path = model_path+'/metadata.json'
            with open(path,'r') as file: 
                metadata = json.load(file)
            if metadata:
                total_data = metadata[1]["GENERAL_INFO"]["DATOS_ENTRENAMIENTO"]
                total_data = (int(total_data)+data.size)
                metadata[1]["GENERAL_INFO"]["FECHA_MODIFICACION"] = datetim_e
                metadata[1]["GENERAL_INFO"]["DATOS_ENTRENAMIENTO"] = str(total_data)
                
                with open(path,'w') as file: 
                    json.dump(metadata, file, indent=4)

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
    
    def predictions(self, model_path,model_metadata,predictions):
        model, df, scaler, steps, p_default, root_dir = self.loadUtils(model_path,model_metadata)
        dataPrepare, future_data, first_day, last_day = self.prepareData(steps,predictions,model,df,scaler)
        self.GraphicDataCreate(dataPrepare, future_data, model_path, first_day, last_day,'p')
    
    def loadUtils(self, model_path,model_metadata):
        """
        Función que se encarga de recargar el modelo o reconstruirlo como estuvo originalmente
        """

        enable = False
        root_dir = os.path.dirname(model_path)
        util_dir = root_dir+'/utils/'
        df = pd.read_csv(util_dir+'LSLDWINDOW.csv')
        scaler = self.reloadScaler(util_dir+'scaler.pkl')
        model = self.reconstrured_modelFunc(model_path)
        steps = int(model_metadata['GENERAL_INFO']['STEPS'])
        p_default = int(model_metadata['GENERAL_INFO']['PREDICTIONS_DEFAULT'])

        # VALIDACION DE OBJETOS
        if (
            df is not None and not df.empty and
            scaler is not None and
            model is not None
        ):
            
            print("////// SE HAN CARGADO TODOS LOS ARCHIVOS NECESARIOS CORRECTAMENTE //////")
            print("Scaler x: ",type(scaler))
            print("df: ",type(df))
            print("modelo: ",type(model))
            enable = True
        else:
            print("////// NO TODOS LOS ARCHIVOS SE CARGARON CORRECTAMENTE //////")

        if enable:
            df = df.rename(columns={"sales_date": "Date"})
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = self.set_index_datetime(df)
            return model, df, scaler, steps, p_default, root_dir
    
    def reloadScaler(self,path):
        """
        Función que permite recuperar escaladores usando Joblib
        """
        load_scaler = joblib.load(path)
        return load_scaler
        
    def prepareData(self,pasos, predictions, model,data,scaler):
        data.reset_index()
        dataPrepare = data

        try: 
            first_day = datetime.strptime(dataPrepare.index.min(),'%Y-%m') + relativedelta(months=1)
            last_day = datetime.strptime(dataPrepare.index.max(), '%Y-%m' ) + relativedelta(months=1)
            future_days = [last_day + relativedelta(months=i) for i in range(predictions)]
            for i in range(len(future_days)):
                future_days[i] = str(future_days[i])[:7]
        except Exception as e: 
            first_day = dataPrepare.index.min() + timedelta(days=1)
            last_day = dataPrepare.index.max() + timedelta(days=1)
            future_days = [last_day + timedelta(days=i) for i in range(predictions)]
            for i in range(len(future_days)):
                future_days[i] = str(future_days[i])[:10]

        future_data = pd.DataFrame(future_days)
        future_data.columns = ['date']
        column = dataPrepare.columns[0]
        new_data = dataPrepare[[column]].copy()
        new_data = self.eliminar_anomalias(new_data)
        values = new_data.values.astype('float32')
        values = values.reshape(-1, 1)
        scaled = scaler.fit_transform(values)
        reframed = self.series_to_supervised(scaled, pasos, 1)
        reframed.drop(reframed.columns[[pasos]], axis=1, inplace=True)
        x_test = reframed.values[-1:, :]
        x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1])).astype('float32')

        results = []
        for i in range(predictions):
            parcial = model.predict(x_test)
            results.append(parcial[0])
            x_test = self.agregarNuevoValor(x_test, parcial[0])

        adimen = [x for x in results]
        inverted = scaler.inverse_transform(adimen)
        future_data[column]= inverted.astype(int)
        future_data = self.set_index_datetime(future_data)
        dataPrepare.index = pd.to_datetime(dataPrepare.index)
        future_data.index = pd.to_datetime(future_data.index)
        return dataPrepare, future_data, first_day, last_day
        
    #Modulo para guardar las gráficas
    def GraphicDataCreate(self,datos, futureData, model_path,first_day, last_day, mode):
        """
        Este módulo puede generar una gráfica y establecer los directorios, de acuerdo 
        al modo en el que esté: e -> ENTRENAMIENTO, r -> RE-ENTRENAMIENTO, p -> predicción 
        """

        path = os.path.dirname(model_path)

        if mode == 'e':
            pass
        elif mode == 'r':
            path = path+'/reTrainingModel_dataPredict'
        elif mode == 'p':
            path = path+'/reconstruredModel_dataPredict'
        if not os.path.exists(path):
            os.makedirs(path)
        
        path = path+'/'+datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        os.makedirs(path)
        plt.rcParams['figure.figsize' ] = (16, 9)
        plt.style.use('fast')
        for i in range(len(datos.columns)):
            data = datos[datos.columns[i]][:]
            plt.plot(data.index, data,label='Historial {p0} - {p1}'.format(p0=str(first_day.year),p1=str(last_day.year-1)))
            plt.plot(futureData.index, futureData[futureData.columns[i]], label='Predicción {p0}'.format(p0=str(last_day.year)))
            plt.xlabel('Fecha')
            plt.ylabel('Ventas')
            plt.title('Predicción de la demanda global {p0} para el año del {p1}'.format(p0=datos.columns[i],p1=str(last_day.year)))
            plt.legend()
            plt.figtext(0.01, 0.01, "Realizado el: "+datetime.now().strftime('%H:%M:%S %d-%m-%Y'), fontsize=10, color="gray")
            plt.figtext(0.60, 0.01, "Gestión de Innovación en Tecnología Informática S.C.P | Grupo Consultores®", fontsize=10, color="gray")
            name = path+'/GraphicalPrediction_on_'+str(datos.columns[i])+".jpg"
            plt.savefig(name, dpi=300)
            plt.close() 
    
    def saveMetadata(self, model_path,features):
        metaData = model_path+'/metadata.json'
        with open(metaData, 'w') as file:
            json.dump(features,file, indent=4)
    # Prediccion===================

    # Reentrenamiento================

    def retraining(self, model_path,model_metadata):
        """
        Función principal de gestión de reentrenamiento de modelos MLP
        """
        model, df, scaler, steps, p_default, root_dir = self.loadUtils(model_path,model_metadata)
        model = self.reconstrured_modelFunc(model_path)
        self.retrainingModel(model_path, model, df,scaler, model_metadata)

    def retrainingModel(self, model_path, model, datos, scaler, model_metadata):
        nval_data = int(model_metadata['GENERAL_INFO']['DATOS_VALIDACION'])
        dsteps = int(model_metadata['GENERAL_INFO']['STEPS'])
        try: 
            first_day = datetime.strptime(datos.index.min(),'%Y-%m') + relativedelta(months=1)
            last_day = datetime.strptime(datos.index.max(), '%Y-%m' ) + relativedelta(months=1)
            future_days = [last_day + relativedelta(months=i) for i in range(self.PASOS)]
            for i in range(len(future_days)):
                future_days[i] = str(future_days[i])[:7]
        except Exception as e: 
            first_day = datos.index.min() + timedelta(days=1)
            last_day = datos.index.max() + timedelta(days=1)
            future_days = [last_day + timedelta(days=i) for i in range(self.PASOS)]
            for i in range(len(future_days)):
                future_days[i] = str(future_days[i])[:10]

        model.compile(loss='mean_absolute_error', optimizer='Adam', metrics=['mse', 'mae'])
        datos = self.eliminar_anomalias(datos)
        x_train, y_train, x_val, y_val, scaler, values = self.create_tensors(datos,scaler,dsteps,nval_data)
        model_dirname = os.path.dirname(model_path)

        parameters = {
            "ID": os.path.basename(model_path),
            "MODEL_NAME": "retrained_model"
        }

        self.entrenar_modelo(
            x_train, y_train, x_val, y_val, scaler, values, datos, model, model_dirname, 'r', parameters,25
        )
        model.save(model_path)

        utils_dir = os.path.join(model_dirname, 'utils')
        os.makedirs(utils_dir, exist_ok=True)
        n_train_days = int(len(x_val))
        validation_data = datos.iloc[n_train_days:]
        validation_data_path = os.path.join(utils_dir, 'LSLDWINDOW.csv')
        validation_data.to_csv(validation_data_path, index=True)
        dataPrepare, future_data, first_day, last_day = self.prepareData(dsteps,dsteps,model,datos,scaler)
        self.GraphicDataCreate(dataPrepare,future_data,model_path,first_day,last_day,'p')
    #Funciones nuevas para reentrenamiento================

    def main(self, data):
        '''
        NOTA: CUANDO EL SISTEMA ENTRENA POR PRIMERA VEZ, LA CANTIDAD DE NEURONAS, BACH_SIZE, CANTIDA DE DÍAS, PREDICCIONES DE PRUEBA
        SERÁN POR DEFECTO 31, CUANDO SE HACE LA PPREDICCIÓN, ESTOS PARÁMETROS PUEDEN CAMBIAR
        '''
        #Core
        datos = data.copy()
        datos = self.set_index_datetime(datos)
        dirmodels_name = './models/'+datetime.now().strftime('%Y-%m-%d')
        if not os.path.exists(dirmodels_name):
            os.makedirs(dirmodels_name, exist_ok=True)
        try: 
            first_day = datos.index.min() + timedelta(days=1)
            last_day = datos.index.max() + timedelta(days=1)
            future_days = [last_day + timedelta(days=i) for i in range(self.PASOS)]
            for i in range(len(future_days)):
                future_days[i] = str(future_days[i])[:10]
        except Exception as e: 
            first_day = datetime.strptime(datos.index.min(),'%Y-%m') + relativedelta(months=1)
            last_day = datetime.strptime(datos.index.max(), '%Y-%m' ) + relativedelta(months=1)
            future_days = [last_day + relativedelta(months=i) for i in range(self.PASOS)]
            for i in range(len(future_days)):
                future_days[i] = str(future_days[i])[:7]
        future_data = pd.DataFrame(future_days, columns=['fecha'])

        model = self.crear_modeloFF()
        
        datetim_e = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        model_path = dirmodels_name+"/model-training"+datetim_e
        model_path_copy = model_path
        os.makedirs(model_path, exist_ok=True)

        #si hay más de un producto
        fathers_Mdir = model_path
        # productID = 12

        data = []
        for column in datos.columns:
            #----------------------------
            # name_column = fathers_Mdir+ f'/{productID}'
            # model_path = name_column
            #----------------------------
            name = 'model_'+datetim_e
            idM = name
            model_name = model_path+'/'+name+'.keras'
            
            data = datos.filter([column])
            data_copy = data.copy()
            data.set_index(datos.index, inplace=True)
            data = self.eliminar_anomalias(data)
            x_train, y_train, x_val, y_val, scaler, values = self.create_x_y_train(data)

            parameters = {
                "ID": idM,
                "MODEL_NAME": name
            }

            model, x_test = self.entrenar_modelo(x_train, y_train, x_val, y_val, scaler, values, data, model,model_path,'t', parameters)
            utils_dir = os.path.join(model_path, 'utils')
            os.makedirs(utils_dir, exist_ok=True)
            scaler_path = os.path.join(utils_dir, 'scaler.pkl')
            joblib.dump(scaler, scaler_path)
            n_train_days = int(len(data_copy) * self.TRAINING_PERCENTAGE)
            print(n_train_days)
            validation_data = data_copy.iloc[n_train_days:]
            validation_data_path = os.path.join(utils_dir, 'LSLDWINDOW.csv')
            validation_data.to_csv(validation_data_path, index=True)
            results = []
            for i in range(self.PASOS):
                parcial = model.predict(x_test)
                results.append(parcial[0])
                x_test = self.agregarNuevoValor(x_test, parcial[0])
            adimen = [x for x in results]
            adimen = np.array(adimen).reshape(-1, 1)
            inverted = scaler.inverse_transform(adimen)
            future_data[column]= inverted.astype(int).flatten()
            model.save(model_name)
        future_data = self.set_index_datetime(future_data)
        datos.index = pd.to_datetime(datos.index)
        future_data.index = pd.to_datetime(future_data.index)
        model_path = model_path_copy
        path = model_path+"/Training/"
        os.makedirs(path)
        plt.rcParams['figure.figsize' ] = (16, 9)
        plt.style.use('fast')

        #Graficar los dataframes
        for i in range(len(datos.columns)):
            data = datos[datos.columns[i]][:]
            plt.plot(data.index, data,label='Historial {p0} - {p1}'.format(p0=str(first_day.year),p1=str(last_day.year-1)))
            plt.plot(future_data.index, future_data[future_data.columns[i]], label='Predicción {p0}'.format(p0=str(last_day.year)))
            plt.xlabel('Fecha')
            plt.ylabel('Ventas')
            plt.title('Predicción de la demanda de {p0} para el año del {p1}'.format(p0=datos.columns[i],p1=str(last_day.year-1)))
            plt.legend()
            plt.figtext(0.01, 0.01, "Realizado el: "+datetime.now().strftime('%H:%M:%S %d-%m-%Y'), fontsize=10, color="gray")
            plt.figtext(0.60, 0.01, "Gestión de Innovación en Tecnología Informática S.C.P | Grupo Consultores®", fontsize=10, color="gray")
            name = path+'/GraphicalPrediction_on_'+str(datos.columns[i])+".jpg"
            plt.savefig(name, dpi=300)
            plt.close()




