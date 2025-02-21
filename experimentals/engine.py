
#Este es el motor principal o "controlador"
#Que permitirá realizar una interacción con los distintos componentes del sistema
import pyodbc
import pandas as pd
import numpy as np
import os
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.pylab as plt
# %matplotlib inline
from keras.models import Sequential
#nuevo
from keras.models import load_model
from keras.layers import Dense, Activation, Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
import keyboard  # Necesario para detectar la tecla
import shutil



class engine:
    def __init__(self, sql_serverConfig,query,):
        # self.server_name = server_name
        # self.database_name = database_name
        self.query = query
        # self.driver = driver
        self.sql_server= sql_serverConfig
        self.PASOS = 12
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
        n_train_days = int(len(data)) - (30+self.PASOS)
        train = values[:n_train_days, :]
        test = values[n_train_days:, :]
        x_train, y_train = train[:, :- 1], train[:, -1]
        x_val, y_val = test[:, :- 1], test[:, -1]
        x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
        x_val = x_val.reshape((x_val.shape[0], 1, x_val.shape[1]))
        return x_train, y_train, x_val, y_val, scaler, values

    def crear_modeloFF(self):
        model = Sequential()
        model.add(Dense(self.PASOS, input_shape=(1,self.PASOS),activation='tanh'))
        model.add(Flatten())
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mean_absolute_error', optimizer='Adam',metrics=['mse' ])
        model.summary()
        return model

    def entrenar_modelo(self,x_train, y_train, x_val, y_val, scaler, values, data, model) :
        EPOCHS = 100
        model.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_val, y_val), batch_size=self.PASOS)
        model.predict(x_val)
        ultimosDias = data[data.index[int(len(data)*0.70)]:]
        values = ultimosDias.values
        values = values.astype('float32' )
        values = values.reshape(-1, 1)
        scaled = values
        reframed = self.series_to_supervised(scaled, self.PASOS, 1)
        reframed.drop(reframed.columns[[12]], axis=1, inplace=True)
        values = ultimosDias.values
        values = values.astype('float32' )
        values = values.reshape(-1, 1)
        scaled = scaler.fit_transform(values)
        reframed = self.series_to_supervised(scaled, self.PASOS, 1)
        reframed.drop(reframed.columns[[12]], axis=1, inplace=True)
        values = reframed.values
        x_test = values[len(values)-1:, :]
        x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]) )
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
        return model
    

    #funcion principal para tomar un modelo y hacer predicciones
    def modelPredicFuncion(self,sqlServerConfig, query, pasos, model_path):
        reconstrured_model = self.reconstrured_modelFunc(model_path)
        future_date, future_data = self.prepareData(sqlServerConfig,query,pasos,reconstrured_model)
        self.GraphicDataCreate(future_date, future_data, model_path)


    def prepareData(self, sqlServerConfig, query, pasos, reconstrured_model):
        with self.get_sqlconnection(sqlServerConfig) as cursor:
            dataPrepare = pd.read_sql_query(query, cursor)
            dataPrepare = self.set_index_datetime(dataPrepare)
            last_day = datetime.strptime(dataPrepare.index.max(), '%Y-%m') + relativedelta(months=1)
            future_days = [last_day + relativedelta(months=i) for i in range(pasos)]
            for i in range(len(future_days)):
                future_days[i] = str(future_days[i])[:7]
            # print("pasaste por aqui")
            future_data = pd.DataFrame(future_days)
            #renombramiento del campo 
            future_data.columns = ['date']
            for column in dataPrepare.columns:
                new_data = dataPrepare.filter([column])
                new_data.set_index(dataPrepare.index, inplace=True)
                new_data = self.eliminar_anomalias(new_data)
                x_train, y_train, x_val, y_val, scaler, values = self.create_x_y_train(new_data)
                # x_test = self.reorderData(scaler, values,new_data,pasos)
                reconstrured_model, x_test = self.entrenar_modelo(x_train,y_train,x_val,y_val,scaler,values,new_data,reconstrured_model)
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
            return dataPrepare, future_data
        
    def GraphicDataCreate(self,datos, futureData, model_path):

        #tomamos la ruta del modelo y eliminamos el nombre del modelo
        path = os.path.dirname(model_path)

        #ubicamos la carpeta de predicciones de modelos reconstruidos
        path = path+'/reconstruredModel_dataPredict'

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
            plt.plot(data.index, data,label='Historial 2015 - 2020')
            plt.plot(futureData.index, futureData[futureData.columns[i]], label='Predicción 2021 - 2022')
            xtics = data.index.union(futureData.index)[::8]
            plt.xticks(xtics)
            plt.xlabel('Fecha')
            plt.ylabel('Ventas')
            plt.title('Predicción de la demanda del {p0} para el año del 2021'.format(p0=datos.columns[i]))
            plt.legend()
            plt.figtext(0.01, 0.01, "Realizado el: "+datetime.now().strftime('%H:%M:%S %d-%m-%Y'), fontsize=10, color="gray")
            plt.figtext(0.60, 0.01, "Gestión de Innovación en Tecnología Informática S.C.P | Grupo Consultores®", fontsize=10, color="gray")
            name = path+'/GraphicalPrediction_on_'+str(datos.columns[i])+".jpg"
            plt.savefig(name, dpi=300)
            plt.close()  # Cerrar la figura
            # plt.show()
#
    def reorderData(self, scaler, values, data, pasos):
        EPOCHS = 100
        # model.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_val, y_val), batch_size=self.PASOS)
        # model.predict(x_val)
        ultimosDias = data[data.index[int(len(data)*0.70)]:]
        values = ultimosDias.values
        values = values.astype('float32' )
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
#Funciones nuevas para prediccion===================


    def main(self):
        #Core
        with self.get_sqlconnection(self.sql_server) as cursor:
            datos = pd.read_sql_query(self.query, cursor)
            datos = self.set_index_datetime(datos)
            # last_day = datetime.strptime(datos.index.max(), '%Y-%m') + relativedelta(month=1)
            # print(last_day)
            # future_days = [(last_day + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(self.PASOS)]
            # print(future_days)
            # last_day = datos.index.max() + timedelta(days=1)
            # future_days = [last_day + timedelta(days=i) for i in range(self.PASOS)]
            last_day = datetime.strptime(datos.index.max(), '%Y-%m' ) + relativedelta(months=1)
            print(last_day)
            future_days = [last_day + relativedelta(months=i) for i in range(self.PASOS)]
            print(future_days)
            for i in range(len(future_days)):
                future_days[i] = str(future_days[i])[:10]
            # future_data = pd.DataFrame(future_days)
            # future_data.columns = ['fecha']
            future_data = pd.DataFrame(future_days, columns=['fecha'])
            model = self.crear_modeloFF()
            dirmodels_name = './models/'+datetime.now().strftime('%Y-%m-%d')
            if not os.path.exists(dirmodels_name):
                os.makedirs(dirmodels_name, exist_ok=True)
            data = []
            total_col = len(datos.columns)
            print("total de columnas"+str(total_col))
            for i,column in enumerate(datos.columns):
                data = datos.filter([column])
                data.set_index(datos.index, inplace=True)
                data = self.eliminar_anomalias(data)
                x_train, y_train, x_val, y_val, scaler, values = self.create_x_y_train(data)
                model, x_test = self.entrenar_modelo(x_train, y_train, x_val, y_val, scaler, values, data, model)
                results = []
                for i in range(self.PASOS):
                    parcial = model.predict(x_test)
                    results.append(parcial[0])
                    x_test = self.agregarNuevoValor(x_test, parcial[0])
                adimen = [x for x in results]
                inverted = scaler.inverse_transform(adimen)
                y_pred = pd.DataFrame(inverted.astype(int))
                future_data[column]= inverted.astype(int)
            # Parte nueva para guardar los modelos
            datetim_e = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            model_path = dirmodels_name+"/model-training"+datetim_e
            os.makedirs(model_path, exist_ok=True)
            model_name = model_path+'/model_training-'+datetim_e+'.keras'
            model.save(model_name)
            future_data = self.set_index_datetime(future_data)

            datos.index = pd.to_datetime(datos.index)
            future_data.index = pd.to_datetime(future_data.index)

            #Creamos un directorio para guardar los datos del primer entrenamiento
            path = model_path+"/trainedModel_dataPredict/"+datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            os.makedirs(path)
            
            #Configuracion de las imagenes
            plt.rcParams['figure.figsize' ] = (16, 9)
            plt.style.use('fast')

            #Graficar los dataframes
            for i in range(len(datos.columns)):
                data = datos[datos.columns[i]][:]
                #Para asignar los valores de los años a los que está sujeto el proyecto se debe guardar en una
                #variable global y luego 
                plt.plot(data.index, data,label='Historial 2015 - 2020')
                plt.plot(future_data.index, future_data[future_data.columns[i]], label='Predicción 2021 - 2022')
                xtics = data.index.union(future_data.index)[::8]

                plt.xticks(xtics)
                plt.xlabel('Fecha')
                plt.ylabel('Ventas')
                plt.title('Predicción de la demanda del {p0} para el año del 2021'.format(p0=datos.columns[i]))
                plt.legend()
                plt.figtext(0.01, 0.01, "Realizado el: "+datetime.now().strftime('%H:%M:%S %d-%m-%Y'), fontsize=10, color="gray")
                plt.figtext(0.60, 0.01, "Gestión de Innovación en Tecnología Informática S.C.P | Grupo Consultores®", fontsize=10, color="gray")
                name = path+'/GraphicalPrediction_on_'+str(datos.columns[i])+".jpg"
                plt.savefig(name, dpi=300)
                plt.close()  # Cerrar la figura para liberar memoria
    


