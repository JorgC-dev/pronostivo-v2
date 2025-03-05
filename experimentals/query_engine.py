# Esta clase será para un sistema de consultas que podrá tomar y gestionar las tablas
import pyodbc
import pandas as pd
import matplotlib as plt

class query_engine:
    def __init__(self,sql_serverConfig):
        self.sql_server = sql_serverConfig
        print(sql_serverConfig)
    
    def get_sqlconnection(self, config_sqlServer):
        status = "query engine starting..."
        try: 
            connection = pyodbc.connect(config_sqlServer)
            status = "Conection successfully!"
        except Exception as e: 
            status = "An error occurred during the connection: "+e
        print(status)
        return connection
    
    def set_index_datatime(self,data):
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
    
    def get_dataFromQuery(self,query):
        with self.get_sqlconnection(self.sql_server) as cursor:
            data = pd.read_sql_query(query,cursor)
            data = self.set_index_datatime(data)

            #Convertir y analizar la data para identificar si tiene estacionalida
            plt.rcParams['figure.figsize'] = (16,9)
            plt.style.use('fast')

            data.plot()
            data.show()

            


