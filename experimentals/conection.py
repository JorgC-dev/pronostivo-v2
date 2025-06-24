"""
Clase que gestiona las conexiones a la base de datos

V1.0
"""
import pyodbc
import pandas as pd



class connection:
    def __init__(self,sql_serverConfig):
        self.sql_server= sql_serverConfig

    def get_sqlconnection(self,console=None):
        if console != None: 
            with console.status("[bold yellow]Inicializando[/]", spinner="arc") as status: 
                try: 
                    connection = pyodbc.connect(self.sql_server)
                    console.print("[bold green]✔ Conexion establecida satisfactoriamente[/]")
                except Exception as e:
                    console.print("[bold red]✘ Se ha producido un error al establecer la conexión[/]")
                    console.print(f'[bold red]Detalles:{e}[/]')
        else:
            try: 
                connection = pyodbc.connect(self.sql_server)
                print("Conexion establecida satisfactoriamente")
            except Exception as e:
                print("Se ha producido un error al establecer la conexión")
                print(f'Detalles:{e}')
        return connection
    
    def getSQL(self, query, console=None):
        """
        Método que hace consultas hacia la base de datos
        """
        if console != None: 
            console.print("[bold yellow] Obteniendo la consulta [/]")
            try: 
                with self.get_sqlconnection() as cursor: 
                    data = pd.read_sql_query(query,cursor)
            except Exception as e: 
                console.print(f'[bold red]Ocurrió un error. Detalles{e}[/]')
                
        else: 
            try: 
                with self.get_sqlconnection() as cursor: 
                    data = pd.read_sql_query(query,cursor)
            except Exception as e: 
                print("Ocurrió un error",e)
        return data