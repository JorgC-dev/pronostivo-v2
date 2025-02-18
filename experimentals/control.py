from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
from rich.table import Table
import os
import json
import keyboard  # Para detectar una entrada de texto en cualquier sistema operativo
from engine import engine
from rich.__main__ import make_test_card

console = Console()

def wait_for_input():
    console.print("[bold cyan]\nEscribe algo y presiona Enter para continuar...[/bold cyan]")
    user_input = input()  # Espera la entrada de una cadena y la almacena
    return user_input

def initialyze():
    set_terminal_size(100, 0)
    console.clear()
    intro()
    if not os.path.exists('config.json'):
        settings()
    else: 
        table = Table("#","Servers_name","Databases_name","Drivers Available")
        with open('config.json', 'r') as f: 
            config = json.load(f)
            if config:
                if len(config) >= 1:
                    for i in range(len(config)):
                        table.add_row(str(i+1),str(config[i]['server_name']),str(config[i]['database_name']), str(config[i]['driver']))
                else:
                    table.add_row("1",str(config['server_name']),str(config['database_name']), str(config['driver']))
                console.print(table)
                opcion = Prompt.ask("[bold blue]•[/] Selecciona una opción", choices=[str(i+1) for i in range(len(config))])
                server_name = config[int(opcion)-1]['server_name']
                database_name = config[int(opcion)-1]['database_name']
                driver = config[int(opcion)-1]['driver']
                # console.print("[bold green]Inicializando...[/bold green]")
                sqlServerConfig = setSQLServerConfig(server_name, database_name, driver)
                connect_engine(sqlServerConfig)

def set_terminal_size(columns=80, rows=24):
    os.system(f'mode con: cols={columns} lines={rows}' if os.name == 'nt' else f'printf "\e[8;{rows};{columns}t"')

def setSQLServerConfig(server_name, database_name,driver):
    config =f"""
        DRIVER={{{driver}}};
        Server={server_name};
        database={database_name};
        Trusted_connection=yes;
        """
    return config


def connect_engine(sql_serverConfig):
    console.print("[bold green]Conectando con la base de datos...[/bold green]")
    query = """
        DECLARE @cols AS NVARCHAR(MAX)
DECLARE @query AS NVARCHAR(MAX)

SET @cols = STUFF((SELECT ', SUM(CASE WHEN [P].[id] = ' + CONVERT(NVARCHAR(10), [SUB]. [id]) + ' THEN [H].[cantidad] ELSE 0 END) AS [' + [SUB].[producto] + ']'
FROM (
SELECT TOP 5
[H].[id_DimProducto] AS id,
[P].[producto] AS producto
FROM [demo_prediccion].[dbo].[hechos] AS [H]
INNER JOIN [demo_prediccion].[dbo].[Dim_productos] AS [P] ON [P].[id] = [H].[id_DimProducto]
GROUP BY [H].[id_DimProducto], [P].[producto]
ORDER BY SUM([H].[cantidad]) DESC, [H].[id_DimProducto] ASC
) AS SUB
FOR XML PATH(''), TYPE).value('.', 'NVARCHAR(MAX)'),1,2,'')

SET @query = '
SELECT
SUBSTRING(CAST([F].[fecha] AS VARCHAR(256)), 0, 8) AS fecha, ' + @cols + '
FROM
[demo_prediccion].[dbo].[hechos] AS [H]
INNER JOIN [demo_prediccion].[dbo].[Dim_fechas] AS [F] ON [F].[id] = [H].[id_DimFechas]
INNER JOIN [demo_prediccion].[dbo].[Dim_productos] AS [P] ON [P].[id] = [H].[id_DimProducto]
GROUP BY
SUBSTRING(CAST([F].[fecha] AS VARCHAR(256)), 0, 8)
ORDER BY
SUBSTRING(CAST([F].[fecha] AS VARCHAR(256)), 0, 8)'

EXEC(@query)
        """
    obj = engine(sql_serverConfig, query)
    try: 
        obj.get_sqlconnection(sql_serverConfig)
        showSettingsModel(obj,sql_serverConfig,query)
    except Exception as e:
        console.print("[bold red]Error al establecer la conexión[/bold red]")
        console.print(f"[bold red]Error: {e}[/bold red]")
        pass



def intro():
    title = Text("EnvPrediccion | Grupo Consultores® 2025", style="bold yellow")
    description = ("Sea bienvenido a este programa de predicción de la demanda de productos, con el cual podrá predecir la demanda de productos de forma personalizada "
                   "Es un sistema robusto, confiable y se adapta a sus necesidades. Esta CLI le guiará en el proceso de configuración inicial del sistema. "
                   "Gracias por usar nuestro sistema")
    console.print(Panel(description, title=title, expand=False))


def settings():
    console.print("[bold green]No encontramos ninguna configuración previa[/bold green]")
    console.print("[bold]Iniciando modo de configuración...[bold]")
    server_name = Prompt.ask("[bold blue]•[/] Ingresa el nombre del server")
    database_name = Prompt.ask("[bold blue]•[/] Ingresa el nombre de la base de datos")
    driver = showDrivers()
    with open('config.json', 'w') as file:
            json.dump({{"server_name": server_name, "database_name": database_name,"driver":driver}}, file)
    return server_name, database_name

def showDrivers():
    drivers_name = ["SQL Server", "MySQL", "PostgreSQL", "Oracle"]
    drivers = ["ODBC Driver 17 for SQL Server", "MySQL ODBC 8.0 Unicode Driver", "PostgreSQL Unicode", "Oracle in XE"]
    table = Table(title="Drivers disponibles")
    table.add_column("Opción", style="cyan")
    table.add_column("Driver", style="magenta")
    for i, driver in enumerate(drivers_name, 1):
        table.add_row(str(i), driver)
    console.print(table)
    driver = Prompt.ask("[bold blue]•[/] Selecciona un driver", choices=[str(i) for i in range(1, len(drivers_name)+1)])
    driver = drivers[int(driver)-1]
    return driver

def showMenu():
    console.clear()
    console.rule("[bold green]Bienvenido al menú de gestión de modelos de predicción[/]")
    console.print("Aquí podrás visualizar todos tus modelos de predicción, así como retomarlos para realizar predicciones y reentrenarlos")



#Menu models
def showSettingsModel(obj,sql_serverConfig,query):
    console.clear()
    models = os.listdir('./models')
    console.rule("[bold green]Gestión de modelos[/bold green]")
    console.print("[bold green]Seleccione un modelo para hacer una acción[/bold green]")
    table = Table(title="Modelos disponibles")
    table.add_column("Opción", style="cyan")
    table.add_column("Modelo", style="magenta")
    for i, model in enumerate(models, 1):
        table.add_row(str(i), model)
    console.print(table)
    model = Prompt.ask("[bold blue]•[/] Selecciona un modelo", choices=[str(i) for i in range(1, len(models)+1)])
    console.clear()
    console.print(f"[bold green]Modelo seleccionado: [/]"+f"""[bold orange]{models[int(model)-1]}[/]""")
    model_path = f"./models/{models[int(model)-1]}"
    obj.modelPredicFuncion(sql_serverConfig,query,12, model_path)



def main(salir = False):

    while salir == False:
        initialyze()
        # showSettingsModel()
        # showSettingsModel()
        # showMenu()
        console.print("[bold cyan]\n¿Desea salir del programa?[/bold cyan]")
        keyboard1 = Prompt.ask("[bold cyan]Presione [bold red]S[/bold red] para salir o cualquier otra tecla para continuar[/bold cyan]")
        if keyboard1 == "S":
            salir = True
        else:
            salir = False
            console.clear()

if __name__ == "__main__":
    main()
