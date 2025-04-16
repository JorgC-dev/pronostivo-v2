from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
from rich.table import Table
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import configparser
# from experimentals.engine import engine
from engine import engine
import pyodbc
import random
import string



console = Console()
keys = ["SERVER","PORT","DATABASE","USER","PASSWORD","DRIVER","OTHER"]
SQL_PATH = './SQL/'
SQL_TQUERY = SQL_PATH+'training_query.sql'
SQL_RQUERY = SQL_PATH+'retraining_query.sql'
SQL_PQUERY = SQL_PATH+'prediction_query.sql'
INI_PATH = 'config.ini'
MODELSDIR_PATH = './models/'

def prepareConection(mode):
    config = configparser.ConfigParser()
    ready = checkAllDirectory()
    if not ready:
    # if not os.path.exists('config.ini'):
        settings()
    else:
        try:
            config.read('config.ini') 
            console.clear()
            console.print("[bold green]GESTOR DE CONFIGURACIÓN DE CONEXIÓN SIPPBST [/]")
            console.print("A continuación se muestran los parámetros de conexión a base de datos registrados")
            table = Table("#","SERVER","PORT","DATABASE","USER","PASSWORD","DRIVER","OTHER")
            if config:
                if len(config) >= 1:
                    num = 1
                    for section in config.sections():
                        server = ""
                        port= ""
                        database =""
                        uid =""
                        password = ""
                        driver = ""
                        other = ""
                        
                        #create variables to asign the key and the value
                        for key, value in config[section].items():
                            #create variables with the info about each one
                            if key == 'server':
                                server = value
                            elif key == 'port':
                                port = str(value)
                            elif key == 'database':
                                database = value
                            elif key == 'user':
                                uid = value
                            elif key == 'password':
                                password = value
                            elif key == 'driver':
                                driver = value
                            elif key == 'other':
                                other = value
                        #add the variables at the table
                        table.add_row(str(num),server,port, database, uid, password, driver, other)
                        num += 1
                    table.add_row("[bold magenta]a[/]","[bold magenta]añadir otro[/]","[bold magenta]■■■■■[/]","[bold magenta]■■■■■[/]","[bold magenta]■■■■■[/]","[bold magenta]■■■■■[/]","[bold magenta]■■■■■[/]","[bold magenta]■■■■■[/]")
                    console.print(table)

                    # print(len(config))
                    
                    #generlo listado de opciones
                    choices = [str(i+1) for i in range(len(config))]
                    choices.append("a")
                    opcion = Prompt.ask("[bold blue]•[/] Selecciona una opción", choices= choices)
                    
                    if opcion == 'a':
                        #add items
                        settings()
                    else:
                        #generate connection setting string
                        rgSelected = list(config.keys())[int(opcion)]
                        properties = config[rgSelected]
                        vstrConnection = ""
                        for key, value in properties.items(): 
                            
                            if key == "driver":
                                vstrConnection += key+"={"+value+"};"
                            elif key == "other":
                                vstrConnection += value+";"
                            elif key == "port":
                                vstrConnection += ","+value+";"
                            elif key == "server":
                                vstrConnection += key+"="+value
                            else: 
                                vstrConnection += key+"="+value+";"
                        print(vstrConnection)
                        engine, vstrConnection,query = check_engine(vstrConnection,mode)
                        return engine, vstrConnection, query
                else:
                    pass
        except Exception as e: 
            console.print("[bold red] Hay un error con el archivo [/]")
            console.print(e)


def setMode(mode, engine, sql_serverConfig,query):
    print("Iniciando gestor de modulos de funcionamiento")
    try:
        if mode == "Entrenar-Crear modelo":
                engine.main()
        if mode == "Hacer Predicciones":
            #preguntar sobre cuantos días de prediccion quiere
            steps = Prompt.ask("¿Cuantos días a futuro?")
            model_path = showSettingsModel()
            predictingModel(engine, sql_serverConfig, query, steps, model_path)
        if mode == "Reentrenamiento":
            model_path = showSettingsModel()
            retrainingModel(engine,sql_serverConfig,query,31,model_path)
    except Exception as e:
        console.print("[bold red] Error!, Verifique que la query sea correcta [/]")
        console.print(f'[bold red] Detalles: {e} [/]')

def set_terminal_size(columns=80, rows=24):
    os.system(f'mode con: cols={columns} lines={rows}' if os.name == 'nt' else f'printf "\e[8;{rows};{columns}t"')


def check_engine(sql_serverConfig,mode):
    
    while True:
        console.clear() 
        console.print("[bold green] Obteniendo la query...[/bold green]")
        #Revisar la carpeta sql y checar que los archivos sql de entrenamiento y reentrenamiento estén disponibles
        # qPrompt = input("Por favor, ingrese la consulta SQL a continuación.")
        # qPrompt = qPrompt.strip()

        if mode == "Entrenar-Crear modelo":
            query_path = SQL_TQUERY
        if mode == "Hacer Predicciones":
            query_path = SQL_PQUERY
        if mode == "Reentrenamiento":
            query_path = SQL_RQUERY
        
        with open(query_path, 'r', encoding='utf-8') as file:
            sql_script = file.read()
            # print(sql_serverConfig)
            console.print("[bold green] Validando contenido...[/bold green]") 
        if sql_script:
            #Si el archivo no está vacío
            obj = engine(sql_serverConfig, str(sql_script))
            if obj:
                try: 
                    obj.get_sqlconnection(sql_serverConfig)
                except Exception as e:
                    console.print("[bold red]¡Error![/bold red]")
                    console.print(f"[bold red]Error: {e}[/bold red]")
                    pass
            else:
                console.print("[bold red]¡Engine no respondió![/bold red]")
            return obj, sql_serverConfig, sql_script
        else: 
            console.print("[bold orange] Archivo vacío, por favor, inserte la consulta y vuelva a intentarlo [/]")



def intro():
    title = Text("SIPPBST v2.0 | Grupo Consultores® 2025", style="bold yellow")
    description = ("Sea bienvenido a este programa de predicción de la demanda de productos, con el cual podrá predecir la demanda de productos de forma personalizada "
                   "Es un sistema robusto, confiable y se adapta a sus necesidades. Esta CLI le guiará en el proceso de configuración inicial del sistema. "
                   "Gracias por usar nuestro sistema")
    console.print(Panel(description, title=title, expand=False))
    console.rule("[bold green] Por favor, indique lo que hará: [/]")
    options = ["Hacer Predicciones","Entrenar-Crear modelo","Reentrenamiento"]
    options = [options.pop()] if not os.path.exists('./models') else options
    for i, option in enumerate(options):
        console.print(f"{str(i+1)} > {option}")
    vseleccion = int(Prompt.ask(choices=[str(i) for i in range(1,len(options)+1)]))
    vseleccion = options[vseleccion-1]
    print(vseleccion)
    return vseleccion




def checkAllDirectory():
    result = False
    try:
        #Model's path 
        if not os.path.exists(MODELSDIR_PATH):
            os.makedirs(MODELSDIR_PATH)
        #Ini's path
        if not os.path.exists(INI_PATH):
            with open(INI_PATH,'w') as file:
                pass
        #SQL's path
        if not os.path.exists(SQL_PATH):
            os.makedirs(SQL_PATH)

            #traininig files
            with open(SQL_TQUERY,'w') as file: 
                pass
            
            #Retraininig files
            with open(SQL_RQUERY,'w') as file:
                pass

            #Prediction file
            with open(SQL_PQUERY,'w') as file:
                pass

        
        result = True
    except Exception as e: 
        result = False
    return result


def settings():

    console.clear()
    console.print("[bold green]GESTOR DE DIRECCIONES DE BASE DE DATOS SIPPBST[/bold green]")
    console.print("Si estas viendo este mensaje es porque aun no has configurado \r\n la cadena de conexión o deseas configurar otra")
    console.print("[bold]Iniciando modo de configuración...[bold]")

    #mostramos los drivers primero
    driver = showDrivers()

    #pedimos que nos ingrese los valores de cada campo
    Kvalues = []
    for keyName in keys:
        if keyName == 'DRIVER':
            Kvalues.append(driver)
        elif keyName == 'PASSWORD':
            prompt = f'[bold blue]•[/] Ingresa el campo {keyName}'
            value = Prompt.ask(prompt,password=True)
            if value:
                Kvalues.append(value)
        else:
            prompt = f'[bold blue]•[/] Ingresa el campo {keyName}'
            value = Prompt.ask(prompt)
            if value:
                Kvalues.append(value)
        idSection =  generateIDSections()

    #una vez generado mandamos
    generate_sections(INI_PATH,idSection,keys,Kvalues)
    prepareConection()

def generateIDSections():
    numero = random.randint(10, 99)
    letra1 = random.choice(string.ascii_uppercase)
    letra2 = random.choice(string.ascii_uppercase)
    id_aleatorio = f"ID{numero}{letra1}{letra2}"
    return id_aleatorio

def generate_sections(file_name, region_name, keys, values=None):
    config = configparser.ConfigParser()
    try:
        config.read(file_name)
    except Exception as e:
        print(f"Error al leer el archivo INI: {e}")

    if region_name not in config:
        config.add_section(region_name)

    for i, key in enumerate(keys):
        if values and i < len(values):
            config.set(region_name, key, str(values[i]))
        else:
            config.set(region_name, key, "")

    with open(file_name, 'w') as configfile:
        config.write(configfile)
    



def showDrivers():
    drivers = pyodbc.drivers()
    table = Table(title="Drivers disponibles")
    table.add_column("Opción", style="cyan")
    table.add_column("Driver", style="magenta")
    for i, driver in enumerate(drivers, 1):
        table.add_row(str(i), driver)
    console.print(table)
    driver = Prompt.ask("[bold blue]•[/] Selecciona un driver", choices=[str(i) for i in range(1, len(drivers)+1)])
    driver = drivers[int(driver)-1]
    return driver

def showMenu():
    console.clear()
    console.rule("[bold green]Bienvenido al menú de gestión de modelos de predicción[/]")
    console.print("Aquí podrás visualizar todos tus modelos de predicción, así como retomarlos para realizar predicciones y reentrenarlos")



#Menu models
def showSettingsModel():
    contiNue = True
    path = './models'
    while contiNue:
        console.clear()
        models_dir = os.listdir(path)
        console.rule("[bold green]Gestión de modelos[/bold green]")
        console.print("[bold green]A continuación se presenta una tabla con los modelos disponibles[/bold green]")
        console.print("*** Seleccione un modelo para continuar ***")
        table = Table(title="Modelos disponibles")
        table.add_column("Opción", style="cyan")
        table.add_column("Modelo", style="magenta")
        table.add_column("Fecha de entrenamiento", style="yellow")

        #Funcion para listar todos los modelos
        models_name = []
        models_path = []
        for ruta_actual, subdirectorio, archivos in os.walk(path):
            for archivo in archivos:
                if archivo.endswith('.keras'):
                    models_path_join = os.path.join(ruta_actual, archivo)
                    models_path.append(models_path_join)
                    models_name.append(archivo)

        for i, model in enumerate(models_name,1):
            table.add_row(str(i),str(model),"Proximamente")
        console.print(table)
        model = Prompt.ask("[bold blue]•[/] Selecciona un modelo", choices=[str(i) for i in range(1, len(models_name)+1)])
        console.print(f"[bold green]Modelo seleccionado: [/]"+f"""[bold orange]{models_name[int(model)-1]}[/]""")
        model_path = models_path[int(model)-1]
        contiNue = False
        return model_path


def predictingModel(obj, sql_serverConfig, query, steps, model_path):
    console.print("[bold cyan]Ruta -> [/]"+model_path)
    console.print("[bold green]Iniciando proceso de predicción[/] "+model_path)
    obj.modelPredicFuncion(sql_serverConfig,query,steps, model_path)

def retrainingModel(obj, sql_serverConfig, query, steps, model_path):
    console.print("[bold cyan] Ruta -> [/]"+model_path)
    console.print("[bold green] Iniciando proceso de Reentrenamiento... [/]")
    obj.modelRetrainingFunction(sql_serverConfig,query,steps,model_path)
    console.print("[bold green]✔ Modelo Reentrenado guardado con éxito [/]")
    console.print("[bold green]✔ Proceso de Reentrenamiento finalizado... [/]")


def main(salir = False):

    while salir == False:
        console.clear()
        vseleccion = intro()
        engine, vstrConnection, query =  prepareConection(vseleccion)
        setMode(vseleccion,engine,vstrConnection,query)
        console.print("[bold cyan]\n¿Desea salir del programa?[/bold cyan]")
        keyboard1 = Prompt.ask("[bold cyan]Presione [bold red]S[/bold red] para salir o cualquier otra tecla para continuar[/bold cyan]")
        if keyboard1 == "S" or keyboard1=="s":
            console.clear()
            salir = True
        else:
            salir = False
            console.clear()

if __name__ == "__main__":
    main()
