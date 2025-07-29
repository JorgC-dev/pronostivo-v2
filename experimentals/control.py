"""
README

CLI principal para SIPPBST, que controla Predicciones, Rentrenamiento, Y creación de modelos en una amplia gama de escenarios.

Esta versión se integra LSTM y MLP. 
"""

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
from rich.table import Table
import os
import json
from collections import defaultdict
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import configparser
from mlp_engine import Mlp_engine 
from lstm_engine import Lstm_engine
from conection import connection
import pyodbc
import random
import string



console = Console()
keys = ["SERVER","PORT","DATABASE","USER","PASSWORD","DRIVER","OTHER"]

#BASE
SQL_PATH = './SQL/'

#MLP
MLP_SQL_PATH = SQL_PATH+'MLP/'
MLP_SQL_TQUERY = MLP_SQL_PATH+'training_query.sql'
MLP_SQL_RQUERY = MLP_SQL_PATH+'retraining_query.sql'
MLP_SQL_PQUERY = MLP_SQL_PATH+'prediction_query.sql'

#LSTM
LSTM_SQL_PATH = SQL_PATH+'LSTM/'
LSTM_SQL_HQUERY = LSTM_SQL_PATH+'historic_query.sql'
LSTM_SQL_PQUERY = LSTM_SQL_PATH+'product_query.sql' 
LSTM_SQL_RQUERY = LSTM_SQL_PATH+'retraining_query.sql'


SQL_DIRECTORIES =  [MLP_SQL_TQUERY,LSTM_SQL_HQUERY,LSTM_SQL_PQUERY]

INI_PATH = 'config.ini'
MODELSDIR_PATH = './models/'

def prepareConection():
    config = configparser.ConfigParser()
    ready = checkAllDirectory()
    if not ready:
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

                        for key, value in config[section].items():
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
                        table.add_row(str(num),server,port, database, uid, password, driver, other)
                        num += 1
                    table.add_row("[bold magenta]a[/]","[bold magenta]añadir otro[/]","[bold magenta]■■■■■[/]","[bold magenta]■■■■■[/]","[bold magenta]■■■■■[/]","[bold magenta]■■■■■[/]","[bold magenta]■■■■■[/]","[bold magenta]■■■■■[/]")
                    console.print(table)
                    choices = [str(i+1) for i in range(len(config))]
                    choices.append("a")
                    opcion = Prompt.ask("[bold blue]•[/] Selecciona una opción", choices= choices)
                    
                    if opcion == 'a':
                        settings()
                    else:
                        rgSelected = list(config.keys())[int(opcion)]
                        properties = config[rgSelected]
                        vstrConnection = ""
                        server = ""
                        for key, value in properties.items(): 
                            if key == "driver":
                                vstrConnection += key+"={"+value+"};"
                            elif key == "other":
                                vstrConnection += value+";"
                            elif key == "server":
                                server = value
                            elif key == "port": 
                                port = value
                                if port != "":
                                    vstrConnection += f'server={server},{port};'
                                else: 
                                    vstrConnection += str(f'server={server};')
                            elif key == "user":
                                if value != "":
                                    vstrConnection += key+"="+value+";"
                            elif key == "password":
                                if value != "":
                                    vstrConnection += key+"="+value+";"
                            else: 
                                vstrConnection += key+"="+value+";"
                        v_connection, vstrConnection = check_engine(vstrConnection)
                        return v_connection, vstrConnection
                else:
                    pass
        except Exception as e: 
            console.print("[bold red] Hay un error con el archivo [/]")
            console.print(e)


def selectTechnology():
    console.rule("[bold green] Por favor, indque la tecnología que usará: [/]")
    table = Table("MLP(Multilayer Perceptron)","LSTM(Long Short Term Memory)")
    table.add_row("✔ Simplicidad y Facilidad de Implementación","✔ Manejo Efectivo de Dependencias Temporales Largas")
    table.add_row("✔ Capacidad para Capturar Relaciones No Lineales","✔ Robustez a Problemas de Gradiente")
    table.add_row("✔ Versatilidad","✔ Aprendizaje Automático de Características Temporales")
    table.add_row("✔ Menos Requisitos de Datos (en comparación con LSTMs)","✔ Idóneas para Datos Secuenciales")
    table.add_row("✔ Entrenamiento Rápido","✔ Capacidad para Modelar Patrones Complejos y Dinámicos")
    console.print(table)
    options = ["MLP(Multilayer-Perceptron)","LSTM(Long Short Term Memory)"]
    for i, option in enumerate(options):
        console.print(f"{str(i+1)} > {option}")
    vseleccion = int(Prompt.ask(choices=[str(i) for i in range(1,len(options)+1)]))
    vseleccion = options[vseleccion-1]
    return vseleccion


def setMode(mode, v_connection=None, sql_serverConfig=None):
    console.print("[bold yellow] Iniciando gestor de modulos de funcionamiento [/]")
    try:
        if mode == "Entrenar-Crear modelo":
            vseleccion = selectTechnology()
            if vseleccion == "MLP(Multilayer-Perceptron)":
                with open(MLP_SQL_TQUERY, 'r', encoding='utf-8') as file:
                    query = file.read()
                    console.print("[bold green] Validando contenido...[/]") 
                if query:
                    mlp_engine = Mlp_engine()
                    data = v_connection.getSQL(query, console)
                    mlp_engine.main(data)
            elif vseleccion == "LSTM(Long Short Term Memory)":
                with console.status("[bold yellow] Realizando consultas y verificando archivos [/]", spinner="arc") as status:
                    status.update("[bold Yellow] {data histórica} Validando contenido... [/]") 
                    with open(LSTM_SQL_HQUERY, 'r', encoding='utf-8') as file:
                        h_query = file.read()
                        status.update("[bold green] Hecho [/]")
                    status.update("[bold green] {data productos} Validando data de productos... [/]")
                    with open(LSTM_SQL_PQUERY, 'r', encoding='utf-8') as file:
                        p_query = file.read()
                        status.update("[bold green] Hecho [/]")
                    if h_query != "" and p_query != "":
                        status.update("[bold yellow] Realizando consultas [/]")
                        lstm_engine = Lstm_engine()
                        historic_data = v_connection.getSQL(h_query, console)
                        status.update("[bold green] Consulta 1... Hecho [/]")
                        product_data = v_connection.getSQL(p_query, console)
                        status.update("[bold green] Consulta 2... Hecho [/]")
                console.print("[bold yellow] Iniciando creación de modelos LSTM[/]")
                category_id = Prompt.ask("[bold blue]•[/] Indique aquí un identificador o categoria a " \
                "la que pertenecerán los modelos, esto será unicamente para reconocimiento de los modelos")
                lstm_engine.main(category_id,historic_data,product_data)
                console.print("[bold green]✔ Finalizado [/]")
        if mode == "Hacer Predicciones":
            model_path, model_metadata, category_metadata = showSettingsModel()
            console.print("[bold cyan]  Ruta completa -> [/]"+model_path)
            console.print("[bold yellow]Iniciando proceso de predicción[/]")
            if model_metadata['GENERAL_INFO']['TECHNOLOGY'] == "LSTM":
                model, data_trat, features, n_features, Scaler_y, steps, Scaler_x, p_default, root_dir, id_product = Lstm_engine().loadUtils(model_path,model_metadata, category_metadata)
                Lstm_engine().predictions(model, data_trat, features, n_features, Scaler_y, id_product,root_dir, steps, 64, Scaler_x, p_default)
            elif model_metadata['GENERAL_INFO']['TECHNOLOGY'] == "MLP":
                predictions = Prompt.ask("[bold yellow]•[/] Indique el total de predicciones que desea realizar")
                Mlp_engine().predictions(model_path,model_metadata,int(predictions))
            console.print("[bold green]✔ done![/]")
        if mode == "Reentrenamiento":
            model_path, model_metadata, category_metadata = showSettingsModel()
            console.print("[bold cyan]  Ruta completa -> [/]"+model_path)
            console.print("[bold yellow] Iniciando proceso de Reentrenamiento [/]")
            if model_metadata['GENERAL_INFO']['TECHNOLOGY'] == "LSTM":
                model, data_trat, features, n_features, Scaler_y, steps, Scaler_x, p_default, root_dir, id_product = Lstm_engine().loadUtils(model_path,model_metadata, category_metadata)
                model = Lstm_engine().retrainingModel_2(model,data_trat,features,n_features,Scaler_y,steps,Scaler_x,p_default,root_dir,id_product,category_metadata)
            elif model_metadata['GENERAL_INFO']['TECHNOLOGY'] == "MLP":
                Mlp_engine().retraining(model_path,model_metadata)
    except Exception as e:
        console.print("[bold red] Error! [/]")
        console.print(str(f'[bold red] Detalles:{e} [/]'))

def set_terminal_size(columns=80, rows=24):
    os.system(f'mode con: cols={columns} lines={rows}' if os.name == 'nt' else f'printf "\e[8;{rows};{columns}t"')


def check_engine(sql_serverConfig):
    while True:
        console.clear() 
        console.print(f'Apuntador de conexión --> [bold cyan]{sql_serverConfig} [/]')
        obj = connection(sql_serverConfig)
        if obj: 
            try: 
                obj.get_sqlconnection(console)
            except Exception as e:
                console.print("[bold red]¡Error![/bold red]")
                console.print(f"[bold red]Detalles: {e}[/bold red]")
                pass
        else: 
            console.print("[bold red]¡Engine no respondió![/bold red]")
        return obj, sql_serverConfig



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
        os.makedirs(SQL_PATH, exist_ok=True)
        os.makedirs(SQL_PATH, exist_ok=True)
        os.makedirs(MLP_SQL_PATH, exist_ok=True)
        os.makedirs(LSTM_SQL_PATH, exist_ok=True)

        #ENTRENAMIENTO
        for directory in SQL_DIRECTORIES:
            try:
                with open(directory, 'r', encoding='utf-8') as file:
                    contenido = file.read()
                    pass
            except Exception as e: 
                with open(directory,'w') as file: 
                    pass

        # RETAINING
        with open(MLP_SQL_RQUERY,'w') as file:
            pass

        # PREDICTION
        with open(MLP_SQL_PQUERY,'w') as file:
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
    driver = showDrivers()
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
        console.print(f"Error al leer el archivo INI: {e}")
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


#Menu models
def showSettingsModel():
    contiNue = True
    path = './models'
    while contiNue:
        console.clear()
        console.rule("[bold green]Gestión de modelos[/bold green]")
        console.print("[bold green]A continuación se presenta una tabla con los modelos disponibles[/bold green]")
        console.print("*** Seleccione un modelo para continuar ***")
        table = Table(title="Modelos disponibles")
        table.add_column("Opción", style="cyan")
        table.add_column("Modelo", style="magenta")
        table.add_column("Último entrenamiento", style="yellow")
        table.add_column("Tecnología",style="#FFA500")
        table.add_column("Cantidad de Features", style="white")

        #LISTAR MODELOS 
        model_metadata = {}
        category_metadata = {}
        models_name = []
        models_path = []

        cat_before = ""
        for ruta_actual, subdirectorio, archivos in os.walk(path):
            for archivo in archivos:
                if archivo.endswith('.keras'):
                    models_path_join = os.path.join(ruta_actual, archivo)
                    models_path.append(models_path_join)
                    models_name.append(archivo)
                    metadata_path = ruta_actual+'/metadata.json'
                    if os.path.isfile(metadata_path):                    
                        try: 
                            metadata_path = os.path.join(ruta_actual,'metadata.json')
                            with open(metadata_path,'r',encoding="utf-8") as file:
                                metadata = json.load(file)
                                for item in metadata: 
                                    if "MODEL_REFERENCY" in item:
                                        model_name = item['MODEL_REFERENCY']['MODEL_NAME']
                                        technology = item['MODEL_REFERENCY']['TECHNOLOGY']
                                    if "GENERAL_INFO" in item:
                                        model_metadata[model_name] = item
                        except Exception as e: 
                            console.print("[bold red] ¡Ocurrió un error al leer la metadata! [/]")
                            console.print(f'[bold red] Detalles: {e} [/]')
                    else: 
                        console.print("[bold red] Metadata no encontrada [/]")

                    if technology == "LSTM":
                        metadata_path = os.path.normpath(os.path.join(ruta_actual,"../../../metadata_CAT.json"))
                        if os.path.isfile(metadata_path):
                            try: 
                                with open(metadata_path, 'r',encoding="utf-8") as file: 
                                    metadata = json.load(file)
                                    n_features = 0
                                    same = True
                                    for item in metadata: 
                                        if "CATEGORY_ID" in item: 
                                            category = item['CATEGORY_ID']
                                            if category != cat_before:
                                                same = False
                                                cat_before = category
                                        if not same: 
                                            if "N_FEATURES" in item:
                                                category = item['CATEGORY_ID']
                                                category_metadata = item
                            except Exception as e: 
                                console.print("[bold red] Ocurrió un error al leer la metadata [/]")
                                console.print(f'[bold red] Detalles {e} [/]')
        for i, model in enumerate(models_name,1):
            model_name, extension =  os.path.splitext(model)
            last_modified = model_metadata[model_name]['GENERAL_INFO']['FECHA_MODIFICACION']
            technology = model_metadata[model_name]['GENERAL_INFO']['TECHNOLOGY']
            if technology == "LSTM":
                n_features = str(category_metadata["N_FEATURES"])
            else: 
                n_features = "1"
            table.add_row(str(i),str(model),last_modified,technology,n_features)
        console.print(table)
        model = Prompt.ask("[bold blue]•[/] Selecciona un modelo", choices=[str(i) for i in range(1, len(models_name)+1)])
        console.print(f"[bold green]✔ Modelo seleccionado: [/]"+f"""[bold orange]{models_name[int(model)-1]}[/]""")
        model_path = models_path[int(model)-1]
        name, extension = os.path.splitext(models_name[int(model)-1])
        contiNue = False
        return model_path, model_metadata[name], category_metadata

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
        if vseleccion != "Hacer Predicciones":
            v_connection, vstrConnection =  prepareConection()
            setMode(vseleccion,v_connection,vstrConnection)
        else:
            setMode(vseleccion)
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
