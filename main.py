import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from fastapi import FastAPI,HTTPException,Form, Query
from fastapi.middleware.cors import CORSMiddleware
#from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request


app = FastAPI()
#uvicorn main:app --reload

# Configura las plantillas para renderizar HTML
#templates = Jinja2Templates(directory="templates")

# Datos para el dropdown list
#opciones = ["Opción 1", "Opción 2", "Opción 3"]
ciudades={'Agoura Hills':1,'Alhambra':2,'Altadena':3,'Arcadia':4,'Artesia':5,'Azusa':6,'Baldwin Park':7,'Bell Gardens':8,'Bellflower':9,'Beverly Hills':10,'Burbank':11,'Calabasas':12,'Canoga Park':13,'Canyon Country':14,'Carson':15,'Castaic':16,'Cerritos':17,'Chatsworth':18,'Claremont':19,'Compton':20,'Covina':21,'Culver City':22,'Diamond Bar':23,'Downey':24,'Duarte':25,'El Monte':26,'El Segundo':27,'Encino':28,'Gardena':29,'Glendale':30,'Glendora':31,'Granada Hills':32,'Hacienda Heights':33,'Harbor City':34,'Hawaiian Gardens':35,'Hawthorne':36,'Hermosa Beach':37,'Huntington Park':38,'Inglewood':39,'La Canada Flintridge':40,'La Crescenta':41,'La Mirada':42,'La Puente':43,'La Verne':44,'Lakewood':45,'Lancaster':46,'Lawndale':47,'Littlerock':48,'Lomita':49,'Long Beach':50,'Los Angeles':51,'Lynwood':52,'Malibu':53,'Manhattan Beach':54,'Marina Del Rey':55,'Maywood':56,'Mission Hills':57,'Monrovia':58,'Montebello':59,'Monterey Park':60,'Newhall':61,'North Hills':62,'North Hollywood':63,'Northridge':64,'Norwalk':65,'Pacific Palisades':66,'Pacoima':67,'Palmdale':68,'Palos Verdes Peninsula':69,'Panorama City':70,'Paramount':71,'Pasadena':72,'Pico Rivera':73,'Playa Del Rey':74,'Playa Vista':75,'Pomona':76,'Porter Ranch':77,'Rancho Palos Verdes':78,'Redondo Beach':79,'Reseda':80,'Rosemead':81,'Rowland Heights':82,'San Dimas':83,'San Fernando':84,'San Gabriel':85,'San Pedro':86,'Santa Clarita':87,'Santa Fe Springs':88,'Santa Monica':89,'Sherman Oaks':90,'Sierra Madre':91,'South El Monte':92,'South Gate':93,'South Pasadena':94,'Stevenson Ranch':95,'Studio City':96,'Sun Valley':97,'Sunland':98,'Sylmar':99,'Tarzana':100,'Temple City':101,'Torrance':102,'Tujunga':103,'Universal City':104,'Valencia':105,'Valley Village':106,'Van Nuys':107,'Venice':108,'Walnut':109,'West Covina':110,'West Hills':111,'West Hollywood':112,'Whittier':113,'Wilmington':114,'Winnetka':115,'Woodland Hills':116}
categorias={'Korean restaurant': 0, 'Fast food restaurant': 1, 'Pizza restaurant': 10, 'Restaurant': 3, 'Mexican restaurant': 26, 'Sushi Restaurant': 17, 'Chinese restaurant': 6, 'Hamburger restaurant': 7, 'American restaurant': 8, 'Sandwich shop': 9, 'Asian restaurant': 11, 'Thai restaurant': 12, 'Vietnamese restaurant': 13, 'Family restaurant': 14, 'Italian restaurant': 15, 'Cajun restaurant': 16, 'Barbecue restaurant': 18, 'Latin American restaurant': 19, 'Taiwanese restaurant': 20, 'Breakfast restaurant': 21, 'Caterer Restaurant': 22, 'Seafood restaurant': 23, 'Restaurant supply store': 24, 'Mediterranean restaurant': 25, 'Health food restaurant': 27, 'Chicken wings restaurant': 28, 'Japanese restaurant': 29, 'Filipino restaurant': 30, 'Fusion restaurant': 31, 'Indian restaurant': 32, 'Asian restaurant Ramen': 33, 'Restaurant Catering food': 34}

@app.get("/")
def read_root():
    return {"message": "Bienvenido, a mi api sobre peliculas"}

#def seleccionar_opcion(opcion: str = Query(..., enum=list(opciones.keys()))):

@app.get("/modelo-de-recomendacion")
def get_recomendacion(ciudad:str=Query(...,description="Seleccionar",enum=list(ciudades.keys())),
                      categoria:str=Query(...,description="Seleccionar",enum=list(categorias.keys())), reseñas_positivas:int=0,reseñas_negativas:int=0):
    """Función que predice el potencial de éxito basado en las calificaciones que se le hacen a los restaurantes en google maps y yelp

    Args:
        ciudad (str,requerido): Ciudades de California
        categoria (str, requerido): Categorias de los diferentes restaurantes de California

    Returns:
        mensaje: Que describe el alto o bajo potencial de éxito de una categoria de comida en una ciudad en california 
    """
    #movies=pd.read_csv("datasets/movies_limpio.csv",sep=',',encoding='UTF-8')
    df_negocios=pd.read_csv('dataset/negocios_reviews_oscar.csv',encoding='utf-8',sep=',')
    df_negocios['success'] = df_negocios['avg_rating'].apply(lambda x: 1 if x >= 3.5 else 0)
    
    grouped = df_negocios.groupby(['cluster', 'cluster_categories']).size().reset_index(name='counts')
    # Convertir las columnas 'ciudad' y 'estado' en un diccionario
    subcategorias_dict = grouped.set_index('cluster_categories')['cluster'].to_dict()
    ciudades={'Agoura Hills':1,'Alhambra':2,'Altadena':3,'Arcadia':4,'Artesia':5,'Azusa':6,'Baldwin Park':7,'Bell Gardens':8,'Bellflower':9,'Beverly Hills':10,'Burbank':11,'Calabasas':12,'Canoga Park':13,'Canyon Country':14,'Carson':15,'Castaic':16,'Cerritos':17,'Chatsworth':18,'Claremont':19,'Compton':20,'Covina':21,'Culver City':22,'Diamond Bar':23,'Downey':24,'Duarte':25,'El Monte':26,'El Segundo':27,'Encino':28,'Gardena':29,'Glendale':30,'Glendora':31,'Granada Hills':32,'Hacienda Heights':33,'Harbor City':34,'Hawaiian Gardens':35,'Hawthorne':36,'Hermosa Beach':37,'Huntington Park':38,'Inglewood':39,'La Canada Flintridge':40,'La Crescenta':41,'La Mirada':42,'La Puente':43,'La Verne':44,'Lakewood':45,'Lancaster':46,'Lawndale':47,'Littlerock':48,'Lomita':49,'Long Beach':50,'Los Angeles':51,'Lynwood':52,'Malibu':53,'Manhattan Beach':54,'Marina Del Rey':55,'Maywood':56,'Mission Hills':57,'Monrovia':58,'Montebello':59,'Monterey Park':60,'Newhall':61,'North Hills':62,'North Hollywood':63,'Northridge':64,'Norwalk':65,'Pacific Palisades':66,'Pacoima':67,'Palmdale':68,'Palos Verdes Peninsula':69,'Panorama City':70,'Paramount':71,'Pasadena':72,'Pico Rivera':73,'Playa Del Rey':74,'Playa Vista':75,'Pomona':76,'Porter Ranch':77,'Rancho Palos Verdes':78,'Redondo Beach':79,'Reseda':80,'Rosemead':81,'Rowland Heights':82,'San Dimas':83,'San Fernando':84,'San Gabriel':85,'San Pedro':86,'Santa Clarita':87,'Santa Fe Springs':88,'Santa Monica':89,'Sherman Oaks':90,'Sierra Madre':91,'South El Monte':92,'South Gate':93,'South Pasadena':94,'Stevenson Ranch':95,'Studio City':96,'Sun Valley':97,'Sunland':98,'Sylmar':99,'Tarzana':100,'Temple City':101,'Torrance':102,'Tujunga':103,'Universal City':104,'Valencia':105,'Valley Village':106,'Van Nuys':107,'Venice':108,'Walnut':109,'West Covina':110,'West Hills':111,'West Hollywood':112,'Whittier':113,'Wilmington':114,'Winnetka':115,'Woodland Hills':116}
    df_negocios['ciudad_id'] = df_negocios['Ciudad'].map(ciudades)
    
    features=['ciudad_id', 'cluster', 'negative_sentiment','positive_sentiment']
    X = df_negocios[features]
    y = df_negocios['success']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Entrenar el modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predecir en el conjunto de prueba
    y_pred = model.predict(X_test)

    # Evaluar el modelo
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    print(report)
    
    new_data = {
    'ciudad_id':ciudades[ciudad], #Venice
    'cluster':subcategorias_dict[categoria], #Asian restaurant Ramen
    'negative_sentiment':reseñas_negativas,
    'positive_sentiment':reseñas_positivas
    }
    new_data_df = pd.DataFrame([new_data])
    prediction = model.predict(new_data_df)
      
    # Interpretar el resultado
    if prediction[0] == 1:
        mensaje="Predicción: Alto potencial de éxito."
        print("Predicción: Alto potencial de éxito.")
    else:
        mensaje="Predicción: Bajo potencial de éxito."
        print("Predicción: El negocio tiene bajo potencial de éxito.")
        
    return mensaje