import streamlit as st
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
# from transformers import pipeline

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from translate import Translator

import pandas as pd
import requests
import openai
import random
import json

import joblib

use_cols = ['Noches','Tip.Hab.Fra.','Régimen factura', 'AD', 'NI','CU','Horario venta',
            'Precio alojamiento','Precio desayuno', 'Precio almuerzo', 'Precio cena',
            'Cantidad Habitaciones','Mes Entrada','Mes Venta','Antelacion']


def load_cancel_data():
    #Leemos el csv para recuperar el dataframe
    return pd.read_csv('_Data/cancelaciones.csv')

def load_booking_data():
    #Leemos el csv reservas_total_preprocesado para recuperar el dataframe
    reservas_total=pd.read_csv('_Data/reservas_total_preprocesado.csv')

    # Convertimos las columnas en formato de fecha
    reservas_total['Fecha entrada'] = pd.to_datetime(reservas_total['Fecha entrada'], dayfirst=True, format = "mixed")
    reservas_total['Fecha venta'] = pd.to_datetime(reservas_total['Fecha venta'], dayfirst=True, format = "mixed")
    reservas_total['Fecha Anulacion'] = pd.to_datetime(reservas_total['Fecha Anulacion'], dayfirst=True, format = "mixed")

    return reservas_total



# Recopilar datos de la nueva reserva:
def new_Booking(df, room_type, noches, adultos, child, cunas, fecha_entrada, fecha_venta, regimen):
    
    def get_horario():
        hora = int(datetime.now().strftime('%H'))
        if (0 <= hora < 6):
            return'Madrugada'
        elif (6 <= hora < 12):
            return 'Mañana'
        elif (12 <= hora < 18):
            return'Tarde'
        else:
            return 'Noche'
    
    #Función para definir la cantidad mínima de habitaciones a reservar en base a huespedes y tipo de habitación
    def habitaciones(adultos, niños, tipo_habitacion):
      cont = 1

      #Si es una SUITE, la capacidad máxima es de 2 adultos y 2 niños o 3 adultos
      if tipo_habitacion == 'SUITE':
        #Si hay más de 2 niños por adulto devolvemos error (0)
        if adultos * 2 < niños:
          return 0

        #Asignamos los niños de 2 en 2 y dos adultos por habitación
        cont = niños // 2 + niños % 2
        adultos -= cont * 2

        #Asignamos habitaciones de 3 adultos
        if  adultos > 0:
          cont += adultos // 3
          adultos = adultos % 3
   
          #Última habitación si sobran adultos
          if adultos > 0:
            cont += 1

      #Si es una habitación DELUXE VISTA COTO, la capacidad máxima es de 2 adultos y 1 niño
      if tipo_habitacion == 'DVC':
        #Si hay más niños que adultos devolvemos error (0)
        if adultos < niños:
          return 0

        #Asignamos una habitación por niño y 2 adultos por habitación
        cont = niños
        adultos -= cont * 2

        #Asignamos habitaciones de 2 adultos
        if  adultos > 0:
          cont += adultos // 2 + adultos % 2

      #Si es una habitación DELUXE VISTA MAR, la capacidad máxima es de 2 adultos. No se permiten niños
      if tipo_habitacion == 'DVM':
        #Si hay niños por adulto devolvemos error (0)
        if niños > 0:
          return 0

        #Asignamos habitaciones de 2 adultos
        cont = adultos // 2 + adultos % 2

      #Si es una habitación INDIVIDUAL, la capacidad máxima es de 1 adulto. No se permiten niños
      if tipo_habitacion == 'IND':
        #Si hay niños por adulto devolvemos error (0)
        if niños > 0:
          return 0

        #Asignamos las habitaciones individuales
        cont = adultos

      #Si es un APARTAMENTO PREMIUM, la capacidad máxima es de 4 adultos y 3 niños
      if tipo_habitacion == 'A':
        #Si hay más de 3 niños por adulto devolvemos error (0)
        if adultos * 3 < niños:
          return 0

        #Asignamos los niños de 3 en 3 y cuatro adultos por habitación
        cont = niños // 3
        niños = niños % 3
        adultos -= cont * 4

        #Si sobran niños asignamos otra habitación con capacidad para 4 adultos más
        if niños > 0:
          cont += 1
          adultos -= 4

        #Si sobran adultos asignamos habitaciones de 4 adultos
        if adultos > 0:
          cont += adultos // 4
          adultos = adultos % 4

          #Última habitación si sobran adultos
          if adultos > 0:
            cont += 1

      #Si es un ESTUDIO estándar o una habitación DOBLE SUPERIOR, independientemente de si es vista COTO o MAR,
      #la capacidad máxima es de 3 adultos y 1 niño o 2 adultos y 2 niños
      if tipo_habitacion in ('EC', 'EM', 'DSC', 'DSM'):
        #Si hay más de 2 niños por adulto devolvemos error (0)
        if adultos * 2 < niños:
          return 0

        #Asignamos los niños de 2 en 2 y dos adultos por habitación
        cont = niños // 2
        adultos -= cont * 2

        #Asignamos habitaciones de 3 en 3
        if adultos > 0:
          cont += adultos // 3
          adultos = adultos % 3

          #Última habitación si sobran adultos
          if adultos > 0:
            cont += 1
        #Si no sobran adultos pero sí un niño, asignaremos una habitación extra
        elif niños % 2 == 1:
          cont += 1

      return cont


    precio_alojamiento = df['Precio alojamiento'].loc[df['Tip.Hab.Fra.'] == room_type].mean()/df['Noches'].loc[df['Tip.Hab.Fra.'] == room_type].mean()*noches
    precio_desayuno=df['Precio desayuno'].loc[df['Régimen factura'] == regimen].mean()
    precio_almuerzo=df['Precio almuerzo'].loc[df['Régimen factura'] == regimen].mean()
    precio_cena= df['Precio cena'].loc[df['Régimen factura'] == regimen].mean()


    obj = {
    'Noches': noches,
    'Tip.Hab.Fra.' : room_type,
    'Régimen factura': regimen,
    'AD': adultos,
    'NI':child,
    'CU':cunas,
    'Horario venta': get_horario(),
    'Precio alojamiento': precio_alojamiento,
    'Precio desayuno': precio_desayuno,
    'Precio almuerzo': precio_almuerzo,
    'Precio cena': precio_cena,
    'Cantidad Habitaciones': habitaciones(adultos,child,room_type),
    'Mes Entrada' : fecha_entrada.strftime('%B'),
    'Mes Venta': fecha_venta.strftime('%B'),
    'Antelacion': (fecha_entrada-fecha_venta).days
    }


    return obj


def new_data_to_model(df, _obj, _use_cols = use_cols):
    #Tomamos nuestra base de entrenamiento para realizar el proceso de normalizaci�n y One Hot Encoding
    _sample = df[_use_cols]

    # Agregar la nueva fila al DataFrame
    _X =  pd.concat([_sample, pd.DataFrame(_obj,index=[0])], ignore_index=True)

    #One Hot Encoding de las variables categ�ricas
    _X = pd.get_dummies(_X, columns=["Tip.Hab.Fra.", "Régimen factura","Horario venta", "Mes Entrada", "Mes Venta"], drop_first=True)

    #Aplicamos el escalador robusto
    robust_scaler = RobustScaler()
    _X[["Precio alojamiento", "Antelacion"]] = robust_scaler.fit_transform(_X[["Precio alojamiento", "Antelacion"]])

    # Aplicamos la normalizaci�n Min Max
    scaler = MinMaxScaler()
    X = scaler.fit_transform(_X)
    return X


#Funci�n para predercir la probabilidad de cancelaci�n de una reserva con un modelo determinado
def predict_prob(X):
    model = joblib.load("random_forest.pkl")
    return model.predict_proba(X[-1].reshape(1, -1))[0,1]

#Fecha maxima para cancelar
def predict_date_score(X, _obj, fecha_venta):
    model = joblib.load("reg_random_forest.pkl")

    _score = model.predict(X[-1].reshape(1, -1))[0]

    return _score

#Función cuota no reembolsable
def func_no_reembolso(fecha_entrada, cancel_prob, score, _cuota_media=0.10, _cuota_maxima=0.25, _umbral_inferior=0.25, _umbral_superior=0.4 ):
        #Condiciones de control
        if 0 <= _cuota_maxima <= 1:
          if 0 <= _cuota_media <= 1:
            if 0 <= _umbral_inferior <= 1:
              if 0 <= _umbral_superior <= 1:
                if _umbral_superior >_umbral_inferior:

                  #Según los distintos umbrales y dependiendo del score, las cancelaciones tendrán unas cuotas y fechas de cancelación 
                  if cancel_prob < _umbral_inferior:
                    st.write(f"Riesgo bajo de cancelación.")
                    if score<0.5:
                      st.write(f"¡¡Aviso de posible cancelación tardía!!")
                      st.write(f"Fecha límite de cancelación: 7 días antes del {fecha_entrada}")
                    else:
                      st.write(f"Fecha límite de cancelación: 24 horas antes del {fecha_entrada}")
                    return 0;
                  elif cancel_prob > _umbral_superior:
                    st.write(f"Riesgo alto de cancelación.")
                    if score<0.5:
                      st.write(f"¡¡Aviso de posible cancelación tardía!!")
                      st.write(f"Fecha límite de cancelación: 30 días antes del {fecha_entrada}")
                    else:
                      st.write(f"Fecha límite de cancelación: 7 días antes del {fecha_entrada}")
                    return _cuota_maxima
                  else:
                    st.write(f"Riesgo moderado de cancelación.")
                    if score<0.5:
                      st.write(f"¡¡Aviso de posible cancelación tardía!!")
                      st.write(f"Fecha límite de cancelación: 14 días antes del {fecha_entrada}")
                    else:
                      st.write(f"Fecha límite de cancelación: 48 horas antes del {fecha_entrada}")
                    return _cuota_media
                else:
                  raise ValueError("El valor de ´umbral_superior´  tiene que ser mayor que ´umbral_inferior´.")
              else:
                raise ValueError("El valor ´umbral_superior´ debe estar entre 0 y 1.")
            else:
              raise ValueError("El valor ´umbral_inferior´ debe estar entre 0 y 1.")
          else:
            raise ValueError("El valor ´cuota_media´ debe estar entre 0 y 1.")
        else:
          raise ValueError("El valor ´cuota_maxima´ debe estar entre 0 y 1.")


def predictions(room_type, noches, adultos, child, cunas, fecha_entrada, fecha_venta, regimen):
    reservas = load_booking_data()
    cancel_data = load_cancel_data()

    obj = new_Booking(reservas, room_type, noches, adultos, child, cunas, fecha_entrada, fecha_venta, regimen)

    X_booking = new_data_to_model(reservas, obj)
    X_cancel = new_data_to_model(cancel_data, obj)

    cancel_prob = predict_prob(X_booking)
    score = predict_date_score(X_cancel, obj, fecha_venta)

    cuota =  func_no_reembolso(fecha_entrada, cancel_prob, score)

    return obj, cancel_prob, score, cuota

def sentiment_analysis(_text):

    nltk.download('vader_lexicon')
    nltk.download('punkt')

    sia = SentimentIntensityAnalyzer()

    translator = Translator(from_lang="es", to_lang="en")
    text = translator.translate(_text)

    palabras_positivas = ["good","happy","big","recommend","nice","great", "excellent", "enjoy", "enjoyed", "perfect", "elegant", "gently", "gentle", "delicious", "unforgettable", "spectacular", "highly","detail"]
    palabras_negativas = ["old","poor","small","uncomfortable","bad","slow", "shit", "suffered", "horrible", "disaster","no","noisy","expensive"]


    def calcular_puntuacion_sentimiento(frase_ingles):
        tokens = nltk.word_tokenize(frase_ingles)
        puntuacion_sentimiento = 0
        for token in tokens:
            if token in palabras_positivas:
                puntuacion_sentimiento += 1
            elif token in palabras_negativas:
                puntuacion_sentimiento -= 1

        return puntuacion_sentimiento
    
    puntuacion = calcular_puntuacion_sentimiento(text)
    sentimiento = sia.polarity_scores(text)

    return sentimiento['compound']


def update_comments_data(_obj):
    df_comments = pd.concat([pd.DataFrame(_obj,index=[0]), pd.read_csv("_Data/comments.csv")])
    df_comments.to_csv("_Data/comments.csv", index=False)


with open("_Data/entorno_chatbot.json") as file:
    env = json.load(file)
    file.close()

with open("_Data/ChatSetup.json") as file:
    setup = json.load(file)
    file.close()





def chatbot_env (env=env):

    openai.api_type = env["api_type"]

    # Azure OpenAI on your own data is only supported by the 2023-08-01-preview API version
    openai.api_version = env["api_version"]

    # Azure OpenAI setup
    openai.api_base = env["api_base"] # Add your endpoint here
    openai.api_key = env["api_key"] # Add your OpenAI API key here
    deployment_id = env["deployment_id"] # Add your deployment ID here

    def setup_byod(deployment_id: str) -> None:
        """Sets up the OpenAI Python SDK to use your own data for the chat endpoint.

        :param deployment_id: The deployment ID for the model to use with your own data.

        To remove this configuration, simply set openai.requestssession to None.
        """

        class BringYourOwnDataAdapter(requests.adapters.HTTPAdapter):

            def send(self, request, **kwargs):
                request.url = f"{openai.api_base}/openai/deployments/{deployment_id}/extensions/chat/completions?api-version={openai.api_version}"
                return super().send(request, **kwargs)

        session = requests.Session()

        # Mount a custom adapter which will use the extensions endpoint for any call using the given `deployment_id`
        session.mount(
            prefix=f"{openai.api_base}/openai/deployments/{deployment_id}",
            adapter=BringYourOwnDataAdapter()
        )

        openai.requestssession = session

    setup_byod(deployment_id)

def get_chat_response(message, env=env, setup=setup):

    setup["role"] = "user"
    setup["content"] = message

        # Azure Cognitive Search setup
    search_endpoint = env["search_endpoint"]; # Add your Azure Cognitive Search endpoint here
    search_key = env["Search_Key"]; # Add your Azure Cognitive Search admin key here
    search_index_name = env["search_index_name"]; # Add your Azure Cognitive Search index name here
    deployment_id = env["deployment_id"] # Add your deployment ID here

    completion = openai.ChatCompletion.create(
        messages=[setup],
        deployment_id=deployment_id,
        dataSources=[  # camelCase is intentional, as this is the format the API expects
            {
                "type": "AzureCognitiveSearch",
                "parameters": {
                    "endpoint": search_endpoint,
                    "key": search_key,
                    "indexName": search_index_name,
                }
            }
        ]
    )
    doc_list = ["[doc1]", "[doc2]", "[doc3]", "[doc4]", "[doc5]", "[doc6]"]
    respuesta = completion["choices"][0]["message"]["content"]

    
    for doc in doc_list:
        if doc in respuesta:
            respuesta = respuesta.replace(doc, "")
    return respuesta
