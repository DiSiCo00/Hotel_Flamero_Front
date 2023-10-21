import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_chat import message

from translate import Translator

from PIL import Image

import pandas as pd

import json
import time


from _Func.html_func import html_sheader
from _Func.html_func import html_score_badges
from _Func.html_func import comment_section

from _Func.data_manage_and_models import update_comments_data

from _Func.data_manage_and_models import stentiment_analizis
from _Func.data_manage_and_models import predictions
from _Func.data_manage_and_models import get_chat_response
from _Func.data_manage_and_models import chatbot_env


with open("_Data/room_type.json", encoding='utf-8') as file:
    room_type_obj = json.load(file)
    file.close()

with open("_Data/raitings.json", encoding='utf-8') as file:
    raitings_obj = json.load(file)
    file.close()

with open("_Data/regimen.json", encoding='utf-8') as file:
    regimen = json.load(file)
    file.close()

def add_style(css_file):
    with open(css_file) as file:
        st.markdown(f"<style>{file.read()}</style>", unsafe_allow_html=True)



st.set_page_config(layout= "wide",
                    page_title = "FlameroHotel")




add_style("_CSS/main.css")



c_main = st.container()

c_body = c_main.container()


with st.sidebar:
    page_selected = option_menu(
    menu_title="Menu",
    options=["Flamero", "Opiniones", "ChatBot"],
    default_index=0,
    )


if page_selected == "Flamero":
    
    img =  Image.open("Images/1.png")
    c_body.image(img, use_column_width = "always" )
    c_body.divider()


    with st.form("booking_info"):
        c_body.markdown('<h3>Compruebe disponibilidad:</h3>', unsafe_allow_html=True)


        entry_date = pd.to_datetime(c_body.date_input(label = "Seleccione la fecha deentrada (Las fechas estan acotadas para los dias disponibles):",
                value = pd.to_datetime('1/6/2024', dayfirst=True),
                min_value=pd.to_datetime('1/6/2024', dayfirst=True),
                max_value=pd.to_datetime('30/9/2024',dayfirst=True),
                on_change=None, format="DD/MM/YYYY"), dayfirst=True)
        
        fecha_venta = pd.to_datetime(c_body.date_input(label = "QUe dia es hoy (Funcionalidad disponible solo para la presentación para cambiar el dia de en que se reserva):",

                max_value=pd.to_datetime('30/9/2024',dayfirst=True),
                on_change=None, format="DD/MM/YYYY"), dayfirst=True)
        
        col_1, col_2, col_3 = c_body.columns(3)

        noches = int(col_1.number_input('Seleccione la cantidad de noches:',min_value=1))

        adultos = int(col_2.number_input('Cantidad de adultos:',min_value=1))

        child = int(col_3.number_input('Cantidad de menores de edad:',min_value=0))

        if col_3.checkbox("Necesita cunas en la habitacion?", disabled=bool(not child)):
            cunas = int(col_3.number_input('Cuantas?:',min_value=1))
        else:
            cunas = 0

        if child>0:
            room_type_id_pointer = col_1.radio('Seleccione un tipo de habitacion que desea:',
                            ['INDIVIDUAL','ESTUDIO COTO','ESTUDIO MAR','DOBLE SUPERIOR COTO', 'DOBLE SUPERIOR MAR',
                            'APARTAMENTO PREMIUM','DELUXE VISTA COTO', 'DELUXE VISTA COTO', 'SUITE'])
        else:
            room_type_id_pointer = col_1.radio('Seleccione un tipo de habitacion que desea:',
                            ['DOBLE SUPERIOR COTO', 'DOBLE SUPERIOR MAR', 'DELUXE VISTA COTO', 'ESTUDIO COTO', 'ESTUDIO MAR', 'SUITE'])
        
        room_type = room_type_obj[room_type_id_pointer]["ID"]

        regimen_pointer = col_2.radio('Seleccione un tipo de pensiónm que desea:',
                list(regimen.keys()))
        
        pension = regimen[regimen_pointer]

        
        submitted = st.form_submit_button("Submit")

        c_body.divider()

        
        if submitted:
            with st.spinner("Espera..."):
                msg = st.toast('"Recopilando Información"...')
                time.sleep(2)
                msg.toast("Chequeando disponibilidad...")
                time.sleep(2)
                msg.toast("Chequeando disponibilidad...")
                time.sleep(2)
                msg.toast("Estas de suerte!! Ahora buscaremos lahabitacione adecuada...")
                time.sleep(2)
                cancel_prob, c_date, cuota, obj, score = predictions(room_type, noches, adultos, child, cunas, entry_date, fecha_venta, pension )

                st.success("Tenemos la habitación adecuada para ti", icon="✅")
            c_room_info = st.expander("Ver Habitación")
            with c_room_info:
                desc_col, info_col = c_room_info.columns(2)
                # Cloumna de Datos
                info_col.markdown(f"<h2>{room_type_id_pointer}:</h2>", unsafe_allow_html=True)
                info_col.divider()
                info_col.markdown(f"<h3>Precio de la estancia:</h3> €{round(obj['P_Alojamiento'], 2)}", unsafe_allow_html=True)
                info_col.markdown(f"<h3>Probabilidad de Cancelación:</h3> {round(cancel_prob*100, 2)}%", unsafe_allow_html=True)
                info_col.markdown(f"<h3>Pago adelantado:</h3> €{round(cuota*obj['P_Alojamiento'] , 2)}", unsafe_allow_html=True)
                
                info_col.markdown(f"<h3>Cancelación Gratuita Hasta:</h3> {c_date}", unsafe_allow_html=True)
                info_col.markdown(f"<h3>Cancel_Score:</h3> {round(score, 2)}", unsafe_allow_html=True)

                # Columna de Desceipcion de la Habitación
                room_img =  Image.open(f"{room_type_obj[room_type_id_pointer]['img_path']}")
                desc_col.image(room_img, use_column_width="always")
                desc_col.markdown(f"<h6>{room_type_obj[room_type_id_pointer]['Desc']}</h6>", unsafe_allow_html=True)

elif page_selected == "Opiniones":
    with c_body:
        img2 =  Image.open("Images/2.png")
        st.image(img2, use_column_width = "always" )
        c_body.divider()
        col_raitings, comments_section = c_body.columns((1,2), gap="small" )
        col_raitings.markdown("<h2>Raitings:</h2>", unsafe_allow_html=True)
        for key, value in list(raitings_obj.items()):
            c_raitings = col_raitings.container()
            list , badge = c_raitings.columns(2)
            list.markdown(f"<h4>{key}</h4>", unsafe_allow_html=True)
            badge.markdown(html_score_badges(value["Score"]), unsafe_allow_html=True)
            c_raitings.divider()


        with comments_section:
            comments_section.markdown(html_sheader("Comentarios"), unsafe_allow_html=True)
            comments_section.markdown(f"<div class='comment_section'>{comment_section()}</div>", unsafe_allow_html=True)
            comments_section.divider()
            with comments_section.form(key="Comment_section_form"):
                
                st.subheader("**Quieres compartinos tu experiencia?**")
                user = st.text_input("Escribe tu nombre o un alias con el que desees dejar tu comentario:",
                                    value = "Anónimo")
                text_comment = st.text_area(label="Escribe tu comentario aqui:")
                raiting = int(st.number_input("Califica tu experienia con nosotros entre 1 - 10",
                                        value=10,
                                        placeholder="Type a number...",
                                        max_value=10,
                                        min_value=0))
                submit_com = st.form_submit_button("Enviar")
    if submit_com and text_comment != "":

        update_comments_data({"Score": raiting,
                            "Comentario_Positivo":text_comment,
                            "Usuario":user})
        if stentiment_analizis(text_comment) >= 0.05:
            c_main.balloons()
            c_main.success("Gracias por su comentario")
            time.sleep(5)
        elif stentiment_analizis(text_comment) <= 0.05:
            c_main.info("Agradecemos que hayas compartido tus preocupaciones con nosotros. Lamentamos mucho que hayas tenido esta experiencia", icon="ℹ️")
            time.sleep(5)
        else:
            c_main.success("Gracias por su comentario")
            time.sleep(5)
        st.rerun()

elif page_selected == "ChatBot":
                # Initialize chat history
        chatbot_env()

        with c_body:
            if "messages" not in st.session_state:
                st.session_state.messages = []

            img3 =  Image.open("Images/3.png")
            c_body.image(img3, use_column_width = "always" )
            c_body.divider()

            title_col, button_col = c_body.columns([3,1])
            title_col.title("Asistente Virtual  Flamero")

            if button_col.button("Borrar Conversacion", type="primary"):
                st.session_state.messages = []

            # Display chat messages from history on app rerun
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            prompt = st.chat_input("What is up?")
            # Accept user input
            if prompt != None:
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                # Display user message in chat message container
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    respuesta = Translator(to_lang="es").translate(get_chat_response(prompt))

                    for chunk in respuesta.split():
                        full_response += chunk + " "
                        time.sleep(0.05)
                        # Add a blinking cursor to simulate typing
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                    # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})