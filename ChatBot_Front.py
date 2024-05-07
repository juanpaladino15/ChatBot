import streamlit as st
from ChatBot_Back import predict_class, get_response, intents

st.title("ChatBot POC")

# Inicializar variables de estado
if "messages" not in st.session_state:
    st.session_state.messages = []
if "first_message" not in st.session_state:
    st.session_state.first_message = True

# Iterar sobre los mensajes almacenados en session_state y mostrarlos
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Si es el primer mensaje, mostrar un saludo
if st.session_state.first_message:
    with st.chat_message("assistant"):
        st.markdown("Hola, ¿cómo te puedo ayudar?")
    st.session_state.messages.append({"role": "assistant", "content": "Hola, escribe tu mensaje"})
    st.session_state.first_message = False

# Obtener el mensaje del usuario y procesarlo
if prompt := st.chat_input("¿Cómo puedo ayudarte?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Implementar el algoritmo de IA
    insts = predict_class(prompt)
    res = get_response(insts, intents)

    with st.chat_message("assistant"):
        st.markdown(res)
    st.session_state.messages.append({"role": "assistant", "content": res})
