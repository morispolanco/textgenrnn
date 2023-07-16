import streamlit as st
from tensorflow.keras.utils import multi_gpu_model
from textgenrnn import textgenrnn

def generar_texto_longitud(textgen, longitud):
    generated_text = textgen.generate(return_as_list=True, max_gen_length=longitud)[0]
    return generated_text

def main():
    st.title('Generador de Texto con textgenrnn')
    
    st.sidebar.title('Configuración')
    longitud_texto = st.sidebar.number_input('Longitud del Texto (en caracteres)', min_value=100, max_value=10000, step=1000, value=5000)
    
    if st.sidebar.button('Generar Texto'):
        st.info('Generando texto, por favor espera...')
        textgen = textgenrnn.TextgenRnn()
        texto_generado = generar_texto_longitud(textgen, longitud_texto)
        
        st.success('Texto generado con éxito:')
        st.write(texto_generado)

if __name__ == '__main__':
    main()
