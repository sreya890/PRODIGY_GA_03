import pandas as pd
import streamlit as st
from src.markovmodel import MarkovModel

st.set_page_config(page_title='Markov Text Studio', page_icon='🧠', layout='wide')
st.title('🧠 Markov Text Studio')
st.caption('Learn, generate, and explore probability-based text generation.')

@st.cache_data
def load_default():
    with open('data.txt', encoding='utf-8') as f:
        return f.read()

if 'training_text' not in st.session_state:
    st.session_state.training_text = load_default()
if 'generated' not in st.session_state:
    st.session_state.generated = ''

with st.sidebar:
    st.header('Model settings')
    order = st.select_slider('Markov order', options=[1, 2, 3], value=2)
    temperature = st.slider('Sampling temperature', 0.3, 2.0, 1.0, 0.1)
    length = st.slider('Target word count (approx.)', 10, 200, 50, 5)
    st.caption('Higher order = more context. Higher temperature = more varied choices.')

model = MarkovModel(order=order, temperature=temperature)
model.train(st.session_state.training_text)

tab1, tab2, tab3 = st.tabs(['✍️ Generate', '📊 Probabilities', '📚 Training data'])
with tab1:
    st.subheader('Generate text')
    start_text = st.text_input(f'Optional starting sequence ({order} words)', placeholder='Leave blank for a natural sentence start')
    if st.button('Generate', type='primary'):
        st.session_state.generated = model.generate_text(start_text, length)
    if st.session_state.generated:
        st.text_area('Generated output', st.session_state.generated, height=200)
        st.download_button('Download output', st.session_state.generated, file_name='generated_text.txt')
    with st.expander('Example starting sequences'):
        st.write(', '.join(model.available_starts(15)) or 'Try leaving the field blank.')
with tab2:
    st.subheader('Next-word probability explorer')
    state = st.text_input(f'Enter exactly {order} words', key='prob_state')
    if state:
        probs = model.get_probabilities(state)
        if probs:
            df = pd.DataFrame([{'Next token': word, 'Probability': prob, 'Count': model.model[tuple(model.preprocess(state))][word]} for word, prob in probs.items()]).sort_values('Probability', ascending=False)
            st.dataframe(df, hide_index=True, use_container_width=True)
            st.bar_chart(df.set_index('Next token')['Probability'])
        else:
            st.info('No matching state. Try a sequence from the training text.')
with tab3:
    st.subheader('Training data')
    upload = st.file_uploader('Upload a UTF-8 .txt file', type='txt')
    source = upload.getvalue().decode('utf-8-sig') if upload else st.session_state.training_text
    edited = st.text_area('Review or edit text', source, height=260)
    if st.button('Train on this text'):
        if len(edited.split()) < 10:
            st.error('Provide at least 10 words of training text.')
        else:
            st.session_state.training_text = edited
            st.session_state.generated = ''
            st.rerun()
    stats = model.get_statistics()
    a, b, c = st.columns(3)
    a.metric('Tokens', stats['total_tokens'])
    b.metric('Unique tokens', stats['unique_tokens'])
    c.metric('States', stats['states'])
