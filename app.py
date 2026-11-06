import json
import os
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from physics_tools import calculate_physics

load_dotenv()
st.set_page_config(page_title='AI Physics Agent', page_icon='⚛️', layout='wide', initial_sidebar_state='expanded')

st.markdown('''<style>
.block-container {max-width: 1050px; padding-top: 2rem;}
.hero {padding: 1.5rem 1.7rem; border: 1px solid rgba(140,140,180,.22); border-radius: 20px;
background: linear-gradient(115deg, rgba(112,68,225,.20), rgba(30,120,190,.08)); margin-bottom: 1.1rem;}
.hero h1 {margin:0; font-size:2.2rem;}
.hero p {opacity:.8; margin:.5rem 0 0;}
.mini-card {padding:1rem; border:1px solid rgba(140,140,180,.25); border-radius:14px; min-height:110px;}
.mini-card strong {display:block; font-size:1.05rem; margin-bottom:.3rem;}
.mini-card span {opacity:.75; font-size:.91rem;}
div[data-testid="stChatMessage"] {border:1px solid rgba(140,140,180,.15); border-radius:14px;}
</style>''', unsafe_allow_html=True)

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'pending_prompt' not in st.session_state:
    st.session_state.pending_prompt = None

EXAMPLES = [
    ('⚡ Сила', 'Розрахуй силу, якщо маса тіла 10 кг, а прискорення 5 м/с².'),
    ('🏃 Кінетична енергія', 'Розрахуй кінетичну енергію тіла масою 5 кг зі швидкістю 10 м/с.'),
    ('🚗 Швидкість', 'Яка швидкість тіла, що подолало 150 метрів за 10 секунд?'),
    ('📘 Пояснення', 'Поясни другий закон Ньютона простими словами.'),
]

with st.sidebar:
    st.title('⚛️ Physics Lab')
    st.caption('Навчальний AI-помічник')
    st.divider()
    st.subheader('🧪 Швидкі приклади')
    for title, question in EXAMPLES:
        if st.button(title, key=title, use_container_width=True):
            st.session_state.pending_prompt = question
            st.rerun()
    st.divider()
    st.subheader('📐 Доступні формули')
    st.latex(r'F = ma')
    st.latex(r'E_k = \\frac{mv^2}{2}'.replace('\\\\', '\\'))
    st.latex(r'v = \\frac{s}{t}'.replace('\\\\', '\\'))
    st.caption('Також: імпульс, робота, густина')
    st.divider()
    if st.button('🗑️ Очистити історію', use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_prompt = None
        st.rerun()
    transcript = '\n\n'.join(('Користувач' if m['role']=='user' else 'Агент') + ':\n' + m['content'] for m in st.session_state.messages)
    st.download_button('⬇️ Завантажити історію', data=transcript or 'Історія порожня',
                       file_name='physics_chat.txt', mime='text/plain', use_container_width=True)
    st.caption('API-ключ зберігається лише локально у .env')

st.markdown('<div class="hero"><h1>⚛️ AI Physics Agent</h1><p>Розв’язуйте фізичні задачі разом зі штучним інтелектом. Точні розрахунки — через власний Python-інструмент.</p></div>', unsafe_allow_html=True)

cols = st.columns(3)
for col, title, subtitle in zip(cols, ['🤖 Gemini LLM', '🧮 Власний tool', '💬 Інтерактивний чат'],
                                 ['Розуміє запитання та пояснює', 'Виконує точні обчислення', 'Зберігає історію сесії']):
    with col:
        st.markdown(f'<div class="mini-card"><strong>{title}</strong><span>{subtitle}</span></div>', unsafe_allow_html=True)
st.write('')

api_key = os.getenv('GEMINI_API_KEY', '').strip()
if not api_key:
    st.error('Додайте GEMINI_API_KEY у файл .env та перезапустіть застосунок.')
    st.stop()
model_name = os.getenv('GEMINI_MODEL', 'gemini-3-flash-preview').strip()
client = genai.Client(api_key=api_key)

function_declaration = types.FunctionDeclaration(
    name='calculate_physics',
    description='Calculate physics values using SI units. Supported: force (mass, acceleration), kinetic_energy (mass, velocity), momentum (mass, velocity), speed (distance, time), work (force, distance), density (mass, volume). Work assumes parallel force and displacement.',
    parameters={
        'type': 'OBJECT',
        'properties': {
            'quantity': {'type': 'STRING', 'enum': ['force','kinetic_energy','momentum','speed','work','density']},
            **{key: {'type':'NUMBER'} for key in ['mass','acceleration','velocity','distance','time','force','volume']},
        },
        'required': ['quantity'],
    },
)
config = types.GenerateContentConfig(
    system_instruction=(
        'Ти навчальний помічник із фізики. Відповідай українською мовою. '
        'Для числових задач, які підтримує calculate_physics, обов’язково викликай цей інструмент. '
        'Пояснюй кроки: дано, формула, підстановка, відповідь з одиницями SI. '
        'Не вигадуй результати інструмента. Якщо бракує даних — уточнюй. '
        'Перед викликом переводь значення у SI. Для роботи A=F*s зазначай припущення про паралельність сили.'
    ),
    tools=[types.Tool(function_declarations=[function_declaration])],
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    temperature=0.2,
)

st.subheader('💬 Чат із агентом')
if not st.session_state.messages:
    st.info('Оберіть приклад ліворуч або напишіть власне запитання нижче.')

for message in st.session_state.messages:
    with st.chat_message(message['role'], avatar='🧑' if message['role']=='user' else '🤖'):
        st.markdown(message['content'])
        if message.get('tool_log'):
            with st.expander('🔧 Виклики власного інструмента'):
                for item in message['tool_log']:
                    result = item.get('result', {})
                    if 'error' not in result:
                        st.success(f"Результат: {result.get('result')} {result.get('unit', '')}")
                        st.caption(f"Формула: {result.get('formula', '')}")
                    else:
                        st.warning(result['error'])
                st.json(message['tool_log'])

prompt = st.chat_input('Напишіть запитання з фізики...')
if st.session_state.pending_prompt is not None:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

if prompt:
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message('user', avatar='🧑'):
        st.markdown(prompt)
    with st.chat_message('assistant', avatar='🤖'):
        with st.spinner('Агент аналізує задачу...'):
            try:
                history = [types.Content(role='user' if m['role']=='user' else 'model',
                                         parts=[types.Part(text=m['content'])])
                           for m in st.session_state.messages]
                response = client.models.generate_content(model=model_name, contents=history, config=config)
                tool_log = []
                for _ in range(5):
                    calls = response.function_calls or []
                    if not calls:
                        break
                    if not response.candidates or not response.candidates[0].content:
                        raise RuntimeError('Модель не повернула виклик інструмента.')
                    history.append(response.candidates[0].content)
                    replies = []
                    for call in calls:
                        args = dict(call.args or {})
                        if call.name == 'calculate_physics':
                            params = dict(args)
                            quantity = params.pop('quantity', '')
                            result = calculate_physics(quantity, **params)
                        else:
                            result = {'error': 'Невідомий інструмент'}
                        tool_log.append({'tool':call.name, 'arguments':args, 'result':result})
                        replies.append(types.Part.from_function_response(name=call.name, response={'output':result}))
                    history.append(types.Content(role='user', parts=replies))
                    response = client.models.generate_content(model=model_name, contents=history, config=config)
                else:
                    raise RuntimeError('Перевищено ліміт викликів інструмента.')
                answer = response.text or 'Модель не повернула текстову відповідь.'
                st.markdown(answer)
                if tool_log:
                    with st.expander('🔧 Виклики власного інструмента', expanded=True):
                        for item in tool_log:
                            result = item['result']
                            if 'error' not in result:
                                st.success(f"Результат: {result.get('result')} {result.get('unit','')}")
                                st.caption(f"Формула: {result.get('formula','')}")
                            else:
                                st.warning(result['error'])
                        st.json(tool_log)
                st.session_state.messages.append({'role':'assistant','content':answer,'tool_log':tool_log})
            except Exception as exc:
                st.error(f'Помилка запиту: {exc}')
                st.info('Перевірте назву моделі, інтернет-з’єднання, доступність API та квоту.')
