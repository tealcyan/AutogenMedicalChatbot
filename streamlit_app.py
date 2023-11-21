import streamlit as st
import openai
import asyncio
from googleapiclient.discovery import build
import requests
from bs4 import BeautifulSoup
import sys
import autogen
from autogen import UserProxyAgent, ConversableAgent, oai, config_list_from_json, AssistantAgent, GroupChatManager

st.write("""# Healthcare Chatbot 🩺""")
# Initialize session state for chat history
# Create a Streamlit sidebar to input the API key
st.sidebar.write("## API Key")
api_key = st.sidebar.text_input("Enter your OpenAI API key:")
if "messages" not in st.session_state:
    st.session_state.messages = []
if "junior_doctor_mode" not in st.session_state:
    st.session_state.junior_doctor_mode = False

class TrackableUserProxyAgent(AssistantAgent):
    def _process_received_message(self, message, sender, silent):
        with st.chat_message("assistant"):
            st.markdown(message['content'])
        st.session_state.messages.append({'role': 'assistant', 'content': message['content']})
        return super()._process_received_message(message, sender, silent)
    # def get_human_input(self,prompt):
    #     user_input = st.session_state.get('user_input', '
    
# Load your API key from Streamlit secrets
if api_key:
    openai.api_key = api_key
else:
    st.warning("Please enter your OpenAI API key in the sidebar to use the app.")
# Define your functions here: assess_symp, symptoms, home_remedies, give_remedy, jun_doc_mode, etc.
def assess_symp(symptom):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.2,
        messages=[
            {"role": "system",
             "content": "You are a medical advisor and you have to asses wehther the user needs clinical expertise or not. If the symptoms are not severe and can be treated at home, say They don't need medical expertise. Reply in one word answer when content related to health! Yes or No if the user input says I'm bored or something unrelated to health, say something along the lines of: I'm sorry I can't help you with that, if you have any health related concerns please let me know."},
            {"role": "user", "content": symptom}
        ]
    )
    return completion.choices[0].message['content'], symptom


def symptoms(symp):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.2,
        messages=[
            {"role": "system",
             "content": "From the given input extract only the relevant medical symptoms and list them out separated with commas with no additional words "},
            {"role": "user", "content": symp}
        ]
    )
    return completion.choices[0].message['content']


def home_remedies(tokens):
    api_key = "AIzaSyDMcPs5y2Q58i8vp4SpjWmHxp35WvRrJfw"
    cse_id = "c1b34026eca3d42d6"
    texts = {}

    def google_search(search_term, api_key, cse_id, **kwargs):
        service = build("customsearch", "v1", developerKey=api_key)
        res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
        return res['items']

    def extract_text_from_url(url):
        try:
            page = requests.get(url)
            soup = BeautifulSoup(page.content, "html.parser")
            text = ' '.join(map(lambda p: p.text, soup.find_all('p')))
            return text
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None

    results = google_search(f"{tokens} relief", api_key, cse_id, num=5)

    for result in results:
        url = result['link']
        text = extract_text_from_url(url)
        if text:
            texts[url] = text
    return texts


def give_remedy(tokens):
    texts = str(home_remedies(tokens))
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                temperature=0.2,
                messages=[
                    {"role": "system",
                     "content": f"Act like a medical advisor and based on the given symptoms: {tokens} suggest what the patient can do to get better at home and how they can monitor their symptoms, do not state what might be the cause of it. provide the source from which you are extracting the remedy."},
                    {"role": "user", "content": texts}
                ],
                stream=True
        ):
            try:
                full_response += (response.choices[0].delta.content or "")
                message_placeholder.markdown(full_response + "▌")
            except:
                full_response += ""
                message_placeholder.markdown(full_response + "▌")
            
        message_placeholder.markdown(full_response)


def jun_doc_mode(tokens, user_input):
    junior_doc = AssistantAgent(name="junior_doc",
                                         llm_config=llm_config,
                                         is_termination_msg=lambda x: x.get("content", "").rstrip().endswith(
                                             "TERMINATE") or x.get("content", "").strip() == "",
                                         system_message=f"act like a medical assitant and ask appropriate, relevant follow up questions ONE AT A TIME to the human_user based on the symptoms {tokens} they mentioned, for example how long they have had it for, and other symptom they noticed, how severe it is and any other relevant question. you should employ a structured approach to gather the patient's clinical history, which might involve asking questions about symptoms, medical history, medications, allergies, and recent changes in health. take into consideration what has already been asked in the context that is provided to you and what info you've already gathered and then tread accordingly. Ask questions one by one, you will be given all the previous question you asked: {str(st.session_state.messages)} once you are done asking questions, and have gathered enough information say THANK YOU and end the entire message with a TERMINATE", )
    human_user = TrackableUserProxyAgent(
        name="human_user",
        human_input_mode="ALWAYS",
        max_consecutive_auto_reply=1,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        system_message="""
    Reply TERMINATE once the junior_doc says THANK YOU""")
    terminator = autogen.UserProxyAgent("terminator",
                                        system_message="Your job is to terminate the chat if the doctor says THANK YOU or TERMINATE ",
                                        human_input_mode="NEVER",
                                        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith(
                                            "TERMINATE") or x.get("content", "").strip() == "",
                                        )

    grp_chat = autogen.GroupChat(agents=[junior_doc, human_user, terminator], messages=[], max_round=50)
    manager = autogen.GroupChatManager(groupchat=grp_chat, llm_config=llm_config,
                                       is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
                                       system_message="""
      Reply TERMINATE once the junior_doc says THANK YOU""")
    # Create an event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
   

    # Define an asynchronous function
    async def initiate_chat():
        await human_user.a_initiate_chat(
            manager,
            message=user_input,
            )

    # Run the asynchronous function within the event loop
    loop.run_until_complete(initiate_chat())

with st.chat_message("assistant"):
    st.markdown("How can I help you today?")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
user_input = st.chat_input("What is up?")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({'role':'user','content':user_input})
    config = [{"model": "gpt-4", "api_key": openai.api_key}]
    llm_config = {"config_list": config, "temperature": 0.1}
    if st.session_state.junior_doctor_mode==True:
        tokens = symptoms(user_input)
        jun_doc_mode(tokens, user_input)
        with st.chat_message("assistant"):
            st.markdown("Advice while waiting for the doctor:")
        remedies = give_remedy(tokens)
        with st.chat_message("assistant"):
            st.markdown(remedies)
    ans, user_input = assess_symp(user_input)
    if ans == "No":
        tokens = symptoms(user_input)
        give_remedy(tokens)
    elif ans == "Yes":
        tokens = symptoms(user_input)
        st.session_state.junior_doctor_mode=True
        jun_doc_mode(tokens, user_input)
        with st.chat_message("assistant"):
            st.markdown("Advice while waiting for the doctor:")
        remedies = give_remedy(tokens)
        with st.chat_message("assistant"):
            st.markdown(remedies)
