# import statements

import os
import streamlit as st
from elasticsearch import Elasticsearch
from langchain_openai import AzureChatOpenAI
from datetime import datetime
import json
from dotenv import load_dotenv

load_dotenv()

now = datetime.now()
dt_string = now.strftime("%m%d%Y %H:%M:%S")

# Elasticsearch Connect Parameters with API Key Authentication
def es_connect(cloud_id, api_key):
    es = Elasticsearch(
        cloud_id=cloud_id,
        api_key=api_key
    )
    return es

# Search Queries to be executed
def search(query_txt, cloud_id, api_key, index_name):

    es = es_connect(cloud_id, api_key)
    query = {
        "text_expansion": {
            "ml.inference.body_content_expanded.predicted_value": {
                "model_text": query_txt,
                "model_id": ".elser_model_2_linux-x86_64",
                "boost": 3
            }
        }
    }
    fields = ["body_content", "url", "title"]

    resp = es.search(index=index_name, fields=fields, query=query, size=1, source=False)
    
    # Check if any hits were returned
    if len(resp['hits']['hits']) > 0:
        body = resp['hits']['hits'][0]['fields']['body_content'][0]
        url = resp['hits']['hits'][0]['fields']['url'][0]
        return body, url
    else:
        return None, None



def search_elser(query_txt, cloud_id, api_key, index_name):

    es = es_connect(cloud_id, api_key)
    query = {
        "text_expansion": {
            "ml.inference.body_content_expanded.predicted_value": {
                "model_text": query_txt,
                "model_id": ".elser_model_2_linux-x86_64",
                "boost": 3
            }
        }
    }
    fields = ["body_content", "url", "title"]
    resp = es.search(index=index_name, fields=fields, query=query, size=10, source=False)
    hit = resp['hits']['hits']
    return hit

def search_bm25(query_txt, cloud_id, api_key, index_name):

    es = es_connect(cloud_id, api_key)
    query = {
        "match": {
            "body_content": query_txt
        }
    }
    fields = ["body_content", "url", "title"]
    resp = es.search(index=index_name, fields=fields, query=query, size=10, source=False)
    hit = resp['hits']['hits']
    return hit

# Integration with Azure OpenAI

def chat_gpt(prompt, deployment_name):
    # Initialize AzureChatOpenAI client
    client = AzureChatOpenAI(
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        deployment_name=deployment_name,
        openai_api_version="2024-02-01",
        max_tokens=4096,  # Adjust this as needed
        temperature=0.3,
    )

    # Send a completion call to generate an answer
    response = client.invoke([prompt])

    return response.content


def listToString(s):
    # initialize an empty string
    str1 = " "

    # return string
    return (str1.join(s))


# Main Starts here
def main(ivalue=None):
    cloud_id = os.getenv('CLOUD_ID')
    api_key = os.getenv('ES_API_KEY')
    deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
    index_name = "search-360ace"

    st.title("RAG, ESRE, Keyword Search - On a crawled blogsite")
    with st.form("chat_form"):
        query = st.text_input("Search: ")
        submit_button = st.form_submit_button("Send")
    negResponse = "Not able to find the requested search term"

    if submit_button:

        gpt_col, elser_col, bm25col = st.columns(3)
        gpt_col.subheader("RAG Output")
        elser_col.subheader("ESRE Search")
        bm25col.subheader("Keyword Search")

        resp, url = search(query, cloud_id, api_key, index_name)
        
        if resp is None:
            gpt_col.write("No results found in Elasticsearch.")
        else:
            prompt = f"Answer this question: {query}\nUsing only the information from {resp}\nIf the answer is not contained in the supplied doc, reply '{negResponse}'. Do not make up any answers"
            answer = chat_gpt(prompt, deployment_name)
            
            if negResponse in answer:
                gpt_col.write(f"Chat: {answer.strip()}")
            else:
                gpt_col.write(f"Chat: {answer.strip()}\n\nArticle-url: {url}")

        try:
            hit = search_elser(query, cloud_id, api_key, index_name)
            hit_str = json.dumps(hit)
            hit_dict = json.loads(hit_str)

            for dict in hit_dict:
                msg1 = listToString(dict['fields']['title'])
                msg2 = listToString(dict['fields']['url'])
                elser_col.write(f"{msg1}\n{msg2}")

        except Exception as error:
            elser_col.write("Nothing returned", error)
            print(error)

        try:
            hit = search_bm25(query, cloud_id, api_key, index_name)
            hit_str = json.dumps(hit)
            hit_dict = json.loads(hit_str)

            for dict in hit_dict:
                msg1 = listToString(dict['fields']['title'])
                msg2 = listToString(dict['fields']['url'])
                bm25col.write(f"{msg1} \n {msg2}")

        except Exception as error:
            elser_col.write("Nothing returned", error)
            print(error)



def click_button_ok():
    print("do nothing")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    st.set_page_config(layout="wide")

    def add_bg_from_url():
        st.markdown(
            f"""
             <style>
             .stApp {{
                 background-attachment: fixed;
                 background-size: cover
             }}
             </style>
             """,
            unsafe_allow_html=True
        )

    add_bg_from_url()

    main()
