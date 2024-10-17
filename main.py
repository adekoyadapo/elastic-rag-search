# Import statements

import os
import streamlit as st
from elasticsearch import Elasticsearch
from langchain_openai import AzureChatOpenAI
from datetime import datetime
import json
import elasticapm
from dotenv import load_dotenv
import logging
import streamlit.components.v1 as components

load_dotenv()


now = datetime.now()
dt_string = now.strftime("%m%d%Y %H:%M:%S")

def inject_rum_js(service_name, server_url):
    rum_js = f"""
    <html>
    <head>
    <script src="https://unpkg.com/@elastic/apm-rum@5.16.0/dist/bundles/elastic-apm-rum.umd.min.js"></script>
    </head>
    <body>
    <script>
      elasticApm.init({{
        serviceName: '{service_name}',
        serverUrl: '{server_url}',
        serviceVersion: '1.0.0',
        pageLoadTransactionName: 'Page Load',
        pageLoadTraceId: 'page-load-trace',
        distributedTracing: true,
        captureHeaders: true,
        captureBody: 'errors',
        environment: 'production'
      }});
    </script>
    </body>
    </html>
    """
    st.markdown(rum_js, unsafe_allow_html=True)
    logging.info("RUM script injected successfully.")

# Elasticsearch Connect Parameters with API Key Authentication
def es_connect(cloud_id, api_key):
    apmClient.begin_transaction("es-connect")
    logging.info("Connecting to Elasticsearch with cloud ID: %s", cloud_id)
    try:
        es = Elasticsearch(
            cloud_id=cloud_id,
            api_key=api_key
        )
        logging.info("Successfully connected to Elasticsearch")
        elasticapm.set_transaction_outcome("success")
    except Exception as e:
        logging.error("Failed to connect to Elasticsearch", exc_info=True)
        elasticapm.set_transaction_outcome("failure")
        raise e
    finally:
        apmClient.end_transaction("es-connect")
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
    fields = ["body_content", "url", "title", "meta_description", "_score"]

    resp = es.search(index=index_name, fields=fields, query=query, size=1, source=False)
    logging.info("Search response received from Elasticsearch")

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
                "boost": 1
            }
        }
    }
    
    fields = ["body_content", "url", "title", "meta_description", "_score"]
    resp = es.search(index=index_name, fields=fields, query=query, size=10, source=False)
    
    logging.info("Search response received from Elasticsearch for elser")
    hit = resp['hits']['hits']
    
    return hit


def search_bm25(query_txt, cloud_id, api_key, index_name):
    es = es_connect(cloud_id, api_key)
    
    query = {
        "bool": {
            "must": [{
                "multi_match": {
                    "query": query_txt,
                    "fields": ["title^3", "body_content^2"],  # Boosting title and body content
                    "type": "best_fields",  # Use fields that match best
                    "minimum_should_match": "70%"  # Ensure 70% of terms must match
                }
            }]
        }
    }
    
    fields = ["body_content", "url", "title", "meta_description", "_score"]
    resp = es.search(index=index_name, fields=fields, query=query, size=10, source=False)
    logging.info("Search response received from Elasticsearch for BM25")    
    hit = resp['hits']['hits']
    
    return hit


# Integration with Azure OpenAI
def chat_gpt(prompt, deployment_name):
    apmClient.begin_transaction("chat-gpt")
    try:
        client = AzureChatOpenAI(
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment_name=deployment_name,
            openai_api_version="2024-02-01",
            max_tokens=4096,
            temperature=0.3,
        )

        response = client.invoke([prompt])
        token_usage = response.usage_metadata
        logging.info("Received response from ChatGPT with %d tokens used", token_usage['total_tokens'])

        # Labeling token usage and cost
        input_tokens = token_usage['input_tokens']
        output_tokens = token_usage['output_tokens']
        total_tokens = token_usage['total_tokens']

        # Pricing calculation
        input_cost = input_tokens / 1000 * 0.002
        output_cost = output_tokens / 1000 * 0.002
        total_cost = total_tokens / 1000 * 0.002

        elasticapm.label(openai_input_tokens=input_tokens, openai_output_tokens=output_tokens, openai_total_tokens=total_tokens)
    
        elasticapm.label(openai_input_cost=input_cost, openai_output_cost=output_cost, openai_total_cost=total_cost)

        elasticapm.set_transaction_outcome("success")

        return response.content, token_usage
    except Exception as e:
        elasticapm.set_transaction_outcome("failure")
        raise e
    finally:
        apmClient.end_transaction("chat-gpt")

def listToString(s):
    str1 = " "
    return str1.join(s)

# Main Starts here
def main(ivalue=None):
    # Environment variables
    cloud_id = os.getenv('CLOUD_ID')
    api_key = os.getenv('ES_API_KEY')
    deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
    index_name = "search-360ace"
    
    logging.info("Starting main application flow")
    inject_rum_js(service_name="search-rum", server_url=os.getenv("APM_SERVER_URL"))

    # Streamlit UI
    st.title("RAG | ESRE | Keyword Search - On a Blogsite")

    # Sidebar for layout selection
    layout_option = st.sidebar.radio(
        "Select View:",
        ("Side by Side", "Individual Tabs")
    )

    negResponse = "Not able to find the requested search term"

    # Handle layout option
    if layout_option == "Side by Side":
        with st.form("chat_form"):
            query = st.text_input("Search: ")
            submit_button = st.form_submit_button("Send")

        if submit_button:
            # Split layout into three columns
            gpt_col, elser_col, bm25col = st.columns(3)
            gpt_col.subheader("RAG Output")
            elser_col.subheader("ESRE Search")
            bm25col.subheader("Keyword Search")

            # RAG Search
            resp, url = search(query, cloud_id, api_key, index_name)
            if resp is None:
                gpt_col.write("No results found in Elasticsearch.")
            else:
                prompt = (f"Answer this question: {query}\nUsing only the information from {resp}\n"
                          f"If the answer is not contained in the supplied doc, reply '{negResponse}'. Do not make up any answers")
                answer, token_usage = chat_gpt(prompt, deployment_name)

                gpt_col.markdown(f"**Chat**: {answer.strip()}  \n"
                                 f"**Article URL**: {url}  \n"
                                 f"**Tokens used**: {token_usage['total_tokens']} "
                                 f"(Input: {token_usage['input_tokens']}, Output: {token_usage['output_tokens']})")

            # ESRE Search
            try:
                hits = search_elser(query, cloud_id, api_key, index_name)
                for hit in hits:
                    title = listToString(hit['fields']['title'])
                    url = listToString(hit['fields']['url'])
                    score = hit['_score']
                    elser_col.markdown(f"**Title**: {title}  \n"
                                       f"**URL**: {url}  \n"
                                       f"**Score**: {score}  \n\n")
            except Exception as error:
                elser_col.write(f"Nothing returned: {error}")
                print(error)

            # BM25 Search
            try:
                hits = search_bm25(query, cloud_id, api_key, index_name)
                for hit in hits:
                    title = listToString(hit['fields']['title'])
                    url = listToString(hit['fields']['url'])
                    score = hit['_score']
                    bm25col.markdown(f"**Title**: {title}  \n"
                                     f"**URL**: {url}  \n"
                                     f"**Score**: {score}  \n\n")
            except Exception as error:
                bm25col.write(f"Nothing returned: {error}")
                print(error)

    elif layout_option == "Individual Tabs":
        tab1, tab2, tab3 = st.tabs(["Conversational Search", "ESRE Search", "Keyword Search"])

        # RAG Search Tab
        with tab1:
            st.subheader("RAG Output")
            with st.form("rag_form"):
                query = st.text_input("Enter your conversational query for RAG:")
                submit_button = st.form_submit_button("Search")
            
            if submit_button:
                resp, url = search(query, cloud_id, api_key, index_name)
                if resp is None:
                    st.write("No results found in Elasticsearch.")
                else:
                    prompt = (f"Answer this question: {query}\nUsing only the information from {resp}\n"
                              f"If the answer is not contained in the supplied doc, reply '{negResponse}'. Do not make up any answers")
                    answer, token_usage = chat_gpt(prompt, deployment_name)
                    
                    st.markdown(f"**Chat**: {answer.strip()}  \n"
                                f"**Article URL**: {url}  \n"
                                f"**Tokens used**: {token_usage['total_tokens']} "
                                f"(Input: {token_usage['input_tokens']}, Output: {token_usage['output_tokens']})")

        # ESRE Search Tab
        with tab2:
            st.subheader("ESRE Search")
            with st.form("esre_form"):
                query = st.text_input("Enter your search query for ESRE:")
                submit_button = st.form_submit_button("Search")
            
            if submit_button:
                try:
                    hits = search_elser(query, cloud_id, api_key, index_name)
                    for hit in hits:
                        title = listToString(hit['fields']['title'])
                        url = listToString(hit['fields']['url'])
                        score = hit['_score']
                        st.markdown(f"**Title**: {title}  \n"
                                    f"**URL**: {url}  \n"
                                    f"**Score**: {score}  \n\n")
                except Exception as error:
                    st.write(f"Nothing returned: {error}")
                    print(error)

        # BM25 Search Tab
        with tab3:
            st.subheader("Keyword Search")
            with st.form("keyword_form"):
                query = st.text_input("Enter your search query for Keyword Search:")
                submit_button = st.form_submit_button("Search")
            
            if submit_button:
                try:
                    hits = search_bm25(query, cloud_id, api_key, index_name)
                    for hit in hits:
                        title = listToString(hit['fields']['title'])
                        url = listToString(hit['fields']['url'])
                        score = hit['_score']
                        st.markdown(f"**Title**: {title}  \n"
                                    f"**URL**: {url}  \n"
                                    f"**Score**: {score}  \n\n")
                except Exception as error:
                    st.write(f"Nothing returned: {error}")
                    print(error)

# apmClient.end_transaction("search")
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    st.set_page_config(layout="wide")
    apmClient = elasticapm.Client(
      service_name="search-compare",
      secret_token=os.getenv("APM_TOKEN"),
      server_url=os.getenv("APM_SERVER_URL"),
      verify_server_cert="false",
      environment="production",
      logging=True
      )
    
    elasticapm.instrument()

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
    elasticapm.instrument()  # Only call this once, as early as possible.
    apmClient.begin_transaction(transaction_type="script")
    logging.info("Starting the search app")
    main()
    apmClient.end_transaction(name=__name__, result="success")