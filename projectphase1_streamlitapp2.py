from langchain_chroma import Chroma
from langchain.schema import Document
import requests
from newspaper import Article
import pandas as pd
import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter,TokenTextSplitter
from langchain.schema import Document
from langchain.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
#from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain
#import tiktoken
#from transformers import AutoTokenizer
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings
#from google.colab import drive
#drive.mount('/content/drive')
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import Literal
import requests
from newspaper import Article
import pandas as pd
import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter,TokenTextSplitter
from langchain.schema import Document
from langchain.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
#from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain
import time
import shutil
import chromadb
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Dict, Literal, Optional
from pydantic import field_validator, BaseModel, Field
from typing import List, Dict, Literal, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.llms import HuggingFacePipeline
from transformers import pipeline



from huggingface_hub import login
login("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")

#if no relevant document found, give answer from general knowledge.
#newsScrapper function
def newsscrapper(a,b):
    #headers, which is typically used when making HTTP requests to simulate a browser's behavior
    headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'}
    url=a
    url_number=b
    #session uses the Python requests library, which provides a way to persist parameters, cookies, and headers across multiple HTTP requests.
    session = requests.Session()
    try:
        response = session.get(url, headers=headers, timeout=10)
        #HTTP response status codes indicate whether a specific HTTP request has been successfully completed
        status=response.status_code
    
        if status==200:
            # if the status code is 200 then website can be scrapped
            maintext_placeholder.text(f"Scraping website url {url_number} with status code {response.status_code}")
            time.sleep(1)
            # instance of the Article class is created.
            #this class is used to extract and parse articles from HTML content
            #empty string '' is passed to initialize it, but it will be set with actual HTML content in the next steps.
            article=Article('')
            #set_html() method of the Article object is called to set its HTML content. 
            #response.text contains the HTML content of the webpage fetched from the request.
            article.set_html(response.text)
            #parse() method processes the HTML content to extract useful information about the article,
            #such as the title, text, authors, and publication date.
            article.parse()
            #these variable contains the data extracted from the website
            news_text=article.text
            #to remove empty space or white line in the text
            #cleaned_text = '\n'.join([line.strip() for line in news_text.split('\n') if line.strip()])
            #Split the news_text into individual lines
            lines = news_text.split('\n')
            #Create an empty list to store the cleaned lines
            cleaned_lines = []
            #Iterate through each line
            for line in lines:
                # Strip leading and trailing whitespace from the line
                stripped_line = line.strip()
                
                # If the stripped line is not empty, add it to the cleaned_lines list
                if stripped_line:
                    cleaned_lines.append(stripped_line)

            #Join the cleaned lines back into a single string with newline characters
            cleaned_text = '\n'.join(cleaned_lines)

            #print(cleaned_text)
            news_authors=','.join(article.authors)
            news_title=article.title
            news_publication_date=str(article.publish_date)
            # Extract the metadata
            #meta_data = article.meta_data
            # Extract the category (if available)
            #news_category = ''.join(meta_data.get('category', 'Category not found'))
            news_url=url
            #data contains the Article text 
            data=f"Article Title: {news_title} \n Publication Date:{news_publication_date} \n {cleaned_text}"
            #while metadata contain the information tag related to Article text
            metadata = {
                "source": f"{news_url}",
                "Author": f"{news_authors}",
               "Publication date": f"{news_publication_date}",
                "Article Title": f"{news_title}",
                }
            #the data and meta data is converted in to document so it can processed through LLM
            data_document = Document(page_content=data, metadata=metadata)
            return data_document
        else:    
            #if status code is not 200 then it will print the failure to scrap the website
            maintext_placeholder.text(f"Failed to scrape the website url {url_number}. HTTP Status Code: {response.status_code}")
            time.sleep(1)
            return None
                

    except Exception as e:
        #print out any error occured during scrapping of news article
        maintext_placeholder.text(f"Error occurred while fetching article URL {url_number} at {url}: {e}")
        time.sleep(1)
        return None
#removing repeitative data
def metadata_cleaning(x):
  return list(dict.fromkeys(x))


#create heading for streamlit website and create a sidebar foir streamlit
st.title("Automated News summarization chatbot using RAG based large language Model")
st.write("Contains 600 news articles published between December 2024 and February 2025, categorized into Politics, Business, Sports, Technology, and Environment.")
st.sidebar.title("News_Article_URL")
#create an input bar on streamlit where user can give url of website
url1=st.sidebar.text_input(f"URL 1:")
url2=st.sidebar.text_input(f"URL 2:")
#url_process create button for user to press inorder to scrap the website
url_process=st.sidebar.button("Process_URLs")
maintext_placeholder=st.empty()
#name of embedding model that will embed our news text and query
embedding_model_name ="mixedbread-ai/mxbai-embed-large-v1"
#embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings_model = HuggingFaceEmbeddings(model_name=embedding_model_name,model_kwargs={"device": "cpu"})
#configuration setting for our LLM
Groq_API_TOKEN="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"
llm = ChatGroq(
    model="llama-3.1-8b-instant",  
    api_key=Groq_API_TOKEN,  
    temperature=0,
    max_tokens=1024
)

#-------------------------------------local LLM----------------------------------------------
#loading Local LLama from google drive
model_id = "meta-llama/Llama-3.2-3B-Instruct"
save_path = "/content/drive/MyDrive/model/Llama3.2-3B-Instruct"

@st.cache_resource
def load_llama_model():
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return pipeline(
     "text-generation",
     model=model,
     tokenizer=tokenizer,
     max_new_tokens=1000,
     temperature=0,
     do_sample=False
 )

#tokenizer = AutoTokenizer.from_pretrained(save_path)
#model = AutoModelForCausalLM.from_pretrained(save_path)

#pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=1000,
#     temperature=0,
#     do_sample=False
# )

llm_generator = HuggingFacePipeline(pipeline=load_llama_model())


#it is path where vector database will be stored when it will be created
vector_database_filepath="/content/drive/MyDrive/temp_vectordatabase"
#if user will press the button on streamlit then if function will work
if url_process:
    maintext_placeholder.text("URL_Loading...")
    time.sleep(1)
    #scrap url link1 on streamlit
    Url_link1=[newsscrapper(url1,1)]
    #scrap url link2 on streamlit
    Url_link2=[newsscrapper(url2,2)]
    #if user enter both url then if function will work
    if Url_link1 !=[None] and Url_link2 !=[None]:
        #combines both url document
        Url_all_document=Url_link1 + Url_link2
        maintext_placeholder.text("URL_Data...Loaded...✔✔✔")
        time.sleep(1)
        maintext_placeholder.text("URL_Data_Text_splitting...Started...✔✔✔")
        time.sleep(1)
        #configuration setting for text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000,
            chunk_overlap=0
        )
        Url_textall=[]
        #splits the text in to chunks so it can be processed by LLM without reaching LLM token limit
        for doc in Url_all_document:
            chunk=text_splitter.split_documents([doc])
            Url_textall.extend(chunk)
        maintext_placeholder.text("URL_Data_splitText_Vector_Embedding...Started...✔✔✔")
        time.sleep(1)
        #create a vector embedding and stores it in a database
        vectorstore_Huggingface = FAISS.from_documents(Url_textall, embeddings_model)
        #vectorstore_Huggingface.save_local("vectorstore_dataset")
        #saves the vector database on your computer as a file
        vectorstore_Huggingface.save_local(vector_database_filepath)
        maintext_placeholder.text("Vector_Embedding_stored_✔✔✔")
        time.sleep(1)
    #elif function works when user have not entered both URL but only URL 1    
    elif Url_link1 !=[None] and Url_link2 ==[None]:
        Url_all_document=Url_link1
        maintext_placeholder.text("URL1_Data...Loaded...✔✔✔")
        time.sleep(1)
        maintext_placeholder.text("URL1_Data_Text_splitting...Started...✔✔✔")
        time.sleep(1)
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000,
            chunk_overlap=0
        )
        Url_textall=[]
        for doc in Url_all_document:
            chunk=text_splitter.split_documents([doc])
            Url_textall.extend(chunk)
        maintext_placeholder.text("URL1_Data_splitText_Vector_Embedding...Started...✔✔✔")
        time.sleep(1)
        vectorstore_Huggingface = FAISS.from_documents(Url_textall, embeddings_model)
        #vectorstore_Huggingface.save_local("vectorstore_dataset")
        vectorstore_Huggingface.save_local(vector_database_filepath)
        maintext_placeholder.text("Vector_Embedding_stored_✔✔✔")
        time.sleep(1)
    #elif function works when user have not entered both URL but only URL 2    
    elif Url_link2 !=[None] and Url_link1 ==[None]:
        Url_all_document=Url_link2
        maintext_placeholder.text("URL2_Data...Loaded...✔✔✔")
        time.sleep(1)
        maintext_placeholder.text("URL2_Data_Text_splitting...Started...✔✔✔")
        time.sleep(1)
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000,
            chunk_overlap=0
        )
        Url_textall=[]
        for doc in Url_all_document:
            chunk=text_splitter.split_documents([doc])
            Url_textall.extend(chunk)
        maintext_placeholder.text("URL2_Data_splitText_Vector_Embedding...Started...✔✔✔")
        time.sleep(1)
        vectorstore_Huggingface = FAISS.from_documents(Url_textall, embeddings_model)
        #vectorstore_Huggingface.save_local("vectorstore_dataset")
        vectorstore_Huggingface.save_local(vector_database_filepath)
        maintext_placeholder.text("Vector_Embedding_stored_✔✔✔")
        time.sleep(1)
    else:#else function works when user have not entered both URL   
        maintext_placeholder.text("No URL links are provided.")
        #it will delete the previouse vector database file stored on the computer
        if os.path.exists(vector_database_filepath):
            # Delete the folder and its contents
            shutil.rmtree(vector_database_filepath)
            maintext_placeholder.text(f"Deleted vector database at {vector_database_filepath}")
            time.sleep(2)
        else:
            maintext_placeholder.text(f"Vector database not found at {vector_database_filepath}")
            time.sleep(2)
#create input box where user can ask question from LLM    
question=maintext_placeholder.text_input("Question:")


    #if user pressed entered on written text then this if statement will work
if question:
    Question_query = question.replace("Question:", "").strip()
    maintext_placeholder.text(f"question:{Question_query}")
    time.sleep(5)

#-----------------------------------------------Query Router--------------------------------------------------------------------------
    # Define the routing prompt
    router_prompt = ChatPromptTemplate.from_template(
        """You are an AI assistant that decides whether a user query requires retrieval-augmented generation (RAG).
    If the query needs external knowledge from a news article vector database, respond with "RAG".
    If the query can be answered directly without retrieval, respond with "NO RAG".

    User query: {question}"""
    )

    # Define the model pipeline
    llm_router = (
        router_prompt 
        | llm
        | StrOutputParser()
    )

    maintext_placeholder.text("Query_Routing_Started")

    # Example usage
    Router_query = Question_query
    Router_response = llm_router.invoke({"question": Router_query})

    maintext_placeholder.text("Query_Routing_✔✔✔")
    maintext_placeholder.text(f"Router:{Router_response}")
    time.sleep(1)

    print(Router_response)  # Expected output: "RAG" or "NO RAG"

    if Router_response=="RAG":
        #----------------------------------------------QUERY Translation-------------------------------------------------------------------------
        template = """You are an AI language model assistant. Your task is to generate five 
        different versions of the given user question and to reform the original question into a better worded structured question in order to retrieve relevant documents from a News article vector 
        database which contains 5 different categories (Politics, Sports, Economy and Business, Environment, Technology) and contains data from October 2024 to February 2025. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search. 
        Provide these alternative questions separated by newlines.

        ### Instruction:
        1. First, rewrite the original query in a **more structured and detailed way** and only mention dates if original query had one.
        2. Then, generate **five alternative variations** that rephrase the query from different perspectives.

        example output format:
        what are the latest news updates on cricket matches that took place in Pakistan during January 2025?
        what recent cricket events occurred in Pakistan in January 2025, and what were the key outcomes?
        etc

        Original question: {question}

        ### Output Format:
        (no need to add labels like query1, query2, or reformed query, and do not say anything extra, just give output in this format)
        reformed query,query1,query2,query3,query4,query5
        """


        query_rewrite_prompt = PromptTemplate(
            input_variables=["question"],
            template=template
        )

        maintext_placeholder.text("Query_translation_Started")
        time.sleep(1)

        query_rewriter = query_rewrite_prompt | llm
        multiquery_response = query_rewriter.invoke(Question_query)
        print(multiquery_response.content)

        response_text = multiquery_response.content
        queries = response_text.split("\n")

        # Remove the 'reformed query:' text from the first element
        reformed_query = queries[0].replace("reformed query:", "").strip()

        # Extract the alternative queries (query1, query2, etc.)
        alternative_queries = []
        for query in queries[1:]:  # Loop through queries starting from index 1
            cleaned_query = query.strip()  # Remove leading/trailing whitespace
            alternative_queries.append(cleaned_query)  # Add to the list

        # Store the results in a list
        query_list = [reformed_query] + alternative_queries
        maintext_placeholder.text("Query_translation_✔✔✔")
        maintext_placeholder.text(f"Query_translation:{query_list}")
        time.sleep(1)
        # Print the list
        #print(query_list)

        #this if statement works if there is vector database file present on computer
        if os.path.exists(vector_database_filepath):
            maintext_placeholder.text(f"website_data")
            time.sleep(2)
            #vectorstore_load=FAISS.load_local("vectorstore_dataset",embeddings,allow_dangerous_deserialization=True)
            #loads the vector database from the computer for use.
            vectorstore_load=FAISS.load_local(vector_database_filepath,embeddings_model,allow_dangerous_deserialization=True)
            # Create a retriever from the FAISS database
            retriever = vectorstore_load.as_retriever(search_kwargs={"k": 5})  # Fetch top 3 matches

            # 4. Get relevant documents (chunks of text)
            retrieved_docs = retriever.get_relevant_documents(Question_query)

            # 5. Combine the document content into a context string
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])

            # Optional: You can also extract metadata (like source, title)
            #sources = [doc.metadata.get("Source", "Unknown") for doc in retrieved_docs]
            

            #-----------------------------------------------Generator----------------------------------------------------------
            generator_prompt = ChatPromptTemplate.from_template(
            """You are an AI news assistant that answers user queries based on news articles.
            You will be given a user query and relevant context retrieved from a vector database.

            Your task is to answer the query **using only the context** provided.

            Only output the answer. Do not repeat the query or the context.
            /n/n/n
            Question:
            {question}
            /n/n/n
            Context:
            {context}
            /n/n/n
            Answer:"""
            )
            query =Question_query
            rag_chain = generator_prompt | llm_generator
            response = rag_chain.invoke({
                "question": query,
                "context": context
            })
            response_text = response
            queries = response_text.split("/n/n/n")
            reformed_query = queries[3].replace("Answer:", "").strip()
            Answer_gen=reformed_query
            st.header("Question:")
            st.write(query)
            st.subheader("Answer:")
            answer_output = Answer_gen
            st.write(answer_output)
            #pritnting the output
            #sources = response.get("sources", "")
            #if will work only if source url link is present in meta data of document 
            #if sources:
            #    st.subheader("Sources:")
            #    sources_list = sources.split("\n")  # Split the sources by newline
            #    for source in sources_list:
            #        st.write(source)
        else:#if no vector database is present on computer it will answer only general question
            vector_db = Chroma(persist_directory="/content/drive/MyDrive/chroma_db",embedding_function=embeddings_model)
            #---------------------------------------------------------Query Structuring------------------------------------------------------------
            class MetadataCondition(BaseModel):
                Category: List[str] = Field(
                    ...,
                    description=(
                        "Only one category must be selected from: "
                        "[Politics, Sports, Business_and_Economics, Environment and Climate, Technology]. "
                        "If more than one category applies or it cannot be classified, use ['ALL']."
                    )
                )
                Publish_Time: Optional[Dict[Literal["$gte", "$lte"], int]] = Field(
                    default=None,
                    description="Only '$gte' and '$lte' are valid keys. Use YYYYMMDD format (e.g., 20241205).the dates should be in between the range of 20241201 to 20250228.only use this if dates are mentioned in the query otherwise leave it"
                )

                @field_validator("Category")
                def validate_category(cls, v):
                    # Ensure that if more than one category is selected, it defaults to 'ALL'
                    if len(v) > 1:
                        return ["ALL"]
                    return v


            structured_llm = llm.with_structured_output(MetadataCondition)
            prompt = ChatPromptTemplate.from_messages([
            ("system", """You're a helpful assistant that analyzes user queries to determine relevant metadata filters for a vector database.

            Return a dictionary containing:
            - 'Category': A list with exactly **one** of the following categories: ["Politics", "Sports", "Business_and_Economics", "Environment and Climate", "Technology"]. If the query relates to **more than one category** or is **unclear**, return ["ALL"].
            - 'Publish_Time': (optional) A dictionary with at most two keys: "$gte" and/or "$lte", and values in the format YYYYMMDD (e.g., 20241202). Only include this field **if the query explicitly mentions a date, month, or year**. Do NOT infer timeframes like "recent", "last year", or "last month".

            --- Examples ---

            Query: "What were the main discussions during the government meeting last week?"
            Output:
            {{ 
            "Category": ["Politics"]
            }}

            Query: "Tell me about events that happened in January 2025 in the tech world"
            Output:
            {{ 
            "Category": ["Technology"],
            "Publish_Time": {{
                "$gte": 20250101,
                "$lte": 20250131
            }}
            }}

            Now analyze this user query:
            """),
                ("human", "User query: {question}")
            ])
            query_structurer = prompt | structured_llm

            news_context=[]
            news_metedata=[]
            for x in range(len(query_list)):
                query_structing_metadata=query_list[x]
                response = query_structurer.invoke({"question": query_structing_metadata})
                maintext_placeholder.text("Query Structuring...........✔✔✔")
                maintext_placeholder.text(f"Query_translation:{response}")
                time.sleep(5)
                # Assuming 'response' is an instance of MetadataCondition
                news_category = response.Category  # Accessing the 'Category' attribute
                news_publish_time = response.Publish_Time  # Accessing the 'Publish_Time' attribute
                if news_category == ["ALL"] and news_publish_time:
                    # Use the values of 'Publish_Time' from the response to create the filter
                    news_filters = {
                        "$and": [
                            {"Publish_Time": {"$gte": news_publish_time["$gte"]}},
                            {"Publish_Time": {"$lte": news_publish_time["$lte"]}}
                        ]
                    }

                # If the category is "ALL" and there is no publish time
                elif news_category == ["ALL"] and news_publish_time is None:
                    news_filters = None

                # If the category is not "ALL" and there is a publish time
                elif news_category != ["ALL"] and news_publish_time:
                    news_filters = {
                        "$and": [
                            {"Category": news_category[0]},
                            {"Publish_Time": {"$gte": news_publish_time["$gte"]}},
                            {"Publish_Time": {"$lte": news_publish_time["$lte"]}}
                        ]
                    }

                # If the category is not "ALL" and there is no publish time
                elif news_category != ["ALL"] and news_publish_time is None:
                    news_filters = {"Category": news_category[0]}

                maintext_placeholder.text("Filtered Selected...........✔✔✔")
                maintext_placeholder.text(f"Filter:{news_filters}")
                time.sleep(5)

                #---------------------------------------------------------Vector_Database---------------------------------------    

                results = vector_db.similarity_search(
                    query_structing_metadata, 
                    k=5,
                    filter =news_filters 
                )

                # Print retrieved documents
                for i, doc in enumerate(results):
                    news_context.append(f"Title: {doc.metadata['News_Title']}\nDate: {doc.metadata['Publish_Time']}\n{doc.page_content}")
                    #print(f"Document {i+1}: {doc.page_content}")
                    news_metedata.append(f"{doc.metadata['Source']}\n")
                    #print(f"Metadata: {doc.metadata["Source"]}")
                    #print("-" * 50)
                #print(type(doc.metadata["Source"]))
            metadata_cleaned=metadata_cleaning(news_metedata)    
            news_context_join="\n\n".join(news_context)    
            news_metedata_join="".join(metadata_cleaned)
            #user_question="\n".join(query_list)
            maintext_placeholder.text("Data Retrieved...........✔✔✔")
            maintext_placeholder.text(f"Context:{news_context_join}")
            time.sleep(1)
            maintext_placeholder.text(f"Source:{news_metedata_join}")
            time.sleep(1)
            #-----------------------------------------------Generator----------------------------------------------------------
            generator_prompt = ChatPromptTemplate.from_template(
            """You are an AI news assistant that answers user queries based on news articles.
            You will be given a user query and relevant context retrieved from a vector database.

            Your task is to answer the query **using only the context** provided.

            Only output the answer. Do not repeat the query or the context.
            /n/n/n
            Question:
            {question}
            /n/n/n
            Context:
            {context}
            /n/n/n
            Answer:"""
            )
            query =query_list[0]
            rag_chain = generator_prompt | llm_generator
            response = rag_chain.invoke({
                "question": query,
                "context": news_context_join
            })
            response_text = response
            queries = response_text.split("/n/n/n")
            reformed_query = queries[3].replace("Answer:", "").strip()
            Answer_gen=reformed_query
            query_question = response_text.split("/n/n/n")
            reformed_query_question = query_question[1].replace("question", "").strip()
            questions_gen=reformed_query_question
            maintext_placeholder.text(f"RAG_Successfully_Completed")
            time.sleep(1)
            st.header("Question:")
            st.write(questions_gen)
            st.subheader("Answer:")
            answer_output = Answer_gen
            st.write(answer_output)
            #pritnting the output
            sources_output =news_metedata_join
            #if will work only if source url link is present in meta data of document 
            if sources_output:
                st.subheader("Sources:")
                sources_list = sources_output.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)

                
    
    elif Router_response=="NO RAG.":
        general_answer = llm.predict(f"Answer the question: {Question_query}")
        st.write(general_answer)







    
    













