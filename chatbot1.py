# prototype chatbot withe four key functions:
# - generate tearsheet
# - query database and display data
# - query database and plot data
# - chat with vectorstore

# local imports
import tearsheet_utils as tshu

# system imports
import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

from langchain.tools import tool
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser

from typing import Optional, List
from pydantic.v1 import BaseModel, Field


# load vectordb
vectordb = tshu.create_or_load_vectorstore('data/chroma',
    tshu.load_persona_html(), 
    override=False)
 

# define chat_with_docs function
class ChatWithDocsInput(BaseModel):
    query: str = Field(..., description="Question to ask the document store.")
    client_name: str = Field(..., description="Name of client to query for.")

@tool(args_schema=ChatWithDocsInput)
def chat_with_docs(query: str,  client_name: str) -> dict:
    # question: should document details go here or in the chat agent prompt?
    """
    Search the document store with the given query for the given client name. 
    The document store filters for the top documents given the query. Document
    types include:
    
    'equilar' - For details on stock or equity transactions, stock sold, annual compensation.
    'google' - For recent news articles listed on google.
    'linkedin' - For employment history, education, board memberships, and bio.
    'pitchbook' - For deals as lead partner, investor bio.
    'relsci' - For current or prior board memberships, top donations.
    'wealthx' - For individual and family net worth, interests, passions, hobbies.
    """

    # create filter and run query
    filter_ = tshu.create_filter(client_name, 'all')
    response = tshu.qa_metadata_filter(query, vectordb, filter_, top_k=3)

    #return f'called chat_with_docs for client {client_name}' #response
    return response


# define generate_and_send_tearsheet function
class GenSendTearsheetInput(BaseModel):
    client_name: str = Field(..., description="Name of client to generate tearsheet for.")
    email: str = Field(..., description="Email that will receive tearsheet this function sends.")

@tool(args_schema=GenSendTearsheetInput)
def gen_send_tearsheet(client_name: str, email: str) -> dict:
    """
    Generate the tearsheet for the given client_name. Send to the given email
    address.
    """

    # generate
    #html, output_path = tshu.generate_tearsheet(client_name, vectordb)

    # todo: format / send email
    #prep_and_send_email(output_path, img, email)

    return f'called gen_send_tearsheet for client {client_name} and recipient {email}' #response


# include is the client_name and recipient email address
# function wraps around tshu.generate_tearsheet(client, vectordb)

# define table_from_db

# define plot_from_db


if __name__ == '__main__':
    import chatbot1 as m

    tools = [m.chat_with_docs, m.gen_send_tearsheet]
    functions = [m.format_tool_to_openai_function(f) for f in tools]
    model = m.ChatOpenAI(temperature=0).bind(functions=functions)
    prompt = m.ChatPromptTemplate.from_messages([
        ("system", "You are helpful but sassy assistant"),
        ("user", "{input}"),
    ])
    chain = prompt | model | m.OpenAIFunctionsAgentOutputParser()

    # map tool names to callables 
    tool_map = {
        "chat_with_docs": m.chat_with_docs, 
        "gen_send_tearsheet": m.gen_send_tearsheet,
        }

    # test that questions map to the right functions
    # eval output with resultN.log
    result1 = chain.invoke({"input": "where does Robert King work?"})
    result2 = chain.invoke({"input": "what is Robert King's family net worth?"})
    result3 = chain.invoke({"input": "What deals as lead partner did Robert King do?"})
    result4 = chain.invoke({"input": "What recent news is there about Robert King?"})
    result5 = chain.invoke({"input": "Write a tearsheet about Jared Livinglife and send it to siyer@target.com"})

    for i, result in enumerate([result1, result2, result3, result4, result5]):
        observation = tool_map[result.tool].run(result.tool_input)
        print(f'\n{i}, {result.log} : {observation}\n')

    # test that functions run correctly
    m.chat_with_docs('where does robert work?', 'Robert King', ['linkedin'])