import os
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()
api_key = os.getenv('GCP_API_KEY')

# Ensure the API key is loaded
if not api_key:
    raise ValueError("GCP API key is not set in the environment variables")

# Set up the database connection
db = SQLDatabase.from_uri("sqlite:///Chinook.db")

# Print database dialect and usable table names for verification
print("Database Dialect:", db.dialect)
print("Usable Table Names:", db.get_usable_table_names())

# Run a test query
print("Sample Query Result:", db.run("SELECT * FROM Artist LIMIT 10;"))

# Initialize the language model
llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=api_key)

# Create the SQL query chain
query_chain = create_sql_query_chain(llm, db)

# Test the chain with a sample question
response = query_chain.invoke({"question": "How many employees are there?"})
print("Query Response:", response)

# Run the query obtained from the response
result = db.run(response)
print("Query Result:", result)

# Set up the QuerySQLDataBaseTool and chain for execution
execute_query = QuerySQLDataBaseTool(db=db)
write_query_chain = create_sql_query_chain(llm, db)

# Chain for executing queries and handling results
chain = write_query_chain | execute_query

# Test the chain with another sample question
response = chain.invoke({
    "question": "HOW MANY EMPLOYEES ARE THERE?"
})
print("Invoices Query Response:", response)

# Define a prompt for generating answers from query results
answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

# Create the final chain to handle the entire process
final_chain = (
    RunnablePassthrough.assign(query=write_query_chain).assign(
        result=itemgetter("query") | execute_query
    )
    | answer_prompt
    | llm
    | StrOutputParser()
)

# Test the final chain with a question
final_response = final_chain.invoke({
    "question": "HOW MANY EMPLOYEES ARE THERE?"
})
print("Final Response:", final_response)
