# Q/A System over SQL Data using LangChain

This project demonstrates a Question/Answer (Q/A) system that interacts with a SQL database using LangChain. The system leverages Google Generative AI to generate SQL queries from natural language questions and execute those queries against the database.

## Requirements

- Python 3.8+
- `langchain_community` package
- `langchain_google_genai` package
- `langchain_core` package
- `python-dotenv` package
- `sqlite` package (for SQLite database)

## Setup

1. **Clone the repository and navigate to the project directory:**

    ```bash
    git clone <repository-url>
    cd <project-directory>
    ```

2. **Install the required Python packages:**

    ```bash
    pip install langchain_community langchain_google_genai langchain_core python-dotenv
    ```

3. **Set up the environment variables:**

    Create a `.env` file in the project directory and add your Google Cloud API key:

    ```env
    GCP_API_KEY=your-google-cloud-api-key
    ```

4. **Ensure you have the SQLite database (`Chinook.db`) in the project directory.** If not, you can download it from the official source or use your own SQLite database.


## Conclusion

This project provides a comprehensive example of creating a Q/A system over SQL data using LangChain and Google Generative AI. By following the setup and usage instructions, you can customize and extend the system to suit your specific needs.

