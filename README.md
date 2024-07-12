# Document-Q-A-using-Google-Gemma-Groq-API

## Detailed Information

### Imports and Environment Setup:

- The script imports necessary libraries including `streamlit`, `os`, `langchain` components, `dotenv` for environment variable management, and `time` for response timing.
- Environment variables are loaded using `load_dotenv()`, and API keys for GROQ and Google are retrieved from the environment.

### Streamlit Application Title:

- The title of the Streamlit application is set using `st.title("Gemma Model Document Q&A")`.

### Language Model Initialization:

- The `ChatGroq` language model is initialized with the provided GROQ API key and model name `Llama3-8b-8192`.

### Chat Prompt Template:

- A `ChatPromptTemplate` is created to define the format for the questions and context provided to the language model.

### Vector Embedding Function:

The `vector_embedding` function checks if vector embeddings are already stored in the Streamlit session state. If not, it performs the following steps:

1. Initializes the Google Generative AI Embeddings model.
2. Loads PDF documents from the specified directory using `PyPDFDirectoryLoader`.
3. Splits the documents into chunks using `RecursiveCharacterTextSplitter`.
4. Creates a FAISS vector store with the document embeddings.

### User Input and Embedding Button:

- A text input field is provided for the user to enter a question.
- A button is provided to trigger the `vector_embedding` function. If clicked, the function is executed, and a message is displayed indicating that the vector store database is ready.

### Question Processing and Response Generation:

If a question is entered by the user, the script performs the following steps:

1. Creates a document chain using the `create_stuff_documents_chain` function with the language model and prompt.
2. Retrieves the document vectors from the session state and creates a retriever.
3. Creates a retrieval chain using `create_retrieval_chain`.
4. Measures the response time and invokes the retrieval chain with the user’s question.
5. Displays the answer to the user.
6. Expands a section to show document similarity search results, displaying the relevant document chunks.

### Summary

This code integrates document embedding and retrieval with a language model to answer user questions based on the provided documents. The Streamlit interface allows for user interaction and visualization of the results.
