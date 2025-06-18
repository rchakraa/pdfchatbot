# pdfchatbot
PDF chatbot using Qwen 3-0.6B model. Upload multiple PDFs and ask questions about their content with conversational memory, vector search, and source document references. Built with Streamlit, LangChain, and Pinecone.


Pinecone Setup

Create a free account at Pinecone
Create a new index with:

Dimensions: 384 (for sentence-transformers/all-MiniLM-L6-v2)
Metric: cosine

Copy your API key and index name to the .env file

Set up environment variables
Create a .env file in the project root: 
        PINECONE_API_KEY=your_pinecone_api_key_here
        INDEX_NAME=your_pinecone_index_name

