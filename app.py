import streamlit as st
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.llms.base import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Optional, List, Any

# Fix PyTorch path issue
if hasattr(torch.classes, '__path__'):
    torch.classes.__path__ = [os.path.join(torch.__path__[0], 'classes')]

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="PDF Chatbot with Qwen",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF Chatbot with Qwen")
st.markdown("Ask questions about your PDF document using Qwen 3-0.6B model!")

# Custom LLM class for Qwen - Fixed Version
class QwenLLM(LLM):
    model: Any = None
    tokenizer: Any = None
    
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
    
    @property
    def _llm_type(self) -> str:
        return "qwen"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        # Format the prompt for chat
        messages = [{"role": "user", "content": prompt}]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # Disabled thinking mode to prevent getting stuck
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        # Generate response with improved parameters
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=500,  # Reduced for faster response
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                early_stopping=True
            )
        
        # Get the full generated text
        full_response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        
        # Extract just the generated part (after the input prompt)
        input_text = self.tokenizer.decode(model_inputs.input_ids[0], skip_special_tokens=False)
        generated_text = full_response[len(input_text):].strip()
        
        # Clean up the response
        content = self._clean_response(generated_text)
        
        return content
    
    def _clean_response(self, generated_text: str) -> str:
        """Clean up the generated response"""
        
        # Remove common special tokens
        content = generated_text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        
        # Handle thinking content if present (fallback)
        if "<think>" in content and "</think>" in content:
            # Split by </think> and take everything after it
            parts = content.split("</think>")
            if len(parts) > 1:
                final_content = parts[-1].strip()
                if final_content:
                    return final_content
                else:
                    # If nothing after </think>, try to extract from thinking section
                    thinking_content = parts[0].replace("<think>", "").strip()
                    return thinking_content if thinking_content else "I need more information to answer your question."
            else:
                # Remove <think> tags if </think> is missing
                content = content.replace("<think>", "").strip()
        
        # Remove any remaining special tokens
        content = content.replace("<think>", "").replace("</think>", "").strip()
        
        # Return cleaned content or fallback message
        return content if content else "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def initialize_chatbot():
    """Initialize the chatbot components"""
    try:
        # Get API keys from environment variables
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        
        if not pinecone_api_key:
            st.error("Please set PINECONE_API_KEY in your .env file")
            return None
        
        # Initialize Pinecone
        pc = Pinecone(api_key=pinecone_api_key)
        
        # Load PDF
        with st.spinner("Loading PDF document..."):
            loader = PyPDFLoader('Rahul C (3).pdf')
            documents = loader.load()
        
        # Split documents with better parameters
        with st.spinner("Processing document chunks..."):
            text_splitter = CharacterTextSplitter(
                chunk_size=1000,  # Increased for better context
                chunk_overlap=200
            )
            texts = text_splitter.split_documents(documents)
        
        # Initialize embeddings
        with st.spinner("Loading embeddings model..."):
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        
        # Initialize vector store
        with st.spinner("Setting up vector database..."):
            index_name = os.getenv("INDEX_NAME")
            vectordb = PineconeVectorStore.from_documents(
                documents=texts,
                embedding=embeddings,
                index_name=index_name,
                pinecone_api_key=pinecone_api_key
            )
        
        #Qwen LLM
        with st.spinner("Loading Qwen language model..."):
            llm = QwenLLM()
        
        #QA chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm,
            vectordb.as_retriever(search_kwargs={'k': 5}),
            return_source_documents=True
        )
        
        st.success("✅ Chatbot with Qwen initialized successfully!")
        return qa_chain
        
    except Exception as e:
        st.error(f"Error initializing chatbot: {e}")
        st.exception(e)  # This will show the full traceback
        return None

# Initialize chatbot
if st.session_state.qa_chain is None:
    st.session_state.qa_chain = initialize_chatbot()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your PDF"):
    if st.session_state.qa_chain is None:
        st.error("Please initialize the chatbot first.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.qa_chain.invoke({
                        'question': prompt, 
                        'chat_history': st.session_state.chat_history
                    })
                    response = result['answer']
                    
                    # Update chat history
                    st.session_state.chat_history.append((prompt, response))
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    st.markdown(response)
                    
                    # Optionally show source documents
                    if 'source_documents' in result and result['source_documents']:
                        with st.expander("📄 Source Documents"):
                            for i, doc in enumerate(result['source_documents']):
                                st.write(f"**Source {i+1}:**")
                                st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                                st.write("---")
                    
                except Exception as e:
                    error_msg = f"Error generating response: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ Information")
    st.markdown("""
    This chatbot uses **Qwen 3-0.6B** model to answer questions about your PDF document.
    
    **Features:**
    - Local Qwen model (thinking mode disabled for stability)
    - Vector search through PDF content
    - Conversational memory
    - Source document references
    
    **Setup Required:**
    1. Ensure `Rahul C (3).pdf` is in the project directory
    2. Set PINECONE_API_KEY in your .env file
    3. Install required packages:
       ```
       pip install streamlit transformers torch
       pip install langchain langchain-community
       pip install pinecone-client langchain-pinecone
       pip install sentence-transformers pypdf
       ```
    """)
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()
    
    # Model info
    st.header("🤖 Model Info")
    st.markdown("""
    **Model:** Qwen 3-0.6B
    **Capabilities:** 
    - Conversational AI
    - Document understanding
    - RAG (Retrieval Augmented Generation)
    
    **Optimizations:**
    - Thinking mode disabled for faster responses
    - Reduced max tokens for better performance
    - Early stopping enabled
    """)