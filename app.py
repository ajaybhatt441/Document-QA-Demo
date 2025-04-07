import streamlit as st
import os
import json
import fitz  # PyMuPDF
import re
from pathlib import Path
import tempfile
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import networkx as nx
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set page configuration
st.set_page_config(page_title="Book Q&A System", layout="wide")

# Authentication function
def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == "AJAY@2025":
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password
        else:
            st.session_state["password_correct"] = False

    # Return True if password is correct or has been previously verified
    if "password_correct" in st.session_state and st.session_state["password_correct"]:
        return True

    # Show input for password
    st.title("📚 Book Q&A System")
    st.subheader("Password Protected")
    st.text_input(
        "Please enter the password", 
        type="password", 
        on_change=password_entered, 
        key="password"
    )
    
    if "password_correct" in st.session_state:
        if not st.session_state["password_correct"]:
            st.error("😕 Incorrect password")
    return False

# Get API key from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("API key not found in environment variables. Please check your .env file.")
    st.stop()

# Set API key for Google Generative AI
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)

# Create necessary directories
def setup_directories():
    # Create the processed_books directory if it doesn't exist
    data_dir = Path("processed_books")
    data_dir.mkdir(exist_ok=True)
    return data_dir

# Function to process PDF and extract chapters
def process_pdf(file_path):
    """Extract text and chapter information from PDF"""
    st.write(f"Processing PDF...")
    st.session_state.current_step = 2  # Update current step
    
    doc = fitz.open(file_path)
    chapters = {}
    current_text = []
    current_chapter = "Introduction"  # Default chapter name
    chapter_pattern = re.compile(r"(?:chapter|section)\s+\d+[:]?\s*(.*)", re.IGNORECASE)
    
    progress_bar = st.progress(0)
    total_pages = len(doc)
    
    for page_num in range(total_pages):
        # Update progress bar
        progress_bar.progress((page_num + 1) / total_pages)
        
        page = doc.load_page(page_num)
        text = page.get_text()
        
        # Check for chapter headings
        lines = text.split('\n')
        for line in lines:
            match = chapter_pattern.match(line.strip())
            if match or line.strip().upper() == line.strip() and len(line.strip()) > 5 and len(line.strip()) < 50:
                # Save previous chapter
                if current_text:
                    if current_chapter in chapters:
                        chapters[current_chapter]["text"] += " ".join(current_text)
                    else:
                        chapters[current_chapter] = {
                            "text": " ".join(current_text),
                            "start_page": chapters[current_chapter]["start_page"] if current_chapter in chapters else page_num
                        }
                    current_text = []
                
                # Start new chapter
                if match:
                    current_chapter = match.group(1).strip()
                else:
                    current_chapter = line.strip()
                    
                if current_chapter not in chapters:
                    chapters[current_chapter] = {"text": "", "start_page": page_num}
            else:
                current_text.append(line)
        
        # Add page separator
        current_text.append(f"[PAGE {page_num + 1}]")
    
    # Add the last chapter
    if current_text and current_chapter:
        if current_chapter in chapters:
            chapters[current_chapter]["text"] += " ".join(current_text)
        else:
            chapters[current_chapter] = {
                "text": " ".join(current_text),
                "start_page": chapters[current_chapter]["start_page"] if current_chapter in chapters else len(doc) - 1
            }
    
    doc.close()
    st.write(f"Found {len(chapters)} chapters")
    return chapters

# Function to process TXT file
def process_txt(file_path):
    """Extract text and attempt to identify chapters from TXT file"""
    st.write(f"Processing TXT file...")
    st.session_state.current_step = 2  # Update current step
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        content = file.read()
    
    # Split content into lines for processing
    lines = content.split('\n')
    
    chapters = {}
    current_text = []
    current_chapter = "Introduction"  # Default chapter name
    chapter_pattern = re.compile(r"(?:chapter|section)\s+\d+[:]?\s*(.*)", re.IGNORECASE)
    line_count = 0
    total_lines = len(lines)
    
    progress_bar = st.progress(0)
    
    for i, line in enumerate(lines):
        # Update progress
        if i % 100 == 0:  # Update every 100 lines for performance
            progress_bar.progress(min(i / total_lines, 1.0))
        
        line_count += 1
        # Check for chapter headings
        match = chapter_pattern.match(line.strip())
        
        # Consider all-caps lines of reasonable length as potential chapter headers
        is_header = (line.strip().upper() == line.strip() and 
                     len(line.strip()) > 5 and 
                     len(line.strip()) < 50)
        
        if match or is_header:
            # Save previous chapter
            if current_text:
                if current_chapter in chapters:
                    chapters[current_chapter]["text"] += " ".join(current_text)
                else:
                    chapters[current_chapter] = {
                        "text": " ".join(current_text),
                        "start_page": chapters[current_chapter]["start_page"] if current_chapter in chapters else max(1, line_count // 40)  # Estimate page
                    }
                current_text = []
            
            # Start new chapter
            if match:
                current_chapter = match.group(1).strip()
            else:
                current_chapter = line.strip()
                
            if current_chapter not in chapters:
                chapters[current_chapter] = {"text": "", "start_page": max(1, line_count // 40)}  # Estimate page numbers (40 lines per page)
        else:
            if line.strip():  # Only add non-empty lines
                current_text.append(line)
        
        # Add pseudo-page markers every ~40 lines
        if line_count % 40 == 0:
            page_num = line_count // 40
            current_text.append(f"[PAGE {page_num}]")
    
    # Add the last chapter
    if current_text and current_chapter:
        if current_chapter in chapters:
            chapters[current_chapter]["text"] += " ".join(current_text)
        else:
            chapters[current_chapter] = {
                "text": " ".join(current_text),
                "start_page": chapters[current_chapter]["start_page"] if current_chapter in chapters else max(1, line_count // 40)
            }
    
    # If no chapters were identified, create a single chapter with all content
    if not chapters or len(chapters) == 1 and "Introduction" in chapters:
        st.write("No clear chapter structure detected. Processing as a single document.")
        estimated_pages = max(1, line_count // 40)
        chapters = {
            "Full Text": {
                "text": content,
                "start_page": 1
            }
        }
    
    st.write(f"Identified {len(chapters)} sections/chapters")
    return chapters

# Function to create chunks
def create_chunks(chapters):
    """Create chunks from chapters with metadata"""
    st.write("Creating chunks from chapters...")
    st.session_state.current_step = 3  # Update current step
    
    chunks = []
    chapter_info = {}
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    for chapter_name, info in chapters.items():
        chapter_text = info["text"]
        start_page = info["start_page"]
        
        # Extract page numbers from text
        page_numbers = []
        for match in re.finditer(r'\[PAGE (\d+)\]', chapter_text):
            page_numbers.append(int(match.group(1)))
        
        # Replace page markers to clean text
        clean_text = re.sub(r'\[PAGE \d+\]', '', chapter_text)
        
        # Create chunks
        chapter_chunks = text_splitter.create_documents([clean_text])
        
        # Add metadata to each chunk
        for i, chunk in enumerate(chapter_chunks):
            # Determine the page number for this chunk
            page = start_page + 1
            if page_numbers:
                # Find the closest page marker before this chunk
                text_so_far = clean_text[:clean_text.find(chunk.page_content)]
                for p in page_numbers:
                    page_marker = f"[PAGE {p}]"
                    if page_marker in chapter_text[:chapter_text.find(chunk.page_content)]:
                        page = p
            
            chunk.metadata = {
                "chapter": chapter_name,
                "page": page,
                "chunk_id": f"{chapter_name}_{i}"
            }
            chunks.append(chunk)
        
        # Store chapter information
        chapter_info[chapter_name] = {
            "start_page": start_page + 1,
            "chunk_ids": [f"{chapter_name}_{i}" for i in range(len(chapter_chunks))]
        }
    
    st.write(f"Created {len(chunks)} chunks")
    return chunks, chapter_info

# Function to create ontology from text
def create_ontology(chunks, book_name):
    """Create an ontology (knowledge graph) from the book chunks"""
    st.write("Creating ontology from book content...")
    
    # Combine text from chunks for analysis
    combined_text = " ".join([chunk.page_content for chunk in chunks])
    
    # Use Gemini to extract key concepts and relationships
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)  # Lower temperature for more structured output
    
    ontology_prompt = ChatPromptTemplate.from_template("""
    You are an expert at analyzing text and extracting structured knowledge.
    I need you to create an ontology from the following text.
    
    Please identify:
    1. The 10-15 most important concepts/entities in the text
    2. The relationships between these concepts
    
    Return ONLY a valid JSON object with this exact structure:
    {{
      "concepts": [
        {{"id": "concept1", "name": "Full Name of Concept", "description": "Brief description"}},
        ...
      ],
      "relationships": [
        {{"source": "concept1", "target": "concept2", "type": "relationship type", "description": "description of relationship"}}
        ...
      ]
    }}
    
    Text to analyze:
    {text}
    
    IMPORTANT: Return ONLY the JSON without any explanatory text, markdown code blocks, or other content.
    Make sure the JSON is properly formatted with double quotes around all keys and string values.
    """
    )
    
    # Create a smaller sample of text to work with (to avoid token limits)
    sample_text = combined_text[:1000000]  # Reduced to 80k characters to help model processing
    
    try:
        response = ontology_prompt | model | StrOutputParser()
        result = response.invoke({"text": sample_text})
        
        # Clean the result to ensure proper JSON format
        cleaned_result = result.strip()
        
        # Remove any markdown code block indicators if present
        if cleaned_result.startswith("```json"):
            cleaned_result = cleaned_result[7:]
        if cleaned_result.startswith("```"):
            cleaned_result = cleaned_result[3:]
        if cleaned_result.endswith("```"):
            cleaned_result = cleaned_result[:-3]
            
        cleaned_result = cleaned_result.strip()
        
        # Try to parse the JSON
        try:
            ontology_data = json.loads(cleaned_result)
            
            # Validate the structure
            if not isinstance(ontology_data, dict) or "concepts" not in ontology_data or "relationships" not in ontology_data:
                raise ValueError("Invalid JSON structure: missing required keys")
            
            # Create a graph visualization
            G = nx.DiGraph()
            
            # Add nodes (concepts)
            for concept in ontology_data["concepts"]:
                G.add_node(concept["id"], label=concept["name"], description=concept["description"])
            
            # Add edges (relationships)
            for rel in ontology_data["relationships"]:
                G.add_edge(rel["source"], rel["target"], 
                          label=rel["type"], 
                          description=rel["description"])
            
            return ontology_data, G
            
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse ontology response as JSON: {str(e)}")
            st.error("Raw response (first 500 chars): " + cleaned_result[:500] + "...")
            
            # Fall back to a simpler approach - try to extract just concepts
            fallback_data = {
                "concepts": [
                    {"id": f"concept{i}", "name": f"Key Concept {i}", "description": "Extracted from text"}
                    for i in range(1, 6)  # Just create 5 placeholder concepts
                ],
                "relationships": []
            }
            
            # Create a basic graph
            G = nx.DiGraph()
            for concept in fallback_data["concepts"]:
                G.add_node(concept["id"], label=concept["name"], description=concept["description"])
            
            st.warning("Using simplified fallback ontology due to parsing error.")
            return fallback_data, G
            
    except Exception as e:
        st.error(f"Error creating ontology: {str(e)}")
        st.error("Trying fallback approach...")
        
        # Use a very simple approach as fallback
        try:
            # Create a simpler prompt that asks for just a few concepts
            simple_prompt = ChatPromptTemplate.from_template("""
            Extract 5-10 key concepts from this text. 
            Format each concept as a simple line with a dash, followed by the concept name. 
            Don't include any explanation or additional text.
            
            Text to analyze:
            {text}
            """)
            
            concept_response = simple_prompt | model | StrOutputParser()
            concept_result = concept_response.invoke({"text": sample_text})
            
            # Parse the concepts from the result
            concept_lines = [line.strip() for line in concept_result.split('\n') if line.strip().startswith('-')]
            concepts = [line[1:].strip() for line in concept_lines]
            
            # Create a basic ontology structure
            basic_ontology = {
                "concepts": [
                    {"id": f"concept{i}", "name": concept, "description": "Extracted from text"}
                    for i, concept in enumerate(concepts[:10], 1)
                ],
                "relationships": []
            }
            
            # Create a basic graph
            G = nx.DiGraph()
            for concept in basic_ontology["concepts"]:
                G.add_node(concept["id"], label=concept["name"], description=concept["description"])
            
            st.warning("Created a simplified ontology with basic concepts.")
            return basic_ontology, G
            
        except Exception as e2:
            st.error(f"Fallback approach also failed: {str(e2)}")
            return None, None

# Function to visualize ontology graph
def visualize_ontology(G):
    """Create a visualization of the ontology graph"""
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=1500, font_size=10, font_weight='bold')
    
    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    # Save the figure
    plt.tight_layout()
    
    # Save to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(temp_file.name, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return temp_file.name

# Function to create vectorstore
def create_vectorstore(chunks):
    """Create and save FAISS vectorstore from chunks"""
    st.write("Creating vector store...")
    st.session_state.current_step = 4  # Update current step
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    return vectorstore

# Function to save documents as JSON
def save_documents_as_json(chunks, file_path):
    """Save document chunks as JSON instead of pickle"""
    serializable_chunks = []
    for chunk in chunks:
        serializable_chunks.append({
            "page_content": chunk.page_content,
            "metadata": chunk.metadata
        })
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(serializable_chunks, f, ensure_ascii=False, indent=2)

# Function to save processed book data
def save_processed_book(book_name, chunks, chapter_info, vectorstore, ontology_data=None):
    """Save processed book data to disk"""
    st.write(f"Saving processed book data for: {book_name}")
    st.session_state.current_step = 5  # Update current step
    
    # Create data directory if it doesn't exist
    data_dir = Path("processed_books")
    data_dir.mkdir(exist_ok=True)
    book_dir = data_dir / book_name
    book_dir.mkdir(exist_ok=True)
    
    # Save chunks and chapter info as JSON instead of pickle
    save_documents_as_json(chunks, book_dir / "chunks.json")
    
    with open(book_dir / "chapter_info.json", "w", encoding="utf-8") as f:
        json.dump(chapter_info, f, ensure_ascii=False, indent=2)
    
    # Save vectorstore
    vectorstore.save_local(str(book_dir / "vectorstore"))
    
    # Save ontology data if available
    if ontology_data:
        with open(book_dir / "ontology.json", "w", encoding="utf-8") as f:
            json.dump(ontology_data, f, ensure_ascii=False, indent=2)
    
    # Save metadata about the book (without API key)
    metadata = {
        "num_chunks": len(chunks),
        "num_chapters": len(chapter_info),
        "chapters": list(chapter_info.keys()),
        "has_ontology": ontology_data is not None
    }
    
    with open(book_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    st.write(f"Book processed and saved to {book_dir}")
    return str(book_dir)

# Function to process a book
def process_book(file_path, book_name, file_type):
    """Process a book from PDF or TXT to structured format"""
    with st.spinner("Processing book..."):
        st.session_state.current_step = 1  # Set initial step
        
        # Extract chapters based on file type
        if file_type == 'pdf':
            chapters = process_pdf(file_path)
        else:  # txt
            chapters = process_txt(file_path)
        
        # Create chunks with metadata
        chunks, chapter_info = create_chunks(chapters)
        
        # Create ontology from chunks
        ontology_data, ontology_graph = create_ontology(chunks, book_name)
        
        # Create vector store
        vectorstore = create_vectorstore(chunks)
        
        # Save all processed data
        book_dir = save_processed_book(book_name, chunks, chapter_info, vectorstore, ontology_data)
        
        # Reset step
        st.session_state.current_step = 0
        
        return book_dir, ontology_graph

# Function to convert JSON documents back to LangChain Documents
def json_to_documents(json_docs):
    documents = []
    for doc in json_docs:
        documents.append(
            Document(
                page_content=doc["page_content"],
                metadata=doc["metadata"]
            )
        )
    return documents

# Function to load available books
def get_available_books():
    data_dir = Path("processed_books")
    if not data_dir.exists():
        return []
    
    return [d.name for d in data_dir.iterdir() if d.is_dir() and (d / "metadata.json").exists()]

# Function to load book data
def load_book_data(book_name):
    book_dir = Path("processed_books") / book_name
    
    # Load metadata
    with open(book_dir / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    # Load chapter info
    with open(book_dir / "chapter_info.json", "r", encoding="utf-8") as f:
        chapter_info = json.load(f)
    
    # Load chunks from JSON
    with open(book_dir / "chunks.json", "r", encoding="utf-8") as f:
        chunks_json = json.load(f)
    
    # Convert JSON back to Document objects
    chunks = json_to_documents(chunks_json)
    
    # Load vectorstore
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.load_local(str(book_dir / "vectorstore"), embeddings, allow_dangerous_deserialization=True)
    
    # Load ontology if available
    ontology_data = None
    if metadata.get("has_ontology", False) and (book_dir / "ontology.json").exists():
        with open(book_dir / "ontology.json", "r", encoding="utf-8") as f:
            ontology_data = json.load(f)
    
    return metadata, chapter_info, chunks, vectorstore, ontology_data

# Function to combine multiple vectorstores
def combine_vectorstores(vectorstores, book_names):
    # We'll create a new combined vectorstore
    all_documents = []
    
    # For each vector store, we need to retrieve a sample of documents
    for i, vs in enumerate(vectorstores):
        book_name = book_names[i]
        
        # Load all chunks for this book from disk instead
        book_dir = Path("processed_books") / book_name
        with open(book_dir / "chunks.json", "r", encoding="utf-8") as f:
            chunks_json = json.load(f)
        
        # Convert to Document objects and add book name
        chunks = json_to_documents(chunks_json)
        for chunk in chunks:
            chunk.metadata["book_name"] = book_name
            all_documents.append(chunk)
    
    # Create a new vectorstore with all documents
    if all_documents:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        combined_vs = FAISS.from_documents(all_documents, embeddings)
        return combined_vs
    
    return None

# Function to set up the RAG pipeline
def setup_rag_pipeline(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # RAG prompt
    template = """You are a helpful assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise.
    Each piece of context includes information about which book it comes from.
    Mention the book name(s) in your answer to provide proper attribution.
    
    Question: {question}
    
    Context:
    {context}
    
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Initialize the Gemini model
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    
    # Create the RAG pipeline
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    
    return rag_chain

# Function to extract source information from relevant documents
def get_source_info(docs):
    sources = []
    for doc in docs:
        sources.append({
            "book": doc.metadata.get("book_name", "Unknown"),
            "chapter": doc.metadata["chapter"],
            "page": doc.metadata["page"]
        })
    return sources

# Initialize session states
if "books_loaded" not in st.session_state:
    st.session_state.books_loaded = False
if "current_books" not in st.session_state:
    st.session_state.current_books = []
if "combined_vectorstore" not in st.session_state:
    st.session_state.combined_vectorstore = None
if "books_metadata" not in st.session_state:
    st.session_state.books_metadata = {}
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Query Books"
if "current_step" not in st.session_state:
    st.session_state.current_step = 0
if "ontology_graphs" not in st.session_state:
    st.session_state.ontology_graphs = {}

# Setup directories
setup_directories()

# Main App - Check password before showing content
if check_password():
    # Main App content
    st.title("📚 Book Q&A System")

    # Create tabs for different functions
    tab1, tab2, tab3 = st.tabs(["Query Books", "Upload New Book", "Ontology Viewer"])

    # Add steps to sidebar
    with st.sidebar:
        st.subheader("Processing Steps")
        
        # Define the steps
        steps = [
            "📁 Upload Document",
            "📋 Extract Chapters",
            "✂️ Create Text Chunks",
            "🔍 Build Vector Database",
            "💾 Save Processed Book"
        ]
        
        # Display steps with different styling for current step
        for i, step in enumerate(steps):
            step_num = i + 1
            
            if st.session_state.current_step == step_num:
                # Highlight current step
                st.markdown(f"### **Step {step_num}: {step}** 🔄")
            else:
                st.markdown(f"Step {step_num}: {step}")
        
        # Display processing status
        if st.session_state.current_step > 0:
            st.success("Processing in progress...")
        
        # Book selection section
        st.divider()
        st.subheader("Select Books")
        
        # Book selection - now with multi-select
        available_books = get_available_books()
        if available_books:
            selected_books = st.multiselect("Choose books to query:", available_books)
            
            if st.button("Load Selected Books") and selected_books:
                with st.spinner(f"Loading {len(selected_books)} books..."):
                    # Reset previous state
                    st.session_state.books_metadata = {}
                    vectorstores = []
                    st.session_state.ontology_graphs = {}
                    
                    # Load each book
                    for book_name in selected_books:
                        metadata, chapter_info, chunks, vectorstore, ontology_data = load_book_data(book_name)
                        st.session_state.books_metadata[book_name] = {
                            "metadata": metadata,
                            "chapter_info": chapter_info,
                            "ontology_data": ontology_data
                        }
                        vectorstores.append(vectorstore)
                        
                        # Recreate ontology graph if data exists
                        if ontology_data:
                            G = nx.DiGraph()
                            # Add nodes (concepts)
                            for concept in ontology_data["concepts"]:
                                G.add_node(concept["id"], label=concept["name"], description=concept["description"])
                            
                            # Add edges (relationships)
                            for rel in ontology_data["relationships"]:
                                G.add_edge(rel["source"], rel["target"], 
                                        label=rel["type"], 
                                        description=rel["description"])
                            
                            st.session_state.ontology_graphs[book_name] = G
                    
                    # Combine vectorstores
                    combined_vs = combine_vectorstores(vectorstores, selected_books)
                    
                    # Store in session state
                    st.session_state.current_books = selected_books
                    st.session_state.combined_vectorstore = combined_vs
                    st.session_state.books_loaded = True
                    
                    st.success(f"Successfully loaded {len(selected_books)} books")
        else:
            st.warning("No processed books found. Please upload a book first in the 'Upload New Book' tab.")

    # Tab 1: Query Books
    with tab1:
        st.header("Query Books")
        
        # Main area for Q&A
        if st.session_state.books_loaded and st.session_state.current_books:
            # Display book information
            st.subheader("📚 Loaded Books")
            
            # Create metrics for all loaded books
            total_chapters = 0
            total_chunks = 0
            
            # Book information in expandable sections
            for book_name in st.session_state.current_books:
                book_data = st.session_state.books_metadata[book_name]
                total_chapters += book_data["metadata"]["num_chapters"]
                total_chunks += book_data["metadata"]["num_chunks"]
                
                with st.expander(f"Book: {book_name}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Chapters", book_data["metadata"]["num_chapters"])
                    with col2:
                        st.metric("Text Chunks", book_data["metadata"]["num_chunks"])
                    with col3:
                        has_ontology = book_data["metadata"].get("has_ontology", False)
                        st.metric("Ontology", "Yes" if has_ontology else "No")
                    
                    st.subheader("Chapters in this book")
                    for chapter in book_data["metadata"]["chapters"]:
                        st.write(f"- {chapter}")
            
            # Show totals
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Books", len(st.session_state.current_books))
            with col2:
                st.metric("Total Chapters", total_chapters)
            with col3:
                st.metric("Total Text Chunks", total_chunks)
            
            # Q&A section
            st.subheader("Ask a Question")
            
            # Input for question
            question = st.text_input("Enter your question about the book(s):")
            
            if question and st.button("Ask"):
                with st.spinner("Finding answer..."):
                    # Setup RAG pipeline
                    rag_chain = setup_rag_pipeline(st.session_state.combined_vectorstore)
                    
                    # Get the answer
                    answer = rag_chain.invoke(question)
                    
                    # Get the source documents for attribution
                    retriever = st.session_state.combined_vectorstore.as_retriever(search_kwargs={"k": 5})
                    docs = retriever.get_relevant_documents(question)
                    sources = get_source_info(docs)
                    
                    # Display the answer
                    st.subheader("Answer")
                    st.write(answer)
                    
                    # Display sources in an expandable section
                    with st.expander("Sources"):
                        for i, source in enumerate(sources):
                            st.write(f"Source {i+1}: {source['book']}, Chapter: {source['chapter']}, Page: {source['page']}")
            
            # Show a separator
            st.divider()
    
    # Tab 2: Upload New Book
    with tab2:
        st.header("Upload New Book")
        
        # Upload section
        uploaded_file = st.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"])
        
        if uploaded_file is not None:
            # Get file details
            file_type = uploaded_file.name.split('.')[-1].lower()
            book_name = st.text_input("Enter a name for this book:", 
                                    value=uploaded_file.name.split('.')[0])
            
            if st.button("Process Book"):
                # Save the uploaded file to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_type}') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    # Process the book
                    book_dir, ontology_graph = process_book(tmp_path, book_name, file_type)
                    
                    # Store the ontology graph if available
                    if ontology_graph:
                        st.session_state.ontology_graphs[book_name] = ontology_graph
                    
                    # Clean up the temporary file
                    os.unlink(tmp_path)
                    
                    # Show success message
                    st.success(f"Book '{book_name}' processed successfully! You can now select it from the sidebar.")
                    
                    # If ontology was created, suggest viewing it
                    if ontology_graph:
                        if st.button("View Ontology Visualization"):
                            st.session_state.active_tab = "Ontology Viewer"
                            st.experimental_rerun()
                    
                except Exception as e:
                    st.error(f"Error processing book: {str(e)}")
                    # Clean up the temporary file
                    os.unlink(tmp_path)
    
    # Tab 3: Ontology Viewer
    with tab3:
        st.header("Knowledge Graph Viewer")
        
        # Show ontology visualizations if available
        if st.session_state.ontology_graphs:
            # Let user select which book's ontology to view
            book_options = list(st.session_state.ontology_graphs.keys())
            selected_book = st.selectbox("Select a book to view its knowledge graph:", book_options)
            
            if selected_book in st.session_state.ontology_graphs:
                G = st.session_state.ontology_graphs[selected_book]
                
                # Generate visualization
                viz_path = visualize_ontology(G)
                
                # Display the visualization
                st.image(viz_path, caption=f"Knowledge Graph for {selected_book}", use_container_width=True)
                
                # Display concepts and relationships
                st.subheader("Key Concepts")
                
                # Get ontology data for the selected book
                ontology_data = st.session_state.books_metadata[selected_book]["ontology_data"]
                
                # Create tables for concepts and relationships
                if ontology_data:
                    # Concepts table
                    concept_data = []
                    for concept in ontology_data["concepts"]:
                        concept_data.append([concept["name"], concept["description"]])
                    
                    st.table({"Concept": [c[0] for c in concept_data], 
                             "Description": [c[1] for c in concept_data]})
                    
                    # Relationships table
                    st.subheader("Relationships Between Concepts")
                    rel_data = []
                    for rel in ontology_data["relationships"]:
                        # Find source and target concept names
                        source_name = next((c["name"] for c in ontology_data["concepts"] if c["id"] == rel["source"]), rel["source"])
                        target_name = next((c["name"] for c in ontology_data["concepts"] if c["id"] == rel["target"]), rel["target"])
                        
                        rel_data.append([source_name, rel["type"], target_name, rel["description"]])
                    
                    st.table({"Source": [r[0] for r in rel_data],
                             "Relationship": [r[1] for r in rel_data],
                             "Target": [r[2] for r in rel_data],
                             "Description": [r[3] for r in rel_data]})
                else:
                    st.warning("No detailed ontology data available for this book.")
        else:
            st.info("No knowledge graphs available. Process a book with ontology creation to view it here.")
    
    # Add a Chat tab or section
    if "active_tab" in st.session_state and st.session_state.active_tab == "Chat":
        st.subheader("Chat with your Books")
        
        # Initialize chat history if not exists
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        user_input = st.chat_input("Ask something about your books:")
        
        if user_input:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Display user message
            with st.chat_message("user"):
                st.write(user_input)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    if st.session_state.books_loaded:
                        # Use chat-optimized prompt for better conversation
                        chat_prompt = ChatPromptTemplate.from_template("""
                        You are a helpful assistant that answers questions about books.
                        Use the following pieces of retrieved context to answer the question.
                        If you don't know the answer, just say that you don't know.
                        Keep the answer conversational and helpful.
                        
                        Chat History:
                        {chat_history}
                        
                        Question: {question}
                        
                        Context:
                        {context}
                        
                        Answer:
                        """)
                        
                        # Setup RAG pipeline for chat
                        retriever = st.session_state.combined_vectorstore.as_retriever(search_kwargs={"k": 5})
                        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
                        
                        # Format chat history
                        chat_history_str = "\n".join([
                            f"{msg['role'].capitalize()}: {msg['content']}" 
                            for msg in st.session_state.chat_history[:-1]  # Exclude current message
                        ])
                        
                        # Get context documents
                        docs = retriever.get_relevant_documents(user_input)
                        context_str = "\n".join([
                            f"From '{doc.metadata.get('book_name', 'Unknown')}', Chapter '{doc.metadata['chapter']}', Page {doc.metadata['page']}:\n{doc.page_content}\n"
                            for doc in docs
                        ])
                        
                        # Generate response
                        response = chat_prompt.format(
                            chat_history=chat_history_str,
                            question=user_input,
                            context=context_str
                        )
                        
                        response_text = model.invoke(response).content
                        st.write(response_text)
                        
                        # Add assistant response to chat history
                        st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                    else:
                        st.write("Please load books before starting a chat.")
                        st.session_state.chat_history.append({"role": "assistant", "content": "Please load books before starting a chat."})
    
    # Add a button to return to the main Query interface from Chat
    if "active_tab" in st.session_state and st.session_state.active_tab == "Chat":
        if st.button("Return to Query Interface"):
            st.session_state.active_tab = "Query Books"
            st.experimental_rerun()
    
    # Main function to run the app
    def main():
        st.sidebar.header("Book Q&A System")
        st.sidebar.write("Upload books and ask questions about their content.")
        
        # Add additional information in the sidebar
        st.sidebar.divider()
        st.sidebar.subheader("About")
        st.sidebar.info("""
        This application allows you to:
        - Upload PDF or TXT books
        - Process them into searchable content
        - Ask questions about one or multiple books
        - View the knowledge graph of concepts
        """)
    
    # Run main function
    if __name__ == "__main__":
        main()