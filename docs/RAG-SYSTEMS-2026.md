# 🔮 RAG Systems - Production Guide 2026

<div align="center">

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="600">

**Building Production-Grade Retrieval Augmented Generation**

</div>

---

## 🎯 What is RAG?

**Retrieval Augmented Generation (RAG)** combines information retrieval with language generation. Instead of relying only on the LLM's training data, RAG retrieves relevant documents and uses them as context for generation.

### Why RAG in 2026?

- 📚 **Accurate**: Ground responses in real documents
- 🔄 **Up-to-date**: Access latest information without retraining
- 💰 **Cost-Effective**: Smaller models + retrieval = better results
- 🎯 **Specific**: Domain knowledge without fine-tuning
- 🔐 **Private**: Keep sensitive data in your infrastructure

---

## 🏗️ Modern RAG Architecture (2026)

```
User Query
    ↓
Query Enhancement (Rewrite, Expand, Decompose)
    ↓
Hybrid Retrieval (Vector + Keyword + Knowledge Graph)
    ↓
Reranking (Cross-Encoder, ColBERT)
    ↓
Context Compression & Selection
    ↓
LLM Generation (with Retrieved Context)
    ↓
Response Validation & Citation
    ↓
Final Response
```

---

## 🚀 Advanced RAG Techniques (2026)

### 1. **Query Optimization**

#### HyDE (Hypothetical Document Embeddings)
```python
from langchain.chains import HypotheticalDocumentEmbedder

# Generate hypothetical answer first
hyde_chain = HypotheticalDocumentEmbedder.from_llm(
    llm=ChatOpenAI(model="gpt-4-turbo"),
    base_embeddings=OpenAIEmbeddings()
)

# Use hypothetical answer for retrieval
query = "How does photosynthesis work?"
hyde_embeddings = await hyde_chain.aembed_query(query)
docs = vector_store.similarity_search_by_vector(hyde_embeddings, k=5)
```

#### Multi-Query Retrieval
```python
from langchain.retrievers import MultiQueryRetriever

# Generate multiple query variations
retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=ChatOpenAI(model="gpt-4-turbo"),
    include_original=True
)

# Retrieves using all query variations
docs = await retriever.aget_relevant_documents(
    "What are the benefits of exercise?"
)
```

#### Step-Back Prompting
```python
async def step_back_retrieval(query: str):
    """First ask high-level question, then specific"""
    
    # Generate step-back question
    step_back_prompt = f"""
    Given: {query}
    
    Generate a more general, high-level question that would help
    answer this specific question.
    """
    
    general_query = await llm.apredict(step_back_prompt)
    
    # Retrieve for both queries
    general_docs = await retriever.aget_relevant_documents(general_query)
    specific_docs = await retriever.aget_relevant_documents(query)
    
    # Combine and deduplicate
    all_docs = general_docs + specific_docs
    unique_docs = list({doc.page_content: doc for doc in all_docs}.values())
    
    return unique_docs
```

---

### 2. **Advanced Chunking Strategies**

#### Semantic Chunking
```python
from langchain.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

# Split based on semantic similarity
splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",  # or "standard_deviation"
    breakpoint_threshold_amount=95
)

chunks = splitter.create_documents([long_document])
```

#### Parent-Child Chunking
```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

# Small chunks for retrieval, large chunks for context
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

store = InMemoryStore()
retriever = ParentDocumentRetriever(
    vectorstore=vector_store,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

retriever.add_documents(documents)
```

#### Sentence Window Retrieval
```python
from llama_index.node_parser import SentenceWindowNodeParser
from llama_index.indices.postprocessor import MetadataReplacementPostProcessor

# Store small sentences, return large windows
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,  # sentences before and after
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)

nodes = node_parser.get_nodes_from_documents(documents)

# Post-processor to replace with full window
postprocessor = MetadataReplacementPostProcessor(
    target_metadata_key="window"
)
```

---

### 3. **Hybrid Search (Vector + Keyword)**

```python
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import BM25Retriever

# Vector search
vector_retriever = vector_store.as_retriever(search_kwargs={"k": 10})

# Keyword search (BM25)
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 10

# Combine with weights
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.3, 0.7],  # 30% keyword, 70% vector
    c=60  # rank fusion constant
)

docs = ensemble_retriever.get_relevant_documents(query)
```

---

### 4. **Reranking for Precision**

#### Cross-Encoder Reranking
```python
from sentence_transformers import CrossEncoder

# Initialize reranker
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

async def rerank_documents(query: str, docs: List[Document], top_k: int = 3):
    """Rerank retrieved documents using cross-encoder"""
    
    # Score all query-document pairs
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)
    
    # Sort by score
    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Return top-k
    return [doc for doc, score in scored_docs[:top_k]]
```

#### Cohere Rerank (2026 SOTA)
```python
import cohere

co = cohere.Client("YOUR_API_KEY")

async def cohere_rerank(query: str, docs: List[str], top_n: int = 3):
    """Use Cohere Rerank 3 for best results"""
    
    results = co.rerank(
        model="rerank-english-v3.0",  # or rerank-multilingual-v3.0
        query=query,
        documents=docs,
        top_n=top_n,
        return_documents=True
    )
    
    return [result.document.text for result in results.results]
```

---

### 5. **Advanced Embedding Strategies**

```python
# Late Chunking (2026 technique)
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en')
tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v2-base-en')

def late_chunking_embed(text: str, chunk_size: int = 512):
    """Embed full document, then split embeddings"""
    
    # Tokenize full document
    tokens = tokenizer(text, return_tensors='pt', truncation=False)
    
    # Get embeddings for full document
    with torch.no_grad():
        outputs = model(**tokens)
        embeddings = outputs.last_hidden_state[0]  # [seq_len, hidden_dim]
    
    # Split embeddings based on chunk boundaries
    chunk_embeddings = []
    for i in range(0, len(embeddings), chunk_size):
        chunk = embeddings[i:i+chunk_size]
        chunk_embedding = torch.mean(chunk, dim=0)  # Average pooling
        chunk_embeddings.append(chunk_embedding)
    
    return chunk_embeddings
```

---

### 6. **Context Compression**

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Extract only relevant parts
compressor = LLMChainExtractor.from_llm(
    llm=ChatOpenAI(model="gpt-4-turbo")
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vector_store.as_retriever()
)

# Returns only relevant excerpts
compressed_docs = compression_retriever.get_relevant_documents(query)
```

---

## 🎯 Production RAG Pipeline (Complete Example)

```python
from typing import List, Dict, Any
import asyncio
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.retrievers import EnsembleRetriever
from langchain.chains import RetrievalQA
import cohere

class ProductionRAGSystem:
    """Production-grade RAG with all 2026 techniques"""
    
    def __init__(self):
        # Initialize components
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
        self.reranker = cohere.Client(api_key="YOUR_KEY")
        
        # Vector store
        self.vector_store = PineconeVectorStore(
            index_name="production-index",
            embedding=self.embeddings
        )
        
    async def query(self, question: str) -> Dict[str, Any]:
        """Complete RAG pipeline"""
        
        # 1. Query Enhancement
        enhanced_queries = await self.enhance_query(question)
        
        # 2. Hybrid Retrieval
        all_docs = []
        for query in enhanced_queries:
            docs = await self.hybrid_retrieve(query, k=20)
            all_docs.extend(docs)
        
        # Deduplicate
        unique_docs = self.deduplicate_docs(all_docs)
        
        # 3. Reranking
        reranked_docs = await self.rerank(question, unique_docs, top_n=5)
        
        # 4. Context Compression
        compressed_context = await self.compress_context(
            question, reranked_docs
        )
        
        # 5. Generation
        response = await self.generate_answer(
            question, compressed_context
        )
        
        # 6. Add citations
        response_with_citations = self.add_citations(
            response, reranked_docs
        )
        
        return {
            "answer": response_with_citations,
            "sources": [doc.metadata for doc in reranked_docs],
            "confidence": self.calculate_confidence(response, reranked_docs)
        }
    
    async def enhance_query(self, query: str) -> List[str]:
        """Generate query variations"""
        prompt = f"""
        Generate 3 different ways to ask this question:
        "{query}"
        
        Return as JSON list.
        """
        
        response = await self.llm.apredict(prompt)
        variations = json.loads(response)
        
        return [query] + variations  # Include original
    
    async def hybrid_retrieve(self, query: str, k: int = 10) -> List[Document]:
        """Combine vector + keyword search"""
        
        # Vector search
        vector_docs = await self.vector_store.asimilarity_search(query, k=k)
        
        # Keyword search (BM25) - simplified
        keyword_docs = self.bm25_search(query, k=k)
        
        # Merge with rank fusion
        return self.reciprocal_rank_fusion(
            [vector_docs, keyword_docs],
            k=k
        )
    
    async def rerank(self, query: str, docs: List[Document], 
                     top_n: int = 5) -> List[Document]:
        """Rerank using Cohere"""
        
        doc_texts = [doc.page_content for doc in docs]
        
        results = self.reranker.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=doc_texts,
            top_n=top_n,
            return_documents=True
        )
        
        # Return reranked documents
        reranked = []
        for result in results.results:
            original_doc = docs[result.index]
            original_doc.metadata['rerank_score'] = result.relevance_score
            reranked.append(original_doc)
        
        return reranked
    
    async def compress_context(self, query: str, 
                               docs: List[Document]) -> str:
        """Extract only relevant parts"""
        
        context_parts = []
        for doc in docs:
            prompt = f"""
            Query: {query}
            
            Document: {doc.page_content}
            
            Extract only the parts relevant to answering the query.
            If nothing is relevant, return "NOT RELEVANT".
            """
            
            relevant_part = await self.llm.apredict(prompt)
            
            if relevant_part != "NOT RELEVANT":
                context_parts.append(relevant_part)
        
        return "\n\n".join(context_parts)
    
    async def generate_answer(self, question: str, 
                             context: str) -> str:
        """Generate final answer"""
        
        prompt = f"""
        Context:
        {context}
        
        Question: {question}
        
        Provide a detailed answer based ONLY on the context above.
        If you cannot answer from the context, say "I don't have enough information."
        
        Answer:
        """
        
        return await self.llm.apredict(prompt)
    
    def add_citations(self, answer: str, sources: List[Document]) -> str:
        """Add inline citations"""
        
        # Simple implementation - add source numbers
        cited_answer = answer
        for i, source in enumerate(sources, 1):
            source_title = source.metadata.get('title', 'Unknown')
            cited_answer += f"\n[{i}] {source_title}"
        
        return cited_answer
    
    def calculate_confidence(self, answer: str, 
                            sources: List[Document]) -> float:
        """Estimate confidence score"""
        
        # Factors: number of sources, rerank scores, answer length
        if not sources:
            return 0.0
        
        avg_rerank_score = sum(
            doc.metadata.get('rerank_score', 0) for doc in sources
        ) / len(sources)
        
        source_factor = min(len(sources) / 5, 1.0)  # More sources = better
        
        return (avg_rerank_score * 0.7 + source_factor * 0.3)
```

---

## 📊 RAG Evaluation Metrics (2026)

```python
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision
)

# Evaluate RAG system
results = evaluate(
    dataset=eval_dataset,
    metrics=[
        answer_relevancy,  # Answer addresses question?
        faithfulness,      # Answer grounded in context?
        context_recall,    # All relevant docs retrieved?
        context_precision  # Only relevant docs retrieved?
    ]
)

print(results)
```

---

## 🔥 2026 Best Practices

1. **Always use hybrid search** (vector + keyword)
2. **Rerank retrieved documents** before generation
3. **Compress context** to stay within token limits
4. **Add citations** for transparency
5. **Monitor retrieval quality** continuously
6. **Use semantic chunking** over fixed-size
7. **Implement query enhancement** (HyDE, multi-query)
8. **Cache embeddings** for frequently accessed docs

---

*Building the future of intelligent information retrieval* 🚀
