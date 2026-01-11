"""
Indexing & Retrieval Module - Handles embedding generation and semantic search.
"""
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Set


class DualQueryIndexer:
    """Manages semantic indexing and dual-query retrieval for narrative segments."""
    
    def __init__(self, model_name: str = 'BAAI/bge-base-en-v1.5'):
        """
        Initialize the indexer with sentence transformer model.
        
        Args:
            model_name: HuggingFace model identifier for embeddings
        """
        self.model = SentenceTransformer(model_name , device= "cpu")
        self.index = None
        self.chunks = []
        self.chunk_positions = []
        self.dimension = None
    
    def build_index(self, chunks: List[Tuple[str, int]]) -> None:
        """
        Build FAISS index from text chunks.
        
        Args:
            chunks: List of (chunk_text, position) tuples
        """
        self.chunks = [chunk[0] for chunk in chunks]
        self.chunk_positions = [chunk[1] for chunk in chunks]
        
        print(f"Encoding {len(self.chunks)} chunks with {self.model.get_sentence_embedding_dimension()}-dim embeddings...")
        
        # Generate embeddings
        embeddings = self.model.encode(
            self.chunks,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True  # Normalize for inner product = cosine similarity
        )
        
        # Create FAISS index with inner product (IP) for normalized vectors
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings.astype('float32'))
        
        print(f"Index built with {self.index.ntotal} vectors")
    
    def dual_query_search(
        self,
        backstory_claim: str,
        character_name: str,
        top_k: int = 8
    ) -> List[Tuple[str, float, int]]:
        """
        Perform dual-query retrieval with deduplication.
        
        Executes two searches:
        1. Raw backstory claim
        2. Character name + backstory claim
        
        Then deduplicates and returns top_k results.
        
        Args:
            backstory_claim: The claim to verify
            character_name: Name of the character
            top_k: Number of top results to return
            
        Returns:
            List of (chunk_text, similarity_score, position) tuples
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Query 1: Raw backstory claim
        query1 = backstory_claim
        
        # Query 2: Character-focused query
        query2 = f"{character_name} {backstory_claim}"
        
        # Encode both queries
        query_embeddings = self.model.encode(
            [query1, query2],
            normalize_embeddings=True
        )
        
        # Search with more candidates than needed (to handle deduplication)
        search_k = top_k * 3
        
        # Search for both queries
        distances1, indices1 = self.index.search(
            query_embeddings[0:1].astype('float32'),
            search_k
        )
        distances2, indices2 = self.index.search(
            query_embeddings[1:2].astype('float32'),
            search_k
        )
        
        # Combine results with deduplication
        seen_indices: Set[int] = set()
        combined_results = []
        
        # Interleave results from both queries
        all_results = []
        for i in range(search_k):
            if i < len(indices1[0]):
                all_results.append((indices1[0][i], distances1[0][i]))
            if i < len(indices2[0]):
                all_results.append((indices2[0][i], distances2[0][i]))
        
        # Sort by similarity (distance) and deduplicate
        all_results.sort(key=lambda x: x[1], reverse=True)
        
        for idx, score in all_results:
            if idx not in seen_indices:
                seen_indices.add(idx)
                combined_results.append((
                    self.chunks[idx],
                    float(score),
                    self.chunk_positions[idx]
                ))
                
                if len(combined_results) >= top_k:
                    break
        
        return combined_results
    
    def get_retrieval_features(self, results: List[Tuple[str, float, int]]) -> dict:
        """
        Extract statistical features from retrieval results.
        
        Args:
            results: List of (chunk_text, similarity_score, position) tuples
            
        Returns:
            Dictionary of features (max_similarity, mean_similarity, etc.)
        """
        if not results:
            return {
                'max_similarity': 0.0,
                'mean_similarity': 0.0,
                'min_similarity': 0.0,
                'std_similarity': 0.0
            }
        
        scores = [score for _, score, _ in results]
        
        return {
            'max_similarity': float(np.max(scores)),
            'mean_similarity': float(np.mean(scores)),
            'min_similarity': float(np.min(scores)),
            'std_similarity': float(np.std(scores))
        }
    
    def save_index(self, index_path: str, metadata_path: str) -> None:
        """Save FAISS index and metadata to disk."""
        if self.index is None:
            raise ValueError("No index to save")
        
        faiss.write_index(self.index, index_path)
        
        # Save metadata
        import pickle
        metadata = {
            'chunks': self.chunks,
            'chunk_positions': self.chunk_positions,
            'dimension': self.dimension
        }
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Index saved to {index_path}")
    
    def load_index(self, index_path: str, metadata_path: str) -> None:
        """Load FAISS index and metadata from disk."""
        import pickle
        
        self.index = faiss.read_index(index_path)
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.chunks = metadata['chunks']
        self.chunk_positions = metadata['chunk_positions']
        self.dimension = metadata['dimension']
        
        print(f"Index loaded from {index_path} with {len(self.chunks)} chunks")
