"""
Text Processing Module - Handles Project Gutenberg text cleaning and semantic chunking.
"""
import re
from typing import List, Tuple
import tiktoken


class TextProcessor:
    """Processes raw novel text into semantically meaningful chunks."""
    
    def __init__(self, chunk_size: int = 1200, overlap: int = 200):
        """
        Initialize the text processor.
        
        Args:
            chunk_size: Target size of each chunk in tokens
            overlap: Number of overlapping tokens between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def strip_gutenberg_metadata(self, text: str) -> str:
        """
        Remove Project Gutenberg headers and footers, starting at first Chapter/Part.
        
        Args:
            text: Raw text from Project Gutenberg file
            
        Returns:
            Cleaned text starting from first chapter
        """
        # Find the first occurrence of Chapter or Part heading
        chapter_pattern = r'(?:^|\n)(?:CHAPTER|Chapter|PART|Part)\s+(?:I\b|1\b|ONE\b)'
        match = re.search(chapter_pattern, text, re.MULTILINE | re.IGNORECASE)
        
        if match:
            start_pos = match.start()
            text = text[start_pos:]
        
        # Remove Project Gutenberg footer
        footer_patterns = [
            r'\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*',
            r'End of (?:the )?Project Gutenberg.*',
            r'\*\*\* END OF THE PROJECT GUTENBERG.*'
        ]
        
        for pattern in footer_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                text = text[:match.start()]
                break
        
        return text.strip()
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in text."""
        return len(self.encoding.encode(text))
    
    def semantic_paragraph_chunking(self, text: str) -> List[Tuple[str, int]]:
        """
        Split text into semantic chunks using paragraph boundaries.
        
        Aggregates paragraphs into chunks of ~chunk_size tokens with overlap
        to preserve causal narrative links.
        
        Args:
            text: Cleaned novel text
            
        Returns:
            List of tuples (chunk_text, start_position)
        """
        # Split on double newlines to get paragraphs
        paragraphs = re.split(r'\n\n+', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_start_pos = 0
        text_position = 0
        
        for i, paragraph in enumerate(paragraphs):
            para_tokens = self.count_tokens(paragraph)
            
            # If adding this paragraph exceeds chunk_size, finalize current chunk
            if current_tokens + para_tokens > self.chunk_size and current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append((chunk_text, chunk_start_pos))
                
                # Create overlap: keep last paragraphs that fit in overlap size
                overlap_chunk = []
                overlap_tokens = 0
                
                for j in range(len(current_chunk) - 1, -1, -1):
                    para_tok = self.count_tokens(current_chunk[j])
                    if overlap_tokens + para_tok <= self.overlap:
                        overlap_chunk.insert(0, current_chunk[j])
                        overlap_tokens += para_tok
                    else:
                        break
                
                current_chunk = overlap_chunk
                current_tokens = overlap_tokens
                
                # Update chunk start position
                if overlap_chunk:
                    # Find position of first paragraph in overlap
                    overlap_text = overlap_chunk[0]
                    chunk_start_pos = text.find(overlap_text, chunk_start_pos)
                else:
                    chunk_start_pos = text.find(paragraph, text_position)
            
            current_chunk.append(paragraph)
            current_tokens += para_tokens
            
        # Add final chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append((chunk_text, chunk_start_pos))
        
        return chunks
    
    def process_novel(self, filepath: str) -> List[Tuple[str, int]]:
        """
        Complete processing pipeline for a novel file.
        
        Args:
            filepath: Path to the novel text file
            
        Returns:
            List of semantically chunked text segments with positions
        """
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            raw_text = f.read()
        
        cleaned_text = self.strip_gutenberg_metadata(raw_text)
        chunks = self.semantic_paragraph_chunking(cleaned_text)
        
        print(f"Processed novel: {len(chunks)} chunks created from {self.count_tokens(cleaned_text)} tokens")
        
        return chunks
