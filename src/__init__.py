"""
Package initialization for Narrative Consistency Auditor.
"""
from .processor import TextProcessor
from .indexer import DualQueryIndexer
from .reasoner import NarrativeReasoner

__all__ = ['TextProcessor', 'DualQueryIndexer', 'NarrativeReasoner']
__version__ = '1.0.0'
