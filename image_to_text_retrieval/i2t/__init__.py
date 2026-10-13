"""Image-to-Text Retrieval cu CLIP: dintr-o imagine se regăsesc cele mai apropiate texte."""

from .index import SearchResult, TextIndex, load_texts

__all__ = ["TextIndex", "SearchResult", "load_texts"]
