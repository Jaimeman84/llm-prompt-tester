from typing import List, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import textstat
from spellchecker import SpellChecker
import re

class OutputEvaluator:
    """Class for evaluating LLM outputs using various metrics."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.spell = SpellChecker()
        
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts.
        
        Args:
            text1 (str): First text
            text2 (str): Second text
            
        Returns:
            float: Similarity score between 0 and 1
        """
        tfidf_matrix = self.vectorizer.fit_transform([text1, text2])
        return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    def calculate_readability(self, text: str) -> dict:
        """Calculate readability metrics for the given text."""
        if not text or text.isspace():
            return {
                'flesch_reading_ease': 0.0,
                'flesch_kincaid_grade': 0.0,
                'gunning_fog': 0.0,
                'smog': 0.0,
                'automated_readability_index': 0.0,
                'coleman_liau_index': 0.0,
                'linsear_write': 0.0,
                'dale_chall': 0.0,
                'spache': 0.0
            }

        try:
            # Calculate readability scores
            readability = textstat.flesch_reading_ease(text)
            grade_level = textstat.flesch_kincaid_grade(text)
            gunning_fog = textstat.gunning_fog(text)
            smog = textstat.smog_index(text)
            ari = textstat.automated_readability_index(text)
            cli = textstat.coleman_liau_index(text)
            lw = textstat.linsear_write_formula(text)
            dc = textstat.dale_chall_readability_score(text)
            spache = textstat.spache_readability(text)

            # Ensure all scores are floats and clamp Flesch reading ease to 0-100
            return {
                'flesch_reading_ease': float(min(max(readability, 0.0), 100.0)),
                'flesch_kincaid_grade': float(grade_level),
                'gunning_fog': float(gunning_fog),
                'smog': float(smog),
                'automated_readability_index': float(ari),
                'coleman_liau_index': float(cli),
                'linsear_write': float(lw),
                'dale_chall': float(dc),
                'spache': float(spache)
            }
        except Exception as e:
            print(f"Error calculating readability metrics: {e}")
            return {
                'flesch_reading_ease': 0.0,
                'flesch_kincaid_grade': 0.0,
                'gunning_fog': 0.0,
                'smog': 0.0,
                'automated_readability_index': 0.0,
                'coleman_liau_index': 0.0,
                'linsear_write': 0.0,
                'dale_chall': 0.0,
                'spache': 0.0
            }
    
    def check_spelling(self, text: str) -> dict:
        """Check spelling using pyspellchecker.
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Dictionary containing spell check results
        """
        # Split text into words and remove punctuation
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Find misspelled words
        misspelled = self.spell.unknown(words)
        
        # Get corrections for misspelled words
        corrections = {}
        for word in misspelled:
            corrections[word] = self.spell.candidates(word)
        
        return {
            'error_count': len(misspelled),
            'misspelled_words': list(misspelled),
            'corrections': corrections
        }
    
    def calculate_overall_score(self, 
                              text: str, 
                              reference: Optional[str] = None,
                              weights: Optional[dict] = None) -> float:
        """Calculate overall quality score combining multiple metrics.
        
        Args:
            text (str): Input text
            reference (str, optional): Reference text for similarity
            weights (dict, optional): Custom weights for different metrics
            
        Returns:
            float: Overall score between 0 and 1
        """
        if not text.strip():
            return 0.0
        
        if weights is None:
            weights = {
                'similarity': 0.4,
                'readability': 0.3,
                'spelling': 0.3
            }
            
        scores = {}
        
        # Calculate similarity if reference is provided
        if reference:
            scores['similarity'] = self.calculate_similarity(text, reference)
        
        # Calculate readability
        readability = self.calculate_readability(text)
        scores['readability'] = min(readability['flesch_reading_ease'] / 100, 1.0)
        
        # Calculate spelling score
        spelling = self.check_spelling(text)
        # Normalize spelling score (assume max 10 errors as baseline)
        scores['spelling'] = 1.0 - min(spelling['error_count'] / 10, 1.0)
        
        # Calculate weighted average
        total_score = 0
        total_weight = 0
        
        for metric, score in scores.items():
            if metric in weights:
                total_score += score * weights[metric]
                total_weight += weights[metric]
                
        # Ensure score is between 0 and 1
        final_score = total_score / total_weight if total_weight > 0 else 0.0
        return max(0.0, min(1.0, final_score)) 