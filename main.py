"""
Narrative Consistency Auditor - Main orchestration and final arbitration.

This module integrates text processing, semantic retrieval, LLM reasoning,
and machine learning to determine narrative consistency.
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Import our modules
from src.processor import TextProcessor
from src.indexer import DualQueryIndexer
from src.reasoner import NarrativeReasoner


class NarrativeConsistencyAuditor:
    """
    High-performance system for auditing narrative consistency.
    
    Combines semantic retrieval, LLM reasoning, and ML-based arbitration
    to determine if character backstories are consistent with novels.
    """
    
    def __init__(self, novel_path: str = None):
        """
        Initialize the auditor.
        
        Args:
            novel_path: Path to the novel text file
        """
        # Load environment variables
        load_dotenv()
        
        # Initialize components
        self.processor = TextProcessor(chunk_size=1200, overlap=200)
        self.indexer = DualQueryIndexer(model_name='BAAI/bge-base-en-v1.5')
        self.reasoner = NarrativeReasoner()
        
        # ML model for final arbitration
        self.classifier = None
        self.is_trained = False
        
        # Novel data
        self.novel_path = novel_path
        self.chunks = None
        
        print("=" * 70)
        print("NARRATIVE CONSISTENCY AUDITOR - INITIALIZED")
        print("=" * 70)
    
    def load_and_index_novel(self, novel_path: str = None) -> None:
        """
        Process and index a novel for retrieval.
        
        Args:
            novel_path: Path to the novel file (overrides constructor path)
        """
        if novel_path:
            self.novel_path = novel_path
        
        if not self.novel_path:
            raise ValueError("Novel path not specified")
        
        print(f"\n[1/2] Processing novel: {self.novel_path}")
        self.chunks = self.processor.process_novel(self.novel_path)
        
        print(f"\n[2/2] Building semantic index...")
        self.indexer.build_index(self.chunks)
        
        print(f"\n✓ Novel indexed successfully!")
    
    def verify_backstory(
        self,
        character_name: str,
        backstory_claim: str,
        return_details: bool = False
    ) -> Dict[str, Any]:
        """
        Verify if a backstory claim is consistent with the novel.
        
        Args:
            character_name: Name of the character
            backstory_claim: The backstory claim to verify
            return_details: Whether to return detailed analysis
            
        Returns:
            Dictionary containing prediction and analysis details
        """
        if self.chunks is None:
            raise ValueError("Novel not loaded. Call load_and_index_novel() first")
        
        print(f"\n{'=' * 70}")
        print(f"VERIFYING: {character_name} - {backstory_claim}")
        print(f"{'=' * 70}")
        
        # Step 1: Dual-query retrieval
        print("\n[Step 1/3] Retrieving relevant context...")
        retrieved_chunks = self.indexer.dual_query_search(
            backstory_claim=backstory_claim,
            character_name=character_name,
            top_k=8
        )
        
        retrieval_features = self.indexer.get_retrieval_features(retrieved_chunks)
        print(f"  ✓ Retrieved {len(retrieved_chunks)} segments")
        print(f"  ✓ Max similarity: {retrieval_features['max_similarity']:.4f}")
        print(f"  ✓ Mean similarity: {retrieval_features['mean_similarity']:.4f}")
        
        # Step 2: LLM reasoning
        print("\n[Step 2/3] Analyzing with LLM...")
        llm_result = self.reasoner.analyze_consistency(
            character_name=character_name,
            backstory_claim=backstory_claim,
            context_chunks=retrieved_chunks
        )
        
        print(f"  ✓ LLM Verdict: {llm_result['verdict']}")
        print(f"  ✓ LLM Confidence: {llm_result['confidence']:.4f}")
        
        # Step 3: Final arbitration with ML
        print("\n[Step 3/3] Final arbitration...")
        features = self._extract_features(retrieval_features, llm_result)
        
        if self.is_trained and self.classifier:
            prediction = self._predict_with_classifier(features)
            prediction_source = "ML Classifier"
        else:
            # Fallback: use LLM verdict if classifier not trained
            prediction = 1 if llm_result['verdict'] == 'CONSISTENT' else 0
            prediction_source = "LLM (No ML model)"
        
        result = {
            'character_name': character_name,
            'backstory_claim': backstory_claim,
            'prediction': prediction,
            'prediction_label': 'CONSISTENT' if prediction == 1 else 'CONTRADICTORY',
            'prediction_source': prediction_source,
            'retrieval_features': retrieval_features,
            'llm_verdict': llm_result['verdict'],
            'llm_confidence': llm_result['confidence'],
            'llm_analysis': llm_result['analysis']
        }
        
        if return_details:
            result['retrieved_chunks'] = retrieved_chunks
        
        print(f"\n{'=' * 70}")
        print(f"FINAL PREDICTION: {result['prediction_label']}")
        print(f"{'=' * 70}\n")
        
        return result
    
    def _extract_features(
        self,
        retrieval_features: Dict[str, float],
        llm_result: Dict[str, Any]
    ) -> np.ndarray:
        """
        Extract features for ML classifier.
        
        Args:
            retrieval_features: Features from retrieval system
            llm_result: Result from LLM analysis
            
        Returns:
            Feature vector as numpy array
        """
        # Encode LLM verdict
        verdict_encoding = {
            'CONSISTENT': 1.0,
            'CONTRADICT': -1.0,
            'UNCLEAR': 0.0
        }
        
        features = [
            retrieval_features['max_similarity'],
            retrieval_features['mean_similarity'],
            retrieval_features['min_similarity'],
            retrieval_features['std_similarity'],
            verdict_encoding.get(llm_result['verdict'], 0.0),
            llm_result['confidence']
        ]
        
        return np.array(features).reshape(1, -1)
    
    def _predict_with_classifier(self, features: np.ndarray) -> int:
        """Make prediction using trained classifier."""
        prediction = self.classifier.predict(features)[0]
        proba = self.classifier.predict_proba(features)[0]
        
        print(f"  ✓ ML Prediction: {prediction}")
        print(f"  ✓ ML Confidence: {max(proba):.4f}")
        
        return int(prediction)
    
    def train_arbitrator(
        self,
        training_data_path: str,
        test_size: float = 0.2,
        save_model_path: str = None
    ) -> Dict[str, Any]:
        """
        Train the Random Forest classifier for final arbitration.
        
        Args:
            training_data_path: Path to CSV with columns: char, content, label
            test_size: Fraction of data for testing
            save_model_path: Path to save trained model
            
        Returns:
            Training metrics and results
        """
        if self.chunks is None:
            raise RuntimeError(
                "Novel index not built. Call load_and_index_novel() before training."
            )
        
        print(f"\n{'=' * 70}")
        print("TRAINING ML ARBITRATOR")
        print(f"{'=' * 70}")
        
        # Load training data
        print(f"\nLoading training data from {training_data_path}...")
        df = pd.read_csv(training_data_path)
        
        required_cols = ['char', 'content', 'label']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Training data must have columns: {required_cols}")
        
        print(f"  ✓ Loaded {len(df)} training examples")
        
        # Convert label to binary (consistent=1, contradict=0)
        df['label'] = df['label'].apply(lambda x: 1 if str(x).lower() == 'consistent' else 0)
        
        # Extract features for each example
        print("\nExtracting features for training data...")
        X_list = []
        y_list = []
        
        for idx, row in df.iterrows():
            try:
                # Get retrieval and LLM features
                retrieved_chunks = self.indexer.dual_query_search(
                    backstory_claim=row['content'],
                    character_name=row['char'],
                    top_k=4
                )
                
                retrieval_features = self.indexer.get_retrieval_features(retrieved_chunks)
                
                llm_result = self.reasoner.analyze_consistency(
                    character_name=row['char'],
                    backstory_claim=row['content'],
                    context_chunks=retrieved_chunks
                )
                
                features = self._extract_features(retrieval_features, llm_result)
                X_list.append(features.flatten())
                y_list.append(row['label'])
                
                if (idx + 1) % 10 == 0:
                    print(f"  Processed {idx + 1}/{len(df)} examples...")
                
            except Exception as e:
                print(f"  Warning: Skipping example {idx} due to error: {e}")
                continue
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        print(f"\n✓ Feature extraction complete: {X.shape}")
        
        # Train/test split
        if len(X) == 0:
            raise RuntimeError(
                "No training samples generated. Check CSV columns and feature extraction."
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"\nTrain set: {len(X_train)} | Test set: {len(X_test)}")
        
        # Train Random Forest
        print("\nTraining Random Forest Classifier...")
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.classifier.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        print("\nEvaluating model...")
        y_pred_train = self.classifier.predict(X_train)
        y_pred_test = self.classifier.predict(X_test)
        
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        print(f"\n{'=' * 70}")
        print(f"TRAINING RESULTS")
        print(f"{'=' * 70}")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy:  {test_acc:.4f}")
        print("\nTest Set Classification Report:")
        print(classification_report(y_test, y_pred_test, target_names=['Contradictory', 'Consistent']))
        
        # Feature importance
        feature_names = [
            'max_similarity', 'mean_similarity', 'min_similarity',
            'std_similarity', 'llm_verdict', 'llm_confidence'
        ]
        importances = self.classifier.feature_importances_
        
        print("\nFeature Importances:")
        for name, importance in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
            print(f"  {name:20s}: {importance:.4f}")
        
        # Save model
        if save_model_path:
            self.save_model(save_model_path)
            print(f"\n✓ Model saved to {save_model_path}")
        
        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'feature_importances': dict(zip(feature_names, importances))
        }
    
    def save_model(self, path: str) -> None:
        """Save trained classifier to disk."""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        with open(path, 'wb') as f:
            pickle.dump(self.classifier, f)
    
    def load_model(self, path: str) -> None:
        """Load trained classifier from disk."""
        with open(path, 'rb') as f:
            self.classifier = pickle.load(f)
        self.is_trained = True
        print(f"✓ Model loaded from {path}")


def main():
    """Main execution function with example usage."""
    print("\n" + "=" * 70)
    print(" " * 15 + "NARRATIVE CONSISTENCY AUDITOR")
    print("=" * 70)
    
    # Initialize auditor
    auditor = NarrativeConsistencyAuditor()
    
    # Example: Load novels from the Dataset/Books folder
    dataset_path = Path(__file__).parent / "Dataset" / "Books"
    
    # Check for available novels
    if dataset_path.exists():
        novels = list(dataset_path.glob("*.txt"))
        if novels:
            print(f"\nFound {len(novels)} novel(s) in Dataset/Books/")
            
            # Index all novels into one combined index
            print("\n" + "=" * 70)
            print("INDEXING ALL NOVELS")
            print("=" * 70)
            
            all_chunks = []
            for novel_path in novels:
                print(f"\nProcessing: {novel_path.name}")
                chunks = auditor.processor.process_novel(str(novel_path))
                all_chunks.extend(chunks)
            
            print(f"\nBuilding combined index with {len(all_chunks)} total chunks...")
            auditor.chunks = all_chunks
            auditor.indexer.build_index(all_chunks)
            
            # Test with actual test.csv data
            test_csv = Path(__file__).parent / "Dataset" / "test.csv"
            if test_csv.exists():
                print("\n" + "=" * 70)
                print("TESTING WITH ACTUAL TEST DATA")
                print("=" * 70)
                
                test_df = pd.read_csv(test_csv)
                print(f"\nLoaded {len(test_df)} test cases from Dataset/test.csv")
                
                # Test first 3 examples
                for idx, row in test_df.head(3).iterrows():
                    print(f"\n{'─' * 70}")
                    print(f"Test #{idx + 1}: {row['char']}")
                    print(f"Book: {row['book_name']}")
                    print(f"Claim: {row['content'][:100]}...")
                    try:
                        result = auditor.verify_backstory(row['char'], row['content'])
                        print(f"✓ Prediction: {result['prediction_label']}")
                        print(f"✓ LLM Verdict: {result['llm_verdict']}")
                    except Exception as e:
                        print(f"✗ Error: {e}")
            
            # Show training info
            train_csv = Path(__file__).parent / "Dataset" / "train.csv"
            if train_csv.exists():
                print("\n" + "=" * 70)
                print("TRAINING DATA AVAILABLE")
                print("=" * 70)
                train_df = pd.read_csv(train_csv)
                print(f"✓ Found {len(train_df)} training examples in Dataset/train.csv")
                print(f"✓ Labels: {train_df['label'].value_counts().to_dict()}")
                print("\nTo train the ML arbitrator, run:")
                print("  from main import NarrativeConsistencyAuditor")
                print("  auditor = NarrativeConsistencyAuditor()")
                print("  auditor.load_and_index_novel('Dataset/Books/The Count of Monte Cristo.txt')")
                print("  auditor.train_arbitrator('Dataset/train.csv', save_model_path='model.pkl')")
            
        else:
            print("\nNo novels found in Dataset/Books/")
            print("Please add a text file to Dataset/Books/ directory")
    else:
        print("\nDataset/Books/ directory not found")
        print("Please create it and add novel text files")
    
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
