import json
import time
import pandas as pd
from typing import List, Dict, Any
from rag_backend import RAGSystem, Config
import numpy as np
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

class AblationStudy:
    """Comprehensive ablation study for RAG system"""
    
    def __init__(self, config: Config):
        self.config = config
        self.results = []
        
        # Test queries for evaluation
        self.test_queries = [
            "How do I complete KYC verification on JioPay?",
            "What are the different payment methods available?",
            "How do I add money to my JioPay wallet?",
            "How do I request a refund for a failed transaction?",
            "What security measures does JioPay have in place?",
            "What are the transaction limits on JioPay?",
            "How do I link my bank account to JioPay?",
            "How do I pay utility bills through JioPay?",
            "What should I do if I forget my JioPay PIN?",
            "How do I enable transaction notifications?"
        ]
    
    def run_chunking_ablation(self):
        """Test different chunking strategies"""
        print("Running Chunking Ablation Study...")
        
        chunking_configs = [
            {"method": "fixed", "size": 256, "overlap": 0},
            {"method": "fixed", "size": 256, "overlap": 64},
            {"method": "fixed", "size": 512, "overlap": 0},
            {"method": "fixed", "size": 512, "overlap": 64},
            {"method": "fixed", "size": 1024, "overlap": 0},
            {"method": "fixed", "size": 1024, "overlap": 128},
            {"method": "semantic", "size": None, "overlap": None},
            {"method": "structural", "size": None, "overlap": None},
        ]
        
        chunking_results = []
        
        for config in chunking_configs:
            print(f"Testing {config['method']} chunking...")
            
            # Update configuration
            if config['size']:
                self.config.CHUNK_SIZE = config['size']
            if config['overlap']:
                self.config.CHUNK_OVERLAP = config['overlap']
            
            # Initialize RAG system with new config
            rag_system = RAGSystem(self.config)
            rag_system.ingest_data(self.config.DATA_FILE, config['method'])
            
            # Evaluate on test queries
            metrics = self.evaluate_system(rag_system, f"chunking_{config['method']}")
            
            result = {
                "strategy": config['method'],
                "size": config.get('size', '-'),
                "overlap": config.get('overlap', '-'),
                **metrics
            }
            
            chunking_results.append(result)
        
        # Save results
        df = pd.DataFrame(chunking_results)
        df.to_csv('chunking_ablation_results.csv', index=False)
        print("Chunking ablation results saved to 'chunking_ablation_results.csv'")
        
        return df
    
    def run_embedding_ablation(self):
        """Test different embedding models"""
        print("Running Embedding Model Ablation Study...")
        
        embedding_configs = [
            {"provider": "openai", "model": "text-embedding-3-small"},
            {"provider": "openai", "model": "text-embedding-3-large"},
            {"provider": "sentence_transformers", "model": "all-MiniLM-L6-v2"},
            {"provider": "sentence_transformers", "model": "all-mpnet-base-v2"},
        ]
        
        embedding_results = []
        
        for config in embedding_configs:
            print(f"Testing {config['provider']} - {config['model']}...")
            
            try:
                # Initialize RAG system with new embedding config
                rag_system = RAGSystem(self.config)
                rag_system.embedding_provider.provider = config['provider']
                rag_system.embedding_provider.model_name = config['model']
                
                if config['provider'] == 'sentence_transformers':
                    from sentence_transformers import SentenceTransformer
                    rag_system.embedding_provider.model = SentenceTransformer(config['model'])
                
                # Ingest data with new embeddings
                rag_system.ingest_data(self.config.DATA_FILE, "structural")
                
                # Evaluate
                metrics = self.evaluate_system(rag_system, f"embedding_{config['model']}")
                
                result = {
                    "provider": config['provider'],
                    "model": config['model'],
                    **metrics
                }
                
                embedding_results.append(result)
                
            except Exception as e:
                print(f"Error with {config['model']}: {str(e)}")
                continue
        
        # Save results
        df = pd.DataFrame(embedding_results)
        df.to_csv('embedding_ablation_results.csv', index=False)
        print("Embedding ablation results saved to 'embedding_ablation_results.csv'")
        
        return df
    
    def run_retrieval_ablation(self):
        """Test different retrieval configurations"""
        print("Running Retrieval Configuration Ablation Study...")
        
        top_k_values = [3, 5, 7, 10]
        retrieval_results = []
        
        # Initialize base system
        rag_system = RAGSystem(self.config)
        rag_system.ingest_data(self.config.DATA_FILE, "structural")
        
        for top_k in top_k_values:
            print(f"Testing top_k = {top_k}...")
            
            # Update retrieval parameter
            original_top_k = rag_system.config.TOP_K_RETRIEVAL
            rag_system.config.TOP_K_RETRIEVAL = top_k
            
            # Evaluate
            metrics = self.evaluate_system(rag_system, f"retrieval_k{top_k}")
            
            result = {
                "top_k": top_k,
                **metrics
            }
            
            retrieval_results.append(result)
            
            # Restore original value
            rag_system.config.TOP_K_RETRIEVAL = original_top_k
        
        # Save results
        df = pd.DataFrame(retrieval_results)
        df.to_csv('retrieval_ablation_results.csv', index=False)
        print("Retrieval ablation results saved to 'retrieval_ablation_results.csv'")
        
        return df
    
    def evaluate_system(self, rag_system: RAGSystem, experiment_name: str) -> Dict[str, float]:
        """Evaluate RAG system performance"""
        latencies = []
        token_usage = []
        retrieval_scores = []
        answer_quality_scores = []
        
        print(f"Evaluating {len(self.test_queries)} test queries...")
        
        for i, query in enumerate(self.test_queries):
            print(f"Query {i+1}/{len(self.test_queries)}: {query[:50]}...")
            
            try:
                start_time = time.time()
                result = rag_system.query(query)
                latency = (time.time() - start_time) * 1000
                
                latencies.append(latency)
                token_usage.append(result.get('tokens_used', 0))
                
                # Retrieval evaluation
                sources = result.get('sources', [])
                if sources:
                    avg_score = np.mean([s.get('score', 0) for s in sources])
                    retrieval_scores.append(avg_score)
                else:
                    retrieval_scores.append(0)
                
                # Simple answer quality heuristic
                    answer = result.get('answer', '') if isinstance(result, dict) and 'answer' in result else ''
                quality_score = min(len(answer) / 100, 1.0)  # Normalize by length
                if 'I don\'t have information' in answer:
                    quality_score *= 0.5  # Penalize non-answers
                
                answer_quality_scores.append(quality_score)
                
            except Exception as e:
                print(f"Error processing query: {str(e)}")
                latencies.append(10000)  # High penalty for errors
                token_usage.append(0)
                retrieval_scores.append(0)
                answer_quality_scores.append(0)
        
        # Calculate metrics
        metrics = {
            "avg_latency_ms": np.mean(latencies),
            "p95_latency_ms": np.percentile(latencies, 95),
            "avg_tokens": np.mean(token_usage),
            "avg_retrieval_score": np.mean(retrieval_scores),
            "avg_answer_quality": np.mean(answer_quality_scores),
            "success_rate": sum(1 for l in latencies if l < 10000) / len(latencies)
        }
        
        print(f"Results for {experiment_name}:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.2f}")
        
        return metrics
    
    def generate_report(self):
        """Generate comprehensive ablation study report"""
        print("\n" + "="*80)
        print("COMPREHENSIVE RAG ABLATION STUDY REPORT")
        print("="*80)
        
        try:
            # Run all ablations
            chunking_df = self.run_chunking_ablation()
            embedding_df = self.run_embedding_ablation()
            retrieval_df = self.run_retrieval_ablation()
            
            # Generate visualizations
            self.create_visualizations(chunking_df, embedding_df, retrieval_df)
            
            # Print summary
            print("\n" + "-"*60)
            print("SUMMARY RESULTS")
            print("-"*60)
            
            print("\nBest Chunking Strategy:")
            best_chunking = chunking_df.loc[chunking_df['avg_answer_quality'].idxmax()]
            print(f"  Strategy: {best_chunking['strategy']}")
            print(f"  Size: {best_chunking['size']}")
            print(f"  Overlap: {best_chunking['overlap']}")
            print(f"  Answer Quality: {best_chunking['avg_answer_quality']:.3f}")
            print(f"  Avg Latency: {best_chunking['avg_latency_ms']:.1f}ms")
            
            if not embedding_df.empty:
                print("\nBest Embedding Model:")
                best_embedding = embedding_df.loc[embedding_df['avg_answer_quality'].idxmax()]
                print(f"  Provider: {best_embedding['provider']}")
                print(f"  Model: {best_embedding['model']}")
                print(f"  Answer Quality: {best_embedding['avg_answer_quality']:.3f}")
                print(f"  Avg Latency: {best_embedding['avg_latency_ms']:.1f}ms")
            
            print("\nBest Retrieval Configuration:")
            best_retrieval = retrieval_df.loc[retrieval_df['avg_answer_quality'].idxmax()]
            print(f"  Top-K: {best_retrieval['top_k']}")
            print(f"  Answer Quality: {best_retrieval['avg_answer_quality']:.3f}")
            print(f"  Avg Latency: {best_retrieval['avg_latency_ms']:.1f}ms")
            
            print("\n" + "="*80)
            print("Ablation study completed! Check CSV files and plots for detailed results.")
            
        except Exception as e:
            print(f"Error generating report: {str(e)}")
    
    def create_visualizations(self, chunking_df, embedding_df, retrieval_df):
        """Create visualization plots"""
        try:
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Chunking performance
            if not chunking_df.empty:
                axes[0, 0].bar(range(len(chunking_df)), chunking_df['avg_answer_quality'])
                axes[0, 0].set_title('Answer Quality by Chunking Strategy')
                axes[0, 0].set_xlabel('Strategy')
                axes[0, 0].set_ylabel('Answer Quality Score')
                axes[0, 0].set_xticks(range(len(chunking_df)))
                axes[0, 0].set_xticklabels([f"{row['strategy']}\n{row['size']}" 
                                          for _, row in chunking_df.iterrows()], 
                                         rotation=45, ha='right')
            
            # Latency comparison
            if not chunking_df.empty:
                axes[0, 1].scatter(chunking_df['avg_latency_ms'], chunking_df['avg_answer_quality'])
                axes[0, 1].set_title('Latency vs Answer Quality')
                axes[0, 1].set_xlabel('Average Latency (ms)')
                axes[0, 1].set_ylabel('Answer Quality Score')
            
            # Retrieval performance
            if not retrieval_df.empty:
                axes[1, 0].plot(retrieval_df['top_k'], retrieval_df['avg_answer_quality'], 'o-')
                axes[1, 0].set_title('Answer Quality vs Top-K Retrieval')
                axes[1, 0].set_xlabel('Top-K Value')
                axes[1, 0].set_ylabel('Answer Quality Score')
            
            # Token usage
            if not chunking_df.empty:
                axes[1, 1].bar(range(len(chunking_df)), chunking_df['avg_tokens'])
                axes[1, 1].set_title('Token Usage by Chunking Strategy')
                axes[1, 1].set_xlabel('Strategy')
                axes[1, 1].set_ylabel('Average Tokens')
                axes[1, 1].set_xticks(range(len(chunking_df)))
                axes[1, 1].set_xticklabels([f"{row['strategy']}" 
                                          for _, row in chunking_df.iterrows()], 
                                         rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig('ablation_study_results.png', dpi=300, bbox_inches='tight')
            print("Visualization saved as 'ablation_study_results.png'")
            
        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")

def main():
    """Run comprehensive ablation study"""
    config = Config()
    
    if not config.OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set your OpenAI API key in the .env file.")
        return
    
    study = AblationStudy(config)
    study.generate_report()

if __name__ == "__main__":
    main()