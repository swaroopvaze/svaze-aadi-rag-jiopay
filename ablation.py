import os
import time
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rag_backend import RAGSystem, Config


class AblationStudy:
    """Comprehensive ablation study for RAG system"""

    def __init__(self, config: Config):
        self.config = config
        self.results: List[Dict[str, Any]] = []

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
            "How do I enable transaction notifications?",
        ]

        # Resolve data file path from config or default
        self.data_file = getattr(self.config, "DATA_FILE", "jiopay_data.json")
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(
                f"Data file not found: {self.data_file}. Please ensure it exists."
            )

        # Determine if OpenAI is available
        self.has_openai_key = bool(getattr(self.config, "OPENAI_API_KEY", ""))

    def _make_rag_system(self, prefer_provider: Optional[str] = None) -> RAGSystem:
        """Create a RAGSystem instance with safe defaults.
        If no OpenAI key, default to sentence_transformers."""
        rag_system = RAGSystem(self.config)

        # Decide provider
        provider = prefer_provider
        if provider is None:
            provider = "openai" if self.has_openai_key else "sentence_transformers"

        if provider == "openai" and not self.has_openai_key:
            # Fallback if key missing
            provider = "sentence_transformers"

        # Configure embedding provider
        rag_system.embedding_provider.provider = provider
        if provider == "sentence_transformers":
            # Default lightweight, widely available local model
            rag_system.embedding_provider.model_name = "all-MiniLM-L6-v2"
            try:
                from sentence_transformers import SentenceTransformer

                rag_system.embedding_provider.model = SentenceTransformer(
                    rag_system.embedding_provider.model_name
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize SentenceTransformer: {str(e)}"
                )

        return rag_system

    def run_chunking_ablation(self) -> pd.DataFrame:
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

        results: List[Dict[str, Any]] = []

        for cfg in chunking_configs:
            method = cfg["method"]
            size = cfg.get("size", None)
            overlap = cfg.get("overlap", None)
            print(f"- Testing {method} chunking (size={size}, overlap={overlap})")

            # Update configuration even when zero (use is not None checks)
            if size is not None:
                self.config.CHUNK_SIZE = size
            if overlap is not None:
                self.config.CHUNK_OVERLAP = overlap

            # Initialize system with safe provider defaults
            rag_system = self._make_rag_system()

            # Ingest data
            try:
                rag_system.ingest_data(self.data_file, method)
            except Exception as e:
                print(f"  Skipping due to ingest error: {str(e)}")
                continue

            # Evaluate on test queries
            metrics = self.evaluate_system(rag_system, f"chunking_{method}")

            results.append(
                {
                    "strategy": method,
                    "size": size if size is not None else "-",
                    "overlap": overlap if overlap is not None else "-",
                    **metrics,
                }
            )

        df = pd.DataFrame(results)
        if not df.empty:
            df.to_csv("chunking_ablation_results.csv", index=False)
            print("Chunking ablation results saved to 'chunking_ablation_results.csv'")
        else:
            print("No chunking results to save.")
        return df

    def run_embedding_ablation(self) -> pd.DataFrame:
        """Test different embedding models"""
        print("Running Embedding Model Ablation Study...")

        embedding_configs = [
            {"provider": "openai", "model": "text-embedding-3-small"},
            {"provider": "openai", "model": "text-embedding-3-large"},
            {"provider": "sentence_transformers", "model": "all-MiniLM-L6-v2"},
            {"provider": "sentence_transformers", "model": "all-mpnet-base-v2"},
        ]

        results: List[Dict[str, Any]] = []

        for cfg in embedding_configs:
            provider = cfg["provider"]
            model = cfg["model"]

            if provider == "openai" and not self.has_openai_key:
                print(f"- Skipping {provider}:{model} (no OPENAI_API_KEY)")
                continue

            print(f"- Testing {provider}:{model}...")

            try:
                rag_system = self._make_rag_system(prefer_provider=provider)
                rag_system.embedding_provider.provider = provider
                rag_system.embedding_provider.model_name = model

                if provider == "sentence_transformers":
                    from sentence_transformers import SentenceTransformer

                    rag_system.embedding_provider.model = SentenceTransformer(model)

                rag_system.ingest_data(self.data_file, "structural")
                metrics = self.evaluate_system(rag_system, f"embedding_{model}")

                results.append({"provider": provider, "model": model, **metrics})

            except Exception as e:
                print(f"  Error with {provider}:{model}: {str(e)}")
                continue

        df = pd.DataFrame(results)
        if not df.empty:
            df.to_csv("embedding_ablation_results.csv", index=False)
            print("Embedding ablation results saved to 'embedding_ablation_results.csv'")
        else:
            print("No embedding results to save.")
        return df

    def run_retrieval_ablation(self) -> pd.DataFrame:
        """Test different retrieval configurations"""
        print("Running Retrieval Configuration Ablation Study...")

        top_k_values = [3, 5, 7, 10]
        results: List[Dict[str, Any]] = []

        # Initialize base system with safe defaults
        rag_system = self._make_rag_system()
        try:
            rag_system.ingest_data(self.data_file, "structural")
        except Exception as e:
            print(f"  Ingest failed for retrieval ablation: {str(e)}")
            return pd.DataFrame()

        for top_k in top_k_values:
            print(f"- Testing top_k = {top_k}...")
            original_top_k = rag_system.config.TOP_K_RETRIEVAL
            rag_system.config.TOP_K_RETRIEVAL = top_k
            try:
                metrics = self.evaluate_system(rag_system, f"retrieval_k{top_k}")
                results.append({"top_k": top_k, **metrics})
            finally:
                rag_system.config.TOP_K_RETRIEVAL = original_top_k

        df = pd.DataFrame(results)
        if not df.empty:
            df.to_csv("retrieval_ablation_results.csv", index=False)
            print("Retrieval ablation results saved to 'retrieval_ablation_results.csv'")
        else:
            print("No retrieval results to save.")
        return df

    def evaluate_system(self, rag_system: RAGSystem, experiment_name: str) -> Dict[str, float]:
        """Evaluate RAG system performance"""
        latencies: List[float] = []
        token_usage: List[int] = []
        retrieval_scores: List[float] = []
        answer_quality_scores: List[float] = []

        print(f"Evaluating {len(self.test_queries)} test queries...")

        for i, query in enumerate(self.test_queries):
            print(f"  Query {i + 1}/{len(self.test_queries)}: {query[:60]}...")
            start_time = time.time()
            try:
                result = rag_system.query(query)
                latency = (time.time() - start_time) * 1000.0
                latencies.append(latency)

                # Defensive dict access
                if not isinstance(result, dict):
                    result = {}

                token_usage.append(int(result.get("tokens_used", 0)))

                # Retrieval evaluation
                sources = result.get("sources", []) or []
                if isinstance(sources, list) and len(sources) > 0:
                    avg_score = np.mean([float(s.get("score", 0)) for s in sources])
                    retrieval_scores.append(float(avg_score))
                else:
                    retrieval_scores.append(0.0)

                # Simple answer quality heuristic (length-based proxy with penalty)
                answer = result.get("answer", "")
                quality_score = min(len(answer) / 100.0, 1.0)
                if "I don't have information" in answer or "I do not have information" in answer:
                    quality_score *= 0.5  # Penalize non-answers
                answer_quality_scores.append(float(quality_score))

            except Exception as e:
                print(f"    Error processing query: {str(e)}")
                latencies.append(10000.0)  # High penalty for errors
                token_usage.append(0)
                retrieval_scores.append(0.0)
                answer_quality_scores.append(0.0)

        # Calculate metrics
        metrics = {
            "avg_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
            "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
            "avg_tokens": float(np.mean(token_usage)) if token_usage else 0.0,
            "avg_retrieval_score": float(np.mean(retrieval_scores)) if retrieval_scores else 0.0,
            "avg_answer_quality": float(np.mean(answer_quality_scores)) if answer_quality_scores else 0.0,
            "success_rate": float(sum(1 for l in latencies if l < 10000.0) / len(latencies))
            if latencies
            else 0.0,
        }

        print(f"  Results for {experiment_name}:")
        for key, value in metrics.items():
            print(f"    {key}: {value:.3f}")

        return metrics

    def generate_report(self):
        """Generate comprehensive ablation study report"""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE RAG ABLATION STUDY REPORT")
        print("=" * 80)

        try:
            # Run all ablations
            chunking_df = self.run_chunking_ablation()
            embedding_df = self.run_embedding_ablation()
            retrieval_df = self.run_retrieval_ablation()

            # Generate visualizations
            self.create_visualizations(chunking_df, embedding_df, retrieval_df)

            # Print summary
            print("\n" + "-" * 60)
            print("SUMMARY RESULTS")
            print("-" * 60)

            if not chunking_df.empty:
                print("\nBest Chunking Strategy:")
                best_chunking = chunking_df.loc[chunking_df["avg_answer_quality"].idxmax()]
                print(f"  Strategy: {best_chunking['strategy']}")
                print(f"  Size: {best_chunking['size']}")
                print(f"  Overlap: {best_chunking['overlap']}")
                print(f"  Answer Quality: {best_chunking['avg_answer_quality']:.3f}")
                print(f"  Avg Latency: {best_chunking['avg_latency_ms']:.1f}ms")
            else:
                print("\nNo chunking results available.")

            if not embedding_df.empty:
                print("\nBest Embedding Model:")
                best_embedding = embedding_df.loc[embedding_df["avg_answer_quality"].idxmax()]
                print(f"  Provider: {best_embedding['provider']}")
                print(f"  Model: {best_embedding['model']}")
                print(f"  Answer Quality: {best_embedding['avg_answer_quality']:.3f}")
                print(f"  Avg Latency: {best_embedding['avg_latency_ms']:.1f}ms")
            else:
                print("\nNo embedding results available.")

            if not retrieval_df.empty:
                print("\nBest Retrieval Configuration:")
                best_retrieval = retrieval_df.loc[retrieval_df["avg_answer_quality"].idxmax()]
                print(f"  Top-K: {best_retrieval['top_k']}")
                print(f"  Answer Quality: {best_retrieval['avg_answer_quality']:.3f}")
                print(f"  Avg Latency: {best_retrieval['avg_latency_ms']:.1f}ms")
            else:
                print("\nNo retrieval results available.")

            print("\n" + "=" * 80)
            print("Ablation study completed! Check CSV files and plots for detailed results.")

        except Exception as e:
            print(f"Error generating report: {str(e)}")

    def create_visualizations(
        self, chunking_df: pd.DataFrame, embedding_df: pd.DataFrame, retrieval_df: pd.DataFrame
    ):
        """Create visualization plots"""
        try:
            plt.style.use("seaborn-v0_8")
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # Chunking performance
            if not chunking_df.empty:
                axes[0, 0].bar(range(len(chunking_df)), chunking_df["avg_answer_quality"])
                axes[0, 0].set_title("Answer Quality by Chunking Strategy")
                axes[0, 0].set_xlabel("Strategy")
                axes[0, 0].set_ylabel("Answer Quality Score")
                axes[0, 0].set_xticks(range(len(chunking_df)))
                axes[0, 0].set_xticklabels(
                    [f"{row['strategy']}\n{row['size']}" for _, row in chunking_df.iterrows()],
                    rotation=45,
                    ha="right",
                )
            else:
                axes[0, 0].set_title("Answer Quality by Chunking Strategy (no data)")

            # Latency comparison
            if not chunking_df.empty:
                axes[0, 1].scatter(chunking_df["avg_latency_ms"], chunking_df["avg_answer_quality"])
                axes[0, 1].set_title("Latency vs Answer Quality")
                axes[0, 1].set_xlabel("Average Latency (ms)")
                axes[0, 1].set_ylabel("Answer Quality Score")
            else:
                axes[0, 1].set_title("Latency vs Answer Quality (no data)")

            # Retrieval performance
            if not retrieval_df.empty:
                axes[1, 0].plot(retrieval_df["top_k"], retrieval_df["avg_answer_quality"], "o-")
                axes[1, 0].set_title("Answer Quality vs Top-K Retrieval")
                axes[1, 0].set_xlabel("Top-K Value")
                axes[1, 0].set_ylabel("Answer Quality Score")
            else:
                axes[1, 0].set_title("Answer Quality vs Top-K Retrieval (no data)")

            # Token usage
            if not chunking_df.empty:
                axes[1, 1].bar(range(len(chunking_df)), chunking_df["avg_tokens"])
                axes[1, 1].set_title("Token Usage by Chunking Strategy")
                axes[1, 1].set_xlabel("Strategy")
                axes[1, 1].set_ylabel("Average Tokens")
                axes[1, 1].set_xticks(range(len(chunking_df)))
                axes[1, 1].set_xticklabels(
                    [f"{row['strategy']}" for _, row in chunking_df.iterrows()],
                    rotation=45,
                    ha="right",
                )
            else:
                axes[1, 1].set_title("Token Usage by Chunking Strategy (no data)")

            plt.tight_layout()
            plt.savefig("ablation_study_results.png", dpi=300, bbox_inches="tight")
            print("Visualization saved as 'ablation_study_results.png'")

        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")


def main():
    """Run comprehensive ablation study"""
    config = Config()

    # Proceed even if OpenAI key is absent; we will fallback to local embeddings
    if not getattr(config, "OPENAI_API_KEY", ""):
        print("Warning: OPENAI_API_KEY not found. Falling back to local embeddings.")

    study = AblationStudy(config)
    study.generate_report()


if __name__ == "__main__":
    main()