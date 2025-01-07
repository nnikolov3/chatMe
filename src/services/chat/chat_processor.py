import asyncio
import multiprocessing
from pathlib import Path
from typing import Dict, List, Optional
from ollama import AsyncClient
import logging
from ..query.query_processor import QueryProcessor
from src.services.document.vector_processor import VectorProcessor

logger = logging.getLogger(__name__)

# Define models and their initial weights
models = ["qwen2.5:7b", "mistral", "llama3.2"]
model_weights = {"qwen2.5:7b": 0.33333, "mistral": 0.33333, "llama3.2": 0.33333}


class ChatProcessor:
    def __init__(self, db_path: str, max_workers: int = multiprocessing.cpu_count()):
        """Initialize the ChatProcessor with a configurable worker pool.

        Args:
            db_path: Path to the ChromaDB database
            max_workers: Maximum number of worker threads for async operations
        """
        self.db_path = Path(db_path)
        self.max_workers = max_workers
        self.vector_processor = VectorProcessor(
            str(db_path)
        )  # Assuming VectorProcessor takes a string path

    def _create_prompt(self, question: str, context: str) -> str:
        requirements = (
            "Use all relevant information available. Be specific and cite relevant parts. "
            "Say 'I don't know' if the context lacks necessary information. "
            "Be thorough and detailed"
            "Before return the answer, confirm it makes sense."
        )
        return f"""Answer based on the provided context.
        Context: {context}
        Question: {question}
        Requirements:{requirements}
        """

    async def _get_responses(self, prompt: str) -> Dict[str, str]:
        content = prompt
        my_message = [{"role": "user", "content": content}]
        responses = {}
        client = AsyncClient()  # Instantiate client outside the context manager

        tasks = [client.chat(model=model, messages=my_message) for model in models]
        results = await asyncio.gather(*tasks)

        for model, result in zip(models, results):
            responses[model] = result.message.content

        return responses

    async def _get_context(self, question: str) -> Optional[Dict]:
        """Retrieve relevant context from processed documents using embeddings."""
        query_processor = QueryProcessor(self.db_path)
        try:
            results = await query_processor.parallel_query(question)

            if not results:
                return None

            # Using the similarity attribute directly since these are QueryResult objects
            result = max(results, key=lambda x: x.similarity)
            return result

        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return None

    async def weighted_voting(
        self, responses: Dict[str, str], weights: Dict[str, float]
    ) -> str:
        """Enhanced weighted voting system for mocadel responses with semantic similarity."""
        if not responses:
            return "No responses from models."

        # Preprocess responses
        preprocessed_responses = {}
        for model, response in responses.items():
            # Split into sentences and clean them
            sentences = [s.strip() for s in response.split(".") if s.strip()]
            # Remove very short sentences and duplicates
            sentences = [s for s in sentences if len(s.split()) > 3]
            preprocessed_responses[model] = list(dict.fromkeys(sentences))

        # Gather all unique sentences
        all_sentences = set(
            sentence
            for sentences in preprocessed_responses.values()
            for sentence in sentences
        )

        # Calculate weighted scores with additional factors
        sentence_scores = {}
        sentence_models = {}  # Track which models contributed to each sentence

        for sentence in all_sentences:
            # Initialize score and supporting models
            base_score = 0
            supporting_models = []

            # Calculate base score from model weights
            for model, sentences in preprocessed_responses.items():
                if sentence in sentences:
                    base_score += weights[model]
                    supporting_models.append(model)

            # Bonus for agreement across multiple models
            agreement_bonus = (len(supporting_models) / len(models)) * 0.5

            # Length penalty for extremely long or short sentences
            words = len(sentence.split())
            length_penalty = 1.0
            if words < 5:  # Too short
                length_penalty = 0.7
            elif words > 50:  # Too long
                length_penalty = 0.8

            # Calculate final score
            final_score = (base_score + agreement_bonus) * length_penalty

            sentence_scores[sentence] = final_score
            sentence_models[sentence] = supporting_models

        # Select and order sentences
        sorted_sentences = sorted(
            sentence_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Build final response with intelligent selection
        final_sentences = []
        current_length = 0
        max_sentences = 20  # Adjustable parameter
        max_words = 500  # Adjustable parameter

        for sentence, score in sorted_sentences:
            # Stop if we've reached our limits
            if (
                len(final_sentences) >= max_sentences
                or current_length + len(sentence.split()) > max_words
            ):
                break

            # Check if this sentence adds new information
            if not final_sentences or not self._is_redundant(sentence, final_sentences):
                final_sentences.append(sentence)
                current_length += len(sentence.split())

        # Combine sentences with proper punctuation
        final_response = ". ".join(final_sentences)
        if final_response and not final_response.endswith("."):
            final_response += "."

        return final_response

    def _is_redundant(
        self,
        new_sentence: str,
        existing_sentences: List[str],
        similarity_threshold: float = 0.8,
    ) -> bool:
        """Check if a new sentence is too similar to existing ones."""
        new_words = set(new_sentence.lower().split())

        for existing in existing_sentences:
            existing_words = set(existing.lower().split())

            # Calculate Jaccard similarity
            intersection = len(new_words.intersection(existing_words))
            union = len(new_words.union(existing_words))

            if union > 0 and intersection / union > similarity_threshold:
                return True

        return False

    async def get_voted_response(self, question: str, context: Optional[str]) -> str:
        prompt = self._create_prompt(question, context)
        responses = await self._get_responses(prompt)
        return await self.weighted_voting(responses, model_weights)

    async def get_user_input(self):
        # This method needs implementation. Here's a simple version:
        user_input = input("Enter your question: ")
        return user_input

    async def chat_processor(self):
        question = await self.get_user_input()
        context = await self._get_context(question)
        result = await self.get_voted_response(question, context)
        return result


if __name__ == "__main__":
    chat_processor = ChatProcessor(db_path="path/to/your/db")
    result = asyncio.run(chat_processor.chat_processor())
    print(f"Response: {result}")
