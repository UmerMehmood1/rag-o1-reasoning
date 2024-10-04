import base64
import os
import sys
from typing import Any, List, Dict

import nest_asyncio  # type: ignore
import torch
import yaml  # type: ignore
from IPython.display import HTML, display
from langchain_core.documents.base import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import load_prompt
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod
from langgraph.graph.state import CompiledStateGraph
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore

from utils.logging_utils import log_method
from utils.model_wrappers.api_gateway import APIGateway

class RAGComponents:
    def __init__(self, configs: dict) -> None:
        self.configs = configs
        self.init_llm()
        self.init_embeddings()
        self.init_base_llm_chain()

    @staticmethod
    def load_config(filename: str) -> dict:
        """Loads a YAML configuration file."""
        try:
            with open(filename, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f'The YAML configuration file {filename} was not found.')
        except yaml.YAMLError as e:
            raise RuntimeError(f'Error parsing YAML file: {e}')

    def init_llm(self) -> None:
        """Initializes the Large Language Model (LLM)."""
        self.llm = APIGateway.load_llm(
            type=self.configs['api'],
            streaming=True,
            coe=self.configs['llm']['coe'],
            do_sample=self.configs['llm']['do_sample'],
            max_tokens_to_generate=self.configs['llm']['max_tokens_to_generate'],
            temperature=self.configs['llm']['temperature'],
            select_expert=self.configs['llm']['select_expert'],
            process_prompt=False,
        )

    def init_embeddings(self) -> None:
        """Initializes the embeddings for the model."""
        self.embeddings = APIGateway.load_embedding_model(
            type=self.configs['embedding_model']['type'],
            batch_size=self.configs['embedding_model']['batch_size'],
            coe=self.configs['embedding_model']['coe'],
            select_expert=self.configs['embedding_model']['select_expert'],
        )

    def init_base_llm_chain(self) -> None:
        """Initializes the base LLM chain."""
        base_llm_prompt: Any = load_prompt(self.configs['prompts']['base_llm_prompt'])
        self.base_llm_chain = base_llm_prompt | self.llm | StrOutputParser()  # type: ignore

    def rerank_docs(self, query: str, docs: List[Document], final_k: int) -> List[Document]:
        """Rerank a list of documents based on relevance to the query."""
        tokenizer = AutoTokenizer.from_pretrained(self.configs['retrieval']['reranker'])
        reranker = AutoModelForSequenceClassification.from_pretrained(self.configs['retrieval']['reranker'])
        
        pairs = [[query, d.page_content] for d in docs]
        
        with torch.no_grad():
            inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            scores = reranker(**inputs, return_dict=True).logits.view(-1).float()

        scores_sorted_idx = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
        return [docs[k] for k in scores_sorted_idx[:final_k]]

    @log_method
    def llm_generation(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generates a response from the selected LLM."""
        print('---GENERATING FROM INTERNAL KNOWLEDGE---')
        question: str = state['question']
        generation: str = self.base_llm_chain.invoke({'question': question})
        return {'question': question, 'generation': generation}

    def display_graph(self, app: CompiledStateGraph) -> None:
        """Displays a graph generated using Mermaid."""
        nest_asyncio.apply()  # Required for Jupyter Notebook to run async functions
        img_bytes = app.get_graph().draw_mermaid_png(
            curve_style=CurveStyle.LINEAR,
            wrap_label_n_words=9,
            output_file_path=None,
            draw_method=MermaidDrawMethod.PYPPETEER,
            background_color='white',
            padding=10,
        )
        self._display_image(img_bytes)

    def _display_image(self, image_bytes: bytes, width: int = 512) -> None:
        """Displays an image from a byte string."""
        decoded_img_bytes = base64.b64encode(image_bytes).decode('utf-8')
        html = f'<img src="data:image/png;base64,{decoded_img_bytes}" style="width: {width}px;" />'
        display(HTML(html))
