from typing import List, TypedDict


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
        generation_count: how many times the generate node has run
    """

    question: str
    generation: str
    web_search: bool
    documents: List[str]
    generation_count: int