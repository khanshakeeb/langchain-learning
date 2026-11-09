from types import SimpleNamespace

from dotenv import load_dotenv

load_dotenv()

import graph.graph as graph_module
from graph.consts import MAX_RETRIES


class AlwaysFailGrader:
    """Stand-in grader that rejects every answer, to force the retry loop."""

    def invoke(self, _input):
        return SimpleNamespace(binary_score=False)


def test_graph_stops_after_max_retries(monkeypatch) -> None:
    monkeypatch.setattr(graph_module, "hallucination_grader", AlwaysFailGrader())

    result = graph_module.app.invoke({"question": "what is agent memory?"})

    # 1 initial generation + MAX_RETRIES retries, then the graph ends
    assert result["generation_count"] == MAX_RETRIES + 1
    assert result["generation"]
