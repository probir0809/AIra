# search.py


from ddgs import DDGS
from loguru import logger
from typing import List, Dict


class SearchTool:
    """
    Web search tool using DuckDuckGo.
    No API key required — completely free.

    The agent calls this when the question requires
    real-time or general knowledge not in the documents.
    """

    name = "web_search"
    description = (
        "Searches the web for current information. "
        "Use this when the question is about recent events, "
        "general knowledge, or anything NOT likely to be in the documents. "
        "Input must be a clear search query string. "
        "Example inputs: 'latest Python version 2025', 'what is gradient descent'"
    )

    def __init__(self, max_results: int = 5):
        self.max_results = max_results
        logger.info(f"SearchTool initialized — max_results={max_results}")

    def run(self, query: str) -> str:
        """
        Searches DuckDuckGo and returns formatted results.
        Returns a string so the agent LLM can read it directly.
        """
        logger.info(f"Web search: '{query}'")

        try:
            results = self._search(query)

            if not results:
                logger.warning(f"No results found for: '{query}'")
                return f"No results found for query: '{query}'"

            formatted = self._format_results(results)
            logger.info(f"Search returned {len(results)} results for: '{query}'")
            return formatted

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return f"Search error: {e}"

    def _search(self, query: str) -> List[Dict]:
        """
        Calls DuckDuckGo and returns raw results.
        Each result has: title, href, body.
        """
        with DDGS() as ddgs:
            results = list(
                ddgs.text(
                    query,
                    max_results=self.max_results
                )
            )
        return results

    def _format_results(self, results: List[Dict]) -> str:
        """
        Formats results into a readable string for the LLM.
        Each result shows its number, title, URL, and snippet.
        """
        lines = [f"Web search results ({len(results)} found):\n"]

        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            url = result.get("href", "")
            snippet = result.get("body", "No description")

            lines.append(f"[{i}] {title}")
            lines.append(f"    URL: {url}")
            lines.append(f"    {snippet}")
            lines.append("")

        return "\n".join(lines)