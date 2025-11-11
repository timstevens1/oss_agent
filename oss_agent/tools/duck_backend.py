import httpx
from bs4 import BeautifulSoup

import logging
import chz

from gpt_oss.tools.simple_browser.page_contents import process_html, PageContents
from gpt_oss.tools.simple_browser.backend import Backend, ClientSession

from duckduckgo_mcp_server.server import DuckDuckGoSearcher
import datetime
import asyncio

logger = logging.getLogger(__name__)

BASE_URL = "https://html.duckduckgo.com/html"
HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

VIEW_SOURCE_PREFIX = "view-source:"



class Context:

    async def info(self, string):
        logger.info(string)

    async def error(self, string):
        logger.error(string)





@chz.chz(typecheck=True)
class DuckBackend(Backend):
    source: str = chz.field(doc="Search the web with duck duck go")
    searcher = DuckDuckGoSearcher()
    context = Context()
    

    async def search(
        self,
        query: str,
        topn: int,
        session: ClientSession,
    ) -> PageContents:
        data = await self.searcher.search(query, self.context, topn)
        titles_and_urls = [
            (result.title, result.link, result.snippet)
            for result in data
        ]
        html_page = f"""
        <html><body>
        <h1>Search Results</h1>
        <ul>
        {"".join([f"<li><a href='{url}'>{title}</a> {summary}</li>" for title, url, summary in titles_and_urls])}
        </ul>
        </body></html>
        """

        return process_html(
            html=html_page,
            url="",
            title=query,
            display_urls=True,
            session=session,
        )


    async def fetch(self, url: str, session: ClientSession) -> PageContents:
        is_view_source = url.startswith(VIEW_SOURCE_PREFIX)
        if is_view_source:
            url = url[len(VIEW_SOURCE_PREFIX) :]

        async with httpx.AsyncClient() as client:
            response = await client.get(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    },
                    follow_redirects=True,
                    timeout=30.0,
                )
            response.raise_for_status()
        return process_html(
            html=response.text,
            url=url,
            title=None,
            display_urls=True,
            session=session
        )

    def get_api_key():
        return ""
