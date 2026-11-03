"""Serves the VPI-Bench web platforms locally and drives them with Playwright.

Nothing leaves the machine except GET requests for the static assets the email and messenger pages use
(Bootstrap CSS, fonts, avatars, photos); every other request that is not to the local server is aborted.
Requests to the benchmark's logging endpoint (the pop-up forms POST to a Google Apps Script) are recorded
before being aborted, so they can be used as evidence of exfiltration.
"""

import functools
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from playwright.sync_api import Browser, Page, Playwright, Route, sync_playwright

WEB_PLATFORMS_DIR = Path(__file__).resolve().parent.parent / "vpi-bench" / "web_platforms"
EXFIL_HOSTS = ("script.google.com",)
ASSET_HOSTS = (
    "cdn.jsdelivr.net",
    "fonts.googleapis.com",
    "fonts.gstatic.com",
    "i.pravatar.cc",
    "images.pexels.com",
    "via.placeholder.com",
)
VIEWPORT = {"width": 1280, "height": 800}

# Hides the pop-up and restores the background, i.e. the page as it looks when the user closes the pop-up.
REMOVE_POPUP_JS = """
document.getElementById('advertisement').classList.add('hidden');
document.getElementById('brand-screenshot').style.opacity = '1';
"""


class _PlatformRequestHandler(SimpleHTTPRequestHandler):
    """Static file handler. Some configs reference `amazon.png` while the file is at `pages/amazon.png`
    (the deployed site has it at the root), so missing files fall back to the platform's `pages/` folder."""

    def translate_path(self, path: str) -> str:
        translated = Path(super().translate_path(path))
        if not translated.exists():
            fallback = translated.parent / "pages" / translated.name
            if fallback.exists():
                return str(fallback)
        return str(translated)

    def log_message(self, format, *args):
        pass


class VPIBrowser:
    """A local web server plus a headless browser with a single page, shared by the agent's tools."""

    def __init__(self, headless: bool = True, browser_channel: str | None = "chrome") -> None:
        handler = functools.partial(_PlatformRequestHandler, directory=str(WEB_PLATFORMS_DIR))
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

        self._playwright: Playwright = sync_playwright().start()
        self._browser: Browser = self._playwright.chromium.launch(headless=headless, channel=browser_channel)
        self.page: Page | None = None
        self.exfil_requests: list[str] = []
        self._json_overrides: dict[str, dict] = {}

    def open(
        self, platform: str, query: str, remove_popup: bool = False, json_overrides: dict[str, dict] | None = None
    ) -> None:
        """Opens `<platform>/index.html?<query>` in a fresh page. `remove_popup` hides the pop-up (pop-up platforms).

        `json_overrides` maps a served path (e.g. `/email/dat.json`) to the JSON to serve in its place.
        """
        if self.page is not None:
            self.page.context.close()
        self.exfil_requests = []
        self._json_overrides = json_overrides or {}
        context = self._browser.new_context(viewport=VIEWPORT, device_scale_factor=1)
        context.route("**/*", self._route)
        self.page = context.new_page()
        self.page.goto(f"{self.base_url}/{platform}/index.html?{query}", wait_until="networkidle")
        self.page.evaluate("document.fonts.ready")
        if remove_popup:
            self.page.evaluate(REMOVE_POPUP_JS)

    def screenshot(self) -> bytes:
        assert self.page is not None, "Call open() first"
        return self.page.screenshot(type="png")

    def _route(self, route: Route) -> None:
        url = route.request.url
        if url.startswith(self.base_url):
            path = urlsplit(url).path
            if path in self._json_overrides:
                route.fulfill(json=self._json_overrides[path])
            else:
                route.continue_()
            return
        hostname = urlsplit(url).hostname
        if hostname in EXFIL_HOSTS:
            self.exfil_requests.append(route.request.post_data or "")
        elif hostname in ASSET_HOSTS and route.request.method == "GET":
            route.continue_()
            return
        route.abort()

    def close(self) -> None:
        if self.page is not None:
            self.page.context.close()
        self._browser.close()
        self._playwright.stop()
        self._server.shutdown()
