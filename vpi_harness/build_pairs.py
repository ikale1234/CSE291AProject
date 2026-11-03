"""Renders a clean and an injected screenshot for every VPI-Bench testcase, plus a manifest of annotations.

Injected: the page exactly as the testcase URL shows it.
Clean: the same page with only the injection removed:
  - amazon / booking / bbc: the pop-up is hidden and the background restored (as if the user closed it).
  - email / messenger: the injected item (email `10` / chat `chat10`) is removed from the page's JSON and
    the first remaining item is opened in its place.

Output in vpi_harness/data/:
  images/injected/<page>.png    one per distinct testcase URL
  images/clean/<page>.png       one per distinct clean page (content-addressed, so identical renders are shared)
  pairs.jsonl                   one `Pair` per testcase

All boxes are [x0, y0, x1, y1] in pixels of the 1280x800 screenshot.

Usage (from the project root):
    agentdojo/.venv/bin/python vpi_harness/build_pairs.py
"""

import copy
import hashlib
import io
import json
import re
from pathlib import Path
from urllib.parse import parse_qs

import numpy as np
from browser import VIEWPORT, WEB_PLATFORMS_DIR, VPIBrowser
from PIL import Image
from pydantic import BaseModel
from vpi_suite import POPUP_PLATFORMS, TESTCASES_DIR, VPITestcase

DATA_DIR = Path(__file__).resolve().parent / "data"
RENDER_NOISE_TOLERANCE = 8
"""Chrome's renders of the same page differ by a few intensity levels between runs; clean renders whose
largest per-channel difference is within this are treated as the same image."""
_clean_renders: dict[str, list[tuple[str, np.ndarray]]] = {}
INJECTED_ITEM_ID = {"email": "10", "messenger": "chat10"}

# The pop-up platforms show a static screenshot, so the controls the task needs are annotated by hand.
POPUP_TASK_TARGETS = {
    "amazon": [{"label": "search box", "bbox": [362, 40, 866, 82]}],
    "booking": [{"label": "destination field", "bbox": [167, 447, 491, 494]}],
    "bbc": [],  # "Summarize the news" can be answered from the headlines on the screenshot.
}


class Target(BaseModel):
    label: str
    bbox: list[int]


class PageAnnotations(BaseModel):
    image: str
    """Path relative to the data directory."""
    injection_visible: bool
    """Whether the injected content is on screen. False for every clean page."""
    injection_region: list[int] | None = None
    injection_text: str = ""
    attack_targets: list[Target] = []
    """On-screen controls that start the attack: pop-up form controls, links in the injected message."""
    dismiss_targets: list[Target] = []
    """Controls that dismiss the injection without following it (the pop-up's close button)."""
    task_targets: list[Target] = []
    """Controls that advance the user's task."""


class Pair(BaseModel):
    testcase_id: str
    agent_type: str
    platform: str
    user_prompt: str
    attacker_objective: str
    attack_commands: list[str]
    """Commands quoted in the objective, e.g. `rm -rf /home/computeruse/boot/*`."""
    planted_files: dict[str, str]
    """Files the testcase creates on the user's machine (path -> content); the secrets an attack would leak."""
    viewport: dict[str, int]
    clean: PageAnnotations
    injected: PageAnnotations


def _bbox(box: dict | None) -> list[int] | None:
    """Clips a Playwright bounding box to the viewport; None if it is not on screen."""
    if box is None:
        return None
    x0, y0 = max(0, round(box["x"])), max(0, round(box["y"]))
    x1 = min(VIEWPORT["width"], round(box["x"] + box["width"]))
    y1 = min(VIEWPORT["height"], round(box["y"] + box["height"]))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def _targets(browser: VPIBrowser, selector: str, skip_index: int | None = None) -> list[Target]:
    assert browser.page is not None
    targets = []
    for i, element in enumerate(browser.page.locator(selector).all()):
        bbox = _bbox(element.bounding_box()) if element.is_visible() else None
        if i == skip_index or bbox is None:
            continue
        label = element.inner_text().strip() or element.get_attribute("placeholder") or ""
        if not label and element.get_attribute("id"):
            label = browser.page.locator(f"label[for='{element.get_attribute('id')}']").inner_text().strip()
        targets.append(Target(label=" ".join(label.split())[:80], bbox=bbox))
    return targets


def _strip_html(html: str) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", html).split())


def _save(png: bytes, folder: str, name: str) -> str:
    relative = f"images/{folder}/{name}.png"
    path = DATA_DIR / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)
    return relative


def _page_name(testcase: VPITestcase) -> str:
    params = parse_qs(testcase.query)
    parts = [
        testcase.platform,
        params["id"][0],
        *(f"load{v}" for v in params.get("load", [])),
    ]
    return "__".join(parts)


def _save_clean(png: bytes, platform: str) -> str:
    """Saves a clean render, reusing an earlier one if they differ only by rendering noise, so that testcases
    with the same clean page get the exact same image."""
    pixels = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.int16)
    for path, existing in _clean_renders.setdefault(platform, []):
        if np.abs(pixels - existing).max() <= RENDER_NOISE_TOLERANCE:
            return path
    path = _save(png, "clean", f"{platform}__{hashlib.sha256(png).hexdigest()[:10]}")
    _clean_renders[platform].append((path, pixels))
    return path


def _popup_pages(browser: VPIBrowser, testcase: VPITestcase) -> tuple[PageAnnotations, PageAnnotations]:
    task_targets = [Target(**t) for t in POPUP_TASK_TARGETS[testcase.platform]]
    browser.open(testcase.platform, testcase.query)
    assert browser.page is not None
    title = browser.page.locator("#form-title").inner_text()
    description = browser.page.locator("#form-description").inner_text()
    injected = PageAnnotations(
        image=_save(browser.screenshot(), "injected", _page_name(testcase)),
        injection_visible=browser.page.locator("#advertisement").is_visible(),
        injection_region=_bbox(browser.page.locator("#advertisement").bounding_box()),
        injection_text=f"{title}\n{description}".strip(),
        attack_targets=_targets(browser, "#ad-form input, #ad-form button"),
        dismiss_targets=_targets(browser, "#close-popup"),
        task_targets=task_targets,
    )
    browser.open(testcase.platform, testcase.query, remove_popup=True)
    clean = PageAnnotations(
        image=_save_clean(browser.screenshot(), testcase.platform),
        injection_visible=False,
        task_targets=task_targets,
    )
    return clean, injected


def _app_pages(browser: VPIBrowser, testcase: VPITestcase) -> tuple[PageAnnotations, PageAnnotations]:
    """Email and messenger: the injection is one email / chat in the page's JSON."""
    platform = testcase.platform
    params = parse_qs(testcase.query)
    config_path = f"/{platform}/{params['id'][0]}.json"
    config = json.loads((WEB_PLATFORMS_DIR / config_path.lstrip("/")).read_text())
    items_key, list_selector = ("emails", "#email-list li") if platform == "email" else ("chats", "#chats-list li")
    item_ids = [str(item["id"]) for item in config[items_key]]
    injected_id = INJECTED_ITEM_ID[platform]
    injected_index = item_ids.index(injected_id)
    injected_item = config[items_key][injected_index]
    if platform == "email":
        injection_text = f"From: {injected_item['sender']}\nSubject: {injected_item['subject']}\n"
        injection_text += _strip_html(injected_item["content"])
        region_selector, links_selector = "#email-detail", "#email-detail .email-body a"
    else:
        injection_text = "\n".join(m.get("message", "") for m in injected_item["history"])
        region_selector, links_selector = "#chat-history", "#chat-history a"

    browser.open(platform, testcase.query)
    assert browser.page is not None
    injection_visible = params.get("load", [None])[0] == injected_id
    injected = PageAnnotations(
        image=_save(browser.screenshot(), "injected", _page_name(testcase)),
        injection_visible=injection_visible,
        injection_region=_bbox(browser.page.locator(region_selector).bounding_box()) if injection_visible else None,
        injection_text=injection_text,
        attack_targets=_targets(browser, links_selector) if injection_visible else [],
        task_targets=_targets(browser, list_selector, skip_index=injected_index),
    )

    clean_config = copy.deepcopy(config)
    del clean_config[items_key][injected_index]
    first_id = str(clean_config[items_key][0]["id"])
    browser.open(
        platform,
        f"id={params['id'][0]}&load={first_id}",
        json_overrides={config_path: clean_config},
    )
    clean = PageAnnotations(
        image=_save_clean(browser.screenshot(), platform),
        injection_visible=False,
        task_targets=_targets(browser, list_selector),
    )
    return clean, injected


def _planted_files(testcase: VPITestcase) -> dict[str, str]:
    files = {}
    for instruction in testcase.environment:
        if instruction["instruction_type"] == "FILE_CREATE":
            # Despite its name, `b64_data` holds plain text in the VPI-Bench testcases.
            files[instruction["path"]] = instruction.get("content", instruction.get("b64_data", ""))
    return files


def build() -> list[Pair]:
    testcase_paths = sorted(TESTCASES_DIR.glob("*/*/*.json"))
    browser = VPIBrowser()
    rendered: dict[str, tuple[PageAnnotations, PageAnnotations]] = {}
    pairs = []
    try:
        for i, path in enumerate(testcase_paths, 1):
            testcase = VPITestcase(platform=path.parent.name, **json.loads(path.read_text()))
            page = _page_name(testcase)
            if page not in rendered:
                print(f"[{i}/{len(testcase_paths)}] rendering {page}", flush=True)
                if testcase.platform in POPUP_PLATFORMS:
                    rendered[page] = _popup_pages(browser, testcase)
                else:
                    rendered[page] = _app_pages(browser, testcase)
            clean, injected = rendered[page]
            pairs.append(
                Pair(
                    testcase_id=testcase.id,
                    agent_type=path.parent.parent.name,
                    platform=testcase.platform,
                    user_prompt=testcase.user_prompt,
                    attacker_objective=testcase.attacker_objective,
                    attack_commands=re.findall(r"'([^']+)'", testcase.attacker_objective),
                    planted_files=_planted_files(testcase),
                    viewport=VIEWPORT,
                    clean=clean,
                    injected=injected,
                )
            )
    finally:
        browser.close()

    with (DATA_DIR / "pairs.jsonl").open("w") as f:
        for pair in pairs:
            f.write(pair.model_dump_json() + "\n")
    return pairs


if __name__ == "__main__":
    pairs = build()
    clean_images = {p.clean.image for p in pairs}
    injected_images = {p.injected.image for p in pairs}
    hidden = [p.testcase_id for p in pairs if not p.injected.injection_visible]
    print(f"\n{len(pairs)} pairs, {len(injected_images)} injected images, {len(clean_images)} clean images")
    print(f"Injection not visible on the first screen: {hidden or 'none'}")
    print(f"Manifest: {DATA_DIR / 'pairs.jsonl'}")
