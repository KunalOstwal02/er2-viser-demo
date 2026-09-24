"""Gemini Robotics ER 2 client, prompts and JSON response parsing."""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from io import BytesIO
from typing import Literal, Protocol

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field, ValidationError

log = logging.getLogger(__name__)

MODEL_ID = os.environ.get("ER2_MODEL_ID", "gemini-robotics-er-2-preview")

SYSTEM_PROMPT = """\
You are the embodied-reasoning brain of a Franka Panda robot arm with a parallel-jaw gripper,
working on a tabletop. You see two camera images:
  Image 1 (FRONT camera): faces the robot from the far side of the table. The robot base is at
  the top of the image. Objects on the robot's left appear on the RIGHT of this image.
  Image 2 (SIDE camera): an extra view from the robot's right side, for depth and occlusion.

Accomplish the user's task by choosing exactly ONE action per turn. After each action you get
fresh images and the result, so you can correct mistakes.

Actions:
  "pick"  – grasp an object. points: [[y, x]] ON the object to pick, in Image 1.
  "place" – put down the held object. points: [[y, x]] in Image 1 where it should go.
            To put it in a bowl/bin, point at the inside of that container.
            To STACK it on another object, point at the top face of the object underneath.
            To put it on the table, point at an empty patch of table away from other objects.
  "home"  – move the arm back to its home pose (e.g. if it blocks the view).
  "done"  – the task is complete (or impossible); explain in "thought".

Rules:
  * Points are [y, x] normalised to 0–1000 in Image 1 (y down, x right).
  * You can hold only one object at a time; always "place" before the next "pick".
  * Only use objects you can see. Do not repeat an action that already succeeded.
  * Flat coloured mats are zones on the table, not objects; objects standing on them are "on the mat".
  * Use the side camera to judge relative heights and sizes.

Reply with ONLY a JSON object:
{"thought": "<short reasoning>", "action": "pick|place|home|done",
 "points": [[y, x]], "label": "<object or location name>"}
"""

ASSESS_PROMPT = """\
You were controlling a robot arm to perform this task: "{task}".
The image shows the final state of the table (front camera; robot at the top).
Did the robot fully accomplish the task? Reply with ONLY a JSON object:
{{"success": true|false, "explanation": "<one or two sentences>"}}
"""


class Action(BaseModel):
    thought: str = ""
    action: Literal["pick", "place", "home", "done"]
    points: list[list[float]] = Field(default_factory=list)
    label: str = ""

    def point_tuples(self) -> list[tuple[float, float]]:
        out = []
        for p in self.points:
            if len(p) >= 2:
                out.append((float(np.clip(p[0], 0, 1000)), float(np.clip(p[1], 0, 1000))))
        return out


class Assessment(BaseModel):
    success: bool
    explanation: str = ""


class ER2Error(RuntimeError):
    pass


def extract_json(text: str) -> dict:
    """Parse the first JSON object in ``text`` (tolerates code fences / chatter / lists)."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # ER-2 sometimes appends a stray "}" or chatter after a valid object: decode the first
        # complete JSON value and ignore whatever follows it.
        starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
        if not starts:
            raise ER2Error("response contained no JSON object") from None
        try:
            data, _ = json.JSONDecoder().raw_decode(text[min(starts):])
        except json.JSONDecodeError as error:
            raise ER2Error(f"invalid JSON: {error}") from None
    if isinstance(data, list) and data and isinstance(data[0], dict):
        data = data[0]
    if not isinstance(data, dict):
        raise ER2Error("response JSON is not an object")
    return data


def parse_action(text: str) -> Action:
    data = extract_json(text)
    # Tolerate a single point given as [y, x] or {"point": [y, x]}.
    if "point" in data and "points" not in data:
        data["points"] = [data.pop("point")]
    pts = data.get("points")
    if isinstance(pts, list) and len(pts) == 2 and all(isinstance(v, (int, float)) for v in pts):
        data["points"] = [pts]
    if isinstance(data.get("action"), str):
        data["action"] = data["action"].strip().lower()
    try:
        return Action.model_validate(data)
    except ValidationError as error:
        raise ER2Error(f"response did not match the action schema: {error.errors()[0]['msg']}") from None


def parse_assessment(text: str) -> Assessment:
    try:
        return Assessment.model_validate(extract_json(text))
    except ValidationError as error:
        raise ER2Error(f"bad assessment: {error.errors()[0]['msg']}") from None


def _jpeg_b64(image: np.ndarray) -> str:
    buf = BytesIO()
    Image.fromarray(image).save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class Client(Protocol):
    def decide(self, text: str, images: list[np.ndarray]) -> str: ...
    def assess(self, text: str, image: np.ndarray) -> str: ...


class GeminiER2:
    """Stateless per-turn requests (full context each turn) → raw response text."""

    def __init__(self, model_id: str = MODEL_ID, timeout_s: float = 60.0) -> None:
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ER2Error("GEMINI_API_KEY is not set")
        from google import genai

        self._client = genai.Client(api_key=key)
        del key
        self.model_id = model_id
        self.timeout_s = timeout_s
        self._transport = os.environ.get("ER2_TRANSPORT", "auto")  # auto | interactions | generate

    def decide(self, text: str, images: list[np.ndarray]) -> str:
        return self._request(SYSTEM_PROMPT, text, images)

    def assess(self, text: str, image: np.ndarray) -> str:
        return self._request(None, text, [image])

    def _request(self, system: str | None, text: str, images: list[np.ndarray]) -> str:
        last: Exception | None = None
        for attempt in range(3):
            try:
                if self._transport in ("auto", "interactions"):
                    try:
                        return self._interactions(system, text, images)
                    except Exception as error:  # noqa: BLE001
                        if self._transport == "interactions" or _is_auth(error):
                            raise
                        log.warning("interactions API failed (%s); falling back to generate_content", error)
                        self._transport = "generate"
                return self._generate(system, text, images)
            except Exception as error:  # noqa: BLE001
                if _is_auth(error):
                    raise ER2Error("Gemini rejected the API key or model access") from None
                last = error
                log.warning("ER-2 request failed (attempt %d): %s", attempt + 1, error)
                time.sleep(1.5 * (attempt + 1))
        raise ER2Error(f"ER-2 request failed: {type(last).__name__}: {last}")

    def _interactions(self, system: str | None, text: str, images: list[np.ndarray]) -> str:
        content = [{"type": "text", "text": text}] + [
            {"type": "image", "mime_type": "image/jpeg", "data": _jpeg_b64(im)} for im in images
        ]
        kwargs = dict(model=self.model_id, input=[{"type": "user_input", "content": content}],
                      response_mime_type="application/json", store=False)
        if system:
            kwargs["system_instruction"] = system
        response = self._client.interactions.create(**kwargs)
        out = getattr(response, "output_text", None)
        if not out:
            raise ER2Error("empty interaction response")
        return out

    def _generate(self, system: str | None, text: str, images: list[np.ndarray]) -> str:
        from google.genai import types

        parts = [types.Part.from_text(text=text)] + [
            types.Part.from_bytes(data=base64.b64decode(_jpeg_b64(im)), mime_type="image/jpeg") for im in images
        ]
        config = types.GenerateContentConfig(system_instruction=system, response_mime_type="application/json",
                                             temperature=0.3)
        response = self._client.models.generate_content(model=self.model_id, contents=parts, config=config)
        if not response.text:
            raise ER2Error("empty response")
        return response.text


def _is_auth(error: Exception) -> bool:
    code = getattr(error, "code", None) or getattr(error, "status_code", None)
    msg = str(error).lower()
    return code in (401, 403) or "api key" in msg or "permission" in msg or "unauthenticated" in msg


class ReplayClient:
    """Feeds back the raw responses of a recorded episode, in order."""

    def __init__(self, turns: list[str], assessment: str | None) -> None:
        self._turns = list(turns)
        self._assessment = assessment

    def decide(self, text: str, images: list[np.ndarray]) -> str:
        del text, images
        if not self._turns:
            return json.dumps({"thought": "End of recording.", "action": "done", "points": []})
        time.sleep(0.6)  # keep the replay's pacing readable
        return self._turns.pop(0)

    def assess(self, text: str, image: np.ndarray) -> str:
        del text, image
        return self._assessment or json.dumps({"success": False, "explanation": "No assessment recorded."})
