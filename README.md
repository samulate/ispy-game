# I Spy With My Little AI

A photo guessing game using YOLO object detection and sentence embeddings.

## Run locally with uv

From the project directory:

```bash
uv sync --locked
uv run python app.py
```

Upload a photo with recognizable objects, then
guess the object from its color clue.

uv installs Python 3.12 automatically if needed. This app requires Python 3.12
because it uses `imghdr`, which was removed in Python 3.13. The first launch
downloads `yolo11x-seg.pt` and `all-MiniLM-L6-v2`; allow time for these downloads
and model loading before opening the page.

Dependencies are managed in `pyproject.toml` and pinned in `uv.lock`. Use
`uv add <package>` when adding a dependency.
