import streamlit as st
import cv2
import numpy as np
import pandas as pd
import json
import io
import tempfile
import os
from collections import defaultdict
from datetime import datetime

# -------------------------------
# App configuration
# -------------------------------
st.set_page_config(page_title="Video + Bounding Boxes Viewer", layout="wide")

# Example tracking data in JSONL format (one JSON object per line)
DEFAULT_EVENTS_JSONL = """
{"event_id": "1562891a-04d9-438b-beb0-c0829c9a2aa6", "event_type": "park_start", "event_time": 1755809460000, "publication_time": 1757019381503, "event_session_id": "2cfa9bb9-8d04-5bc8-be16-22d82c823e06", "global_id": 1, "vehicle_type": "other", "blocked_lane_types": "parking", "purpose": "unspecified", "bbox_x1": 904, "bbox_y1": 245, "bbox_x2": 1164, "bbox_y2": 459, "confidence": 0.9822140336036682, "camera_id": "00000000-0000-0000-0000-000000000001", "curb_zone_id": "09644318-805d-538c-abe0-ac718c080092", "latitude": 40.789, "longitude": -73.974}
{"event_id": "f8034af1-7511-431e-8d56-1faceff1c6f3", "event_type": "park_start", "event_time": 1755809499200, "publication_time": 1757019381505, "event_session_id": "d6fd7a28-7df8-5d1d-bcaa-4943b13ffe43", "global_id": 2, "vehicle_type": "van", "blocked_lane_types": "parking", "purpose": "delivery", "bbox_x1": 240, "bbox_y1": 176, "bbox_x2": 477, "bbox_y2": 295, "confidence": 0.9819761514663696, "camera_id": "00000000-0000-0000-0000-000000000001", "curb_zone_id": "e1b69828-45a3-55c8-bbe7-8b967724052b", "latitude": 40.789, "longitude": -73.974}
{"event_id": "88219ef3-bb53-4055-946a-6e1d47821253", "event_type": "park_start", "event_time": 1755809500000, "publication_time": 1757019381507, "event_session_id": "2eccd749-a343-562e-a317-63dc0c82ce56", "global_id": 3, "vehicle_type": "other", "blocked_lane_types": "parking", "purpose": "unspecified", "bbox_x1": 516, "bbox_y1": 144, "bbox_x2": 771, "bbox_y2": 240, "confidence": 0.9795910120010376, "camera_id": "00000000-0000-0000-0000-000000000001", "curb_zone_id": "e1b69828-45a3-55c8-bbe7-8b967724052b", "latitude": 40.789, "longitude": -73.974}
{"event_id": "fffc9986-8713-4b8e-91f4-e8dbc6bf5ccd", "event_type": "park_start", "event_time": 1755809460000, "publication_time": 1757019381508, "event_session_id": "ba17aa86-d9a3-5aab-8303-d128346a3709", "global_id": 4, "vehicle_type": "other", "blocked_lane_types": "parking", "purpose": "unspecified", "bbox_x1": 1109, "bbox_y1": 211, "bbox_x2": 1269, "bbox_y2": 349, "confidence": 0.9744930863380432, "camera_id": "00000000-0000-0000-0000-000000000001", "curb_zone_id": "09644318-805d-538c-abe0-ac718c080092", "latitude": 40.789, "longitude": -73.974}
""".strip()

# -------------------------------
# Helpers
# -------------------------------

def save_uploaded_video(uploaded_file) -> str:
    """Persist the uploaded video to a temporary file and return the path."""
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name

@st.cache_data(show_spinner=False)
def get_video_meta(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {"fps": fps, "frame_count": frame_count, "width": width, "height": height}

@st.cache_data(show_spinner=False)
def parse_events_jsonl(text: str) -> list:
    events = []
    if not text:
        return events
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            e = json.loads(ln)
            # Normalize expected keys
            for k in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]:
                if k in e:
                    e[k] = int(e[k])
            if "confidence" in e and e["confidence"] is not None:
                e["confidence"] = float(e["confidence"])
            events.append(e)
        except Exception:
            # Ignore malformed lines
            continue
    return events

@st.cache_data(show_spinner=False)
def parse_events_jsonl_stream(uploaded_file) -> list:
    """Parse a newline-delimited JSON (.jsonl) file without loading it all into memory."""
    events = []
    if uploaded_file is None:
        return events
    try:
        uploaded_file.seek(0)
        for ln in io.TextIOWrapper(uploaded_file, encoding="utf-8", errors="ignore"):
            ln = ln.strip()
            if not ln:
                continue
            try:
                e = json.loads(ln)
                for k in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]:
                    if k in e:
                        e[k] = int(e[k])
                if "confidence" in e and e["confidence"] is not None:
                    e["confidence"] = float(e["confidence"])
                events.append(e)
            except Exception:
                continue
    finally:
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
    return events

@st.cache_data(show_spinner=False)
def map_events_to_frames(events: list, fps: float, video_start_ms: int) -> list:
    """Return a copy of events augmented with a computed 'frame_idx' field.
    If an event already includes 'frame' or 'frame_idx', that is used.
    """
    mapped = []
    for e in events:
        e2 = dict(e)
        if "frame_idx" in e2:
            mapped.append(e2)
            continue
        if "frame" in e2:
            e2["frame_idx"] = int(e2["frame"])  # allow alternate naming
            mapped.append(e2)
            continue
        # Fallback: compute from epoch-millis event_time and video_start_ms
        if "event_time" in e2 and video_start_ms is not None:
            dt_ms = int(e2["event_time"]) - int(video_start_ms)
            frame_idx = int(round((dt_ms / 1000.0) * fps))
            e2["frame_idx"] = frame_idx
            mapped.append(e2)
    return mapped

@st.cache_data(show_spinner=False)
def group_by_frame(events_with_frames: list) -> dict:
    by_frame = defaultdict(list)
    for e in events_with_frames:
        if "frame_idx" in e and isinstance(e["frame_idx"], int):
            by_frame[e["frame_idx"]].append(e)
    return by_frame

def read_frame(video_path: str, frame_idx: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return frame  # BGR

def draw_boxes(frame_bgr: np.ndarray, events: list, show_labels: bool = True) -> np.ndarray:
    out = frame_bgr.copy()
    for e in events:
        x1, y1, x2, y2 = e.get("bbox_x1"), e.get("bbox_y1"), e.get("bbox_x2"), e.get("bbox_y2")
        if None in (x1, y1, x2, y2):
            continue
        # Choose a deterministic color by global_id
        gid = int(e.get("global_id", 0))
        # Simple color hashing
        color = ((37 * gid) % 255, (17 * gid) % 255, (97 * gid) % 255)  # BGR
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        if show_labels:
            label = f"{e.get('event_type','')}: {e.get('confidence', 0):.2f}"
            cv2.putText(out, label, (int(x1), max(0, int(y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out

# -------------------------------
# UI
# -------------------------------
st.title("Video Clip Viewer with Bounding Boxes")

with st.sidebar:
    st.header("Inputs")
    video_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"])    
    events_file = st.file_uploader("Upload tracking JSONL (.jsonl) (optional)", type=["jsonl", "json"])     
    st.caption("Tracking format: one JSON object per line, with bbox_x1/y1/x2/y2 and event_time (ms epoch).")

    paste_toggle = st.checkbox("Or paste tracking JSONL", value=events_file is None)
    events_text = ""
    if paste_toggle:
        events_text = st.text_area("Tracking data (JSONL)", value=DEFAULT_EVENTS_JSONL, height=180)

    show_labels = st.checkbox("Show labels", value=True)
    tolerance = st.number_input("Frame tolerance (± frames)", min_value=0, max_value=30, value=0, step=1)

# Load / save video
video_path = None
meta = None
if video_file is not None:
    video_path = save_uploaded_video(video_file)
    meta = get_video_meta(video_path)

# Parse events
raw_events = []
if events_file is not None:
    raw_events = parse_events_jsonl_stream(events_file)
elif paste_toggle and events_text:
    raw_events = parse_events_jsonl(events_text)

# Derive default video start ms from earliest event if present
min_event_ms = min([e.get("event_time") for e in raw_events if e.get("event_time") is not None], default=None)
with st.sidebar:
    if min_event_ms is not None:
        default_start_ms = int(min_event_ms)
    else:
        default_start_ms = 0
    video_start_ms = st.number_input(
        "Video start timestamp (ms since epoch)",
        value=int(default_start_ms),
        step=1000,
        help=(
            "Used to map event_time → frame index. If your events already contain a 'frame' or 'frame_idx', this is ignored."
        ),
    )

# Map events to frames once FPS is known
mapped_events = []
by_frame = {}
if meta is not None and raw_events:
    mapped_events = map_events_to_frames(raw_events, meta["fps"], video_start_ms)
    by_frame = group_by_frame(mapped_events)

# Layout
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Video & Annotations")
    if video_path and meta:
        fps = meta["fps"]
        frame_count = meta["frame_count"]
        width, height = meta["width"], meta["height"]
        duration_sec = frame_count / fps if fps > 0 else 0

        st.markdown(
            f"**FPS:** {fps:.3f} | **Frames:** {frame_count} | **Resolution:** {width}×{height} | **Duration:** {duration_sec:.2f}s"
        )

        # Frame selection controls
        frame_idx = st.slider("Select frame", min_value=0, max_value=max(0, frame_count - 1), value=0, step=1)
        frame_idx = st.number_input("Or type frame index", min_value=0, max_value=max(0, frame_count - 1), value=int(frame_idx), step=1)

        # Read and annotate current frame
        frame_bgr = read_frame(video_path, int(frame_idx))
        if frame_bgr is None:
            st.warning("Could not read this frame.")
        else:
            # Determine which events to draw
            events_now = []
            if by_frame:
                # exact or within tolerance
                for f in range(int(frame_idx) - tolerance, int(frame_idx) + tolerance + 1):
                    events_now.extend(by_frame.get(f, []))
            annotated = draw_boxes(frame_bgr, events_now, show_labels=show_labels)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)
    else:
        st.info("Upload a video to begin.")

with col_right:
    st.subheader("Events")
    if mapped_events:
        # Show a small summary and a per-frame filter
        st.markdown(f"Total events parsed: **{len(mapped_events)}**")
        if meta is not None:
            with st.expander("Event → frame mapping details", expanded=False):
                if min_event_ms is not None:
                    dt = datetime.utcfromtimestamp(min_event_ms / 1000.0)
                    st.write(f"Earliest event_time: {min_event_ms} (UTC {dt.isoformat()}Z)")
                st.write(f"Video start (ms): {video_start_ms}")
                st.write(f"FPS used: {meta['fps']:.3f}")
        
        # If a frame is selected on the left, filter table to the tolerance window
        if meta is not None and 'frame_idx' in locals():
            lower = max(0, int(frame_idx) - tolerance)
            upper = int(frame_idx) + tolerance
            filtered = [e for e in mapped_events if lower <= int(e.get('frame_idx', -1)) <= upper]
        else:
            filtered = mapped_events
        
        if filtered:
            df = pd.DataFrame(filtered)
            # Keep common columns up front
            front_cols = [
                "frame_idx", "event_id", "event_type", "confidence",
                "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
                "event_time", "global_id", "vehicle_type", "purpose",
            ]
            cols = [c for c in front_cols if c in df.columns] + [c for c in df.columns if c not in front_cols]
            st.dataframe(df[cols], use_container_width=True, hide_index=True)
        else:
            st.write("No events for the current selection.")
    else:
        st.info("Upload or paste tracking data to see events.")

st.markdown("---")
st.caption(
    "Tip: If your tracking rows already include a 'frame' or 'frame_idx' column, the epoch 'event_time' → frame mapping is skipped."
)
