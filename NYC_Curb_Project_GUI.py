import streamlit as st
import cv2
import numpy as np
import pandas as pd
import json
import io
import xml.etree.ElementTree as ET
try:
    import requests
except Exception:
    requests = None
import tempfile
import os
import time
from collections import defaultdict
from datetime import datetime

# -------------------------------
# App configuration
# -------------------------------
st.set_page_config(page_title="Video + Bounding Boxes Viewer", layout="wide")

# Session state defaults
if 'playing' not in st.session_state:
    st.session_state.playing = False
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = 0
if 'last_tick' not in st.session_state:
    st.session_state.last_tick = 0.0
if 'play_speed' not in st.session_state:
    st.session_state.play_speed = 5.0
if 'looping' not in st.session_state:
    st.session_state.looping = False
if 'loop_range' not in st.session_state:
    st.session_state.loop_range = (0, 0)

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

def frame_to_ms(frame_idx: int, video_start_ms: int, fps: float) -> int:
    try:
        if fps <= 0:
            return int(video_start_ms)
        return int(video_start_ms + (int(frame_idx) / float(fps)) * 1000)
    except Exception:
        return int(video_start_ms)

def ms_to_frame(timestamp_ms: int, video_start_ms: int, fps: float) -> int:
    try:
        if fps <= 0:
            return 0
        dt_ms = int(timestamp_ms) - int(video_start_ms)
        return int(round((dt_ms / 1000.0) * float(fps)))
    except Exception:
        return 0

# ---- CVAT polygon helpers ----

def hex_to_bgr(hex_color: str):
    if not hex_color:
        return (0, 255, 0)
    s = hex_color.strip().lstrip('#')
    if len(s) == 3:
        s = ''.join([c*2 for c in s])
    try:
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
        return (b, g, r)
    except Exception:
        return (0, 255, 0)

@st.cache_data(show_spinner=False)
def parse_cvat_xml(xml_text: str):
    """Parse CVAT XML polygons → {frame_idx: [{label, points, color_hex}]} and label color map."""
    by_frame = defaultdict(list)
    label_colors = {}
    if not xml_text:
        return by_frame, label_colors
    root = ET.fromstring(xml_text)
    # collect label colors
    for lbl in root.findall('.//meta/job/labels/label'):
        name = (lbl.findtext('name') or '').strip()
        color = (lbl.findtext('color') or '#00FF00').strip()
        if name:
            label_colors[name] = color
    # collect polygons
    for trk in root.findall('.//track'):
        label = trk.attrib.get('label', '')
        color = label_colors.get(label, '#00FF00')
        for poly in trk.findall('polygon'):
            if poly.attrib.get('outside', '0') == '1':
                continue
            frame_idx = int(poly.attrib.get('frame', '0'))
            pts_attr = (poly.attrib.get('points') or '').strip()
            pts = []
            for pair in pts_attr.split(';'):
                if not pair:
                    continue
                try:
                    x_str, y_str = pair.split(',')
                    pts.append((int(round(float(x_str))), int(round(float(y_str)))))
                except Exception:
                    continue
            if len(pts) >= 3:
                by_frame[frame_idx].append({
                    'label': label,
                    'points': pts,
                    'color': color,
                })
    return by_frame, label_colors

@st.cache_data(show_spinner=False)
def flatten_cvat_polygons(by_frame_dict: dict) -> list:
    """Flatten {frame_idx: [polys]} into a single list of polygons (ignoring frame)."""
    polys = []
    if not by_frame_dict:
        return polys
    for _f, lst in by_frame_dict.items():
        polys.extend(lst)
    return polys
    root = ET.fromstring(xml_text)
    # collect label colors
    for lbl in root.findall('.//meta/job/labels/label'):
        name = (lbl.findtext('name') or '').strip()
        color = (lbl.findtext('color') or '#00FF00').strip()
        if name:
            label_colors[name] = color
    # collect polygons
    for trk in root.findall('.//track'):
        label = trk.attrib.get('label', '')
        color = label_colors.get(label, '#00FF00')
        for poly in trk.findall('polygon'):
            if poly.attrib.get('outside', '0') == '1':
                continue
            frame_idx = int(poly.attrib.get('frame', '0'))
            pts_attr = (poly.attrib.get('points') or '').strip()
            pts = []
            for pair in pts_attr.split(';'):
                if not pair:
                    continue
                try:
                    x_str, y_str = pair.split(',')
                    pts.append((int(round(float(x_str))), int(round(float(y_str)))))
                except Exception:
                    continue
            if len(pts) >= 3:
                by_frame[frame_idx].append({
                    'label': label,
                    'points': pts,
                    'color': color,
                })
    return by_frame, label_colors

@st.cache_data(show_spinner=False)
def fetch_url_text(url: str) -> str:
    if requests is None:
        raise RuntimeError("'requests' not available")
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.text

def overlay_polygons(frame_bgr: np.ndarray, polys: list, alpha: float = 0.35, edge_thickness: int = 2) -> np.ndarray:
    if not polys:
        return frame_bgr
    alpha = float(max(0.0, min(1.0, alpha)))
    out = frame_bgr.copy()
    overlay = frame_bgr.copy()
    mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    for p in polys:
        pts = np.array(p.get('points', []), dtype=np.int32).reshape((-1, 1, 2))
        if pts.size == 0:
            continue
        color_bgr = hex_to_bgr(p.get('color', '#00FF00'))
        cv2.fillPoly(overlay, [pts], color_bgr)
        cv2.fillPoly(mask, [pts], 255)
    # Blend only where mask is set
    blended = cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0)
    out[mask > 0] = blended[mask > 0]
    # Draw edges on top with full opacity
    for p in polys:
        pts = np.array(p.get('points', []), dtype=np.int32).reshape((-1, 1, 2))
        if pts.size == 0:
            continue
        color_bgr = hex_to_bgr(p.get('color', '#00FF00'))
        if edge_thickness > 0:
            cv2.polylines(out, [pts], isClosed=True, color=color_bgr, thickness=int(edge_thickness), lineType=cv2.LINE_AA)
    return out

def read_frame(video_path: str, frame_idx: int):
    """Read a frame using a persistent VideoCapture for smoother playback."""
    cap = st.session_state.get('cap', None)
    if cap is None or st.session_state.get('cap_path') != video_path:
        # Initialize or reinitialize capture
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        cap = cv2.VideoCapture(video_path)
        st.session_state['cap'] = cap
        st.session_state['cap_path'] = video_path
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_idx)))
    ok, frame = cap.read()
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


def ms_to_frame(ts_ms: int, start_ms: int, fps: float) -> int:
    if fps <= 0:
        return 0
    return int(round(((int(ts_ms) - int(start_ms)) / 1000.0) * float(fps)))


def frame_to_ms(frame_idx: int, start_ms: int, fps: float) -> int:
    if fps <= 0:
        return int(start_ms)
    return int(start_ms) + int(round((int(frame_idx) / float(fps)) * 1000.0))

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

    # --- CVAT XML annotations ---
    show_polygons = st.checkbox("Show CVAT polygons", value=True)
    with st.expander("Annotations (CVAT XML)", expanded=False):
        xml_file = st.file_uploader("Upload CVAT XML", type=["xml"])    
        xml_url = st.text_input("Or GitHub raw URL to XML", value="", help="Paste the raw URL to the XML file in your GitHub repo.")
        poly_alpha = st.slider("Polygon fill opacity", 0.0, 1.0, 0.35, 0.05)
        poly_edge_thickness = st.number_input("Polygon edge thickness", min_value=0, max_value=10, value=2, step=1)
        poly_all_frames = st.checkbox("Show polygons on all frames", value=True, help="Draw all parsed polygons on every frame (ignores per-frame indices).")

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

# Parse CVAT XML annotations
cvat_polys_by_frame = {}
label_colors = {}
xml_error = None
if 'xml_file' in locals() and xml_file is not None:
    try:
        xml_text = xml_file.read().decode('utf-8', errors='ignore')
        cvat_polys_by_frame, label_colors = parse_cvat_xml(xml_text)
    except Exception as e:
        xml_error = str(e)
elif 'xml_url' in locals() and xml_url:
    try:
        xml_text = fetch_url_text(xml_url)
        cvat_polys_by_frame, label_colors = parse_cvat_xml(xml_text)
    except Exception as e:
        xml_error = str(e)
# Flatten polygons for all-frames overlay
cvat_polys_all = flatten_cvat_polygons(cvat_polys_by_frame) if cvat_polys_by_frame else []
with st.sidebar:
    if xml_error:
        st.warning(f"XML parse/fetch issue: {xml_error}")
        st.warning(f"XML parse/fetch issue: {xml_error}")

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
col_left, col_right = st.columns([3, 1])

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
        current_frame = int(st.session_state.get("current_frame", 0))

        # Slider directly above the video
        new_frame = st.slider(
            "Frame",
            min_value=0,
            max_value=max(0, frame_count - 1),
            value=current_frame,
            step=1,
            key="frame_slider",
        )
        if new_frame != current_frame:
            st.session_state["current_frame"] = int(new_frame)
            current_frame = int(new_frame)
        
        # Controls row: -15s | Play/Pause | +15s
        ctrl_cols = st.columns([1, 1, 1, 2])
        def _seek(seconds: float):
            if fps and fps > 0:
                delta = int(round(seconds * float(fps)))
                st.session_state["current_frame"] = int(
                    max(0, min(frame_count - 1, st.session_state["current_frame"] + delta))
                )
                (st.rerun() if hasattr(st, "rerun") else st.experimental_rerun())
        
        with ctrl_cols[0]:
            if st.button("−15 sec", use_container_width=True):
                _seek(-15)
        
        with ctrl_cols[1]:
            # Keep Play/Pause behavior
            if st.button("▶ Play" if not st.session_state.playing else "⏸ Pause", use_container_width=True):
                st.session_state.playing = not st.session_state.playing
                (st.rerun() if hasattr(st, "rerun") else st.experimental_rerun())

        with ctrl_cols[2]:
            if st.button("+15 sec", use_container_width=True):
                _seek(+15)

        with ctrl_cols[3]:
            # Speed selector (renders every run, persists via key)
            st.session_state.play_speed = st.select_slider(
                "Speed",
                options=[0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
                value=float(st.session_state.get('play_speed', 5.0)),
            )
        
        # Status line
        cur_ms = frame_to_ms(int(st.session_state["current_frame"]), int(video_start_ms), float(fps))
        st.caption(
            f"Frame {st.session_state['current_frame']}/{max(0, frame_count - 1)} · {cur_ms} ms since epoch · {cur_ms/1000.0:.2f} s"
        )
        
        # --- Read & render current frame (video window) ---
        current_frame = int(st.session_state["current_frame"])
        frame_bgr = read_frame(video_path, current_frame)
        if frame_bgr is None:
            st.warning("Could not read this frame.")
        else:
            # Determine which events to draw
            events_now = []
            if by_frame:
                for f in range(int(current_frame) - tolerance, int(current_frame) + tolerance + 1):
                    events_now.extend(by_frame.get(f, []))
            canvas = draw_boxes(frame_bgr, events_now, show_labels=show_labels)
            if show_polygons and (cvat_polys_by_frame or cvat_polys_all):
                polys_to_draw = (
                    cvat_polys_all
                    if ("poly_all_frames" in locals() and poly_all_frames)
                    else cvat_polys_by_frame.get(int(current_frame), [])
                )
                if polys_to_draw:
                    canvas = overlay_polygons(
                        canvas, polys_to_draw, alpha=float(poly_alpha), edge_thickness=int(poly_edge_thickness)
                    )
            st.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), use_container_width=True)

            # Advance playback if enabled
            if st.session_state.playing and fps > 0:
                # Compute next frame respecting loop mode
                next_f = current_frame + 1
                if st.session_state.get('looping', False):
                    a, b = st.session_state.get('loop_range', (0, frame_count - 1))
                    a = int(max(0, min(a, frame_count - 1)))
                    b = int(max(0, min(b, frame_count - 1)))
                    if a > b:
                        a, b = b, a
                    if next_f > b:
                        next_f = a
                else:
                    if next_f >= frame_count:
                        next_f = frame_count - 1
                        st.session_state.playing = False
                st.session_state['current_frame'] = int(next_f)
                # Sleep to target the selected playback speed and rerun
                delay = max(0.001, 1.0 / (fps * float(st.session_state.get('play_speed', 5.0))))
                time.sleep(delay)
                (st.rerun() if hasattr(st, "rerun") else st.experimental_rerun())
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
        
        # --- All events list & navigation ---
        st.markdown("### All events")
        all_df = pd.DataFrame(mapped_events)
        if not all_df.empty:
            # Prefer a concise ordering
            if 'frame_idx' in all_df.columns:
                all_df = all_df.sort_values(['frame_idx', 'event_time'] if 'event_time' in all_df.columns else ['frame_idx'])
            st.dataframe(all_df, use_container_width=True, hide_index=True)
            # Click-to-navigate via selectbox
            event_options = list(enumerate(all_df.to_dict(orient='records')))
            def _fmt(opt):
                i, e = opt
                f = e.get('frame_idx', '?')
                et = e.get('event_type', 'event')
                gid = e.get('global_id', '')
                eid = e.get('event_id', '')
                return f"[{f}] {et}  id={eid or gid}"
            sel = st.selectbox("Select an event to navigate", options=event_options, format_func=_fmt, index=0 if event_options else None)
            if st.button("Go to selected event", use_container_width=True, disabled=not bool(event_options)):
                _, e = sel
                target = int(e.get('frame_idx', 0))
                st.session_state['current_frame'] = max(0, int(target))
                (st.rerun() if hasattr(st, "rerun") else st.experimental_rerun())
        else:
            st.write("No events for the current selection.")
    else:
        st.info("Upload or paste tracking data to see events.")

st.markdown("---")
st.caption(
    "Tip: If your tracking rows already include a 'frame' or 'frame_idx' column, the epoch 'event_time' → frame mapping is skipped."
)
