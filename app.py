import pandas as pd
import streamlit as st
import config as cfg
from src import performance as perf
from src.density_estimator import classify
from src.detector import Detector
from src.pipeline import Pipeline, save_results
from src.roi import ROI, parse_points
from src.signal_controller import choose_phase, green_time
from src.video_reader import first_frame, list_videos

st.set_page_config(page_title="Adaptive Traffic Management", layout="wide")
st.title("AI-Based Adaptive Traffic Management for Sustainable Smart Cities")


@st.cache_resource
def load_detector():
    return Detector()


st.sidebar.header("Video")
upload = st.sidebar.file_uploader("Add a video", type=[e.strip(".") for e in cfg.VIDEO_EXTENSIONS])
if upload and not (cfg.VIDEO_DIR / upload.name).exists():
    (cfg.VIDEO_DIR / upload.name).write_bytes(upload.getbuffer())
videos = list_videos()
if not videos:
    st.warning(f"No videos found. Copy a traffic video (e.g. cam_5.mp4) into {cfg.VIDEO_DIR}")
    st.stop()
video = st.sidebar.selectbox("Select video", videos, format_func=lambda p: p.name)
skip = st.sidebar.slider("Process every Nth frame", 1, 6, cfg.FRAME_SKIP)
max_frames = st.sidebar.number_input("Frames to process (0 = whole video)", 0, 100000, 300, step=50)

st.sidebar.header("Region of interest")
roi_text = st.sidebar.text_area("ROI polygon (x,y points between 0 and 1)", "0,0.25; 1,0.25; 1,1; 0,1", help="Leave empty to use the whole frame.")
try:
    roi_points = parse_points(roi_text)
except ValueError as error:
    st.sidebar.error(str(error))
    st.stop()
camera_approach = st.sidebar.selectbox("Approach seen by the camera", cfg.APPROACHES)
start = st.sidebar.button("▶ Start analysis", type="primary")
st.sidebar.button("⏹ Stop")
st.sidebar.caption("Changing any control while the analysis runs stops it.")

try:
    preview = first_frame(video)
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()
roi = ROI(roi_points, preview.shape[1], preview.shape[0])
raw_preview = preview.copy()
roi.draw(preview)

live_tab, signal_tab, compare_tab = st.tabs(["Live analysis", "Signal control", "Fixed vs adaptive"])
history = st.session_state.get("history", [])

with live_tab:
    left, right = st.columns(2)
    left.subheader("Input video")
    right.subheader("Detection + tracking")
    raw_slot, out_slot = left.empty(), right.empty()
    raw_slot.image(raw_preview, channels="BGR")
    out_slot.image(preview, channels="BGR")
    slots = [c.empty() for c in st.columns(5)]

    def show_kpis(stats):
        slots[0].metric("Vehicles in ROI now", sum(stats["present"].values()))
        slots[1].metric("Vehicles passed", sum(stats["passed"].values()))
        slots[2].metric("Density", f"{stats['level']} ({stats['load']:.1f} PCU)")
        slots[3].metric("Priority (1 = highest)", stats["priority"])
        slots[4].metric("Suggested green", f"{stats['green']} s")

    if start:
        try:
            pipeline = Pipeline(video, roi_points, skip, load_detector())
        except (FileNotFoundError, ValueError) as error:
            st.error(str(error))
            st.stop()
        history = []
        st.session_state.history, st.session_state.pipeline = history, pipeline
        for raw, processed, stats in pipeline.run(max_frames or None):
            raw_slot.image(raw, channels="BGR")
            out_slot.image(processed, channels="BGR")
            show_kpis(stats)
            history.append(stats)
        if history:
            st.success("Saved " + ", ".join(p.name for p in save_results(pipeline, history)) + f" in {cfg.OUTPUT_DIR}")
        else:
            st.error("No frames could be read from this video.")
    elif history:
        show_kpis(history[-1])

    if history:
        frames = pd.json_normalize(history)
        col1, col2 = st.columns([1, 2])
        col1.pyplot(perf.plot_counts(history[-1]["passed"]))
        col2.pyplot(perf.plot_density(frames))
        st.subheader("Counts by class")
        st.dataframe(pd.DataFrame({"In ROI now": history[-1]["present"], "Passed through ROI": history[-1]["passed"]}))
        log = pd.DataFrame(st.session_state.pipeline.counter.log)
        if not log.empty:
            st.subheader("Movement-wise count")
            st.dataframe(log.groupby(["movement", "class"]).size().unstack(fill_value=0))
            st.caption("Movement = direction of travel inside the ROI as seen in the image. A single camera cannot tell which turn a vehicle takes outside its view.")

with signal_tab:
    st.caption(f"Only the {camera_approach} approach is measured by the camera. The other approaches are simulated inputs, not real observations.")
    last = history[-1] if history else None
    loads, waited = {}, {}
    for col, approach in zip(st.columns(len(cfg.APPROACHES)), cfg.APPROACHES):
        if approach == camera_approach and last:
            loads[approach] = last["load"]
            col.metric(f"{approach} (measured)", f"{last['load']:.1f} PCU")
        else:
            loads[approach] = col.slider(f"{approach} load (simulated, PCU)", 0, 40, 8, key=f"load_{approach}")
        waited[approach] = col.slider(f"{approach}: seconds since last green", 0, 200, 60, key=f"waited_{approach}")
    chosen, reason = choose_phase(loads, waited)
    chosen_level = classify(loads[chosen])
    st.subheader(f"Suggested phase: {chosen} GREEN for {green_time(chosen_level)} s")
    st.caption(f"Reason: {reason}. Density of {chosen}: {chosen_level}.")
    st.dataframe(pd.DataFrame({
        "Load (PCU)": {a: round(loads[a], 1) for a in cfg.APPROACHES},
        "Density": {a: classify(loads[a]) for a in cfg.APPROACHES},
        "Priority": {a: cfg.PRIORITY[classify(loads[a])] for a in cfg.APPROACHES},
        "Signal": {a: "🟢 GREEN" if a == chosen else "🔴 RED" for a in cfg.APPROACHES},
    }))

with compare_tab:
    st.warning("Everything on this tab is SIMULATED: vehicles arrive randomly at the rates below. Only a camera-approach rate taken from the video is measured.")
    measured = None
    if history and history[-1]["time_s"] >= 30:
        measured = sum(history[-1]["passed"].values()) / (history[-1]["time_s"] / 60)
    use_measured = measured is not None and st.checkbox(f"Use rate measured from the video for {camera_approach} ({measured:.1f} veh/min)")
    rates = {}
    for col, approach in zip(st.columns(len(cfg.APPROACHES)), cfg.APPROACHES):
        if approach == camera_approach and use_measured:
            rates[approach] = measured
            col.metric(f"{approach} (measured)", f"{measured:.1f} veh/min")
        else:
            rates[approach] = col.number_input(f"{approach} (veh/min)", 0.0, 30.0, cfg.DEFAULT_RATES[approach], 0.5, key=f"rate_{approach}")
    c1, c2, c3 = st.columns(3)
    duration = c1.slider("Simulated duration (s)", 600, 7200, cfg.SIM_DURATION, 600)
    seeds = c2.slider("Random runs to average", 1, 10, cfg.SIM_SEEDS)
    fixed_green = c3.slider("Fixed-time green per approach (s)", 10, 60, cfg.DEFAULT_GREEN, 5)
    if st.button("Run simulation"):
        table, queues = perf.compare(rates, duration, seeds, fixed_green)
        cfg.OUTPUT_DIR.mkdir(exist_ok=True)
        table.to_csv(cfg.OUTPUT_DIR / "comparison.csv")
        st.session_state.simulation = (table, queues)
    if "simulation" in st.session_state:
        table, queues = st.session_state.simulation
        st.dataframe(table)
        st.pyplot(perf.plot_comparison(table))
        st.pyplot(perf.plot_queue(queues))
        st.caption("Both controllers see identical random arrivals. Vehicles still queued at the end count with their wait so far. Results depend entirely on the inputs above and are not real-world measurements.")
