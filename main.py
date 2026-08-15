import streamlit as st
from PIL import Image

from pipeline import (
    ALGORITHMS,
    MANUAL_THRESHOLD_ALGORITHM,
    OPERATOR_THRESHOLD_ALGORITHMS,
    process_image,
    to_display,
    view_graph,
)

st.set_page_config(page_title="Image Processing Tool", page_icon="🖼️", layout="wide")
st.title("Image Processing Tool")

algorithm_names = list(ALGORITHMS.keys())

with st.sidebar:
    st.header("Settings")

    uploaded_file = st.file_uploader(
        "Upload Image", type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"]
    )

    algorithm = st.selectbox(
        "Select Image Processing Algorithm",
        options=algorithm_names,
        index=algorithm_names.index("Histogram Equalization"),
    )

    low_threshold = high_threshold = operator_threshold = None

    if algorithm == MANUAL_THRESHOLD_ALGORITHM:
        low_threshold = st.number_input("Low Threshold", min_value=0, max_value=255, value=50, step=1)
        high_threshold = st.number_input("High Threshold", min_value=0, max_value=255, value=150, step=1)

    if algorithm in OPERATOR_THRESHOLD_ALGORITHMS:
        operator_threshold = st.number_input("Operator Threshold", min_value=0, max_value=255, value=50, step=1)

    process_clicked = st.button("Process Image", use_container_width=True)
    graph_clicked = st.button("View Graph", use_container_width=True)

if uploaded_file is None:
    st.info("Upload an image from the sidebar to get started.")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")

if algorithm == MANUAL_THRESHOLD_ALGORITHM and low_threshold > high_threshold:
    st.error("Low threshold must not be greater than the high threshold.")
    st.stop()

# Streamlit reruns the whole script on every click, so results are cached against
# the inputs that produced them and dropped as soon as any of those inputs change.
signature = (uploaded_file.file_id, algorithm, low_threshold, high_threshold, operator_threshold)
if st.session_state.get("signature") != signature:
    st.session_state["signature"] = signature
    st.session_state["result"] = None
    st.session_state["graph"] = None

if process_clicked:
    with st.spinner(f"Running {algorithm}..."):
        st.session_state["result"] = process_image(
            image, algorithm, low_threshold, high_threshold, operator_threshold
        )

if graph_clicked:
    with st.spinner(f"Plotting {algorithm}..."):
        st.session_state["graph"] = view_graph(
            image, algorithm, low_threshold, high_threshold, operator_threshold
        )

original_column, result_column = st.columns(2)

with original_column:
    st.subheader("Original Image")
    st.image(image, use_container_width=True)

with result_column:
    st.subheader("Processed Image")
    if st.session_state.get("result") is not None:
        st.image(to_display(st.session_state["result"]), use_container_width=True)
    else:
        st.caption("Click **Process Image** to run the selected algorithm.")

if st.session_state.get("graph") is not None:
    st.subheader("Image before and after")
    st.image(to_display(st.session_state["graph"]), use_container_width=True)
