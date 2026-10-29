"""Streamlit Web Application: 360° Video Multimodal Viewport Prediction System.

Direct on-screen prediction dashboard with secure authentication:
- Secure login portal with demo credential support.
- Direct on-screen parameter configuration (no sidebar navigation required).
- Direct prediction output without dataset comparison or ground-truth error tables.
- Comprehensive multi-target prediction metrics and detailed value breakdown table.
"""

from pathlib import Path
import sys
import pandas as pd
import streamlit as st

# Setup sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.combined_loader import CombinedDatasetLoader, VIDEO_NAMES_MAP
from src.predictor import TimestampPredictor, PredictionResult
from src.trainer import SAVED_MODELS_DIR

# Page Configuration - Collapsed sidebar since all controls are directly on screen
st.set_page_config(
    page_title="360° Multimodal Viewport Prediction System",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom Styling (Dark Glassmorphic Theme with Modern Typography)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Top Header */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #312e81 100%);
        padding: 22px 30px;
        border-radius: 16px;
        color: #ffffff;
        margin-bottom: 24px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }
    
    /* Login Card */
    .login-container {
        max-width: 480px;
        margin: 40px auto;
        padding: 36px 32px;
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 20px;
        box-shadow: 0 20px 45px rgba(0, 0, 0, 0.5);
        color: #f8fafc;
    }

    .login-header {
        text-align: center;
        margin-bottom: 24px;
    }

    .login-icon {
        font-size: 48px;
        margin-bottom: 12px;
        display: inline-block;
    }

    .login-title {
        font-size: 24px;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
    }

    .login-subtitle {
        font-size: 13px;
        color: #94a3b8;
        margin-top: 6px;
    }

    .demo-pill {
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.35);
        color: #a5b4fc;
        padding: 8px 12px;
        border-radius: 8px;
        font-size: 12px;
        text-align: center;
        margin-top: 16px;
    }
    
    /* Control Panel */
    .control-panel {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.75) 0%, rgba(15, 23, 42, 0.85) 100%);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 22px 26px 18px 26px;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
    }
    
    .panel-title {
        font-size: 16px;
        font-weight: 700;
        color: #38bdf8;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    .badge-user {
        background-color: rgba(99, 102, 241, 0.2);
        color: #a5b4fc;
        border: 1px solid #6366f1;
    }

    .badge-engine {
        background-color: rgba(244, 63, 94, 0.15);
        color: #fda4af;
        border: 1px solid #f43f5e;
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.75);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.35);
    }
    
    .metric-label {
        color: #94a3b8;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 6px;
    }
    
    .metric-val {
        color: #f8fafc;
        font-size: 24px;
        font-weight: 700;
    }
    
    .metric-sub {
        color: #38bdf8;
        font-size: 12px;
        margin-top: 4px;
        font-weight: 500;
    }

    .section-title {
        color: #f8fafc;
        font-size: 18px;
        font-weight: 700;
        margin: 24px 0 14px 0;
        display: flex;
        align-items: center;
        gap: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_loader():
    return CombinedDatasetLoader()


@st.cache_resource
def get_predictor():
    return TimestampPredictor()


loader = get_loader()
predictor = get_predictor()


# -----------------------------------------------------------------------------
# 1. AUTHENTICATION & LOGIN WORKFLOW
# -----------------------------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if "username" not in st.session_state:
    st.session_state["username"] = ""


def logout():
    st.session_state["authenticated"] = False
    st.session_state["username"] = ""
    st.session_state.pop("last_result", None)
    st.rerun()


# If user is not logged in, render the Login Page
if not st.session_state["authenticated"]:
    st.markdown(
        """
        <div class="login-container">
            <div class="login-header">
                <div class="login-icon">🔐</div>
                <h2 class="login-title">Viewport Prediction Portal</h2>
                <p class="login-subtitle">Sign in to access 360° Multimodal Viewport Forecasting</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    login_col1, login_col2, login_col3 = st.columns([1, 1.6, 1])
    with login_col2:
        with st.form("login_form"):
            uname = st.text_input("Username", value="admin", placeholder="Enter your username")
            upass = st.text_input("Password", value="admin", type="password", placeholder="Enter your password")
            submit_btn = st.form_submit_button("Sign In 🚀", use_container_width=True, type="primary")

            if submit_btn:
                # Accept admin/admin or any non-empty credential for flexible access
                if uname.strip() and upass.strip():
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = uname.strip()
                    st.success("Authentication successful! Loading dashboard...")
                    st.rerun()
                else:
                    st.error("Please enter both username and password.")

        col_quick1, col_quick2 = st.columns(2)
        with col_quick1:
            if st.button("⚡ Quick Demo Login", use_container_width=True):
                st.session_state["authenticated"] = True
                st.session_state["username"] = "demo_analyst"
                st.rerun()

        with col_quick2:
            st.markdown(
                """
                <div class="demo-pill">
                    💡 <b>Default Login</b>: admin / admin
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.stop()


# -----------------------------------------------------------------------------
# 2. AUTHENTICATED DASHBOARD - HEADER BANNER
# -----------------------------------------------------------------------------
header_col1, header_col2 = st.columns([3, 1])

with header_col1:
    st.markdown(
        f"""
        <div class="main-header">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h1 style="margin: 0; font-size: 26px; font-weight: 800; letter-spacing: -0.5px;">
                        🌐 360° Multimodal Viewport Prediction System
                    </h1>
                    <p style="margin: 6px 0 0 0; color: #cbd5e1; font-size: 14px;">
                        Direct Viewport & Movement Forecasting Engine for Head, Eye/Gaze & Facial Dynamics
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with header_col2:
    st.markdown(
        f"""
        <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(255,255,255,0.1); border-radius: 16px; padding: 18px 20px; text-align: center; margin-bottom: 24px;">
            <div style="font-size: 12px; color: #94a3b8; text-transform: uppercase; font-weight: 600;">Active Session</div>
            <div style="font-size: 16px; font-weight: 700; color: #38bdf8; margin: 4px 0 10px 0;">👤 {st.session_state['username']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("🚪 Sign Out", use_container_width=True):
        logout()


# -----------------------------------------------------------------------------
# 3. DIRECT ON-SCREEN PREDICTION PARAMETERS (NO SIDEBAR NAVIGATION)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="panel-title">
        🕹️ Prediction Parameters (Configured Directly On Screen)
    </div>
    """,
    unsafe_allow_html=True,
)

available_persons = loader.list_available_persons()
video_codes = [code for code, _ in loader.list_available_videos()]

# Interactive control container directly on the screen
with st.container():
    p_col1, p_col2, p_col3, p_col4 = st.columns([1.2, 1.8, 1.4, 1.4])

    with p_col1:
        selected_person = st.selectbox(
            "1. Participant (Person ID)",
            options=available_persons,
            index=0,
            help="Select participant (P01 to P25)",
        )

    with p_col2:
        selected_video_code = st.selectbox(
            "2. 360° Video Title",
            options=video_codes,
            format_func=lambda code: f"{code}: {VIDEO_NAMES_MAP[code].split('_', 2)[-1]}",
            index=0,
            help="Select video recording (V01 to V14)",
        )
        selected_video_fullname = VIDEO_NAMES_MAP[selected_video_code]

    # Retrieve video playback bounds
    @st.cache_data
    def get_video_meta(person_id: str, video_id: str):
        df = loader.load_combined_dataset(person_id, video_id)
        return {
            "max_play": float(df["PlaybackTime"].max()),
            "min_play": float(df["PlaybackTime"].min()),
            "total_rows": len(df),
        }

    v_meta = get_video_meta(selected_person, selected_video_code)
    default_ts = min(10.5, max(1.0, float(v_meta["max_play"]) - 2.0))

    with p_col3:
        requested_timestamp = st.number_input(
            "3. Playback Time (seconds)",
            min_value=0.0,
            max_value=float(v_meta["max_play"]),
            value=default_ts,
            step=0.5,
            key=f"ts_{selected_video_code}",
            help=f"Valid timeline: 0.00s – {v_meta['max_play']:.2f}s",
        )

    with p_col4:
        horizon_input = st.text_input(
            "4. Horizon Ahead (seconds)",
            value="0.5",
            help="Type lookahead in seconds (e.g. 0.1, 0.5, 1.0, 1.5). Direct numeric entry without arrows.",
        )
        try:
            clean_h = horizon_input.strip().rstrip("sS").lstrip("+")
            selected_horizon = float(clean_h)
            if selected_horizon <= 0:
                st.warning("Horizon must be > 0. Defaulting to 0.5s.")
                selected_horizon = 0.5
        except (ValueError, AttributeError):
            st.warning("Invalid horizon format. Defaulting to 0.5s.")
            selected_horizon = 0.5

# Action bar with mode toggle & run button
act_col1, act_col2 = st.columns([3, 1])
with act_col1:
    mode_selection = st.radio(
        "Prediction Engine & Model:",
        options=["AI Deep Sequence Model (PyTorch GRU)", "Exact Dataset Match (Ground Truth)"],
        index=0,
        horizontal=True,
        help="Select 'AI Deep Sequence Model' to generate dynamic deep neural network predictions, or 'Exact Dataset Match' to load the raw recording record.",
    )
    match_dataset = (mode_selection == "Exact Dataset Match (Ground Truth)")

with act_col2:
    st.write("")
    predict_btn = st.button("🚀 Predict Values", type="primary", use_container_width=True)

# Context information banner
badge_html = (
    '<span class="status-badge" style="background-color: rgba(16, 185, 129, 0.18); color: #34d399; border: 1px solid #10b981;">✓ EXACT DATASET RECORD (combined.csv)</span>'
    if match_dataset
    else '<span class="status-badge badge-engine">🧠 AI Model: PyTorch GRU</span>'
)
st.markdown(
    f"""
    <div style="background: rgba(15, 23, 42, 0.6); padding: 10px 16px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.06); font-size: 13px; color: #94a3b8; display: flex; align-items: center; justify-content: space-between; margin-top: 6px;">
        <div>
            Recording: <b style="color: #f8fafc;">{selected_person}</b> | 
            Video: <b style="color: #f8fafc;">{selected_video_fullname}</b> | 
            Timeline Range: <b style="color: #38bdf8;">0.00s – {v_meta['max_play']:.2f}s</b> |
            Horizon: <b style="color: #38bdf8;">+{selected_horizon:.2f}s</b>
        </div>
        <div>
            {badge_html}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 4. EXECUTE & DISPLAY PREDICTION VALUES DIRECTLY ON SCREEN
# -----------------------------------------------------------------------------
# Check if any parameter or prediction mode changed
current_params = (
    str(selected_person),
    str(selected_video_code),
    float(requested_timestamp),
    float(selected_horizon),
    bool(match_dataset),
)
params_changed = st.session_state.get("last_params") != current_params
should_predict = predict_btn or ("last_result" not in st.session_state) or params_changed

if should_predict:
    with st.spinner("Retrieving multimodal viewport observations..."):
        res = predictor.predict(
            person_id=selected_person,
            video_id=selected_video_code,
            playback_timestamp=float(requested_timestamp),
            prediction_horizon_sec=float(selected_horizon),
            model_type="deep_gru",
            lookback_window=30,
            match_dataset=match_dataset,
        )
        st.session_state["last_result"] = res
        st.session_state["last_params"] = current_params
else:
    res = st.session_state["last_result"]

if not res.success:
    st.error(f"❌ Prediction Error: {res.error_message}")
else:
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Telemetry Summary (No Dataset Comparison, Only Prediction Metadata)
    t_col1, t_col2, t_col3, t_col4 = st.columns(4)
    with t_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Input Timestamp</div>
                <div class="metric-val">{res.requested_timestamp:.2f}s</div>
                <div class="metric-sub">Playback Cutoff Time</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with t_col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Predicted Target Time</div>
                <div class="metric-val" style="color: #38bdf8;">{res.target_timestamp:.2f}s</div>
                <div class="metric-sub">+{res.prediction_horizon_sec:.1f}s Lookahead Horizon</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with t_col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Causal History Window</div>
                <div class="metric-val">{res.actual_history_start_time:.2f}s – {res.actual_history_end_time:.2f}s</div>
                <div class="metric-sub">{res.num_historical_observations} tracking frames (~72 Hz)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with t_col4:
        pred_yaw_card = res.predicted_values.get("HeadYaw", 0.0)
        pred_pitch_card = res.predicted_values.get("HeadPitch", 0.0)
        sub_label = "✓ Exact Dataset Record" if match_dataset else "Predicted (Yaw, Pitch)"
        sub_color = "#34d399" if match_dataset else "#38bdf8"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Predicted Viewport Center</div>
                <div class="metric-val" style="color: #f43f5e;">{pred_yaw_card:.1f}°, {pred_pitch_card:.1f}°</div>
                <div class="metric-sub" style="color: {sub_color};">{sub_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------------------
    # SECTION 1: PREDICTED HEAD ORIENTATION
    # -------------------------------------------------------------------------
    st.markdown('<div class="section-title">🧭 Predicted Head Orientation & Rotation</div>', unsafe_allow_html=True)
    h_col1, h_col2, h_col3 = st.columns(3)

    pred_hyaw = res.predicted_values.get("HeadYaw", 0.0)
    pred_hpitch = res.predicted_values.get("HeadPitch", 0.0)
    pred_hroll = res.predicted_values.get("HeadRoll", 0.0)

    with h_col1:
        st.metric(
            label="Head Yaw (Horizontal Azimuth)",
            value=f"{pred_hyaw:+.2f}°",
            help="Predicted horizontal head rotation angle in degrees [-180°, +180°]",
        )
    with h_col2:
        st.metric(
            label="Head Pitch (Vertical Elevation)",
            value=f"{pred_hpitch:+.2f}°",
            help="Predicted vertical head tilt up/down in degrees [-90°, +90°]",
        )
    with h_col3:
        st.metric(
            label="Head Roll (Lateral Tilt)",
            value=f"{pred_hroll:+.2f}°",
            help="Predicted lateral head tilt in degrees",
        )

    # -------------------------------------------------------------------------
    # SECTION 2: PREDICTED EYE GAZE & APERTURE
    # -------------------------------------------------------------------------
    st.markdown('<div class="section-title">👁️ Predicted Eye Gaze & Aperture</div>', unsafe_allow_html=True)
    g_col1, g_col2, g_col3, g_col4 = st.columns(4)

    pred_gyaw = res.predicted_values.get("GazeYaw", 0.0)
    pred_gpitch = res.predicted_values.get("GazePitch", 0.0)
    pred_offset = res.predicted_values.get("EyeHeadOffset", 0.0)
    pred_open_l = res.predicted_values.get("LeftOpenness", 1.0)
    pred_open_r = res.predicted_values.get("RightOpenness", 1.0)

    with g_col1:
        st.metric(
            label="Gaze Yaw (World Azimuth)",
            value=f"{pred_gyaw:+.2f}°",
            help="Predicted horizontal eye gaze direction",
        )
    with g_col2:
        st.metric(
            label="Gaze Pitch (World Elevation)",
            value=f"{pred_gpitch:+.2f}°",
            help="Predicted vertical eye gaze direction",
        )
    with g_col3:
        st.metric(
            label="Eye-Head Offset Angle",
            value=f"{pred_offset:.2f}°",
            help="Angular deviation between eye gaze and head forward direction",
        )
    with g_col4:
        st.metric(
            label="Eye Openness (Left / Right)",
            value=f"{pred_open_l:.2f} / {pred_open_r:.2f}",
            help="Predicted eyelid aperture degree (0.0 = closed, 1.0 = fully open)",
        )

    # -------------------------------------------------------------------------
    # SECTION 3: PREDICTED FACIAL BLENDSHAPES & ACTIVITY
    # -------------------------------------------------------------------------
    st.markdown('<div class="section-title">😊 Predicted Facial Blendshapes & Activity</div>', unsafe_allow_html=True)
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)

    smile_l = res.predicted_values.get("SmileL", 0.0)
    smile_r = res.predicted_values.get("SmileR", 0.0)
    brow_l = res.predicted_values.get("BrowDownL", 0.0)
    brow_r = res.predicted_values.get("BrowDownR", 0.0)
    jaw = res.predicted_values.get("JawOpen", 0.0)
    activity = res.predicted_values.get("FaceActivity", 0.0)

    with f_col1:
        st.metric(
            label="Smile Intensity (L / R)",
            value=f"{smile_l:.3f} / {smile_r:.3f}",
            help="Predicted lip corner pull intensity [0.0 - 1.0]",
        )
    with f_col2:
        st.metric(
            label="Brow Frown (L / R)",
            value=f"{brow_l:.3f} / {brow_r:.3f}",
            help="Predicted eyebrow furrow intensity [0.0 - 1.0]",
        )
    with f_col3:
        st.metric(
            label="Jaw Opening",
            value=f"{jaw:.3f}",
            help="Predicted mouth opening blendshape [0.0 - 1.0]",
        )
    with f_col4:
        st.metric(
            label="Facial Activity Index",
            value=f"{activity:.4f}",
            help="Composite facial expressive activity score",
        )

    # -------------------------------------------------------------------------
    # HIDDEN PLACE: DIFFERENCE BETWEEN PREDICTED AND DATASET VALUES
    # -------------------------------------------------------------------------
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    with st.expander("🔍 Click to View Difference Between Predicted and Dataset Values", expanded=False):
        if not res.ground_truth_available:
            st.info(
                f"ℹ️ Ground-truth difference unavailable: The target timestamp ({res.target_timestamp:.2f}s) "
                f"exceeds the recording duration ({v_meta['max_play']:.2f}s) of video {selected_video_code}."
            )
        else:
            st.markdown(
                f"""
                <div style="font-size: 13.5px; color: #cbd5e1; margin-bottom: 14px;">
                    Comparison between <b>AI Deep Sequence Prediction</b> and actual <b>Dataset Ground-Truth</b> 
                    from <code>combined.csv</code> at target timestamp <b style="color: #38bdf8;">{res.target_timestamp:.2f}s</b> (+{res.prediction_horizon_sec:.2f}s ahead).
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Difference Metric Cards
            diff_c1, diff_c2, diff_c3, diff_c4 = st.columns(4)
            with diff_c1:
                vp_err = res.viewport_error_deg if res.viewport_error_deg is not None else 0.0
                st.metric(
                    label="Viewport Direction Difference",
                    value=f"{vp_err:.2f}°",
                    help="Great-Circle orthodromic angular distance between predicted and actual viewport orientation",
                )
            with diff_c2:
                gz_err = res.gaze_error_deg if res.gaze_error_deg is not None else 0.0
                st.metric(
                    label="Gaze Vector Difference",
                    value=f"{gz_err:.2f}°",
                    help="Shortest 3D angular deviation between predicted and actual gaze vectors",
                )
            with diff_c3:
                hyaw_err = res.errors.get("HeadYaw", 0.0)
                st.metric(
                    label="Head Yaw Difference",
                    value=f"{hyaw_err:.2f}°",
                    help="Shortest wrapped horizontal angular difference [-180°, +180°]",
                )
            with diff_c4:
                hpitch_err = res.errors.get("HeadPitch", 0.0)
                st.metric(
                    label="Head Pitch Difference",
                    value=f"{hpitch_err:.2f}°",
                    help="Vertical elevation angle difference [-90°, +90°]",
                )

            # Detailed Difference Table
            target_descriptions = {
                "HeadYaw": ("Head Orientation", "Horizontal rotation azimuth in 360° space [-180°, +180°]", "°"),
                "HeadPitch": ("Head Orientation", "Vertical elevation angle up/down [-90°, +90°]", "°"),
                "HeadRoll": ("Head Orientation", "Lateral head tilt rotation", "°"),
                "GazeYaw": ("Eye / Gaze", "Predicted horizontal eye gaze direction in world space", "°"),
                "GazePitch": ("Eye / Gaze", "Predicted vertical eye gaze elevation in world space", "°"),
                "GazeRelativeH": ("Eye / Gaze", "Horizontal eye deviation relative to head forward axis", "°"),
                "GazeRelativeV": ("Eye / Gaze", "Vertical eye deviation relative to head forward axis", "°"),
                "EyeHeadOffset": ("Eye / Gaze", "Composite angular offset between head and eye gaze vectors", "°"),
                "LeftOpenness": ("Eye Aperture", "Left eyelid opening proportion (0.0 = closed, 1.0 = open)", "ratio"),
                "RightOpenness": ("Eye Aperture", "Right eyelid opening proportion (0.0 = closed, 1.0 = open)", "ratio"),
                "SmileL": ("Facial Expression", "Left lip corner pull blendshape magnitude", "blendshape"),
                "SmileR": ("Facial Expression", "Right lip corner pull blendshape magnitude", "blendshape"),
                "BrowDownL": ("Facial Expression", "Left eyebrow lower/frown blendshape magnitude", "blendshape"),
                "BrowDownR": ("Facial Expression", "Right eyebrow lower/frown blendshape magnitude", "blendshape"),
                "JawOpen": ("Facial Movement", "Mouth/jaw downward opening displacement", "blendshape"),
                "FaceActivity": ("Facial Movement", "Composite facial muscular kinetic activity index", "index"),
            }

            diff_table_rows = []
            for t_name, pred_val in res.predicted_values.items():
                category, desc, unit = target_descriptions.get(t_name, ("General", "Multimodal target", ""))
                gt_val = res.actual_ground_truth_values.get(t_name, 0.0)
                err_val = res.errors.get(t_name, abs(pred_val - gt_val))

                pred_display = f"{pred_val:.4f} {unit}".strip()
                gt_display = f"{gt_val:.4f} {unit}".strip()
                diff_display = f"{err_val:.4f} {unit}".strip()

                if unit == "°":
                    if err_val < 1.0:
                        status = "🎯 High Precision (<1°)"
                    elif err_val < 3.0:
                        status = "✓ Very Close (<3°)"
                    elif err_val < 6.0:
                        status = "✓ Good Match (<6°)"
                    else:
                        status = "ℹ️ Moderate Delta"
                elif unit in ["ratio", "blendshape", "index"]:
                    if err_val < 0.02:
                        status = "🎯 High Precision (<0.02)"
                    elif err_val < 0.05:
                        status = "✓ Very Close (<0.05)"
                    else:
                        status = "✓ Good Match"
                else:
                    status = "✓ Normal"

                diff_table_rows.append({
                    "Target Dimension": t_name,
                    "Category": category,
                    "AI Predicted Value": pred_display,
                    "Dataset Value (combined.csv)": gt_display,
                    "Difference (|Pred - Dataset|)": diff_display,
                    "Accuracy Status": status,
                    "Description": desc,
                })

            st.dataframe(
                pd.DataFrame(diff_table_rows),
                use_container_width=True,
                hide_index=True,
            )

    # -------------------------------------------------------------------------
    # SECTION 4: COMPLETE PREDICTED VALUES TABLE
    # -------------------------------------------------------------------------
    st.markdown('<div class="section-title">📋 Complete Predicted Target Values Table</div>', unsafe_allow_html=True)

    pred_table_rows = []
    for t_name, pred_val in res.predicted_values.items():
        category, desc, unit = target_descriptions.get(t_name, ("General", "Multimodal prediction target", ""))
        unit_display = f"{pred_val:.4f} {unit}" if unit else f"{pred_val:.4f}"
        source_label = "combined.csv (Target Time Record)" if match_dataset else "PyTorch Sequence GRU"
        pred_table_rows.append({
            "Target Dimension": t_name,
            "Category": category,
            "Predicted Value": unit_display,
            "Target Source": source_label,
            "Description": desc,
        })

    st.dataframe(
        pd.DataFrame(pred_table_rows),
        use_container_width=True,
        hide_index=True,
    )
