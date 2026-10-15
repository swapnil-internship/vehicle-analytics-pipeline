import os
from pathlib import Path

import mysql.connector
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="RoadSight | Vehicle Analytics",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
VIDEO_PATH = BASE_DIR / "outputs" / "tracked_vehicles_web.mp4"

# Professional dark navy / cyan dashboard styling.
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
      html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
      .stApp { background: linear-gradient(180deg, #08111f 0%, #0b1424 45%, #0e192b 100%); color: #edf4ff; }
      [data-testid="stHeader"] { background: rgba(8,17,31,.92); }
      [data-testid="stSidebar"] { background: #0a1423; border-right: 1px solid #22324a; }
      [data-testid="stSidebar"] * { color: #dce8f8; }
      h1, h2, h3 { color: #f4f8ff !important; letter-spacing: -0.035em; }
      p, label, .stCaption { color: #aabbd2; }
      .hero { padding: 1.25rem 1.5rem; border-radius: 18px; margin-bottom: 1rem;
              background: linear-gradient(120deg, #142944 0%, #123b56 58%, #126a78 100%);
              border: 1px solid rgba(115,211,230,.28); }
      .hero-eyebrow { color: #76e4ed; text-transform: uppercase; font-size: .76rem; font-weight: 800; letter-spacing: .16em; }
      .hero-title { color: white; font-size: 2.15rem; font-weight: 800; line-height: 1.2; margin: .35rem 0; }
      .hero-sub { color: #c5d9eb; font-size: .95rem; }
      div[data-testid="stMetric"] { background: linear-gradient(145deg, #13233a, #101c2e); padding: 1rem 1.1rem;
          border: 1px solid #263b56; border-radius: 16px; box-shadow: 0 8px 22px rgba(0,0,0,.12); }
      div[data-testid="stMetricLabel"] { color: #9fb5cf !important; font-weight: 600; }
      div[data-testid="stMetricValue"] { color: #f4f8ff !important; font-weight: 800; }
      div[data-testid="stMetricDelta"] { color: #6de3d0 !important; }
      .section-kicker { color: #69d9e7; font-size: .75rem; text-transform: uppercase; letter-spacing: .15em; font-weight: 800; margin-bottom: -.35rem; }
      .panel-note { background: #101f32; border: 1px solid #263b56; padding: .75rem 1rem; border-radius: 12px; color: #afc2d9; font-size: .88rem; }
      .status-pill { display: inline-block; border: 1px solid #246b66; color: #72e5d5; background: #10312f; border-radius: 99px; padding: .28rem .65rem; font-size: .78rem; font-weight: 700; }
      .stButton > button, .stDownloadButton > button { border-radius: 10px; border: 1px solid #2a7890; background: #123b56; color: white; font-weight: 700; }
      .stButton > button:hover, .stDownloadButton > button:hover { border-color: #64d7e5; background: #17516b; color: white; }
      [data-testid="stDataFrame"] { border: 1px solid #263b56; border-radius: 12px; overflow: hidden; }
      hr { border-color: #25374e; }
      #MainMenu { visibility: hidden; } footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="hero-eyebrow">Computer Vision · Traffic Intelligence</div>
      <div class="hero-title">RoadSight Analytics</div>
      <div class="hero-sub">Vehicle detection, object tracking and line-crossing insights in one place.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar connection controls
st.sidebar.markdown("## ⚙️ Dashboard settings")
st.sidebar.caption("Connect this dashboard to the local MySQL event database.")
ENV_PASSWORD = os.getenv("DB_PASSWORD", "")
if ENV_PASSWORD:
    db_password = ENV_PASSWORD
    st.sidebar.success("Password loaded from environment.")
else:
    db_password = st.sidebar.text_input("MySQL password", type="password")

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", "root"),
    "password": db_password,
    "database": os.getenv("DB_NAME", "vehicle_analytics"),
    "connection_timeout": 5,
}


def load_events():
    if not DB_CONFIG["password"]:
        raise ValueError("Enter the MySQL password in the sidebar or set DB_PASSWORD.")
    connection = mysql.connector.connect(**DB_CONFIG)
    try:
        query = """
            SELECT id, track_id, vehicle_type, vehicle_color,
                   event_timestamp, direction, created_at
            FROM vehicle_events
            ORDER BY id DESC
        """
        return pd.read_sql_query(query, connection)
    finally:
        connection.close()


def apply_plot_theme(fig, height=330):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#dce8f8", size=12),
        margin=dict(l=12, r=12, t=45, b=12),
        title=dict(font=dict(size=15, color="#f4f8ff"), x=0.02, xanchor="left"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#c7d6e8")),
        xaxis=dict(gridcolor="#24364c", zerolinecolor="#24364c", tickfont=dict(color="#aabbd2"), title_font=dict(color="#aabbd2")),
        yaxis=dict(gridcolor="#24364c", zerolinecolor="#24364c", tickfont=dict(color="#aabbd2"), title_font=dict(color="#aabbd2")),
    )
    return fig


@st.cache_data(show_spinner=False, max_entries=2)
def read_video(video_path, modified_time):
    return Path(video_path).read_bytes()

with st.expander("🎬 Annotated detection video", expanded=True):
    if VIDEO_PATH.exists():
        video_size = VIDEO_PATH.stat().st_size / (1024 * 1024)
        modified = VIDEO_PATH.stat().st_mtime
        st.caption(
            f"{VIDEO_PATH.name} · {video_size:.1f} MB · Updated "
            f"{pd.Timestamp.fromtimestamp(modified).strftime('%d %b %Y, %H:%M:%S')}"
        )
        try:
            st.video(read_video(str(VIDEO_PATH), modified))
        except Exception as exc:
            st.warning(f"Could not load the video preview: {exc}")
    else:
        st.info("Annotated browser-compatible video not found. Run the detector, then `python convert_video.py`.")

st.markdown('<div class="section-kicker">Live monitoring</div>', unsafe_allow_html=True)
st.header("Traffic overview")
st.caption("The database refreshes every 10 seconds. These values summarize stored crossing events.")

@st.fragment(run_every="10s")
def live_dashboard():
    top_left, top_right = st.columns([4, 1])
    with top_left:
        st.markdown('<span class="status-pill">● DATABASE MONITORING</span>', unsafe_allow_html=True)
    with top_right:
        if st.button("↻ Refresh now", use_container_width=True, key="refresh_professional"):
            st.rerun(scope="fragment")

    try:
        events = load_events()
    except Exception as exc:
        st.error(f"Could not load events from MySQL: {exc}")
        st.info("Check that MySQL is running and the `vehicle_analytics.vehicle_events` table exists.")
        return

    st.caption("Last refreshed: " + pd.Timestamp.now().strftime("%d %b %Y · %H:%M:%S"))
    if events.empty:
        st.info("No crossing events are stored yet. Run the detector to start collecting events.")
        return

    for column in ("direction", "vehicle_type", "vehicle_color"):
        events[column] = events[column].fillna("Unknown").astype(str)
    events["direction"] = events["direction"].str.upper()
    events["created_at"] = pd.to_datetime(events["created_at"], errors="coerce")
    total = len(events)
    up = int(events["direction"].eq("UP").sum())
    down = int(events["direction"].eq("DOWN").sum())
    vehicle_types = int(events["vehicle_type"].nunique())
    colors = int(events["vehicle_color"].nunique())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Crossing events", f"{total:,}", help="Number of crossing-event records in MySQL.")
    m2.metric("Downward", f"{down:,}", help="Events classified as DOWN.")
    m3.metric("Upward", f"{up:,}", help="Events classified as UP.")
    m4.metric("Vehicle categories", f"{vehicle_types}", help=f"Distinct vehicle labels; {colors} estimated colour labels.")

    st.markdown("<div class='panel-note'>ℹ️ Track IDs can repeat across separate detector runs, so they should not be interpreted as a count of unique physical vehicles. Colour labels are approximate image-based estimates.</div>", unsafe_allow_html=True)
    st.write("")

    overview_tab, breakdown_tab, records_tab = st.tabs(["📊 Overview", "🧭 Detailed analytics", "📋 Event records"])
    with overview_tab:
        c1, c2 = st.columns(2, gap="large")
        direction_counts = events["direction"].value_counts().rename_axis("Direction").reset_index(name="Events")
        with c1:
            fig = px.pie(
                direction_counts, names="Direction", values="Events", hole=0.66,
                title="Crossing direction split",
                color="Direction", color_discrete_map={"UP": "#36cfc9", "DOWN": "#5b8ff9", "UNKNOWN": "#9aa9bd"},
            )
            fig.update_traces(textposition="inside", textinfo="percent+value", marker=dict(line=dict(color="#101b2c", width=3)))
            fig.add_annotation(text=f"{total}<br><sup>events</sup>", x=0.5, y=0.5, showarrow=False, font=dict(size=20, color="#f4f8ff"))
            st.plotly_chart(apply_plot_theme(fig), use_container_width=True, config={"displayModeBar": False})
        with c2:
            type_counts = events["vehicle_type"].value_counts().rename_axis("Vehicle type").reset_index(name="Events")
            fig = px.bar(type_counts, x="Events", y="Vehicle type", orientation="h", title="Events by vehicle type",
                         color="Events", color_continuous_scale=["#17415c", "#1ba6a6", "#69e0d0"], text="Events")
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed", gridcolor="#24364c"))
            st.plotly_chart(apply_plot_theme(fig), use_container_width=True, config={"displayModeBar": False})

        c3, c4 = st.columns(2, gap="large")
        with c3:
            color_counts = events["vehicle_color"].value_counts().rename_axis("Estimated colour").reset_index(name="Events")
            palette = {"white": "#e7edf5", "black": "#46566b", "blue": "#4c9aff", "red": "#ff6b6b", "silver/grey": "#aeb9c8", "silver": "#aeb9c8", "grey": "#8997aa", "dark grey": "#68778c", "other": "#b28dff", "unknown": "#7d8da3"}
            fig = px.pie(color_counts, names="Estimated colour", values="Events", title="Estimated vehicle colours",
                         color="Estimated colour", color_discrete_map=palette, hole=0.42)
            fig.update_traces(textposition="inside", textinfo="percent", marker=dict(line=dict(color="#101b2c", width=2)))
            st.plotly_chart(apply_plot_theme(fig), use_container_width=True, config={"displayModeBar": False})
        with c4:
            if events["created_at"].notna().any():
                timeline = events.dropna(subset=["created_at"]).copy()
                timeline["Time"] = timeline["created_at"].dt.floor("min")
                timeline = timeline.groupby("Time", as_index=False).size().rename(columns={"size": "Events"}).sort_values("Time")
                fig = px.line(timeline, x="Time", y="Events", markers=True, title="Events recorded over time")
                fig.update_traces(line=dict(color="#36cfc9", width=3), marker=dict(size=7, color="#70e7d8"), fill="tozeroy", fillcolor="rgba(54,207,201,0.12)")
                fig.update_xaxes(title="Recorded time")
                fig.update_yaxes(title="Events")
            else:
                fig = go.Figure()
                fig.add_annotation(text="No valid database timestamps available", x=0.5, y=0.5, showarrow=False)
                fig.update_layout(title="Events recorded over time")
            st.plotly_chart(apply_plot_theme(fig), use_container_width=True, config={"displayModeBar": False})

    with breakdown_tab:
        st.subheader("Direction by vehicle type")
        cross = events.groupby(["vehicle_type", "direction"]).size().reset_index(name="Events")
        fig = px.bar(cross, x="vehicle_type", y="Events", color="direction", barmode="group", text="Events",
                     title="Compare upward and downward crossings",
                     color_discrete_map={"UP": "#36cfc9", "DOWN": "#5b8ff9", "UNKNOWN": "#9aa9bd"})
        fig.update_layout(xaxis_title="Vehicle type", yaxis_title="Crossing events", legend_title="Direction")
        st.plotly_chart(apply_plot_theme(fig, 380), use_container_width=True, config={"displayModeBar": False})

        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.subheader("Video timestamp distribution")
            video_times = pd.to_numeric(events["event_timestamp"], errors="coerce")
            video_times = video_times.dropna()
            if not video_times.empty:
                hist = go.Figure(go.Histogram(x=video_times, nbinsx=min(12, max(5, len(video_times))),
                                               marker=dict(color="#8b7cf6", line=dict(color="#101b2c", width=1))))
                hist.update_layout(title="When crossings occurred in the clip", xaxis_title="Video time (seconds)", yaxis_title="Events")
                st.plotly_chart(apply_plot_theme(hist), use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("No numeric video timestamps available.")
        with c2:
            st.subheader("Crossings by estimated colour")
            cc = events["vehicle_color"].value_counts().rename_axis("Colour").reset_index(name="Events").sort_values("Events")
            fig = px.bar(cc, x="Events", y="Colour", orientation="h", title="Colour frequency", text="Events",
                         color="Events", color_continuous_scale=["#293c68", "#8b7cf6", "#c5b8ff"])
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
            st.plotly_chart(apply_plot_theme(fig), use_container_width=True, config={"displayModeBar": False})

    with records_tab:
        st.subheader("Recent vehicle crossing events")
        search = st.text_input("Filter records", placeholder="Search type, colour, direction or track ID…", key="event_filter")
        display = events[["id", "track_id", "vehicle_type", "vehicle_color", "event_timestamp", "direction", "created_at"]].copy()
        display.columns = ["Event ID", "Track ID", "Vehicle type", "Estimated colour", "Video time (s)", "Direction", "Recorded at"]
        if search.strip():
            mask = display.astype(str).apply(lambda col: col.str.contains(search.strip(), case=False, na=False)).any(axis=1)
            display = display[mask]
        st.dataframe(display, use_container_width=True, hide_index=True, height=420)
        csv_data = events.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Export all events as CSV", data=csv_data, file_name="vehicle_events_export.csv", mime="text/csv", key="download_professional_csv")

    st.caption("Data source: MySQL `vehicle_analytics`. Dashboard refresh reads stored events; it does not start or restart the detector.")

live_dashboard()
