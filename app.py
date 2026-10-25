
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Airfare Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Data" / "airfare_data.csv"
MODEL_PATH = BASE_DIR / "Model" / "best_model.joblib"
# ==================== LIQUID GRAPHITE UI ====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    font-family: Inter, sans-serif;
    color: #F4F5F7;
    background:
      radial-gradient(ellipse at 12% 8%, rgba(140,150,170,.11), transparent 34%),
      radial-gradient(ellipse at 90% 15%, rgba(100,120,140,.09), transparent 32%),
      radial-gradient(ellipse at 72% 88%, rgba(180,190,205,.06), transparent 30%),
      linear-gradient(135deg,#070809,#111316 48%,#08090B);
    background-attachment: fixed;
}

[data-testid="stHeader"] {
    background: rgba(7,8,10,.55);
    backdrop-filter: blur(22px);
}

.block-container {
    max-width: 1500px;
    padding-top: 1.65rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(160deg,rgba(27,29,33,.96),rgba(10,11,13,.98));
    border-right: 1px solid rgba(255,255,255,.12);
    backdrop-filter: blur(28px);
}

[data-testid="stSidebar"] * {
    color: #F0F1F3;
}

h1, h2, h3, h4, p, label {
    color: #F4F5F7;
}

.hero {
    padding: 28px 30px;
    border-radius: 25px;
    border: 1px solid rgba(255,255,255,.16);
    background: linear-gradient(135deg,rgba(255,255,255,.09),rgba(255,255,255,.025));
    box-shadow: 0 18px 50px rgba(0,0,0,.28);
    backdrop-filter: blur(28px);
    margin-bottom: 24px;
}

.eyebrow {
    color: #C8D0D8;
    text-transform: uppercase;
    letter-spacing: .20em;
    font-size: .70rem;
    font-weight: 800;
}

.hero h1 {
    font-size: clamp(30px,4vw,48px);
    font-weight: 800;
    letter-spacing: -1.5px;
    margin: 10px 0;
}

.hero p {
    color: #B7BDC7 !important;
    font-size: 15px;
    line-height: 1.7;
}

.glass {
    padding: 21px;
    border-radius: 21px;
    border: 1px solid rgba(255,255,255,.13);
    background: linear-gradient(135deg,rgba(255,255,255,.075),rgba(255,255,255,.025));
    box-shadow: 0 15px 40px rgba(0,0,0,.20);
    backdrop-filter: blur(24px);
    margin-bottom: 18px;
}

.section-note {
    color: #A9AFBA;
    font-size: .86rem;
    margin-bottom: 1rem;
}

.pill {
    display: inline-block;
    border: 1px solid rgba(220,230,240,.20);
    background: rgba(220,230,240,.06);
    color: #D9E0E7;
    border-radius: 999px;
    padding: 6px 11px;
    font-size: .72rem;
    font-weight: 700;
    margin-right: 5px;
}

div[data-testid="stMetric"] {
    background: linear-gradient(135deg,rgba(255,255,255,.075),rgba(255,255,255,.025));
    border: 1px solid rgba(255,255,255,.13);
    padding: 17px 19px;
    border-radius: 19px;
    box-shadow: 0 12px 35px rgba(0,0,0,.18);
}

div[data-testid="stMetricLabel"] {
    color: #B9C0CA !important;
}

div[data-testid="stMetricValue"] {
    color: #F7F8FA !important;
}

.stButton > button,
.stFormSubmitButton > button {
    color: #101214;
    background: linear-gradient(110deg,#E1E5EA,#AEB8C4);
    border: 1px solid rgba(255,255,255,.28);
    border-radius: 12px;
    font-weight: 800;
    min-height: 44px;
    box-shadow: 0 8px 24px rgba(0,0,0,.22);
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: linear-gradient(110deg,#F4F5F7,#C2CBD4);
    color: #050607;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div {
    background: rgba(19,20,23,.94) !important;
    border: 1px solid rgba(255,255,255,.18) !important;
    border-radius: 11px !important;
}

div[data-baseweb="select"] *,
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea {
    color: #F5F6F8 !important;
}

[data-baseweb="popover"],
[data-baseweb="menu"],
ul[role="listbox"],
li[role="option"] {
    background: #1A1C20 !important;
    color: #F5F6F8 !important;
}

li[role="option"]:hover {
    background: #34383E !important;
}

/* Liquid-water scrollbar */
::-webkit-scrollbar {
    width: 11px;
    height: 11px;
}

::-webkit-scrollbar-track {
    background: rgba(255,255,255,.035);
    border-radius: 999px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(
        180deg,
        rgba(238,242,247,.68),
        rgba(133,146,160,.48),
        rgba(220,226,233,.62)
    );
    border: 2px solid rgba(10,11,13,.82);
    border-radius: 999px;
    box-shadow: inset 0 0 5px rgba(255,255,255,.35);
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(180deg,#F1F4F7,#AEB8C3,#E0E5EA);
}

[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,.13);
    border-radius: 14px;
    overflow: hidden;
}

hr {
    border-color: rgba(255,255,255,.12);
}

.footer {
    color: #7F8792;
    text-align: center;
    padding: 25px 0;
    font-size: 11px;
    letter-spacing: .06em;
}
</style>
""", unsafe_allow_html=True)

# ==================== LOAD DATA ====================
@st.cache_data(show_spinner=False)
def load_data(path):
    data = pd.read_csv(path)
    data.columns = [
        str(c).strip().lower().replace(" ", "_")
        for c in data.columns
    ]
    return data


@st.cache_resource(show_spinner=False)
def load_model(path):
    return joblib.load(path)


if not DATA_PATH.exists():
    st.error(
        "airfare_data.csv nahi mila. "
        "CSV ko app.py ke same folder mein rakho."
    )
    st.stop()

try:
    df = load_data(str(DATA_PATH))
except Exception as exc:
    st.error(f"Dataset load error: {exc}")
    st.stop()

required = {
    "airline", "source_city", "departure_time", "stops",
    "arrival_time", "destination_city", "class",
    "duration", "days_left", "price"
}

missing = required - set(df.columns)

if missing:
    st.error("Dataset columns missing: " + ", ".join(sorted(missing)))
    st.write("Available columns:", list(df.columns))
    st.stop()

for col in ["price", "duration", "days_left"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["price", "duration", "days_left"]).copy()

df["route"] = (
    df["source_city"].astype(str)
    + " → "
    + df["destination_city"].astype(str)
)

model = None
model_error = None

if MODEL_PATH.exists():
    try:
        model = load_model(str(MODEL_PATH))
    except Exception as exc:
        model_error = str(exc)


def rupees(value):
    return f"₹{value:,.0f}" if pd.notna(value) else "N/A"


def chart_style(fig, height=340):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#E6E8EB", size=12),
        margin=dict(l=10, r=10, t=28, b=12),
        height=height,
        xaxis=dict(gridcolor="rgba(255,255,255,.08)"),
        yaxis=dict(gridcolor="rgba(255,255,255,.08)"),
        legend=dict(bgcolor="rgba(0,0,0,0)")
    )
    return fig


# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown(
        '<div style="font-size:2.1rem">✈️</div>'
        '<div style="font-size:1.2rem;font-weight:800">'
        'AIRFARE <span style="color:#C4CBD3">INTEL</span></div>'
        '<div style="color:#9097A1;font-size:.68rem;'
        'letter-spacing:.15em;margin-top:5px">'
        'FLIGHT ANALYTICS PLATFORM</div>',
        unsafe_allow_html=True
    )

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Fare Explorer",
            "Fare Estimator",
            "Route Intelligence",
            "Dataset Explorer"
        ]
    )

    st.markdown("---")
    st.markdown(
        '<span class="pill">● DATA CONNECTED</span>',
        unsafe_allow_html=True
    )
    st.caption(f"{len(df):,} historical records")
    st.caption("Python · Streamlit · Plotly")


# ==================== HERO HEADER ====================
st.markdown("""
<div class="hero">
  <div class="eyebrow">AVIATION ANALYTICS • MACHINE LEARNING</div>
  <h1>Airfare Intelligence<span style="color:#D8DDE5">.</span></h1>
  <p>Explore historical flight fares, compare airlines and routes,
  and estimate flight prices with a polished analytics workspace.</p>
  <span class="pill">HISTORICAL DATA</span>
  <span class="pill">FARE COMPARISON</span>
  <span class="pill">LIQUID GRAPHITE UI</span>
</div>
""", unsafe_allow_html=True)


# ==================== DASHBOARD ====================
if page == "Dashboard":
    st.subheader("Executive overview")
    st.markdown(
        '<div class="section-note">A live summary of your historical dataset.</div>',
        unsafe_allow_html=True
    )

    a, b, c, d = st.columns(4)
    a.metric("Flight Records", f"{len(df):,}")
    b.metric("Airlines", f"{df.airline.nunique():,}")
    c.metric("Unique Routes", f"{df.route.nunique():,}")
    d.metric("Average Historical Fare", rupees(df.price.mean()))

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown(
            '<div class="glass"><h3>Median fare by airline</h3>'
            '<p class="section-note">Compare typical historical fares.</p>',
            unsafe_allow_html=True
        )

        chart_data = (
            df.groupby("airline", as_index=False).price.median()
            .sort_values("price", ascending=False)
        )

        fig = px.bar(
            chart_data,
            x="airline",
            y="price",
            color="price",
            color_continuous_scale=["#E2E6EA", "#AAB5C1", "#65717E"],
            labels={"price": "Median fare (₹)", "airline": "Airline"}
        )
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(chart_style(fig), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(
            '<div class="glass"><h3>Travel class mix</h3>'
            '<p class="section-note">Distribution of cabin classes.</p>',
            unsafe_allow_html=True
        )

        class_data = (
            df["class"].value_counts()
            .rename_axis("class")
            .reset_index(name="records")
        )

        fig = px.pie(
            class_data,
            names="class",
            values="records",
            hole=0.68,
            color_discrete_sequence=["#DDE2E8", "#9CA8B5", "#66717E"]
        )
        fig.update_traces(textposition="inside", textinfo="percent")
        st.plotly_chart(chart_style(fig), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown(
            '<div class="glass"><h3>Booking window vs average fare</h3>'
            '<p class="section-note">Average fare by days before departure.</p>',
            unsafe_allow_html=True
        )

        trend = (
            df.groupby("days_left", as_index=False).price.mean()
            .sort_values("days_left")
        )

        fig = px.line(
            trend,
            x="days_left",
            y="price",
            markers=True,
            labels={
                "days_left": "Days before departure",
                "price": "Average fare (₹)"
            }
        )
        fig.update_traces(
            line=dict(color="#D8E0E8", width=3),
            marker=dict(color="#A9B8C6", size=5)
        )
        st.plotly_chart(chart_style(fig, 320), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(
            '<div class="glass"><h3>Highest average-fare routes</h3>'
            '<p class="section-note">Top 10 routes by average fare.</p>',
            unsafe_allow_html=True
        )

        routes = (
            df.groupby("route", as_index=False).price.mean()
            .nlargest(10, "price")
            .sort_values("price")
        )

        fig = px.bar(
            routes,
            x="price",
            y="route",
            orientation="h",
            color="price",
            color_continuous_scale=["#E2E6EA", "#AAB5C1", "#65717E"],
            labels={"price": "Average fare (₹)", "route": "Route"}
        )
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(chart_style(fig, 340), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ==================== FARE EXPLORER ====================
elif page == "Fare Explorer":
    st.subheader("Fare Explorer")
    st.markdown(
        '<div class="section-note">Search historical flights, apply filters, '
        'compare fares and download matching results. '
        'This is not live airline inventory.</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("### 01 · Flight search")

    f1, f2, f3 = st.columns(3)

    with f1:
        origin = st.selectbox(
            "From city",
            ["Any"] + sorted(df.source_city.dropna().astype(str).unique()),
            key="fx_origin"
        )

    with f2:
        destination = st.selectbox(
            "To city",
            ["Any"] + sorted(df.destination_city.dropna().astype(str).unique()),
            key="fx_destination"
        )

    with f3:
        airline_pick = st.selectbox(
            "Airline",
            ["Any"] + sorted(df.airline.dropna().astype(str).unique()),
            key="fx_airline"
        )

    f4, f5, f6 = st.columns(3)

    with f4:
        class_pick = st.selectbox(
            "Cabin class",
            ["Any"] + sorted(df["class"].dropna().astype(str).unique()),
            key="fx_class"
        )

    with f5:
        dep_pick = st.selectbox(
            "Departure time",
            ["Any"] + sorted(df.departure_time.dropna().astype(str).unique()),
            key="fx_dep"
        )

    with f6:
        stop_pick = st.selectbox(
            "Number of stops",
            ["Any"] + sorted(df.stops.dropna().astype(str).unique()),
            key="fx_stops"
        )

    f7, f8 = st.columns(2)

    with f7:
        min_days = int(df.days_left.min())
        max_days = int(df.days_left.max())

        day_range = st.slider(
            "Days before departure",
            min_days,
            max_days,
            (min_days, max_days),
            key="fx_days"
        )

    with f8:
        low_price = float(df.price.min())
        high_price = float(df.price.max())

        if low_price < high_price:
            step_price = max(1.0, float(round((high_price - low_price) / 100)))
            price_range = st.slider(
                "Historical fare range (₹)",
                low_price,
                high_price,
                (low_price, high_price),
                step=step_price,
                key="fx_price"
            )
        else:
            price_range = (low_price, high_price)
            st.caption(f"Only one fare value: {rupees(low_price)}")

    st.markdown("</div>", unsafe_allow_html=True)

    # Apply filters
    result = df.copy()

    if origin != "Any":
        result = result[result.source_city.astype(str) == origin]

    if destination != "Any":
        result = result[result.destination_city.astype(str) == destination]

    if airline_pick != "Any":
        result = result[result.airline.astype(str) == airline_pick]

    if class_pick != "Any":
        result = result[result["class"].astype(str) == class_pick]

    if dep_pick != "Any":
        result = result[result.departure_time.astype(str) == dep_pick]

    if stop_pick != "Any":
        result = result[result.stops.astype(str) == stop_pick]

    result = result[result.days_left.between(day_range[0], day_range[1])]
    result = result[result.price.between(price_range[0], price_range[1])]

    st.markdown("### 02 · Matching fare results")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Matching flights", f"{len(result):,}")
    m2.metric("Average fare", rupees(result.price.mean()) if len(result) else "N/A")
    m3.metric("Lowest fare", rupees(result.price.min()) if len(result) else "N/A")
    m4.metric("Median fare", rupees(result.price.median()) if len(result) else "N/A")

    if result.empty:
        st.warning(
            "No flights match these filters. Broaden your city, class, "
            "days or fare range."
        )

    else:
        best = result.loc[result.price.idxmin()]

        st.markdown(f"""
        <div class="glass">
          <div class="eyebrow">LOWEST HISTORICAL FARE IN CURRENT FILTERS</div>
          <div style="font-size:2.5rem;font-weight:800;letter-spacing:-.04em">
            {rupees(best.price)}
          </div>
          <div style="color:#B9C0CA">
            {best.airline} · {best.source_city} → {best.destination_city}
          </div>
          <div style="color:#9097A1;font-size:.8rem;margin-top:8px">
            Class: {best['class']} · Stops: {best.stops} ·
            Duration: {best.duration:g} hours · {int(best.days_left)} days left
          </div>
        </div>
        """, unsafe_allow_html=True)

        left, right = st.columns(2, gap="large")

        with left:
            st.markdown(
                '<div class="glass"><h3>Fare comparison by airline</h3>',
                unsafe_allow_html=True
            )

            airline_fares = (
                result.groupby("airline", as_index=False).price.mean()
                .sort_values("price")
            )

            fig = px.bar(
                airline_fares,
                x="airline",
                y="price",
                color="price",
                color_continuous_scale=["#E3E7EC", "#A8B2BE", "#65717E"],
                labels={"price": "Average fare (₹)", "airline": "Airline"}
            )
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(chart_style(fig, 330), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown(
                '<div class="glass"><h3>Fare by number of stops</h3>',
                unsafe_allow_html=True
            )

            stops_fares = (
                result.groupby("stops", as_index=False).price.mean()
                .sort_values("price")
            )

            fig = px.bar(
                stops_fares,
                x="stops",
                y="price",
                color="price",
                color_continuous_scale=["#E3E7EC", "#A8B2BE", "#65717E"],
                labels={"price": "Average fare (₹)", "stops": "Stops"}
            )
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(chart_style(fig, 330), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### 03 · Flight options")

        sort_by = st.selectbox(
            "Sort results by",
            [
                "Lowest fare",
                "Highest fare",
                "Shortest duration",
                "Most advance booking"
            ]
        )

        if sort_by == "Lowest fare":
            result = result.sort_values("price")
        elif sort_by == "Highest fare":
            result = result.sort_values("price", ascending=False)
        elif sort_by == "Shortest duration":
            result = result.sort_values("duration")
        else:
            result = result.sort_values("days_left", ascending=False)

        visible_cols = [
            col for col in [
                "airline", "flight", "source_city", "destination_city",
                "departure_time", "arrival_time", "stops", "class",
                "duration", "days_left", "price"
            ]
            if col in result.columns
        ]

        st.dataframe(
            result[visible_cols].head(500),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download matching flight results",
            result[visible_cols].to_csv(index=False).encode("utf-8"),
            "airfare_search_results.csv",
            "text/csv"
        )


# ==================== FARE ESTIMATOR ====================
elif page == "Fare Estimator":
    st.subheader("Machine-learning fare estimator")
    st.markdown(
        '<div class="section-note">Use the saved model to estimate a fare. '
        'This is not a live airline quote.</div>',
        unsafe_allow_html=True
    )

    if model is None:
        st.error("best_model.joblib load nahi hua.")
        if model_error:
            st.code(model_error)
        else:
            st.info("Model file ko app.py ke same folder mein rakho.")

    else:
        with st.form("predict_fare"):
            a, b, c = st.columns(3)

            with a:
                airline = st.selectbox(
                    "Airline",
                    sorted(df.airline.dropna().astype(str).unique())
                )
                source = st.selectbox(
                    "From city",
                    sorted(df.source_city.dropna().astype(str).unique())
                )
                dep = st.selectbox(
                    "Departure time",
                    sorted(df.departure_time.dropna().astype(str).unique())
                )

            with b:
                dest_options = sorted(df.destination_city.dropna().astype(str).unique())
                dest_options = [x for x in dest_options if x != source] or dest_options

                dest = st.selectbox("To city", dest_options)
                arrival = st.selectbox(
                    "Arrival time",
                    sorted(df.arrival_time.dropna().astype(str).unique())
                )
                stops = st.selectbox(
                    "Stops",
                    sorted(df.stops.dropna().astype(str).unique())
                )

            with c:
                cabin = st.selectbox(
                    "Cabin class",
                    sorted(df["class"].dropna().astype(str).unique())
                )
                duration = st.slider(
                    "Duration (hours)",
                    float(df.duration.min()),
                    float(df.duration.max()),
                    float(np.median(df.duration)),
                    step=0.25
                )
                days = st.slider(
                    "Days left",
                    int(df.days_left.min()),
                    int(df.days_left.max()),
                    min(15, int(df.days_left.max()))
                )

            submit = st.form_submit_button("Estimate fare", use_container_width=True)

        if submit:
            row = {
                "airline": airline,
                "source_city": source,
                "departure_time": dep,
                "stops": stops,
                "arrival_time": arrival,
                "destination_city": dest,
                "class": cabin,
                "duration": float(duration),
                "days_left": int(days)
            }

            expected = list(getattr(model, "feature_names_in_", row.keys()))

            if any(col not in row for col in expected):
                st.error(
                    "Model features do not match the form. Expected: "
                    + ", ".join(expected)
                )
            else:
                try:
                    input_df = pd.DataFrame([{k: row[k] for k in expected}])
                    prediction = float(
                        np.asarray(model.predict(input_df)).reshape(-1)[0]
                    )

                    st.markdown(f"""
                    <div class="glass" style="text-align:center">
                      <div class="eyebrow">ESTIMATED TICKET FARE</div>
                      <div style="font-size:3.2rem;font-weight:800">
                        {rupees(prediction)}
                      </div>
                      <p>{airline} · {source} → {dest}</p>
                      <div style="color:#A9AFBA;font-size:.8rem">
                        Model estimate only — not a live quote.
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                except Exception as exc:
                    st.error(f"Prediction error: {exc}")
                    st.info(
                        "Saved model ki training features ya preprocessing "
                        "form se match nahi kar rahi. Training notebook check karna hoga."
                    )


# ==================== ROUTE INTELLIGENCE ====================
elif page == "Route Intelligence":
    st.subheader("Route intelligence")

    a, b, c = st.columns(3)

    with a:
        source = st.selectbox(
            "Origin city",
            ["All"] + sorted(df.source_city.dropna().astype(str).unique())
        )

    with b:
        dest = st.selectbox(
            "Destination city",
            ["All"] + sorted(df.destination_city.dropna().astype(str).unique())
        )

    with c:
        airline = st.selectbox(
            "Airline",
            ["All"] + sorted(df.airline.dropna().astype(str).unique())
        )

    view = df.copy()

    if source != "All":
        view = view[view.source_city.astype(str) == source]

    if dest != "All":
        view = view[view.destination_city.astype(str) == dest]

    if airline != "All":
        view = view[view.airline.astype(str) == airline]

    a, b, c = st.columns(3)
    a.metric("Records", f"{len(view):,}")
    b.metric("Average fare", rupees(view.price.mean()) if len(view) else "N/A")
    c.metric("Median fare", rupees(view.price.median()) if len(view) else "N/A")

    if not view.empty:
        summary = view.groupby(
            ["source_city", "destination_city", "airline"],
            as_index=False
        ).agg(
            flights=("price", "size"),
            average_fare=("price", "mean"),
            median_fare=("price", "median"),
            minimum_fare=("price", "min"),
            maximum_fare=("price", "max")
        ).sort_values("average_fare", ascending=False)

        st.dataframe(
            summary.round(0),
            use_container_width=True,
            hide_index=True
        )


# ==================== DATASET EXPLORER ====================
elif page == "Dataset Explorer":
    st.subheader("Dataset explorer")

    search = st.text_input("Search by city, airline, flight or class")
    view = df.copy()

    if search.strip():
        mask = view.astype(str).apply(
            lambda col: col.str.contains(search.strip(), case=False, na=False)
        )
        view = view[mask.any(axis=1)]

    a, b, c = st.columns(3)
    a.metric("Filtered records", f"{len(view):,}")
    b.metric("Dataset columns", f"{len(view.columns):,}")
    c.metric("Missing cells", f"{int(view.isna().sum().sum()):,}")

    st.dataframe(
        view.drop(columns=["route"], errors="ignore").head(1000),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download filtered CSV",
        view.drop(columns=["route"], errors="ignore")
        .to_csv(index=False).encode("utf-8"),
        "airfare_filtered_data.csv",
        "text/csv"
    )


# ==================== FOOTER ====================
st.markdown(
    '<div class="footer">'
    'AIRFARE INTELLIGENCE SYSTEM · HISTORICAL DATA + ML<br>'
    'Developed by Nasimuddin Ansari<br>'
    'Historical prices are for analysis only and do not represent live availability.'
    '</div>',
    unsafe_allow_html=True
)
