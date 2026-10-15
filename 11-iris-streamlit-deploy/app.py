from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris
import joblib


st.set_page_config(
    page_title="Iris Explorer",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).resolve().with_name("model.pkl")


@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_iris_data():
    iris = load_iris(as_frame=True)
    frame = iris.frame.copy()
    frame["species"] = frame["target"].map(
        {index: name for index, name in enumerate(iris.target_names)}
    )
    return iris, frame


def apply_styles():
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
            :root {
                --ink: #18332d;
                --muted: #71827c;
                --accent: #25866a;
                --line: #e3ebe6;
                --surface: #ffffff;
                --forest: #153c33;
                --lime: #d2e89d;
            }
            html, body, [class*="css"], [data-testid="stAppViewContainer"] {
                font-family: 'DM Sans', sans-serif;
                color: var(--ink) !important;
            }
            [data-testid="stAppViewContainer"] {
                background: #f4f7f5 !important;
            }
            [data-testid="stSidebar"] {
                background: var(--forest) !important;
                border-right: 1px solid rgba(255,255,255,.08);
            }
            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
            [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
                color: #b2c7be !important;
            }
            [data-testid="stSidebar"] hr {
                border-color: rgba(255,255,255,.14);
            }
            .brand {
                font-family: 'Manrope', sans-serif;
                font-size: 1.3rem;
                font-weight: 800;
                letter-spacing: -.04em;
                color: #f6faf6 !important;
                padding: .7rem 0 .15rem;
            }
            .brand span { color: var(--lime); }
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong {
                color: #f2f7f2 !important;
            }
            [data-testid="stSidebar"] label[data-testid="stRadioOption"] {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 10px;
                padding: .52rem .65rem;
                transition: background .18s ease, border-color .18s ease;
            }
            [data-testid="stSidebar"] label[data-testid="stRadioOption"]:hover {
                background: rgba(255,255,255,.07);
                border-color: rgba(255,255,255,.1);
            }
            [data-testid="stSidebar"] label[data-testid="stRadioOption"] p {
                color: #e8f1eb !important;
                font-weight: 600;
            }
            [data-testid="stSidebar"] label[data-testid="stRadioOption"][data-selected="true"] {
                background: rgba(210,232,157,.12);
                border-color: rgba(210,232,157,.2);
            }
            [data-testid="stSidebar"] label[data-testid="stRadioOption"][data-selected="true"] > div > div:first-child {
                border-color: var(--lime) !important;
                background: var(--lime) !important;
            }
            [data-testid="stSidebar"] label[data-testid="stRadioOption"][data-selected="true"] > div > div:first-child > div:first-child {
                background: var(--forest) !important;
            }
            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] .small-note {
                color: #b2c7be !important;
            }
            [data-testid="stAppViewContainer"] .main .block-container {
                max-width: 1320px;
                padding-top: 2.25rem;
                padding-bottom: 3rem;
            }
            header[data-testid="stHeader"] {
                background: transparent !important;
            }
            .eyebrow {
                display: inline-flex;
                align-items: center;
                gap: .5rem;
                color: var(--lime);
                font-size: .7rem;
                font-weight: 700;
                letter-spacing: .16em;
                text-transform: uppercase;
                margin-bottom: .95rem;
            }
            .eyebrow:before {
                content: "";
                width: 18px;
                height: 1px;
                background: var(--lime);
            }
            .page-title {
                font-family: 'Manrope', sans-serif;
                color: #f7fbf7 !important;
                font-size: clamp(2rem, 4vw, 3.15rem);
                font-weight: 800;
                letter-spacing: -.06em;
                line-height: 1.06;
                margin: 0 0 .65rem;
                max-width: 760px;
            }
            .page-subtitle {
                color: #c1d2c9 !important;
                font-size: .98rem;
                line-height: 1.65;
                max-width: 680px;
                margin: 0;
            }
            .page-hero {
                position: relative;
                overflow: hidden;
                isolation: isolate;
                border-radius: 24px;
                padding: clamp(1.65rem, 4vw, 2.7rem);
                margin-bottom: 1.45rem;
                background: linear-gradient(115deg, #153c33 0%, #1d5546 62%, #28715a 100%);
                box-shadow: 0 18px 45px rgba(22, 61, 51, .14);
            }
            .page-hero:after {
                content: "";
                position: absolute;
                z-index: -1;
                width: 340px;
                height: 340px;
                right: -70px;
                top: -190px;
                border: 1px solid rgba(210,232,157,.22);
                border-radius: 50%;
                box-shadow: 0 0 0 34px rgba(210,232,157,.04), 0 0 0 68px rgba(210,232,157,.035);
            }
            .section-title {
                font-family: 'Manrope', sans-serif;
                color: var(--ink);
                font-size: 1.12rem;
                font-weight: 800;
                letter-spacing: -.025em;
                margin: .25rem 0 .2rem;
            }
            .section-copy {
                color: var(--muted);
                font-size: .9rem;
                margin: 0 0 1rem;
            }
            div[data-testid="stMetric"] {
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: 15px;
                padding: 1rem 1.1rem;
                box-shadow: 0 8px 24px rgba(31, 59, 48, .045);
                transition: transform .18s ease, box-shadow .18s ease;
            }
            div[data-testid="stMetric"]:hover {
                transform: translateY(-2px);
                box-shadow: 0 12px 28px rgba(31, 59, 48, .08);
            }
            label[data-testid="stMetricLabel"],
            label[data-testid="stMetricLabel"] p,
            label[data-testid="stMetricLabel"] * {
                color: var(--muted) !important;
                font-size: .82rem;
            }
            div[data-testid="stMetricValue"],
            div[data-testid="stMetricValue"] * {
                color: var(--ink) !important;
                font-family: 'Manrope', sans-serif;
                font-weight: 800;
                letter-spacing: -.04em;
            }
            div[data-testid="stVerticalBlockBorderWrapper"] {
                border: 1px solid var(--line) !important;
                border-radius: 18px !important;
                background: rgba(255,255,255,.92) !important;
                box-shadow: 0 8px 24px rgba(31, 59, 48, .035);
            }
            [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stMarkdownContainer"] h4 {
                color: var(--ink);
                font-family: 'Manrope', sans-serif;
                letter-spacing: -.025em;
            }
            .species-result {
                font-family: 'Manrope', sans-serif;
                color: var(--ink);
                font-weight: 800;
                font-size: 1.8rem;
                letter-spacing: -.04em;
                margin: .2rem 0 .1rem;
                text-transform: capitalize;
            }
            .small-note {
                color: var(--muted);
                font-size: .82rem;
                line-height: 1.55;
            }
            .result-note {
                background: #f2f7f3;
                border: 1px solid #e2eee6;
                border-left: 3px solid var(--accent);
                border-radius: 10px;
                color: #65786f;
                font-size: .88rem;
                line-height: 1.6;
                padding: .8rem .9rem;
            }
            [data-testid="stSlider"] [data-rac][style*="left"] {
                background: var(--accent) !important;
            }
            [data-testid="stSlider"] [data-rac][style*="left"]::before {
                background: var(--accent) !important;
            }
            .stButton > button[kind="primary"] {
                background: var(--accent);
                border-color: var(--accent);
                border-radius: 11px;
                font-weight: 700;
                min-height: 2.8rem;
                box-shadow: 0 7px 16px rgba(37,134,106,.18);
                transition: transform .18s ease, background .18s ease;
            }
            .stButton > button[kind="primary"]:hover {
                background: #176c55;
                border-color: #176c55;
                transform: translateY(-1px);
            }
            [data-testid="stDataFrame"] {
                border: 1px solid var(--line);
                border-radius: 12px;
                overflow: hidden;
            }
            @media (max-width: 700px) {
                [data-testid="stAppViewContainer"] .main .block-container {
                    padding: 1.2rem 1rem 2rem;
                }
                .page-hero { border-radius: 18px; }
                div[data-testid="stMetric"] { padding: .8rem; }
            }
            footer { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(eyebrow, title, subtitle):
    st.markdown(
        f"""
        <section class="page-hero">
            <div class="eyebrow">{eyebrow}</div>
            <h1 class="page-title">{title}</h1>
            <p class="page-subtitle">{subtitle}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_overview(model, iris, frame):
    render_header(
        "IRIS CLASSIFIER · INTERACTIVE DEMO",
        "A closer look at the Iris flower",
        "Explore the classic three-species dataset and use the trained model to identify a flower from its measurements.",
    )

    metric_columns = st.columns(4)
    metrics = [
        ("150", "flower samples"),
        ("3", "species"),
        ("4", "measurements"),
        ("Random forest", "model"),
    ]
    for column, (value, label) in zip(metric_columns, metrics):
        column.metric(label, value)

    st.write("")
    st.markdown('<div class="section-title">Try a prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-copy">Adjust the flower measurements, then ask the model for its best match.</p>',
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        input_column, result_column = st.columns([1.05, 0.95], gap="large")
        feature_names = list(iris.feature_names)
        limits = frame[feature_names].agg(["min", "max", "median"])
        defaults = {
            "sepal length (cm)": 5.8,
            "sepal width (cm)": 3.0,
            "petal length (cm)": 4.35,
            "petal width (cm)": 1.3,
        }
        values = []
        with input_column:
            st.markdown("#### Flower measurements")
            st.caption("All measurements are in centimeters.")
            first, second = st.columns(2)
            for index, feature in enumerate(feature_names):
                column = first if index % 2 == 0 else second
                minimum = float(limits.loc["min", feature])
                maximum = float(limits.loc["max", feature])
                value = column.slider(
                    feature.title(),
                    min_value=minimum,
                    max_value=maximum,
                    value=min(max(defaults[feature], minimum), maximum),
                    step=0.1,
                    key=f"input_{feature}",
                )
                values.append(value)
            predict = st.button("Identify this flower", type="primary", use_container_width=True)

        with result_column:
            st.markdown("#### Model result")
            if predict:
                prediction = int(model.predict([values])[0])
                probabilities = model.predict_proba([values])[0]
                species = str(iris.target_names[prediction])
                confidence = float(probabilities[prediction])
                st.markdown(
                    f'<div class="species-result">{species}</div>'
                    f'<div class="small-note">Most likely species · {confidence:.0%} model confidence</div>',
                    unsafe_allow_html=True,
                )
                st.write("")
                probability_frame = pd.DataFrame(
                    {"Probability": probabilities},
                    index=[str(name).capitalize() for name in iris.target_names],
                )
                st.bar_chart(probability_frame, horizontal=True, color="#21866b", height=180)
                st.caption("Confidence is the model's predicted probability, not a guarantee.")
            else:
                st.markdown(
                    '<div class="species-result">Ready to explore</div>'
                    '<p class="small-note">Set the four measurements and select “Identify this flower” to see the predicted species and class probabilities.</p>',
                    unsafe_allow_html=True,
                )
                st.write("")
                st.markdown(
                    '<div class="result-note">The classifier compares sepal and petal dimensions across three Iris species.</div>',
                    unsafe_allow_html=True,
                )

    st.write("")
    with st.expander("About this model"):
        st.write(
            "This Random Forest classifier was loaded from `model.pkl`. "
            "The sample training script fits it on the complete Iris dataset, so this demo "
            "shows predictions rather than presenting training-set accuracy as a measure of generalization."
        )


def render_summary(iris, frame):
    render_header(
        "DATASET SUMMARY",
        "The data behind the flowers",
        "A compact overview of the measurements, species balance, and patterns in the classic Iris dataset.",
    )

    features = list(iris.feature_names)
    species_count = frame["species"].nunique()
    missing_count = int(frame[features].isna().sum().sum())
    metrics = st.columns(4)
    metrics[0].metric("Total samples", f"{len(frame):,}")
    metrics[1].metric("Species", f"{species_count}")
    metrics[2].metric("Features", f"{len(features)}")
    metrics[3].metric("Missing measurements", f"{missing_count}")

    st.write("")
    chart_column, scatter_column = st.columns([0.85, 1.15], gap="large")
    with chart_column:
        with st.container(border=True):
            st.markdown('<div class="section-title">Species distribution</div>', unsafe_allow_html=True)
            st.markdown(
                '<p class="section-copy">The dataset is evenly balanced across the three classes.</p>',
                unsafe_allow_html=True,
            )
            counts = frame["species"].value_counts().rename_axis("Species").to_frame("Samples")
            counts.index = counts.index.str.capitalize()
            st.bar_chart(counts, horizontal=True, color="#21866b", height=220)
            st.caption("Each species contributes the same number of samples.")

    with scatter_column:
        with st.container(border=True):
            st.markdown('<div class="section-title">Petal measurements</div>', unsafe_allow_html=True)
            st.markdown(
                '<p class="section-copy">Petal length and width separate the species especially clearly.</p>',
                unsafe_allow_html=True,
            )
            chart_data = frame.rename(
                columns={
                    "petal length (cm)": "Petal length (cm)",
                    "petal width (cm)": "Petal width (cm)",
                    "species": "Species",
                }
            )
            st.scatter_chart(
                chart_data,
                x="Petal length (cm)",
                y="Petal width (cm)",
                color="Species",
                size=70,
                height=275,
            )

    st.write("")
    with st.container(border=True):
        st.markdown('<div class="section-title">Measurement summary</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-copy">Average and spread for each measurement, in centimeters.</p>',
            unsafe_allow_html=True,
        )
        summary = frame.groupby("species", observed=True)[features].agg(["mean", "std"]).T
        summary.columns = [name.capitalize() for name in summary.columns]
        summary.index = [
            f"{feature.replace(' (cm)', '').title()} · {statistic}"
            for feature, statistic in summary.index
        ]
        st.dataframe(summary.style.format("{:.2f}"), use_container_width=True)

    with st.expander("Preview the dataset"):
        preview = frame[features + ["species"]].rename(
            columns={feature: feature.title() for feature in features}
        )
        preview["species"] = preview["species"].str.capitalize()
        st.dataframe(preview, use_container_width=True, hide_index=True)

    st.caption(
        "Source: Fisher's Iris dataset, distributed with scikit-learn. "
        "Each sample has sepal length, sepal width, petal length, and petal width measurements."
    )


def main():
    apply_styles()
    try:
        model = load_model()
        iris, frame = load_iris_data()
    except (FileNotFoundError, OSError, ValueError, EOFError) as error:
        st.error(f"Unable to load the Iris model or dataset: {error}")
        st.stop()

    if getattr(model, "n_features_in_", None) != len(iris.feature_names):
        st.error(
            "The saved model does not match the Iris dataset: "
            f"expected {len(iris.feature_names)} input features."
        )
        st.stop()

    with st.sidebar:
        st.markdown('<div class="brand">iris<span>.</span> explorer</div>', unsafe_allow_html=True)
        st.caption("A small, interactive machine-learning demo")
        st.divider()
        page = st.radio(
            "Navigate",
            ["Overview", "Summary"],
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown("**THE IRIS DATASET**")
        st.markdown(
            '<p class="small-note">Four flower measurements. Three species. One classic classification problem.</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="small-note">Model · Random Forest<br>Classes · Setosa, Versicolor, Virginica</p>',
            unsafe_allow_html=True,
        )

    if page == "Overview":
        render_overview(model, iris, frame)
    else:
        render_summary(iris, frame)


if __name__ == "__main__":
    main()