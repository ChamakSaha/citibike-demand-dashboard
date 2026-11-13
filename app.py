"""
NYC Hourly Bike Demand Dashboard (Streamlit)
Chamak Saha

Run locally:  streamlit run app.py
Needs the artifacts/ folder created by the Colab training notebook.
"""
import calendar
import datetime as dt
import json
import warnings
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pandas.tseries.holiday import USFederalHolidayCalendar

warnings.filterwarnings("ignore")
st.set_page_config(page_title="NYC Hourly Bike Demand Dashboard", page_icon="🚲", layout="centered")

ART = Path(__file__).parent / "artifacts"
INK, MUTED, BLUE, ORANGE, GRAY = "#111111", "#6b6b6b", "#2a78d6", "#eb6834", "#a3a3a3"
SEQ = ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
ALL = "ALL"

st.markdown("""
<style>
.stApp {background:#fafafa;}
.block-container {padding-top:1.4rem; max-width:860px;}
h1,h2,h3 {letter-spacing:-.01em;}
/* pill-style tabs (works for old and new Streamlit tab markup) */
[role="tablist"] {display:flex !important; gap:8px; width:100%; border-bottom:none !important;}
[role="tablist"] [role="tab"] {flex:1 1 0; display:flex; justify-content:center; border-radius:999px;
  padding:10px 0 !important; color:#8a8a8a; background:transparent; border:none !important;}
[role="tablist"] [role="tab"] p {font-weight:600; font-size:1rem;}
[role="tablist"] [role="tab"][aria-selected="true"] {background:#efefef; color:#111;}
.react-aria-SelectionIndicator, [data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {display:none !important;}
/* form without box, rounded inputs */
[data-testid="stForm"] {border:none !important; padding:0 !important;}
div[data-baseweb="select"] > div, [data-testid="stDateInput"] div[data-baseweb="input"] {
  border-radius:999px !important; background:#fff !important; border:1px solid #dcdcdc !important;}
[data-testid="stDateInput"] input {background:#fff !important;}
/* black pill button */
[data-testid="stFormSubmitButton"] button, .stButton > button {background:#111 !important; color:#fff !important;
  border-radius:999px !important; border:none !important; padding:.7rem 0 !important; font-weight:600;}
[data-testid="stFormSubmitButton"] button:hover, .stButton > button:hover {background:#333 !important;}
[data-testid="stFormSubmitButton"] button p {color:#fff !important; font-weight:600;}
/* cards */
.card {background:#fff; border:1px solid #e6e6e6; border-radius:16px; padding:18px 20px; margin:10px 0;}
.card .label {color:#555; font-weight:600; font-size:.95rem;}
.card .big {font-size:2.3rem; font-weight:700; color:#111; line-height:1.2; margin-top:6px;}
.card .unit {font-size:1.25rem; font-weight:400; color:#666; margin-left:6px;}
.card .meta {color:#666; font-size:.85rem; margin:4px 0 10px 0;}
.card .note {color:#666; font-size:.9rem; margin-top:10px;}
.badge {display:inline-block; padding:3px 12px; border-radius:999px; font-size:.85rem; font-weight:600;}
.b-high {background:#fde3d8; color:#9a3412;} .b-med {background:#fdf1cc; color:#854d0e;}
.b-low {background:#dcf3e4; color:#166534;}
.kpi {background:#fff; border:1px solid #e6e6e6; border-radius:14px; padding:12px 14px;}
.kpi .k {color:#666; font-size:.8rem;} .kpi .v {font-size:1.35rem; font-weight:700; color:#111;}
.sub {color:#666; margin-top:-6px;}
[data-testid="stVerticalBlockBorderWrapper"] {background:#fff; border-radius:16px;}
</style>
""", unsafe_allow_html=True)




NEEDED = ["model.joblib", "metadata.json", "stations.csv", "city_detail.csv.gz",
          "station_dow_hour.csv.gz", "station_daily.csv.gz", "profile.csv.gz"]
_missing = [f for f in NEEDED if not (ART / f).exists()]
if _missing:
    st.error(f"Missing files in `artifacts/`: {', '.join(_missing)}. Run the Colab notebook first.")
    st.stop()

# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    return joblib.load(ART / "model.joblib")


@st.cache_data(show_spinner="Loading data…")
def load_data():
    meta = json.loads((ART / "metadata.json").read_text())
    stations = pd.read_csv(ART / "stations.csv", dtype={"STATION_ID": str})
    detail = pd.read_csv(ART / "city_detail.csv.gz", parse_dates=["HOUR_TS"])
    sdh = pd.read_csv(ART / "station_dow_hour.csv.gz", dtype={"STATION_ID": str})
    sdaily = pd.read_csv(ART / "station_daily.csv.gz", dtype={"STATION_ID": str}, parse_dates=["DATE"])
    profile = pd.read_csv(ART / "profile.csv.gz", dtype={"STATION_ID": str})
    return meta, stations, detail, sdh, sdaily, profile


model = load_model()
meta, stations, detail, sdh, sdaily, profile = load_data()
NAME = dict(zip(stations["STATION_ID"], stations["NAME"]))
HOLIDAYS = {d.date() for d in USFederalHolidayCalendar().holidays("2020-01-01", "2030-12-31")}
DATA_DATES = pd.to_datetime(meta.get("dates") or pd.date_range("2024-01-01", "2024-01-31"))
JAN_DOW_COUNT = {i: max(1, int((DATA_DATES.dayofweek == i).sum())) for i in range(7)}
STATION_OPTS = stations.sort_values("NAME")["STATION_ID"].tolist()
TOP_STATION = stations.sort_values("TRIPS", ascending=False)["STATION_ID"].iloc[0]


def time_features(ts):
    ts = pd.to_datetime(ts)
    return pd.DataFrame({"HOUR": ts.dt.hour, "DOW": ts.dt.dayofweek,
                         "IS_WEEKEND": (ts.dt.dayofweek >= 5).astype(int),
                         "IS_HOLIDAY": [int(t.date() in HOLIDAYS) for t in ts]}, index=ts.index)


@st.cache_data(max_entries=256, show_spinner=False)
def station_day(sid: str, date_iso: str) -> np.ndarray:
    """Predicted bike demand (trips) for one station for each hour of one date."""
    s = stations[stations["STATION_ID"] == sid].iloc[0]
    hrs = pd.Series(pd.date_range(pd.Timestamp(date_iso), periods=24, freq="h"))
    X = pd.DataFrame({"STATION_ID": sid, "LAT": s["LAT"], "LNG": s["LNG"]}, index=hrs.index)
    X = X.join(time_features(hrs))
    # add the station's historical demand pattern for that weekday and hour
    X = X.merge(profile, on=["STATION_ID", "DOW", "HOUR"], how="left").fillna(0)
    return np.clip(model.predict(X[meta["features"]]), 0, None)


def typical(sid, dow):
    s = sdh[(sdh["STATION_ID"] == sid) & (sdh["DOW"] == dow)]
    return (s.groupby("HOUR")["TRIPS"].sum().reindex(range(24), fill_value=0) / JAN_DOW_COUNT[dow]).to_numpy()


def demand_level(p, sid):
    """Compare with this station's own busiest hours in January."""
    s = sdh[sdh["STATION_ID"] == sid]
    hourly_avg = (s.groupby(["DOW", "HOUR"])["TRIPS"].sum() /
                  s.groupby(["DOW", "HOUR"])["DOW"].first().map(JAN_DOW_COUNT))
    hi, lo = np.quantile(hourly_avg, 0.75), np.quantile(hourly_avg, 0.40)
    if p >= hi:
        return "High demand", "b-high"
    if p >= lo:
        return "Medium demand", "b-med"
    return "Low demand", "b-low"


# ---------------------------------------------------------------------------
st.markdown("## 🚲 NYC Hourly Bike Demand Dashboard")
st.markdown("<div class='sub'>Explore Citi Bike demand patterns across New York City and get the "
            "predicted bike demand for any station, date and hour.</div>",
            unsafe_allow_html=True)
st.write("")

tab_pred, tab_exp, tab_about = st.tabs(["Predict Demand", "Explore Patterns", "About the Models"])

# ======================= PREDICT =======================
with tab_pred:
    st.markdown("### Predict Hourly Bike Demand")
    st.markdown("<div class='sub'>Select a station, date, and time to estimate bike demand.</div>",
                unsafe_allow_html=True)
    st.write("")
    with st.form("predict"):
        sid = st.selectbox("Bike Station", STATION_OPTS, index=STATION_OPTS.index(TOP_STATION),
                           format_func=lambda s: NAME[s])
        c1, c2 = st.columns(2)
        today = dt.date.today()
        default_date = today if dt.date(2024, 1, 1) <= today <= dt.date(2030, 12, 31) else dt.date(2024, 1, 17)
        date = c1.date_input("Select Date", value=default_date, min_value=dt.date(2024, 1, 1),
                             max_value=dt.date(2030, 12, 31), format="MM/DD/YYYY")
        hour = c2.selectbox("Time", list(range(24)), index=17,
                            format_func=lambda h: dt.time(h).strftime("%I:%M %p").lstrip("0"))
        st.caption("🌦️ Weather: not available in the Citi Bike trip data, so predictions assume "
                   "typical conditions.")
        go = st.form_submit_button("✨  Predict Bike Demand", width="stretch")

    if go:
        st.session_state["pred"] = (sid, date, hour)

    if "pred" in st.session_state:
        sid, date, hour = st.session_state["pred"]
        prof = station_day(sid, date.isoformat())
        p = float(prof[hour])
        label, cls = demand_level(p, sid)
        note = ("Bike demand = number of trips starting at this station in that hour. "
                "Predicted by a model trained on Citi Bike trips from January 2024.")
        if date.month != 1:
            note += (f" {calendar.month_name[date.month]} is outside the training month, so it is "
                     "shown at January (winter) levels.")
        if date in HOLIDAYS:
            note += " This date is a U.S. holiday."
        st.markdown(f"""
<div class='card'>
  <div class='label'>Estimated bike demand</div>
  <div class='big'>{p:,.0f}<span class='unit'>trips / hour</span></div>
  <div class='meta'>{NAME[sid]} · {date:%Y-%m-%d} · {hour:02d}:00</div>
  <span class='badge {cls}'>{label}</span>
  <div class='note'>{note}</div>
</div>""", unsafe_allow_html=True)

        st.divider()
        st.markdown("### Hourly Demand Trend")
        st.markdown(f"<div class='sub'>Predicted trips for every hour of {date:%A, %d %b %Y} at "
                    f"{NAME[sid]}, compared with the January average for {DAYS[date.weekday()]}s.</div>",
                    unsafe_allow_html=True)
        dom = ["Predicted", f"January average ({DAYS[date.weekday()]})"]
        df = pd.DataFrame({"Hour": list(range(24)) * 2,
                           "Trips": list(prof) + list(typical(sid, date.weekday())),
                           "Series": [dom[0]] * 24 + [dom[1]] * 24})
        base = alt.Chart(df).encode(
            x=alt.X("Hour:Q", scale=alt.Scale(domain=[0, 23]), axis=alt.Axis(tickMinStep=2, grid=False)),
            y=alt.Y("Trips:Q", title="Trips per hour"),
            color=alt.Color("Series:N", title=None, legend=alt.Legend(orient="top", labelLimit=300),
                            scale=alt.Scale(domain=dom, range=[INK, GRAY])),
            strokeDash=alt.StrokeDash("Series:N", legend=None,
                                      scale=alt.Scale(domain=dom, range=[[1, 0], [4, 4]])),
            tooltip=["Series", "Hour", alt.Tooltip("Trips:Q", format=",.1f")])
        area = alt.Chart(df[df["Series"] == dom[0]]).mark_area(color=INK, opacity=.06).encode(
            x="Hour:Q", y="Trips:Q")
        dot = alt.Chart(pd.DataFrame({"Hour": [hour], "Trips": [p]})).mark_circle(
            size=170, color=ORANGE).encode(x="Hour:Q", y="Trips:Q")
        st.altair_chart((area + base.mark_line(strokeWidth=2.5) + dot)
                        .properties(height=280).configure_view(stroke=None), width="stretch")
        best = int(np.argmax(prof))
        quiet = 6 + int(np.argmin(prof[6:23]))
        st.caption(f"Peak hour: **{best:02d}:00** ({prof[best]:.0f} trips) · "
                   f"Quietest daytime hour: **{quiet:02d}:00** ({prof[quiet]:.0f} trips)")
    else:
        st.info("Choose a station, date and time, then press **Predict Bike Demand**.")

# ======================= EXPLORE =======================
with tab_exp:
    st.markdown("### Explore Demand Patterns")
    st.markdown("<div class='sub'>Actual Citi Bike trips in January 2024.</div>", unsafe_allow_html=True)
    st.write("")
    f1, f2 = st.columns(2)
    e_loc = f1.selectbox("Station", [ALL] + STATION_OPTS, key="e_loc",
                         format_func=lambda s: "All NYC stations" if s == ALL else NAME[s])
    daytype = f2.selectbox("Days", ["All days", "Weekdays", "Weekends"])
    citywide = e_loc == ALL
    f3, f4 = st.columns(2)
    rider = f3.selectbox("Rider type", ["All", "member", "casual"], disabled=not citywide,
                         format_func=str.capitalize)
    bike = f4.selectbox("Bike type", ["All", "classic_bike", "electric_bike"], disabled=not citywide,
                        format_func=lambda b: b.replace("_", " ").capitalize())

    if citywide:
        d = detail
        if rider != "All":
            d = d[d["MEMBER"] == rider]
        if bike != "All":
            d = d[d["BIKE"] == bike]
        hourly = d.groupby("HOUR_TS")["TRIPS"].sum().reindex(
            pd.date_range(DATA_DATES.min(), DATA_DATES.max() + pd.Timedelta(hours=23), freq="h"), fill_value=0)
        h = pd.DataFrame({"TS": hourly.index, "TRIPS": hourly.to_numpy()})
        h["DATE"], h["DOW"], h["HOUR"] = h["TS"].dt.normalize(), h["TS"].dt.dayofweek, h["TS"].dt.hour
        daily = h.groupby("DATE", as_index=False)["TRIPS"].sum()
        dh = h.groupby(["DOW", "HOUR"], as_index=False)["TRIPS"].sum()
    else:
        s = sdaily[sdaily["STATION_ID"] == e_loc].groupby("DATE")["TRIPS"].sum().reindex(
            DATA_DATES, fill_value=0)
        daily = pd.DataFrame({"DATE": s.index, "TRIPS": s.to_numpy()})
        full = pd.MultiIndex.from_product([range(7), range(24)], names=["DOW", "HOUR"])
        dh = (sdh[sdh["STATION_ID"] == e_loc].set_index(["DOW", "HOUR"])["TRIPS"]
              .reindex(full, fill_value=0).reset_index())
    keep = {"Weekdays": range(5), "Weekends": range(5, 7)}.get(daytype, range(7))
    daily = daily[[t.weekday() in keep for t in daily["DATE"]]]
    dh = dh[dh["DOW"].isin(list(keep))].copy()
    dh["AVG"] = dh["TRIPS"] / dh["DOW"].map(JAN_DOW_COUNT)
    dh["Day"] = dh["DOW"].map(lambda i: DAYS[int(i)][:3])
    byh = dh.groupby("HOUR")["AVG"].mean()

    k = st.columns(3)
    for col, (lab, val) in zip(k, [("Total trips", f"{int(daily['TRIPS'].sum()):,}"),
                                   ("Average per day", f"{daily['TRIPS'].mean():,.0f}"),
                                   ("Peak hour", f"{int(byh.idxmax()):02d}:00")]):
        col.markdown(f"<div class='kpi'><div class='k'>{lab}</div><div class='v'>{val}</div></div>",
                     unsafe_allow_html=True)
    st.write("")

    with st.container(border=True):
        st.markdown("**Average trips per hour**")
        wk = dh.assign(Type=np.where(dh["DOW"] >= 5, "Weekend", "Weekday")).groupby(
            ["Type", "HOUR"], as_index=False)["AVG"].mean()
        st.altair_chart(alt.Chart(wk).mark_line(strokeWidth=2.5).encode(
            x=alt.X("HOUR:Q", title="Hour", scale=alt.Scale(domain=[0, 23])),
            y=alt.Y("AVG:Q", title="Trips per hour"),
            color=alt.Color("Type:N", title=None, legend=alt.Legend(orient="top"),
                            scale=alt.Scale(domain=["Weekday", "Weekend"], range=[BLUE, ORANGE])),
            tooltip=["Type", "HOUR", alt.Tooltip("AVG:Q", format=",.1f")]).properties(height=240),
            width="stretch")
    with st.container(border=True):
        st.markdown("**Trips per day in January**")
        st.altair_chart(alt.Chart(daily).mark_bar(color=BLUE, cornerRadiusTopLeft=3,
                                                  cornerRadiusTopRight=3).encode(
            x=alt.X("DATE:T", title=None, axis=alt.Axis(format="%d %b")),
            y=alt.Y("TRIPS:Q", title="Trips"),
            tooltip=[alt.Tooltip("DATE:T", format="%a %d %b"), alt.Tooltip("TRIPS:Q", format=",")])
            .properties(height=220), width="stretch")
    with st.container(border=True):
        st.markdown("**When is demand highest?** (darker = busier)")
        st.altair_chart(alt.Chart(dh).mark_rect(stroke="white", strokeWidth=1, cornerRadius=2).encode(
            x=alt.X("HOUR:O", title="Hour", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Day:N", sort=[d[:3] for d in DAYS], title=None),
            color=alt.Color("AVG:Q", title="Trips/h", scale=alt.Scale(range=SEQ)),
            tooltip=["Day", "HOUR", alt.Tooltip("AVG:Q", format=",.1f")]).properties(height=210),
            width="stretch")
    if citywide:
        with st.container(border=True):
            st.markdown("**10 busiest stations**")
            st.altair_chart(alt.Chart(stations.nlargest(10, "TRIPS")).mark_bar(
                color=BLUE, cornerRadiusTopRight=3, cornerRadiusBottomRight=3).encode(
                y=alt.Y("NAME:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
                x=alt.X("TRIPS:Q", title="Trips in January"),
                tooltip=["NAME", alt.Tooltip("TRIPS:Q", format=",")]).properties(height=300),
                width="stretch")

# ======================= ABOUT =======================
with tab_about:
    st.markdown("### About the Models")
    st.markdown("<div class='sub'>How the bike demand model was trained and how well it performs "
                "on unseen test data.</div>", unsafe_allow_html=True)
    st.write("")

    with st.container(border=True):
        st.markdown("**Data and split**")
        k = st.columns(4)
        for col, (lab, val) in zip(k, [("Trips", f"{meta['n_trips']:,}"),
                                       ("Stations", f"{meta['n_stations']:,}"),
                                       ("Training set", meta["split"]["train"]),
                                       ("Test set", meta["split"]["test"])]):
            col.markdown(f"<div class='kpi'><div class='k'>{lab}</div><div class='v' "
                         f"style='font-size:1.05rem'>{val}</div></div>", unsafe_allow_html=True)
        st.write("")
        st.markdown(f"""
- **Bike demand (target):** number of trips starting at a station in one hour.
- **Inputs:** station location (latitude/longitude), hour, day of week, weekend, U.S. holiday, plus
  four **extracted features** describing the station's historical demand: its overall average,
  its average at that hour, at that hour on weekdays vs weekends, and at that weekday and hour.
  For testing, these averages use only the training days, so there is no leakage.
- **Split:** chronological. Models learn from {meta['n_train_rows']:,} station-hours ({meta['split']['train']})
  and are tested on {meta['n_test_rows']:,} station-hours ({meta['split']['test']}) they never saw.
""")

    res = pd.DataFrame(meta["models"])
    best = meta["best_model"]
    with st.container(border=True):
        st.markdown("**Model performance on the test set**")
        show = res[["model", "mae", "rmse", "r2"]].rename(
            columns={"model": "Model", "mae": "MAE", "rmse": "RMSE", "r2": "R²"})
        show["Model"] = [m + "  ✓ best" if m == best else m for m in show["Model"]]
        st.dataframe(show.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "R²": "{:.3f}"})
                     .apply(lambda r: ["font-weight:700; background:#eef5fd"
                                       if r["Model"].endswith("best") else "" for _ in r], axis=1),
                     hide_index=True, width="stretch")
        st.caption("MAE and RMSE: average error in trips per station-hour (lower is better). "
                   "R²: share of the variation in bike demand explained (higher is better).")
        res["Best"] = res["model"] == best
        st.altair_chart(alt.Chart(res).mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3).encode(
            y=alt.Y("model:N", sort="-x", title=None, axis=alt.Axis(labelLimit=240)),
            x=alt.X("r2:Q", title="R² on test set (higher is better)", scale=alt.Scale(domain=[0, 1])),
            color=alt.condition("datum.Best", alt.value(BLUE), alt.value("#c9c9c9")),
            tooltip=[alt.Tooltip("model:N", title="Model"), alt.Tooltip("mae:Q", format=".3f", title="MAE"),
                     alt.Tooltip("rmse:Q", format=".3f", title="RMSE"),
                     alt.Tooltip("r2:Q", format=".3f", title="R²")]).properties(height=150),
            width="stretch")

    b = res[res["model"] == best].iloc[0]
    with st.container(border=True):
        st.markdown(f"**✅ Selected model: {best}**")
        st.markdown(f"""
Lowest test error (RMSE {b['rmse']:.3f}, MAE {b['mae']:.3f}) with R² {b['r2']:.3f}.
After testing, it was retrained on all of January and is the model used on the **Predict Demand** tab.

**Limitations**
- No weather data in the Citi Bike trips, so weather is not an input.
- The model is trained on January 2024 data only.
""")
    if meta.get("importance"):
        NICE = {"LAT": "Latitude", "LNG": "Longitude", "HOUR": "Hour", "DOW": "Day of week",
                "IS_WEEKEND": "Weekend", "IS_HOLIDAY": "Holiday",
                "ST_MEAN": "Station average", "ST_HOUR_MEAN": "Station average at hour",
                "ST_DAYTYPE_HOUR_MEAN": "Station avg, weekday/weekend × hour",
                "ST_DOW_HOUR_MEAN": "Station avg, weekday × hour"}
        imp = pd.DataFrame({"Feature": [NICE.get(k, k) for k in meta["importance"]],
                            "Importance": list(meta["importance"].values())})
        with st.container(border=True):
            st.markdown(f"**Which features matter most? ({best})**")
            st.caption("Drop in test R² when a feature is shuffled (permutation importance).")
            st.altair_chart(alt.Chart(imp).mark_bar(color=BLUE, cornerRadiusTopRight=3,
                                                    cornerRadiusBottomRight=3).encode(
                y=alt.Y("Feature:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
                x=alt.X("Importance:Q", title="Drop in R²"),
                tooltip=["Feature", alt.Tooltip("Importance:Q", format=".4f")])
                .properties(height=280), width="stretch")
    st.caption("Data: Citi Bike System Data, January 2024 · Built by Chamak Saha")
