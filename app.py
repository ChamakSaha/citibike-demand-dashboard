"""
NYC Citi Bike Demand Dashboard (Streamlit)
Chamak Saha

Run locally:  streamlit run app.py
Needs the artifacts/ folder created by the Colab training notebook.
"""
import calendar
import datetime as dt
import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from pandas.tseries.holiday import USFederalHolidayCalendar

st.set_page_config(page_title="NYC Citi Bike Demand Dashboard", page_icon="🚲", layout="wide")

ART = Path(__file__).parent / "artifacts"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"
SEQ = ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"]   # light → dark blue
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
ALL = "ALL"
NEEDED = ["plan_model.joblib", "next_model.joblib", "metadata.json", "stations.csv",
          "city_hourly.csv", "city_detail.csv.gz", "station_dow_hour.csv.gz",
          "station_daily.csv.gz"]

missing = [f for f in NEEDED if not (ART / f).exists()]
if missing:
    st.error(f"Missing files in `artifacts/`: {', '.join(missing)}. "
             "Run the Colab training notebook and upload its output.")
    st.stop()


# ----------------------------------------------------------------------------
# Loading (cached once per server and shared by all users)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models…")
def load_models():
    return joblib.load(ART / "plan_model.joblib"), joblib.load(ART / "next_model.joblib")


@st.cache_data(show_spinner="Loading data…")
def load_data():
    meta = json.loads((ART / "metadata.json").read_text())
    stations = pd.read_csv(ART / "stations.csv", dtype={"STATION_ID": str})
    city = pd.read_csv(ART / "city_hourly.csv", parse_dates=["hourly_timestamp"])
    detail = pd.read_csv(ART / "city_detail.csv.gz", parse_dates=["HOUR_TS"])
    sdh = pd.read_csv(ART / "station_dow_hour.csv.gz", dtype={"STATION_ID": str})
    sdaily = pd.read_csv(ART / "station_daily.csv.gz", dtype={"STATION_ID": str},
                         parse_dates=["DATE"])
    return meta, stations, city, detail, sdh, sdaily


plan_model, next_model = load_models()
meta, stations, city, detail, sdh, sdaily = load_data()
NAME = dict(zip(stations["STATION_ID"], stations["NAME"]))
HOLIDAYS = set(USFederalHolidayCalendar().holidays("2020-01-01", "2030-12-31").date)
# Number of Mondays, Tuesdays, … in January 2024 (to turn monthly totals into daily averages)
JAN_DOW_COUNT = pd.Series(pd.date_range("2024-01-01", "2024-01-31").dayofweek).value_counts().to_dict()


def loc_label(sid):
    return "🗽 All NYC (citywide)" if sid == ALL else f"{NAME.get(sid, sid)}"


def time_features(ts: pd.Series) -> pd.DataFrame:
    ts = pd.to_datetime(ts)
    return pd.DataFrame({
        "HOUR": ts.dt.hour,
        "DOW": ts.dt.dayofweek,
        "IS_WEEKEND": (ts.dt.dayofweek >= 5).astype(int),
        "IS_HOLIDAY": ts.dt.date.isin(HOLIDAYS).astype(int),
    }, index=ts.index)


@st.cache_data(max_entries=128, show_spinner="Predicting…")
def predict_day(date_iso: str) -> pd.DataFrame:
    """Predicted pickups for every station and every hour of one date (cached per date)."""
    hrs = pd.DataFrame({"HOUR_TS": pd.date_range(pd.Timestamp(date_iso), periods=24, freq="h")})
    grid = stations[["STATION_ID", "LAT", "LNG"]].merge(hrs, how="cross")
    grid = grid.join(time_features(grid["HOUR_TS"]))
    grid["PRED"] = np.clip(plan_model.predict(grid[meta["plan_features"]]), 0, None)
    return grid[["STATION_ID", "HOUR", "PRED"]]


def typical_profile(sid, dow):
    """Actual average pickups per hour on that weekday in January 2024."""
    s = sdh if sid == ALL else sdh[sdh["STATION_ID"] == sid]
    prof = s[s["DOW"] == dow].groupby("HOUR")["TRIPS"].sum().reindex(range(24), fill_value=0)
    return prof / JAN_DOW_COUNT[dow]


def fmt_hour(h):
    return f"{h:02d}:00"


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("🚲 NYC Citi Bike Demand Dashboard")
st.caption(f"Bike pickups per hour at Citi Bike stations, learned from "
           f"{meta['n_trips']:,} trips at {meta['n_stations']:,} stations ({meta['data']}).")

tab_pred, tab_explore, tab_next, tab_batch, tab_model, tab_help = st.tabs(
    ["🔮 Predict demand", "📊 Explore patterns", "⏱️ Next-hour forecast (5B model)",
     "📁 Batch prediction", "🧠 About the models", "❓ How to use"])

# ----------------------------------------------------------------------------
# TAB 1: Predict demand for a date, hour and location
# ----------------------------------------------------------------------------
with tab_pred:
    st.subheader("Choose a date, time and location")
    loc_options = [ALL] + stations.sort_values("NAME")["STATION_ID"].tolist()
    c1, c2, c3 = st.columns([1, 1, 2])
    date = c1.date_input("Date", value=dt.date(2024, 1, 17), format="YYYY-MM-DD")
    hour = c2.selectbox("Hour", list(range(24)), index=8, format_func=fmt_hour)
    loc = c3.selectbox("Location (station)", loc_options, format_func=loc_label,
                       help="Type to search, e.g. 'Broadway' or '8 Ave'.")
    st.caption("Weather is not in the Citi Bike trip data, so predictions reflect typical "
               "January conditions for that weekday, hour and station.")

    day_pred = predict_day(date.isoformat())
    dow = date.weekday()
    if loc == ALL:
        prof = day_pred.groupby("HOUR")["PRED"].sum()
    else:
        prof = day_pred[day_pred["STATION_ID"] == loc].set_index("HOUR")["PRED"]
    typical = typical_profile(loc, dow)
    p_hour = float(prof.loc[hour])

    notes = []
    if date.month != 1:
        notes.append(f"The models learned from **January 2024** (winter). For "
                     f"{calendar.month_name[date.month]}, they show winter-level demand; "
                     "warmer months usually have many more trips.")
    if date in HOLIDAYS:
        notes.append("This date is a U.S. federal holiday: commuting peaks are usually smaller.")
    for n in notes:
        st.info(n)

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"Predicted pickups at {fmt_hour(hour)}", f"{p_hour:,.0f}" if p_hour >= 10 else f"{p_hour:.1f}",
              delta=f"{p_hour - typical.loc[hour]:+,.1f} vs typical January {DAYS[dow]}",
              delta_color="off")
    m2.metric("Predicted pickups, whole day", f"{prof.sum():,.0f}")
    m3.metric("Busiest hour that day", fmt_hour(int(prof.idxmax())))
    m4.metric("Quietest daytime hour (6–22)", fmt_hour(int(prof.loc[6:22].idxmin())))
    st.write(f"**{loc_label(loc)}**, {DAYS[dow]} {date:%d %b %Y}, {fmt_hour(hour)}–{fmt_hour((hour + 1) % 24)}.")

    # 24-hour profile: prediction vs actual January average
    prof_df = pd.DataFrame({
        "Hour": list(range(24)) * 2,
        "Pickups": list(prof.values) + list(typical.values),
        "Series": ["Predicted for this date"] * 24 + [f"Actual average, January {DAYS[dow]}s"] * 24,
    })
    series_dom = ["Predicted for this date", f"Actual average, January {DAYS[dow]}s"]
    color = alt.Color("Series:N", title=None, legend=alt.Legend(orient="top", labelLimit=320),
                      scale=alt.Scale(domain=series_dom, range=[BLUE, GRAY]))
    lines = alt.Chart(prof_df).mark_line(strokeWidth=2).encode(
        x=alt.X("Hour:Q", scale=alt.Scale(domain=[0, 23]), axis=alt.Axis(tickMinStep=1)),
        y=alt.Y("Pickups:Q", title="Pickups per hour"),
        color=color,
        strokeDash=alt.StrokeDash("Series:N", legend=None,
                                  scale=alt.Scale(domain=series_dom, range=[[1, 0], [4, 4]])),
        tooltip=["Series", "Hour", alt.Tooltip("Pickups:Q", format=",.1f")])
    sel = alt.Chart(pd.DataFrame({"Hour": [hour], "Pickups": [p_hour]})).mark_circle(
        size=160, color=ORANGE).encode(x="Hour:Q", y="Pickups:Q",
                                       tooltip=[alt.Tooltip("Pickups:Q", format=",.1f")])
    st.markdown("**Predicted demand across the day** (orange = your hour)")
    st.altair_chart((lines + sel).properties(height=300), width="stretch")

    # Map of every station at the chosen hour
    st.markdown(f"**Predicted pickups at every station, {fmt_hour(hour)}** "
                "(bigger, darker circle = more demand; hover for details)")
    hour_df = day_pred[day_pred["HOUR"] == hour].merge(stations[["STATION_ID", "NAME", "LAT", "LNG"]],
                                                       on="STATION_ID")
    top = max(hour_df["PRED"].quantile(0.99), 1e-6)
    level = np.clip(hour_df["PRED"] / top, 0, 1)
    rgb = np.array([[205, 226, 251], [134, 182, 239], [42, 120, 214], [24, 79, 149], [13, 54, 107]])
    idx = np.minimum((level * 4.999).astype(int), 4)
    hour_df["COLOR"] = [[235, 104, 52, 255] if sid == loc else list(map(int, rgb[i])) + [200]
                        for sid, i in zip(hour_df["STATION_ID"], idx)]
    hour_df["RADIUS"] = 25 + 140 * np.sqrt(level)
    hour_df.loc[hour_df["STATION_ID"] == loc, "RADIUS"] = 200
    hour_df["LABEL"] = hour_df["PRED"].map(lambda v: f"{v:.1f}")
    if loc != ALL:
        center = stations.loc[stations["STATION_ID"] == loc, ["LAT", "LNG"]].iloc[0]
        view = pdk.ViewState(latitude=center["LAT"], longitude=center["LNG"], zoom=13.5)
    else:
        view = pdk.ViewState(latitude=40.735, longitude=-73.97, zoom=11.2)
    st.pydeck_chart(pdk.Deck(
        layers=[pdk.Layer("ScatterplotLayer", data=hour_df,
                          get_position="[LNG, LAT]", get_fill_color="COLOR",
                          get_radius="RADIUS", pickable=True,
                          radius_min_pixels=2, radius_max_pixels=18)],
        initial_view_state=view, map_style="light",
        tooltip={"html": "<b>{NAME}</b><br/>Predicted pickups: {LABEL}"}), height=460)

    with st.expander(f"Top 15 stations at {fmt_hour(hour)}"):
        t15 = hour_df.nlargest(15, "PRED")[["NAME", "PRED"]].rename(
            columns={"NAME": "Station", "PRED": "Predicted pickups"})
        st.dataframe(t15.style.format({"Predicted pickups": "{:.1f}"}), hide_index=True,
                     width="stretch")

# ----------------------------------------------------------------------------
# TAB 2: Explore historical patterns
# ----------------------------------------------------------------------------
with tab_explore:
    st.subheader("Explore actual demand, January 2024")
    f1, f2, f3, f4 = st.columns([2, 1, 1, 1])
    e_loc = f1.selectbox("Location", loc_options, format_func=loc_label, key="e_loc")
    citywide = e_loc == ALL
    rider = f2.selectbox("Rider type", ["All", "member", "casual"], disabled=not citywide)
    bike = f3.selectbox("Bike type", ["All", "classic_bike", "electric_bike"], disabled=not citywide,
                        format_func=lambda b: b.replace("_", " ").capitalize())
    daytype = f4.selectbox("Days", ["All days", "Weekdays", "Weekends"])
    if not citywide:
        st.caption("Rider and bike type filters apply to the citywide view.")

    if citywide:
        d = detail.copy()
        if rider != "All":
            d = d[d["MEMBER"] == rider]
        if bike != "All":
            d = d[d["BIKE"] == bike]
        hourly = d.groupby("HOUR_TS")["TRIPS"].sum().reindex(
            pd.date_range("2024-01-01", periods=744, freq="h"), fill_value=0)
        hourly.index.name = "HOUR_TS"
        hdf = hourly.reset_index()
        hdf["DATE"] = hdf["HOUR_TS"].dt.normalize()
        hdf["DOW"] = hdf["HOUR_TS"].dt.dayofweek
        hdf["HOUR"] = hdf["HOUR_TS"].dt.hour
        daily = hdf.groupby("DATE")["TRIPS"].sum().reset_index()
        dow_hour = hdf.groupby(["DOW", "HOUR"])["TRIPS"].sum().reset_index()
    else:
        daily = sdaily[sdaily["STATION_ID"] == e_loc].groupby("DATE")["TRIPS"].sum().reindex(
            pd.date_range("2024-01-01", "2024-01-31"), fill_value=0).rename_axis("DATE").reset_index()
        dow_hour = sdh[sdh["STATION_ID"] == e_loc][["DOW", "HOUR", "TRIPS"]]
        full = pd.MultiIndex.from_product([range(7), range(24)], names=["DOW", "HOUR"])
        dow_hour = dow_hour.set_index(["DOW", "HOUR"])["TRIPS"].reindex(full, fill_value=0).reset_index()

    wkend = {"Weekdays": [0, 1, 2, 3, 4], "Weekends": [5, 6]}.get(daytype, list(range(7)))
    daily = daily[daily["DATE"].dt.dayofweek.isin(wkend)]
    dow_hour = dow_hour[dow_hour["DOW"].isin(wkend)].copy()
    dow_hour["AVG"] = dow_hour["TRIPS"] / dow_hour["DOW"].map(JAN_DOW_COUNT)
    dow_hour["Day"] = dow_hour["DOW"].map(lambda i: DAYS[i])

    total = int(daily["TRIPS"].sum())
    by_hour = dow_hour.groupby("HOUR")["AVG"].mean()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total trips (selection)", f"{total:,}")
    k2.metric("Average trips per day", f"{daily['TRIPS'].mean():,.0f}" if len(daily) else "–")
    k3.metric("Peak hour", fmt_hour(int(by_hour.idxmax())) if by_hour.sum() else "–")
    if citywide:
        mshare = detail.loc[detail["MEMBER"] == "member", "TRIPS"].sum() / detail["TRIPS"].sum()
        k4.metric("Trips by members (all January)", f"{mshare:.0%}")
    else:
        rank = int((stations["TRIPS"] > stations.loc[stations["STATION_ID"] == e_loc, "TRIPS"].iloc[0]).sum()) + 1
        k4.metric("Station rank by trips", f"#{rank} of {len(stations):,}")

    a1, a2 = st.columns(2)
    a1.markdown("**Trips per day across January**")
    a1.altair_chart(
        alt.Chart(daily).mark_line(color=BLUE, strokeWidth=2,
                                   point=alt.OverlayMarkDef(color=BLUE, size=40))
        .encode(x=alt.X("DATE:T", title=None, axis=alt.Axis(format="%d %b")),
                y=alt.Y("TRIPS:Q", title="Trips"),
                tooltip=[alt.Tooltip("DATE:T", format="%a %d %b"), alt.Tooltip("TRIPS:Q", format=",")])
        .properties(height=280), width="stretch")

    wk = dow_hour.assign(Type=np.where(dow_hour["DOW"] >= 5, "Weekend", "Weekday"))
    wk = wk.groupby(["Type", "HOUR"])["AVG"].mean().reset_index()
    a2.markdown("**Average trips by hour: weekday vs weekend**")
    a2.altair_chart(
        alt.Chart(wk).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=30))
        .encode(x=alt.X("HOUR:Q", title="Hour", scale=alt.Scale(domain=[0, 23])),
                y=alt.Y("AVG:Q", title="Average trips per hour"),
                color=alt.Color("Type:N", title=None, legend=alt.Legend(orient="top"),
                                scale=alt.Scale(domain=["Weekday", "Weekend"], range=[BLUE, ORANGE])),
                tooltip=["Type", "HOUR", alt.Tooltip("AVG:Q", format=",.1f", title="Avg trips")])
        .properties(height=280), width="stretch")

    st.markdown("**When is demand highest? Average trips by day and hour** (darker = busier)")
    st.altair_chart(
        alt.Chart(dow_hour).mark_rect(stroke="white", strokeWidth=1)
        .encode(x=alt.X("HOUR:O", title="Hour"),
                y=alt.Y("Day:N", sort=DAYS, title=None),
                color=alt.Color("AVG:Q", title="Avg trips", scale=alt.Scale(range=SEQ)),
                tooltip=["Day", alt.Tooltip("HOUR:O", title="Hour"),
                         alt.Tooltip("AVG:Q", format=",.1f", title="Avg trips")])
        .properties(height=230), width="stretch")

    if citywide:
        b1, b2 = st.columns(2)
        top_st = stations.nlargest(15, "TRIPS")
        b1.markdown("**15 busiest stations (all January)**")
        b1.altair_chart(
            alt.Chart(top_st).mark_bar(color=BLUE, cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
            .encode(y=alt.Y("NAME:N", sort="-x", title=None), x=alt.X("TRIPS:Q", title="Trips"),
                    tooltip=["NAME", alt.Tooltip("TRIPS:Q", format=",")])
            .properties(height=420), width="stretch")
        mix = detail.groupby(["MEMBER", "BIKE"])["TRIPS"].sum().reset_index()
        mix["BIKE"] = mix["BIKE"].str.replace("_", " ").str.capitalize()
        b2.markdown("**Trips by rider type and bike type**")
        b2.altair_chart(
            alt.Chart(mix).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(x=alt.X("MEMBER:N", title=None, axis=alt.Axis(labelAngle=0)),
                    xOffset="BIKE:N", y=alt.Y("TRIPS:Q", title="Trips"),
                    color=alt.Color("BIKE:N", title=None, legend=alt.Legend(orient="top"),
                                    scale=alt.Scale(range=[BLUE, ORANGE])),
                    tooltip=["MEMBER", "BIKE", alt.Tooltip("TRIPS:Q", format=",")])
            .properties(height=420), width="stretch")

# ----------------------------------------------------------------------------
# TAB 3: Next-hour forecast with the Assignment 5B model
# ----------------------------------------------------------------------------
with tab_next:
    st.subheader("Next-hour citywide forecast (our Assignment 5B Extra Trees model)")
    st.write("This model predicts **citywide** trips in the next hour from recent demand. "
             "Pick an hour in January 2024: the inputs are filled from the real data, and you "
             "can change them to see how the prediction responds.")
    valid = city[city["split"] != "no lags"]
    n1, n2 = st.columns(2)
    nd_date = n1.date_input("Date (8–31 Jan 2024)", value=dt.date(2024, 1, 29),
                            min_value=valid["hourly_timestamp"].min().date(),
                            max_value=dt.date(2024, 1, 31), key="nd_date")
    nd_hour = n2.selectbox("Hour", list(range(24)), index=8, format_func=fmt_hour, key="nd_hour")
    ts = pd.Timestamp(nd_date) + pd.Timedelta(hours=nd_hour)
    row = city[city["hourly_timestamp"] == ts].iloc[0]
    key = ts.strftime("%Y%m%d%H")

    st.markdown("**Model inputs** (edit any value for a what-if)")
    i1, i2, i3, i4 = st.columns(4)
    lag_1 = i1.number_input("Trips in the previous hour (lag_1)", 0, 50_000, int(row["lag_1"]), key=f"l1{key}")
    lag_24 = i2.number_input("Same hour yesterday (lag_24)", 0, 50_000, int(row["lag_24"]), key=f"l24{key}")
    lag_168 = i3.number_input("Same hour last week (lag_168)", 0, 50_000, int(row["lag_168"]), key=f"l168{key}")
    roll = i4.number_input("Average of last 24 hours", 0.0, 50_000.0, float(round(row["rolling_mean_24"], 1)),
                           key=f"r{key}")
    x = pd.DataFrame([{"hour": nd_hour, "day_of_week": ts.dayofweek, "is_weekend": int(ts.dayofweek >= 5),
                       "lag_1": lag_1, "lag_24": lag_24, "lag_168": lag_168, "rolling_mean_24": roll}])
    pred = float(next_model.predict(x[meta["next_features"]])[0])
    edited = (lag_1, lag_24, lag_168) != (int(row["lag_1"]), int(row["lag_24"]), int(row["lag_168"])) \
        or abs(roll - round(row["rolling_mean_24"], 1)) > 1e-6

    r1, r2, r3 = st.columns(3)
    r1.metric(f"Predicted trips, {fmt_hour(nd_hour)}–{fmt_hour((nd_hour + 1) % 24)}", f"{pred:,.0f}")
    r2.metric("Actual trips", f"{int(row['demand']):,}")
    r3.metric("Error", "what-if" if edited else f"{pred - row['demand']:+,.0f}")
    st.caption(("🟢 This hour is in the **test set**: the model never saw it during training."
                if row["split"] == "test" else
                "🟠 This hour was in the **training set**, so the prediction is optimistic. "
                f"Test hours start {pd.Timestamp(meta['next_test_start']):%d %b %H:%M}.")
               + (" Inputs edited: showing a what-if prediction." if edited else ""))

    day_rows = city[(city["hourly_timestamp"].dt.date == nd_date) & (city["split"] != "no lags")]
    day_rows = day_rows.assign(Predicted=next_model.predict(day_rows[meta["next_features"]]))
    long = day_rows.melt(id_vars=["hour"], value_vars=["demand", "Predicted"],
                         var_name="Series", value_name="Trips")
    long["Series"] = long["Series"].replace({"demand": "Actual"})
    st.markdown(f"**Actual vs predicted, {nd_date:%a %d %b %Y}**")
    st.altair_chart(
        alt.Chart(long).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=30))
        .encode(x=alt.X("hour:Q", title="Hour", scale=alt.Scale(domain=[0, 23])),
                y=alt.Y("Trips:Q", title="Trips per hour"),
                color=alt.Color("Series:N", title=None, legend=alt.Legend(orient="top"),
                                scale=alt.Scale(domain=["Actual", "Predicted"], range=[GRAY, BLUE])),
                tooltip=["Series", "hour", alt.Tooltip("Trips:Q", format=",.0f")])
        .properties(height=300), width="stretch")

# ----------------------------------------------------------------------------
# TAB 4: Batch prediction
# ----------------------------------------------------------------------------
def predict_batch(b: pd.DataFrame) -> pd.DataFrame:
    """Planning-model predictions for a table of DATE, HOUR, STATION_ID (or ALL)."""
    b = b.copy()
    b["STATION_ID"] = b["STATION_ID"].astype(str).str.strip()
    b.loc[b["STATION_ID"].str.upper() == ALL, "STATION_ID"] = ALL
    dates = pd.to_datetime(b["DATE"], errors="coerce")
    hrs = pd.to_numeric(b["HOUR"], errors="coerce")
    known = b["STATION_ID"].isin(set(stations["STATION_ID"]) | {ALL})
    ok = dates.notna() & hrs.between(0, 23) & (hrs % 1 == 0) & known
    b["PREDICTED_PICKUPS"] = np.nan
    b["STATUS"] = np.where(~known, "Unknown station", np.where(ok, "OK", "Invalid date/hour"))
    b["STATION_NAME"] = b["STATION_ID"].map(lambda s: loc_label(s) if s in NAME or s == ALL else "")
    for d in dates[ok].dt.date.unique():
        dp = predict_day(d.isoformat())
        city_h = dp.groupby("HOUR")["PRED"].sum()
        lookup = dp.set_index(["STATION_ID", "HOUR"])["PRED"]
        m = ok & (dates.dt.date == d)
        b.loc[m, "PREDICTED_PICKUPS"] = [
            city_h[int(h)] if s == ALL else lookup[(s, int(h))]
            for s, h in zip(b.loc[m, "STATION_ID"], hrs[m])]
    b["PREDICTED_PICKUPS"] = b["PREDICTED_PICKUPS"].round(2)
    return b


with tab_batch:
    st.subheader("Predict many date / hour / station combinations")
    st.write("Upload a CSV with columns **DATE** (`YYYY-MM-DD`), **HOUR** (0–23) and "
             "**STATION_ID** (from the station list below, or `ALL` for citywide). "
             "Up to 20,000 rows and 60 different dates.")
    busiest = stations.nlargest(3, "TRIPS")["STATION_ID"].tolist()
    template = pd.DataFrame({"DATE": ["2024-01-17", "2024-01-17", "2024-01-20", "2025-01-15"],
                             "HOUR": [8, 18, 14, 8],
                             "STATION_ID": [busiest[0], busiest[1], busiest[2], ALL]})
    cA, cB = st.columns(2)
    cA.download_button("⬇️ Example CSV", template.to_csv(index=False), "example_bike_requests.csv", "text/csv")
    cB.download_button("⬇️ Station list (IDs and names)",
                       stations[["STATION_ID", "NAME", "LAT", "LNG", "TRIPS"]].to_csv(index=False),
                       "citibike_stations.csv", "text/csv")
    up = st.file_uploader("Upload your CSV", type="csv")
    if up is not None:
        try:
            b = pd.read_csv(up, dtype=str)
            b.columns = b.columns.str.strip().str.upper()
            lacking = [c for c in ["DATE", "HOUR", "STATION_ID"] if c not in b.columns]
            n_dates = pd.to_datetime(b.get("DATE"), errors="coerce").dt.date.nunique() if "DATE" in b else 0
            if lacking:
                st.error(f"Missing columns: {', '.join(lacking)}")
            elif len(b) > 20_000 or n_dates > 60:
                st.error("Please keep it to 20,000 rows and 60 different dates.")
            else:
                res = predict_batch(b)
                st.success(f"Predicted {int((res['STATUS'] == 'OK').sum()):,} of {len(res):,} rows.")
                st.dataframe(res, hide_index=True, width="stretch")
                st.download_button("⬇️ Download predictions", res.to_csv(index=False),
                                   "bike_demand_predictions.csv", "text/csv")
        except Exception as ex:
            st.error(f"Could not read this file: {ex}")

# ----------------------------------------------------------------------------
# TAB 5: About the models
# ----------------------------------------------------------------------------
with tab_model:
    pm, nm = meta["plan_metrics"], meta["next_metrics"]
    st.subheader("1. Planning model (Predict demand tab)")
    st.markdown(f"""
- **Question:** how many bikes will be picked up at a station in a given hour?
- **Inputs:** station (target-encoded), station latitude/longitude, hour, day of week, weekend, U.S. holiday.
  No past demand is needed, so it works for any date.
- **Model:** histogram gradient boosting with a Poisson loss (suited to counts), scikit-learn.
- **Test:** trained on 1–24 January, tested on {meta['plan_test_period']} (unseen week), then refit on all of January.
""")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Citywide R² (test week)", f"{pm['citywide_r2']:.3f}")
    q2.metric("Citywide MAE", f"{pm['citywide_mae']:,.0f} trips/hour")
    q3.metric("Station-hour MAE", f"{pm['station_hour_mae']:.2f} trips")
    q4.metric("Station-hour RMSE", f"{pm['station_hour_rmse']:.2f} trips")

    st.subheader("2. Next-hour model (Assignment 5B)")
    st.markdown("Extra Trees on hour, day of week, weekend and recent demand "
                "(lag_1, lag_24, lag_168, 24-hour rolling mean), chronological 80:20 split, as in 5B.")
    w1, w2, w3 = st.columns(3)
    w1.metric("R² (test)", f"{nm['r2']:.3f}")
    w2.metric("MAE", f"{nm['mae']:,.1f} trips/hour")
    w3.metric("RMSE", f"{nm['rmse']:,.1f} trips/hour")
    imp = pd.DataFrame({"Feature": list(meta["next_importance"]),
                        "Importance": list(meta["next_importance"].values())})
    st.markdown("**Which inputs matter most to the next-hour model?**")
    st.altair_chart(
        alt.Chart(imp).mark_bar(color=BLUE, cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(y=alt.Y("Feature:N", sort="-x", title=None),
                x=alt.X("Importance:Q", axis=alt.Axis(format="%")),
                tooltip=["Feature", alt.Tooltip("Importance:Q", format=".1%")])
        .properties(height=230), width="stretch")

    st.markdown("**Assignment 5B comparison (default models, citywide hourly demand)**")
    prev = pd.DataFrame([{"Model": k, "MAE": v["mae"], "RMSE": v["rmse"], "R²": v["r2"]}
                         for k, v in meta["previous_5b"].items()])
    st.dataframe(prev.style.format({"MAE": "{:.1f}", "RMSE": "{:.1f}", "R²": "{:.3f}"}),
                 hide_index=True)

    st.markdown("""
**Limitations**
- **No weather:** temperature, rain and snow strongly affect cycling but are not in the trip data.
- **January only:** the models learned winter behaviour; other months are shown at winter level.
- Demand = trips *starting* at a station (pickups). Station capacity and bike availability are not modelled.
- Stations with fewer than 20 trips in January are excluded as too noisy.
""")

# ----------------------------------------------------------------------------
# TAB 6: Help
# ----------------------------------------------------------------------------
with tab_help:
    st.markdown("""
### Quick start
1. Open **🔮 Predict demand**.
2. Pick a **date**, an **hour** and a **location**: one station (type to search) or *All NYC*.
3. Read the predicted pickups for that hour and the whole day; compare with the typical January value.
4. The line chart shows the whole day; the map shows every station at your hour (hover a circle).

### Other tabs
- **📊 Explore patterns:** actual January 2024 demand by day, hour, weekday/weekend, rider and bike type.
- **⏱️ Next-hour forecast:** our Assignment 5B model. Pick an hour, then edit the inputs for a what-if.
- **📁 Batch prediction:** download the example CSV, fill in rows, upload, download predictions.
- **🧠 About the models:** accuracy, feature importance and limitations.

### Things to try
- *All NYC*, a Wednesday at 08:00 vs a Saturday at 08:00: commuting vs leisure.
- 15 January 2024 (Martin Luther King Jr. Day) vs 16 January at 08:00.
- In the next-hour tab, halve "Trips in the previous hour" and watch the forecast drop.
""")

st.caption("Chamak Saha · Data: Citi Bike System Data (citibikenyc.com/system-data) · "
           "For informational purposes only.")
