# main.py — F1 Strategy Simulator API
import os
import pickle
import warnings
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

import uvicorn

warnings.filterwarnings("ignore")

# =========================================================
# APP SETUP
# =========================================================
app = FastAPI(title="F1 Strategy Simulator", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# PYDANTIC MODELS
# =========================================================
class SimulationRequest(BaseModel):
    year: int
    race: str
    driver: str
    pit_laps: List[int]
    compounds: List[str]

class LapResult(BaseModel):
    lap: int
    actual: float
    simulated: float
    diff: float
    cumulative_diff: float
    compound: str
    event: str
    stint_lap: int

class StintSummary(BaseModel):
    stint: int
    compound: str
    start_lap: int
    end_lap: int
    laps: int
    avg_actual: float
    avg_sim: float
    avg_diff: float
    stint_actual: float
    stint_sim: float
    stint_diff: float

class PitSummary(BaseModel):
    lap: int
    actual: float
    simulated: float
    diff: float
    compound: str
    event: str

class SimulationResponse(BaseModel):
    actual_strategy: dict
    sim_strategy: dict
    laps: List[LapResult]
    stint_summary: List[StintSummary]
    pit_summary: List[PitSummary]
    actual_total: float
    sim_total: float
    gain_loss: float
    lap_mae: float
    lap_rmse: float
    mode: str

# =========================================================
# GLOBAL STATE — loaded once at startup
# =========================================================
class ModelState:
    df_raw: pd.DataFrame = None
    df: pd.DataFrame     = None

    model_normal = None
    model_pit    = None

    normal_features: list = []
    pit_features: list    = []

    driver_enc_map: dict  = {}
    team_enc_map: dict    = {}
    circuit_enc_map: dict = {}
    driver_team_map: dict = {}

    driver_circuit_mean: pd.DataFrame           = None
    driver_circuit_year_mean: pd.DataFrame      = None
    driver_circuit_compound_mean: pd.DataFrame  = None
    circuit_compound_year_mean: pd.DataFrame    = None
    circuit_compound_allyr_mean: pd.DataFrame   = None
    circuit_year_median: pd.DataFrame           = None

    sc_lap_times: pd.DataFrame      = None
    pit_under_sc_times: pd.DataFrame = None

    race_total_laps_df: pd.DataFrame = None
    race_medians_df: pd.DataFrame    = None

    global_median: float = 90.0

S = ModelState()

# =========================================================
# COMPOUND MAPS
# =========================================================
COMPOUND_MAP = {
    "SOFT": 0, "MEDIUM": 1, "HARD": 2,
    "INTERMEDIATE": 3, "WET": 4
}
INV_COMPOUND_MAP = {v: k for k, v in COMPOUND_MAP.items()}


def encode_compound(x):
    if pd.isna(x):
        return np.nan
    return COMPOUND_MAP.get(str(x).strip().upper(), np.nan)


def decode_compound(x):
    if pd.isna(x):
        return "UNKNOWN"
    try:
        return INV_COMPOUND_MAP.get(int(x), f"UNK_{x}")
    except Exception:
        return "UNKNOWN"


# =========================================================
# TRACK STATUS HELPERS
# =========================================================
def status_contains_flag(status_val, flag_char: str) -> bool:
    try:
        return flag_char in str(int(float(status_val)))
    except (ValueError, TypeError):
        return False


def is_sc_vsc_status(status_val) -> bool:
    return (
        status_contains_flag(status_val, '4') or
        status_contains_flag(status_val, '6')
    )


# =========================================================
# PREDICTION HELPERS
# =========================================================
def fuel_load_for_lap(lap_number: int, race_total_laps: int) -> float:
    return 1.0 - (lap_number / race_total_laps)


def tyre_life_ratio_for(tyre_life: float) -> float:
    return tyre_life / 20.0


def get_target_encodings(driver: str, circuit: str,
                         compound_enc: int, year: int):
    dc = S.driver_circuit_mean[
        (S.driver_circuit_mean["Driver"]  == driver) &
        (S.driver_circuit_mean["Circuit"] == circuit)
    ]
    dcm = float(dc["driver_circuit_mean"].iloc[0]) \
          if len(dc) > 0 else S.global_median

    dcy = S.driver_circuit_year_mean[
        (S.driver_circuit_year_mean["Driver"]  == driver) &
        (S.driver_circuit_year_mean["Circuit"] == circuit) &
        (S.driver_circuit_year_mean["Year"]    == year)
    ]
    dcym = float(dcy["driver_circuit_year_mean"].iloc[0]) \
           if len(dcy) > 0 else dcm

    dcc = S.driver_circuit_compound_mean[
        (S.driver_circuit_compound_mean["Driver"]   == driver) &
        (S.driver_circuit_compound_mean["Circuit"]  == circuit) &
        (S.driver_circuit_compound_mean["Compound"] == compound_enc)
    ]
    if len(dcc) > 0:
        dccm = float(dcc["driver_circuit_compound_mean"].iloc[0])
    else:
        ccy = S.circuit_compound_year_mean[
            (S.circuit_compound_year_mean["Circuit"]  == circuit) &
            (S.circuit_compound_year_mean["Compound"] == compound_enc) &
            (S.circuit_compound_year_mean["Year"]     == year)
        ]
        if len(ccy) > 0:
            dccm = float(ccy["circuit_compound_year_mean"].iloc[0])
        else:
            cc_all = S.circuit_compound_allyr_mean[
                (S.circuit_compound_allyr_mean["Circuit"]  == circuit) &
                (S.circuit_compound_allyr_mean["Compound"] == compound_enc)
            ]
            dccm = float(cc_all["circuit_compound_allyr_mean"].iloc[0]) \
                   if len(cc_all) > 0 else dcm

    ccy = S.circuit_compound_year_mean[
        (S.circuit_compound_year_mean["Circuit"]  == circuit) &
        (S.circuit_compound_year_mean["Compound"] == compound_enc) &
        (S.circuit_compound_year_mean["Year"]     == year)
    ]
    if len(ccy) > 0:
        ccym = float(ccy["circuit_compound_year_mean"].iloc[0])
    else:
        cc_all = S.circuit_compound_allyr_mean[
            (S.circuit_compound_allyr_mean["Circuit"]  == circuit) &
            (S.circuit_compound_allyr_mean["Compound"] == compound_enc)
        ]
        ccym = float(cc_all["circuit_compound_allyr_mean"].iloc[0]) \
               if len(cc_all) > 0 else dcm

    return dcm, dcym, dccm, ccym


def get_circuit_year_median(circuit: str, year: int) -> float:
    row = S.circuit_year_median[
        (S.circuit_year_median["Circuit"] == circuit) &
        (S.circuit_year_median["Year"]    == year)
    ]
    return float(row["circuit_year_median"].iloc[0]) \
           if len(row) > 0 else S.global_median


def predict_lap_time(
        lap_number: int, tyre_life: float, compound_enc: int,
        race_total_laps: int, driver_enc: int, team_enc: int,
        circuit_enc: int, year: int, stint_lap_number: int,
        driver: str, circuit: str, is_lap1: int = 0) -> float:

    fuel_load  = fuel_load_for_lap(lap_number, race_total_laps)
    tyre_ratio = tyre_life_ratio_for(tyre_life)
    is_opening = int(lap_number <= 3 and stint_lap_number <= 3)
    dcm, dcym, dccm, ccym = get_target_encodings(
        driver, circuit, compound_enc, year
    )

    x = np.array([[
        lap_number, is_lap1, int(lap_number == 2),
        is_opening, stint_lap_number,
        tyre_life, tyre_ratio, fuel_load,
        compound_enc,
        driver_enc, team_enc, circuit_enc, year,
        dcm, dcym, dccm, ccym,
    ]])

    return float(S.model_normal.predict(x)[0])


def predict_pit_lap_time(
        tyre_life_old: float, compound_new_enc: int,
        lap_number: int, race_total_laps: int,
        driver_enc: int, team_enc: int, circuit_enc: int,
        year: int, driver: str, circuit: str) -> float:

    tyre_delta = 1.0 - tyre_life_old
    fuel_load  = fuel_load_for_lap(lap_number, race_total_laps)
    dcm, dcym, dccm, ccym = get_target_encodings(
        driver, circuit, compound_new_enc, year
    )

    x_pit = np.array([[
        tyre_life_old,
        tyre_life_ratio_for(tyre_life_old),
        tyre_delta, tyre_life_old,
        int(tyre_delta < -5),
        lap_number, fuel_load,
        int(lap_number == 2),
        driver_enc, team_enc, circuit_enc, year,
        dcm, dcym,
    ]])

    cym      = get_circuit_year_median(circuit, year)
    pit_loss = float(S.model_pit.predict(x_pit)[0])
    return cym + pit_loss


def get_sc_lap_time(year: int, race: str, lap_number: int):
    row = S.sc_lap_times[
        (S.sc_lap_times["Year"]      == year) &
        (S.sc_lap_times["Race"]      == race) &
        (S.sc_lap_times["LapNumber"] == lap_number)
    ]
    if len(row) == 0:
        return None, False
    return float(row["sc_lap_median"].iloc[0]), True


def get_pit_under_sc_time(year: int, race: str, lap_number: int):
    row = S.pit_under_sc_times[
        (S.pit_under_sc_times["Year"]      == year) &
        (S.pit_under_sc_times["Race"]      == race) &
        (S.pit_under_sc_times["LapNumber"] == lap_number)
    ]
    if len(row) == 0:
        return None, False
    return float(row["pit_sc_lap_median"].iloc[0]), True


def get_circuit_for_race(year: int, race: str) -> str:
    row = S.df_raw[
        (S.df_raw["Year"] == year) &
        (S.df_raw["Race"] == race)
    ]
    return str(row["Circuit"].iloc[0]) if len(row) > 0 else ""


def get_race_total_laps(year: int, race: str) -> int:
    row = S.race_total_laps_df[
        (S.race_total_laps_df["Year"] == year) &
        (S.race_total_laps_df["Race"] == race)
    ]
    if len(row) == 0:
        raise ValueError(f"No total laps for {year} {race}")
    return int(row["race_total_laps"].iloc[0])


# =========================================================
# STRATEGY HELPERS
# =========================================================
def build_lap_compound_map(pit_laps: List[int],
                            compounds_enc: List[int],
                            final_lap: int):
    lap_to_compound = {}
    pit_set = set(pit_laps)
    start   = 1
    bounds  = list(pit_laps) + [final_lap + 1]
    for i, bound in enumerate(bounds):
        for lap in range(start, bound):
            lap_to_compound[lap] = compounds_enc[i]
        start = bound
    return lap_to_compound, pit_set


def extract_actual_strategy(year: int, race: str, driver: str):
    drv = S.df_raw[
        (S.df_raw["Year"]   == year)  &
        (S.df_raw["Race"]   == race)  &
        (S.df_raw["Driver"] == driver)
    ].sort_values("LapNumber").copy()

    if len(drv) == 0:
        return [], []

    drv["Stint"]    = drv["Stint"].fillna(0).astype(int)
    drv["Compound"] = drv["Compound"].fillna("UNKNOWN")

    pit_laps  = []
    compounds = [str(drv.iloc[0]["Compound"]).upper()]
    prev_s    = drv.iloc[0]["Stint"]

    for i in range(1, len(drv)):
        curr_s = drv.iloc[i]["Stint"]
        if curr_s != prev_s:
            pit_laps.append(int(drv.iloc[i]["LapNumber"]))
            compounds.append(str(drv.iloc[i]["Compound"]).upper())
            prev_s = curr_s

    return pit_laps, compounds


# =========================================================
# CORE SIMULATION
# =========================================================
def run_simulation(year: int, race: str, driver: str,
                   sim_pit_laps: List[int],
                   sim_compounds: List[str]) -> dict:

    actual_driver = S.df_raw[
        (S.df_raw["Year"]   == year)  &
        (S.df_raw["Race"]   == race)  &
        (S.df_raw["Driver"] == driver)
    ].sort_values("LapNumber").copy()

    if len(actual_driver) == 0:
        raise ValueError(f"No data for {driver} {year} {race}")

    driver_enc  = int(S.driver_enc_map.get(driver, 0))
    team        = S.driver_team_map.get(driver, "")
    team_enc    = int(S.team_enc_map.get(team, 0))
    circuit     = get_circuit_for_race(year, race)
    circuit_enc = int(S.circuit_enc_map.get(circuit, 0))

    race_total_laps = get_race_total_laps(year, race)
    actual_last_lap = int(actual_driver["LapNumber"].max())

    actual_pit_laps, actual_compounds = extract_actual_strategy(
        year, race, driver
    )

    is_validation = (
        sim_pit_laps  == actual_pit_laps and
        [c.upper() for c in sim_compounds] == [c.upper() for c in actual_compounds]
    )
    mode = ("VALIDATION MODE (actual strategy replay)"
            if is_validation else "USER STRATEGY MODE")

    compounds_enc = [
        COMPOUND_MAP.get(str(c).strip().upper(), 1)
        for c in sim_compounds
    ]

    lap_to_compound, pit_set = build_lap_compound_map(
        sim_pit_laps, compounds_enc, actual_last_lap
    )

    race_sc_laps = set(
        S.sc_lap_times[
            (S.sc_lap_times["Year"] == year) &
            (S.sc_lap_times["Race"] == race)
        ]["LapNumber"].tolist()
    )

    actual_tyre_by_lap = (
        actual_driver.set_index("LapNumber")["TyreLife"].to_dict()
    )
    actual_available = set(actual_driver["LapNumber"].tolist())

    results       = []
    tyre_life     = float(actual_tyre_by_lap.get(1, 1.0))
    stint_lap_num = 1
    current_cmp   = lap_to_compound.get(1, compounds_enc[0])

    for lap in range(1, actual_last_lap + 1):
        if lap not in actual_available:
            continue

        event   = "normal"
        is_lap1 = int(lap == 1)
        is_sc   = lap in race_sc_laps
        is_pit  = lap in pit_set

        if is_pit and is_sc:
            tyre_before = tyre_life
            new_cmp     = lap_to_compound[lap]
            t, found    = get_pit_under_sc_time(year, race, lap)
            sim_lap_time = t if found else predict_pit_lap_time(
                tyre_before, new_cmp, lap, race_total_laps,
                driver_enc, team_enc, circuit_enc, year,
                driver, circuit
            )
            current_cmp   = new_cmp
            tyre_life     = 1.0
            stint_lap_num = 1
            event         = "pit+SC"

        elif is_sc and not is_pit:
            t, found = get_sc_lap_time(year, race, lap)
            sim_lap_time = t if found else predict_lap_time(
                lap, tyre_life, current_cmp, race_total_laps,
                driver_enc, team_enc, circuit_enc, year,
                stint_lap_num, driver, circuit, is_lap1
            )
            if lap > 1:
                tyre_life     += 1.0
                stint_lap_num += 1
            event = "SC/VSC"

        elif is_pit:
            tyre_before = tyre_life
            new_cmp     = lap_to_compound[lap]
            sim_lap_time = predict_pit_lap_time(
                tyre_before, new_cmp, lap, race_total_laps,
                driver_enc, team_enc, circuit_enc, year,
                driver, circuit
            )
            current_cmp   = new_cmp
            tyre_life     = 1.0
            stint_lap_num = 1
            event         = "pit"

        else:
            sim_lap_time = predict_lap_time(
                lap, tyre_life, current_cmp, race_total_laps,
                driver_enc, team_enc, circuit_enc, year,
                stint_lap_num, driver, circuit, is_lap1
            )
            if lap > 1:
                tyre_life     += 1.0
                stint_lap_num += 1
            event = "normal"

        results.append({
            "lap"       : lap,
            "sim"       : sim_lap_time,
            "compound"  : decode_compound(current_cmp),
            "tyre_life" : tyre_life,
            "event"     : event,
            "stint_lap" : stint_lap_num,
        })

    sim_df = pd.DataFrame(results)

    actual_sel = actual_driver[["LapNumber","LapTimeSeconds"]].copy()
    actual_sel.columns = ["lap","actual"]

    comp = actual_sel.merge(
        sim_df.rename(columns={"sim":"simulated"}),
        on="lap", how="inner"
    )
    comp["diff"]            = comp["simulated"] - comp["actual"]
    comp["cumulative_diff"] = comp["diff"].cumsum()

    actual_total = float(comp["actual"].sum())
    sim_total    = float(comp["simulated"].sum())
    gain_loss    = sim_total - actual_total
    lap_mae      = float(mean_absolute_error(comp["actual"], comp["simulated"]))
    lap_rmse     = float(np.sqrt(mean_squared_error(comp["actual"], comp["simulated"])))

    # Pit summary
    pit_rows = comp[comp["event"].str.contains("pit", case=False, na=False)]
    pit_summary = [
        {
            "lap"       : int(row["lap"]),
            "actual"    : float(row["actual"]),
            "simulated" : float(row["simulated"]),
            "diff"      : float(row["diff"]),
            "compound"  : str(row["compound"]),
            "event"     : str(row["event"]),
        }
        for _, row in pit_rows.iterrows()
    ]

    # Stint summary
    stint_bounds = [1] + list(sim_pit_laps) + [actual_last_lap + 1]
    stint_summary = []
    for i in range(len(stint_bounds) - 1):
        s_start = stint_bounds[i]
        s_end   = stint_bounds[i + 1] - 1
        sub     = comp[
            (comp["lap"] >= s_start) &
            (comp["lap"] <= s_end)
        ]
        if len(sub) == 0:
            continue
        stint_summary.append({
            "stint"        : i + 1,
            "compound"     : decode_compound(compounds_enc[i]),
            "start_lap"    : s_start,
            "end_lap"      : s_end,
            "laps"         : len(sub),
            "avg_actual"   : float(sub["actual"].mean()),
            "avg_sim"      : float(sub["simulated"].mean()),
            "avg_diff"     : float(sub["diff"].mean()),
            "stint_actual" : float(sub["actual"].sum()),
            "stint_sim"    : float(sub["simulated"].sum()),
            "stint_diff"   : float(sub["simulated"].sum() - sub["actual"].sum()),
        })

    laps_out = [
        {
            "lap"            : int(row["lap"]),
            "actual"         : float(row["actual"]),
            "simulated"      : float(row["simulated"]),
            "diff"           : float(row["diff"]),
            "cumulative_diff": float(row["cumulative_diff"]),
            "compound"       : str(row["compound"]),
            "event"          : str(row["event"]),
            "stint_lap"      : int(row["stint_lap"]),
        }
        for _, row in comp.iterrows()
    ]

    return {
        "actual_strategy" : {"pit_laps": actual_pit_laps, "compounds": actual_compounds},
        "sim_strategy"    : {"pit_laps": sim_pit_laps, "compounds": sim_compounds},
        "laps"            : laps_out,
        "stint_summary"   : stint_summary,
        "pit_summary"     : pit_summary,
        "actual_total"    : actual_total,
        "sim_total"       : sim_total,
        "gain_loss"       : gain_loss,
        "lap_mae"         : lap_mae,
        "lap_rmse"        : lap_rmse,
        "mode"            : mode,
    }


# =========================================================
# STARTUP — LOAD DATA + TRAIN MODELS
# =========================================================
@app.on_event("startup")
async def startup():
    print("Loading data and training models...")

    # 1. Load raw data
    df_raw = pd.read_csv("Data/f1_2022_2025_combined.csv")
    df_raw = df_raw[df_raw["LapTimeSeconds"].notna()]
    df_raw = df_raw[df_raw["LapTimeSeconds"] > 0]
    df_raw["LapNumber"] = df_raw["LapNumber"].astype(int)
    df_raw["Year"]      = df_raw["Year"].astype(int)
    df_raw["Stint"]     = df_raw["Stint"].fillna(0).astype(int)
    df_raw = df_raw.sort_values(["Year","Race","Driver","LapNumber"])
    df_raw = df_raw.groupby(
        ["Year","Race","Driver","LapNumber"]
    ).last().reset_index()
    S.df_raw = df_raw

    print(f"  Rows loaded: {len(df_raw)}")

    # 2. Categorical encodings
    S.driver_enc_map  = {d: i for i, d in
                         enumerate(sorted(df_raw["Driver"].dropna().unique()))}
    S.team_enc_map    = {t: i for i, t in
                         enumerate(sorted(df_raw["Team"].dropna().unique()))}
    S.circuit_enc_map = {c: i for i, c in
                         enumerate(sorted(df_raw["Circuit"].dropna().unique()))}
    S.driver_team_map = (
        df_raw.sort_values("Year")
              .groupby("Driver")["Team"]
              .last().to_dict()
    )

    # 3. Feature engineering
    df = df_raw.copy()
    df["Compound"]       = df["Compound"].map(COMPOUND_MAP)
    df["TrackStatus"]    = df["TrackStatus"].fillna(1)
    df["fuel_load_norm"] = df["fuel_load"]
    df["Driver_enc"]     = df["Driver"].map(S.driver_enc_map).fillna(0).astype(int)
    df["Team_enc"]       = df["Team"].map(S.team_enc_map).fillna(0).astype(int)
    df["Circuit_enc"]    = df["Circuit"].map(S.circuit_enc_map).fillna(0).astype(int)

    grp = ["Year","Race","Driver"]
    df  = df.sort_values(grp + ["LapNumber"])

    df["is_lap1"]        = (df["LapNumber"] == 1).astype(int)
    df["is_lap2"]        = (df["LapNumber"] == 2).astype(int)
    df["is_opening_lap"] = (
        (df["LapNumber"] <= 3) & (df["Stint"] == 1)
    ).astype(int)
    df["tyre_life_ratio"]  = df["TyreLife"] / 20.0
    df["stint_lap_number"] = df.groupby(
        ["Year","Race","Driver","Stint"]
    )["LapNumber"].transform(lambda x: x - x.min() + 1)

    df["prev_stint"]   = df.groupby(grp)["Stint"].shift(1).fillna(0).astype(int)
    df["is_pit_lap"]   = (df["Stint"] != df["prev_stint"]).astype(int)
    df["prev_tyre"]    = df.groupby(grp)["TyreLife"].shift(1)
    df["tyre_delta"]   = df["TyreLife"] - df["prev_tyre"]
    df["full_pit"]     = (df["tyre_delta"] < -5).astype(int)
    df["laps_on_tyre"] = df["TyreLife"]

    df["is_sc_vsc"] = df["TrackStatus"].apply(
        lambda x: int(is_sc_vsc_status(x))
    ).astype(int)
    df["prev_is_sc_vsc"] = (
        df.groupby(grp)["is_sc_vsc"].shift(1).fillna(0).astype(int)
    )
    df["sc_entry_lap"]   = ((df["is_sc_vsc"]==1)&(df["prev_is_sc_vsc"]==0)).astype(int)
    df["sc_running_lap"] = ((df["is_sc_vsc"]==1)&(df["prev_is_sc_vsc"]==1)).astype(int)
    df["sc_exit_lap"]    = ((df["is_sc_vsc"]==0)&(df["prev_is_sc_vsc"]==1)).astype(int)
    df["pit_under_sc"]   = ((df["is_pit_lap"]==1)&(df["is_sc_vsc"]==1)).astype(int)

    df["event_type"] = 0
    df.loc[df["is_sc_vsc"]    == 1, "event_type"] = 2
    df.loc[df["is_pit_lap"]   == 1, "event_type"] = 1
    df.loc[df["pit_under_sc"] == 1, "event_type"] = 3

    S.df = df

    # 4. SC/VSC overlay
    df_sc_source = df[(df["is_sc_vsc"]==1)&(df["is_pit_lap"]==0)].copy()
    S.sc_lap_times = (
        df_sc_source
        .groupby(["Year","Race","LapNumber"])
        .agg(sc_lap_median=("LapTimeSeconds","median"),
             sc_lap_mean=("LapTimeSeconds","mean"),
             sc_n_cars=("LapTimeSeconds","count"))
        .reset_index()
    )

    S.pit_under_sc_times = (
        df[df["pit_under_sc"]==1]
        .groupby(["Year","Race","LapNumber"])
        .agg(pit_sc_lap_median=("LapTimeSeconds","median"),
             pit_sc_n_cars=("LapTimeSeconds","count"))
        .reset_index()
    )
    print(f"  SC overlay rows : {len(S.sc_lap_times)}")

    # 5. Target encodings
    df_clean = df[
        (df["is_sc_vsc"]     == 0) &
        (df["is_pit_lap"]    == 0) &
        (df["TyreLife"].between(5, 15)) &
        (df["stint_lap_number"].between(5, 15))
    ].copy()

    df_clean_broad = df[
        (df["is_sc_vsc"]  == 0) &
        (df["is_pit_lap"] == 0) &
        (df["TyreLife"].between(3, 12))
    ].copy()

    S.driver_circuit_mean = (
        df_clean_broad.groupby(["Driver","Circuit"])["LapTimeSeconds"]
        .mean().reset_index()
        .rename(columns={"LapTimeSeconds":"driver_circuit_mean"})
    )
    S.driver_circuit_year_mean = (
        df_clean_broad.groupby(["Driver","Circuit","Year"])["LapTimeSeconds"]
        .mean().reset_index()
        .rename(columns={"LapTimeSeconds":"driver_circuit_year_mean"})
    )
    S.driver_circuit_compound_mean = (
        df_clean.groupby(["Driver","Circuit","Compound"])["LapTimeSeconds"]
        .median().reset_index()
        .rename(columns={"LapTimeSeconds":"driver_circuit_compound_mean"})
    )
    S.circuit_compound_year_mean = (
        df_clean.groupby(["Circuit","Compound","Year"])["LapTimeSeconds"]
        .median().reset_index()
        .rename(columns={"LapTimeSeconds":"circuit_compound_year_mean"})
    )
    S.circuit_compound_allyr_mean = (
        df_clean.groupby(["Circuit","Compound"])["LapTimeSeconds"]
        .median().reset_index()
        .rename(columns={"LapTimeSeconds":"circuit_compound_allyr_mean"})
    )
    S.circuit_year_median = (
        df_clean_broad.groupby(["Circuit","Year"])["LapTimeSeconds"]
        .median().reset_index()
        .rename(columns={"LapTimeSeconds":"circuit_year_median"})
    )
    S.global_median = float(df["LapTimeSeconds"].median())

    # Merge into df for training
    df = df.merge(S.driver_circuit_mean,
                  on=["Driver","Circuit"],               how="left")
    df = df.merge(S.driver_circuit_year_mean,
                  on=["Driver","Circuit","Year"],         how="left")
    df = df.merge(S.driver_circuit_compound_mean,
                  on=["Driver","Circuit","Compound"],     how="left")
    df = df.merge(S.circuit_compound_year_mean,
                  on=["Circuit","Compound","Year"],       how="left")
    df = df.merge(S.circuit_compound_allyr_mean,
                  on=["Circuit","Compound"],              how="left")
    df = df.merge(S.circuit_year_median,
                  on=["Circuit","Year"],                  how="left")

    for col in ["driver_circuit_mean","driver_circuit_year_mean",
                "driver_circuit_compound_mean","circuit_compound_year_mean",
                "circuit_compound_allyr_mean","circuit_year_median"]:
        df[col] = df[col].fillna(S.global_median)

    # 6. Pit loss target
    df_pit_raw = df[df["is_pit_lap"] == 1].copy()
    df_pit_raw["pit_loss"] = (
        df_pit_raw["LapTimeSeconds"] - df_pit_raw["circuit_year_median"]
    )
    df_pit_raw = df_pit_raw[df_pit_raw["pit_loss"].between(10, 45)].copy()

    # 7. Race meta
    S.race_total_laps_df = (
        df_raw.groupby(["Year","Race"])["LapNumber"]
              .max().reset_index()
              .rename(columns={"LapNumber":"race_total_laps"})
    )
    S.race_medians_df = (
        df_raw.groupby(["Year","Race"])["LapTimeSeconds"]
              .median().reset_index()
              .rename(columns={"LapTimeSeconds":"race_median_lap"})
    )

    # 8. Feature lists
    S.normal_features = [
        "LapNumber","is_lap1","is_lap2","is_opening_lap","stint_lap_number",
        "TyreLife","tyre_life_ratio","fuel_load_norm","Compound",
        "Driver_enc","Team_enc","Circuit_enc","Year",
        "driver_circuit_mean","driver_circuit_year_mean",
        "driver_circuit_compound_mean","circuit_compound_year_mean",
    ]
    S.pit_features = [
        "TyreLife","tyre_life_ratio","tyre_delta","laps_on_tyre","full_pit",
        "LapNumber","fuel_load_norm","is_lap2",
        "Driver_enc","Team_enc","Circuit_enc","Year",
        "driver_circuit_mean","driver_circuit_year_mean",
    ]

    # 9. Training data
    df_normal_train = df[
        (df["is_sc_vsc"]  == 0) &
        (df["is_pit_lap"] == 0)
    ][S.normal_features + ["LapTimeSeconds"]].dropna().reset_index(drop=True)

    keep_pit = list(dict.fromkeys(
        S.pit_features + ["pit_loss","circuit_year_median"]
    ))
    df_pit_train = df_pit_raw[keep_pit].dropna().reset_index(drop=True)

    Xn = df_normal_train[S.normal_features].values
    yn = df_normal_train["LapTimeSeconds"].values
    Xp = df_pit_train[S.pit_features].values
    yp = df_pit_train["pit_loss"].values

    print(f"  Normal train rows : {len(Xn)}")
    print(f"  Pit train rows    : {len(Xp)}")

    # 10. Train normal model
    Xn_tr, Xn_val, yn_tr, yn_val = train_test_split(
        Xn, yn, test_size=0.1, random_state=42
    )
    S.model_normal = XGBRegressor(
        n_estimators=1000, max_depth=8,
        learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1,
        reg_lambda=1.0, min_child_weight=5,
        random_state=42, n_jobs=-1, verbosity=0,
        early_stopping_rounds=50,
    )
    S.model_normal.fit(
        Xn_tr, yn_tr,
        eval_set=[(Xn_val, yn_val)],
        verbose=False
    )
    from sklearn.metrics import mean_absolute_error as mae
    print(f"  Normal model MAE  : {mae(yn, S.model_normal.predict(Xn)):.3f}s")

    # 11. Train pit model
    S.model_pit = ExtraTreesRegressor(
        n_estimators=500, max_depth=12,
        min_samples_leaf=3, max_features="sqrt",
        random_state=42, n_jobs=-1
    )
    S.model_pit.fit(Xp, yp)
    print(f"  Pit model MAE     : {mae(yp, S.model_pit.predict(Xp)):.3f}s")

    print("✅ All models ready.")


# =========================================================
# API ENDPOINTS
# =========================================================

@app.get("/api/races")
async def get_races():
    """Return all available races with metadata."""
    races = (
        S.df_raw
        .groupby(["Year","Race"])
        .agg(
            total_laps   = ("LapNumber","max"),
            drivers      = ("Driver","nunique"),
            total_rows   = ("LapNumber","count"),
        )
        .reset_index()
        .sort_values(["Year","Race"], ascending=[False, True])
    )
    return races.to_dict("records")


@app.get("/api/race-info")
async def get_race_info(year: int, race: str):
    """Return drivers, compounds, laps, SC info for a race."""
    race_df = S.df_raw[
        (S.df_raw["Year"] == year) &
        (S.df_raw["Race"] == race)
    ]
    if len(race_df) == 0:
        raise HTTPException(404, f"No data for {year} {race}")

    drivers   = sorted(race_df["Driver"].unique().tolist())
    compounds = sorted([
        c for c in race_df["Compound"].dropna().unique().tolist()
        if c not in ["UNKNOWN"]
    ])
    total_laps = int(race_df["LapNumber"].max())

    # Check for SC/VSC
    sc_laps = S.sc_lap_times[
        (S.sc_lap_times["Year"] == year) &
        (S.sc_lap_times["Race"] == race)
    ]["LapNumber"].tolist()

    return {
        "year"       : year,
        "race"       : race,
        "drivers"    : drivers,
        "compounds"  : compounds,
        "total_laps" : total_laps,
        "has_sc_vsc" : len(sc_laps) > 0,
        "sc_vsc_laps": sorted(sc_laps),
    }


@app.get("/api/driver-strategy")
async def get_driver_strategy(year: int, race: str, driver: str):
    """Return the actual historical strategy for a driver."""
    drv = S.df_raw[
        (S.df_raw["Year"]   == year)  &
        (S.df_raw["Race"]   == race)  &
        (S.df_raw["Driver"] == driver)
    ]
    if len(drv) == 0:
        raise HTTPException(404, f"No data for {driver} at {year} {race}")

    pit_laps, compounds = extract_actual_strategy(year, race, driver)
    return {"pit_laps": pit_laps, "compounds": compounds}


@app.post("/api/simulate", response_model=SimulationResponse)
async def simulate(req: SimulationRequest):
    """Run the race simulation with the given strategy."""
    # Validate inputs
    if len(req.compounds) != len(req.pit_laps) + 1:
        raise HTTPException(
            400,
            f"compounds must have len(pit_laps)+1 elements. "
            f"Got {len(req.compounds)} compounds, {len(req.pit_laps)} pit laps."
        )

    try:
        result = run_simulation(
            year         = req.year,
            race         = req.race,
            driver       = req.driver,
            sim_pit_laps = req.pit_laps,
            sim_compounds= req.compounds,
        )
        return SimulationResponse(**result)

    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Simulation error: {str(e)}")


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
    )