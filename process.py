"""
ZTD 車両軌跡データ 分析パイプライン
=====================================
各処理を独立した関数に分割。
main() で呼び出す順番を変えるだけで部分実行が可能。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pathlib import Path
from tqdm import tqdm
import os

os.chdir(r"T:\S-hayashi\2026\EE\ZTD\code")

# ════════════════════════════════════════════════════════
# 設定（全関数から参照する定数）
# ════════════════════════════════════════════════════════
INPUT_FILE = (
    r"T:\S-nakano\━━━2023━━━\ZTD\02_データ\02_データセット"
    r"\a_阪神高速11号池田線（大阪方面）塚本合流付近"
    r"\02_交通データセット\L001_F001_ALL\L001_F001_TRAJECTORY"
    r"\L001_F001_trajectory.csv"
)
OUTPUT_DIR = Path("output")

KP_MIN, KP_MAX   = None, None   # キロポスト範囲（None=全区間）
V_MIN,  V_MAX    = 0,    100    # 速度カラーマップ範囲 (km/h)

BRAKE_THRESHOLD    = -2.9  # 急ブレーキ閾値 (m/s²)  -0.3G
FOLLOW_DIST_MAX    = 150   # 追従とみなす最大車間距離 (m)
RESPONSE_WINDOW    = 5.0   # 急ブレーキ後の反応待ち時間 (秒)
RESPONSE_THRESHOLD = -2.0  # 後続車の「有意な減速」閾値 (m/s²)

# パラメータスキャン範囲
TIME_WINDOWS = np.arange(1.0, 12.0, 1.0)  # 反応時間 1〜11s
DIST_LIMITS  = np.arange(50,  351,  25)   # 追従距離 50〜350m


# ════════════════════════════════════════════════════════
# 0. ランダムデータ生成（動作確認用）
# ════════════════════════════════════════════════════════
def generate_random_data(out_path: str = r"C:\Users\s-hayashi\Downloads\random.csv"):
    """実データの統計量を使ってランダムなサンプルCSVを生成する。"""
    df = pd.read_csv(
        INPUT_FILE, encoding="shift-jis",
        names=["veicle_id", "datetime", "vehicle_type", "velocity",
               "trafic_lane", "longitude", "latitude", "kilopost",
               "vehicle_length", "detected_flag"],
    )
    n = 1000
    num_cols = ["velocity", "longitude", "latitude", "kilopost", "vehicle_length"]
    cat_cols = ["vehicle_type", "trafic_lane", "detected_flag"]

    random_df = pd.concat([
        pd.Series(np.random.choice(range(1, 51), size=n), name="veicle_id"),
        pd.Series(
            pd.to_datetime(np.random.randint(
                df["datetime"].astype("int64").min(),
                df["datetime"].astype("int64").max(), n,
            )), name="datetime",
        ),
        pd.DataFrame({
            col: np.random.normal(df[num_cols].mean()[col], df[num_cols].std()[col], n)
            for col in num_cols
        }),
        pd.DataFrame({
            col: np.random.choice(
                df[col].value_counts().index,
                size=n,
                p=df[col].value_counts(normalize=True).values,
            ) for col in cat_cols
        }),
    ], axis=1)

    random_df.to_csv(out_path, encoding="shift-jis", index=False)
    print(f"ランダムデータ保存: {out_path}")
    return random_df


# ════════════════════════════════════════════════════════
# 1. データ読み込み・前処理
# ════════════════════════════════════════════════════════
def load_and_preprocess() -> pd.DataFrame:
    """CSVを読み込み、型変換・ソート・範囲フィルタを行う。"""
    df = pd.read_csv(
        INPUT_FILE, encoding="shift-jis",
        names=["vehicle_id", "datetime", "vehicle_type", "velocity",
               "traffic_lane", "longitude", "latitude", "kilopost",
               "vehicle_length", "detected_flag"],
    )
    df.columns = df.columns.str.strip()
    df["datetime"] = pd.to_datetime(df["datetime"].astype(str), format="%H%M%S%f")
    df["t_sec"] = (df["datetime"] - df["datetime"].min()).dt.total_seconds()

    if KP_MIN is not None:
        df = df[df["kilopost"] >= KP_MIN]
    if KP_MAX is not None:
        df = df[df["kilopost"] <= KP_MAX]

    df = df.sort_values(["vehicle_id", "t_sec"]).reset_index(drop=True)

    print("=== データ概要 ===")
    print(f"  総レコード数         : {len(df):,}")
    print(f"  ユニーク車両数       : {df['vehicle_id'].nunique():,}")
    print(f"  時刻範囲             : {df['t_sec'].min():.3f}s 〜 {df['t_sec'].max():.3f}s")
    print(f"  キロポスト範囲       : {df['kilopost'].min():.0f}m 〜 {df['kilopost'].max():.0f}m")
    print(f"  速度範囲             : {df['velocity'].min():.1f} 〜 {df['velocity'].max():.1f} km/h")
    print(f"  補間レコード率       : {(df['detected_flag']==0).mean()*100:.1f}%")
    print(f"  大型車比率           : {(df['vehicle_type']==2).mean()*100:.1f}%")
    rpv = df.groupby("vehicle_id").size()
    print(f"  車両あたりレコード数 : 平均 {rpv.mean():.1f}, 最大 {rpv.max()}")
    return df


# ════════════════════════════════════════════════════════
# 2. タイムスペース図
# ════════════════════════════════════════════════════════
def plot_time_space(df: pd.DataFrame):
    """車両軌跡をタイムスペース図（横軸=時刻、縦軸=キロポスト）で描画する。"""
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle("ZTD タイムスペース図（塚本合流付近）", fontsize=14)

    cmap = cm.get_cmap("jet_r")
    norm = mcolors.Normalize(vmin=V_MIN, vmax=V_MAX)
    lane_styles = {1: "-", 2: "--", 3: ":"}

    for vid, grp in tqdm(df.groupby("vehicle_id"), desc="軌跡描画"):
        detected = grp[grp["detected_flag"] == 1]
        interp   = grp[grp["detected_flag"] == 0]

        if len(detected) < 2:
            continue

        for i in range(len(detected) - 1):
            r0, r1 = detected.iloc[i], detected.iloc[i + 1]
            ax.plot(
                [r0["t_sec"], r1["t_sec"]],
                [r0["kilopost"], r1["kilopost"]],
                color=cmap(norm((r0["velocity"] + r1["velocity"]) / 2)),
                linewidth=0.8,
                linestyle=lane_styles.get(int(r0["traffic_lane"]), "-"),
                alpha=0.7,
            )

        if len(interp) >= 2:
            ax.plot(interp["t_sec"], interp["kilopost"],
                    color="gray", linewidth=0.4, alpha=0.3, linestyle=":")

    ax.set_xlabel("時刻 (s)")
    ax.set_ylabel("キロポスト (m)")
    ax.set_ylim(df["kilopost"].min(), df["kilopost"].max())
    ax.set_title("車両軌跡（時間→横）")
    ax.grid(linewidth=0.3, alpha=0.4)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="速度 (km/h)")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "time_space_diagram.png", dpi=150)
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 8))

    det = df[df["detected_flag"] == 1]

    ax.hist(det["velocity"], bins=40, orientation="horizontal",
            color="#378ADD", alpha=0.7, edgecolor="white", linewidth=0.3)

    ax.set_xlabel("車両数")
    ax.set_ylabel("速度 (km/h)")
    ax.set_title("速度分布（detected only）")

    ax.legend()
    ax.grid(axis="x", linewidth=0.3, alpha=0.4)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "velocity_histogram.png", dpi=150)
    plt.show()

# ════════════════════════════════════════════════════════
# 3. 急ブレーキ検出（前処理）
# ════════════════════════════════════════════════════════
def detect_brake_events(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    detected_flag==1 のレコードのみで連続区間の減速度を計算し、
    急ブレーキイベント（BRAKE_THRESHOLD 以下）を抽出する。

    Returns
    -------
    detected_only : 減速度カラム付きの実測レコード全体
    brake_events  : 急ブレーキイベントのみ
    """
    detected_only = df[df["detected_flag"] == 1].copy()
    detected_only = detected_only.sort_values(["vehicle_id", "t_sec"])
    detected_only["dv"] = detected_only.groupby("vehicle_id")["velocity"].diff()
    detected_only["dt"] = detected_only.groupby("vehicle_id")["t_sec"].diff()
    detected_only = detected_only[detected_only["dt"].between(0.001, 0.2)]
    detected_only["decel_ms2"] = (detected_only["dv"] / 3.6) / detected_only["dt"]

    brake_events = detected_only[detected_only["decel_ms2"] <= BRAKE_THRESHOLD]

    print(f"\n=== 急ブレーキイベント（閾値: {BRAKE_THRESHOLD} m/s²） ===")
    print(f"  検出件数   : {len(brake_events)}")
    print(f"  関与車両数 : {brake_events['vehicle_id'].nunique()}")
    if len(brake_events) > 0:
        print(f"  平均減速度 : {brake_events['decel_ms2'].mean():.2f} m/s²")
        print(f"  最大減速度 : {brake_events['decel_ms2'].min():.2f} m/s²")
        print(brake_events[["vehicle_id", "t_sec", "kilopost",
                             "traffic_lane", "velocity", "decel_ms2"]
                            ].head(10).to_string(index=False))

    brake_events.to_csv(OUTPUT_DIR / "brake_events.csv", index=False)
    print(f"急ブレーキCSV保存: {OUTPUT_DIR / 'brake_events.csv'}")
    return detected_only, brake_events


# ════════════════════════════════════════════════════════
# 4. 後続車への伝播判定
# ════════════════════════════════════════════════════════
def calc_propagation(
    detected_only: pd.DataFrame,
    brake_events:  pd.DataFrame,
) -> pd.DataFrame:
    """
    各急ブレーキイベントについて、後続車（同車線・後方 FOLLOW_DIST_MAX 以内）が
    RESPONSE_WINDOW 秒以内に RESPONSE_THRESHOLD 以下の減速をしたかを判定する。

    Returns
    -------
    results_df : brake_vid / brake_t / n_followers / n_infected を持つ DataFrame
    """
    results = []
    for _, brake in tqdm(brake_events.iterrows(), total=len(brake_events), desc="伝播判定"):
        b_vid, b_t, b_kp, b_lane = (
            brake["vehicle_id"], brake["t_sec"],
            brake["kilopost"],   brake["traffic_lane"],
        )
        snapshot = detected_only[
            (detected_only["t_sec"].between(b_t - 0.5, b_t + 0.5)) &
            (detected_only["traffic_lane"] == b_lane) &
            (detected_only["vehicle_id"]   != b_vid)
        ]
        followers = snapshot[
            (snapshot["kilopost"] < b_kp) &
            (snapshot["kilopost"] >= b_kp - FOLLOW_DIST_MAX)
        ]
        if followers.empty:
            results.append({"brake_vid": b_vid, "brake_t": b_t, "brake_kp": b_kp,
                             "lane": b_lane, "brake_decel": brake["decel_ms2"],
                             "n_followers": 0, "n_infected": 0})
            continue

        n_infected = sum(
            1 for f_vid in followers["vehicle_id"].unique()
            if not (future := detected_only[
                (detected_only["vehicle_id"] == f_vid) &
                (detected_only["t_sec"].between(b_t, b_t + RESPONSE_WINDOW))
            ]).empty and future["decel_ms2"].min() <= RESPONSE_THRESHOLD
        )
        results.append({"brake_vid": b_vid, "brake_t": b_t, "brake_kp": b_kp,
                         "lane": b_lane, "brake_decel": brake["decel_ms2"],
                         "n_followers": followers["vehicle_id"].nunique(),
                         "n_infected": n_infected})

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "r_value_raw.csv", index=False)
    print(f"伝播結果CSV保存: {OUTPUT_DIR / 'r_value_raw.csv'}")
    return results_df


# ════════════════════════════════════════════════════════
# 5. R値計算
# ════════════════════════════════════════════════════════
def calc_r_value(results_df: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """
    R = 有意な減速を引き起こした合計 / 有効イベント数

    Returns
    -------
    R      : float  実効再生産数
    valid  : DataFrame  後続車ありのイベントのみ
    """
    valid = results_df[results_df["n_followers"] > 0]
    if valid.empty:
        print("後続車ありのイベントなし。FOLLOW_DIST_MAX や時刻窓を広げてください。")
        return np.nan, valid

    R = valid["n_infected"].sum() / len(valid)
    print(f"\n=== 交通 R 値（実効再生産数） ===")
    print(f"  急ブレーキイベント総数 : {len(results_df)}")
    print(f"  後続車あり（有効）件数 : {len(valid)}")
    print(f"  R 値（平均）           : {R:.3f}")
    print(f"  R 値（中央値）         : {valid['n_infected'].median():.1f}")
    print(f"  標準偏差               : {valid['n_infected'].std():.3f}")
    print(f"  → {'渋滞が自己増幅する状態' if R > 1 else '渋滞が収束する状態'} (R {'>' if R > 1 else '<'} 1)")
    return R, valid


# ════════════════════════════════════════════════════════
# 6. 感度分析（ブレーキ閾値 × 反応閾値）
# ════════════════════════════════════════════════════════
def sensitivity_analysis(detected_only: pd.DataFrame, brake_events: pd.DataFrame):
    """
    ブレーキ閾値・反応閾値を変化させてもR値の大小関係が変わらないことを確認する。
    """
    sensitivity = []
    for b_thresh in [-2.0, -2.9, -4.0]:
        for r_thresh in [-1.0, -2.0, -3.0]:
            b_ev = detected_only[detected_only["decel_ms2"] <= b_thresh]
            if b_ev.empty:
                continue
            sub_results = []
            for _, brake in b_ev.iterrows():
                b_vid, b_t, b_kp, b_lane = (
                    brake["vehicle_id"], brake["t_sec"],
                    brake["kilopost"],   brake["traffic_lane"],
                )
                snapshot = detected_only[
                    (detected_only["t_sec"].between(b_t - 0.5, b_t + 0.5)) &
                    (detected_only["traffic_lane"] == b_lane) &
                    (detected_only["vehicle_id"]   != b_vid)
                ]
                followers = snapshot[
                    (snapshot["kilopost"] < b_kp) &
                    (snapshot["kilopost"] >= b_kp - FOLLOW_DIST_MAX)
                ]
                if followers.empty:
                    sub_results.append(0)
                    continue
                infected = sum(
                    1 for f_vid in followers["vehicle_id"].unique()
                    if not (future := detected_only[
                        (detected_only["vehicle_id"] == f_vid) &
                        (detected_only["t_sec"].between(b_t, b_t + RESPONSE_WINDOW))
                    ]).empty and future["decel_ms2"].min() <= r_thresh
                )
                sub_results.append(infected)

            sub_r = np.mean(sub_results) if sub_results else np.nan
            sensitivity.append({
                "brake_thresh (m/s²)":    b_thresh,
                "response_thresh (m/s²)": r_thresh,
                "n_events":               len(b_ev),
                "R": round(sub_r, 3) if not np.isnan(sub_r) else "—",
            })

    sens_df = pd.DataFrame(sensitivity)
    print(f"\n=== 感度分析（閾値ごとのR値） ===")
    print(sens_df.to_string(index=False))
    sens_df.to_csv(OUTPUT_DIR / "sensitivity.csv", index=False)
    return sens_df


# ════════════════════════════════════════════════════════
# 7. R値可視化（分布・空間・時系列）
# ════════════════════════════════════════════════════════
def plot_r_value(R: float, valid: pd.DataFrame):
    """伝播分布・キロポスト別・時間帯別の3パネル図を出力する。"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("交通R値 分析結果", fontsize=13)

    # 7-A. 伝播件数ヒストグラム
    ax = axes[0]
    ax.hist(valid["n_infected"], bins=range(0, valid["n_infected"].max() + 2),
            color="#378ADD", edgecolor="white", linewidth=0.5, align="left")
    ax.axvline(R, color="#D85A30", linewidth=2, linestyle="--", label=f"R = {R:.2f}")
    ax.axvline(1.0, color="#639922", linewidth=1.5, linestyle=":", label="R = 1 臨界点")
    ax.set_xlabel("1イベントあたりの伝播件数", fontsize=11)
    ax.set_ylabel("急ブレーキイベント数", fontsize=11)
    ax.set_title("伝播件数の分布", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linewidth=0.3, alpha=0.4)

    # 7-B. キロポスト別R値
    ax = axes[1]
    bin_size = 200
    valid = valid.copy()
    valid["kp_bin"] = (valid["brake_kp"] // bin_size * bin_size).astype(int)
    kp_r = valid.groupby("kp_bin")["n_infected"].mean()
    ax.bar(kp_r.index, kp_r.values, width=bin_size * 0.8,
           color=["#D85A30" if v > 1 else "#378ADD" for v in kp_r.values],
           edgecolor="white", linewidth=0.4)
    ax.axhline(1.0, color="#639922", linewidth=1.5, linestyle=":", label="R = 1 臨界点")
    ax.set_xlabel("キロポスト (m)", fontsize=11)
    ax.set_ylabel("区間平均 R 値", fontsize=11)
    ax.set_title("空間分布（赤=R>1 危険区間）", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linewidth=0.3, alpha=0.4)

    # 7-C. 時間帯別R値
    ax = axes[2]
    t_min, t_max = valid["brake_t"].min(), valid["brake_t"].max()
    tw = (t_max - t_min) / 6
    if tw > 0:
        valid["t_bin"] = ((valid["brake_t"] - t_min) // tw).astype(int)
        t_r = valid.groupby("t_bin")["n_infected"].mean()
        ax.plot(t_r.index, t_r.values, marker="o", color="#378ADD",
                linewidth=1.5, markersize=5)
        ax.fill_between(t_r.index, t_r.values, alpha=0.15, color="#378ADD")
        ax.axhline(1.0, color="#639922", linewidth=1.5, linestyle=":", label="R = 1 臨界点")
        ax.set_xticks(t_r.index)
        ax.set_xticklabels(
            [f"{t_min + i * tw:.0f}s" for i in t_r.index], rotation=30, fontsize=8)
        ax.set_xlabel("時間帯", fontsize=11)
        ax.set_ylabel("R 値", fontsize=11)
        ax.set_title("時間的変化", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(linewidth=0.3, alpha=0.4)

    plt.tight_layout()
    out = OUTPUT_DIR / "r_value_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"R値分析図保存: {out}")
    return kp_r


# ════════════════════════════════════════════════════════
# 8. サマリーレポート
# ════════════════════════════════════════════════════════
def save_summary(R: float, valid: pd.DataFrame, results_df: pd.DataFrame, kp_r: pd.Series):
    """分析結果をまとめてCSVに保存し、コンソールに表示する。"""
    summary = {
        "R値（平均）":         round(R, 3),
        "R値（中央値）":       valid["n_infected"].median(),
        "有効イベント数":      len(valid),
        "総急ブレーキ件数":    len(results_df),
        "後続車ありの割合(%)": round(len(valid) / len(results_df) * 100, 1),
        "状態判定":            "渋滞が自己増幅" if R > 1 else "渋滞が収束傾向",
        "最もRが高い区間(kp)": int(kp_r.idxmax()) if len(kp_r) > 0 else "—",
    }
    print(f"\n=== サマリーレポート ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    pd.DataFrame([summary]).to_csv(OUTPUT_DIR / "summary.csv", index=False)
    print(f"\n全出力ファイル:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        print(f"  {f}")


# ════════════════════════════════════════════════════════
# 9. R=1 境界分析（パラメータスキャン）
# ════════════════════════════════════════════════════════
def scan_r1_boundary(
    detected_only: pd.DataFrame,
    brake_events:  pd.DataFrame,
) -> np.ndarray:
    """
    反応時間（TIME_WINDOWS）× 追従距離（DIST_LIMITS）のグリッドでR値を計算し、
    R=1 の臨界境界を求める。

    Returns
    -------
    R_GRID : shape (len(TIME_WINDOWS), len(DIST_LIMITS)) の numpy 配列
    """
    print("\n=== R=1 境界分析（パラメータスキャン） ===")
    R_GRID = np.full((len(TIME_WINDOWS), len(DIST_LIMITS)), np.nan)

    for i, tw in enumerate(tqdm(TIME_WINDOWS, desc="反応時間スキャン")):
        for j, dl in enumerate(DIST_LIMITS):
            scan_results = []
            for _, brake in brake_events.iterrows():
                b_vid, b_t, b_kp, b_lane = (
                    brake["vehicle_id"], brake["t_sec"],
                    brake["kilopost"],   brake["traffic_lane"],
                )
                snapshot = detected_only[
                    (detected_only["t_sec"].between(b_t - 0.5, b_t + 0.5)) &
                    (detected_only["traffic_lane"] == b_lane) &
                    (detected_only["vehicle_id"]   != b_vid)
                ]
                followers = snapshot[
                    (snapshot["kilopost"] < b_kp) &
                    (snapshot["kilopost"] >= b_kp - dl)
                ]
                if followers.empty:
                    continue
                infected = sum(
                    1 for f_vid in followers["vehicle_id"].unique()
                    if not (future := detected_only[
                        (detected_only["vehicle_id"] == f_vid) &
                        (detected_only["t_sec"].between(b_t, b_t + tw))
                    ]).empty and future["decel_ms2"].min() <= RESPONSE_THRESHOLD
                )
                scan_results.append(infected)
            if scan_results:
                R_GRID[i, j] = np.mean(scan_results)

    # テーブル出力
    print(f"\n  {'反応時間(s)':>10}", end="")
    for dl in DIST_LIMITS:
        print(f"  {dl:>4}m", end="")
    print()
    for i, tw in enumerate(TIME_WINDOWS):
        print(f"  {tw:>10.1f}", end="")
        for j in range(len(DIST_LIMITS)):
            val = R_GRID[i, j]
            print(f"  {f'{val:.2f}' if not np.isnan(val) else ' -- ':>5}", end="")
        print()

    pd.DataFrame(
        R_GRID,
        index=pd.Index(TIME_WINDOWS, name="reaction_time_s"),
        columns=pd.Index(DIST_LIMITS, name="follow_dist_m"),
    ).to_csv(OUTPUT_DIR / "r1_boundary.csv")
    print(f"境界CSV保存: {OUTPUT_DIR / 'r1_boundary.csv'}")
    return R_GRID


# ════════════════════════════════════════════════════════
# 10. R=1 境界の可視化
# ════════════════════════════════════════════════════════
def plot_r1_boundary(R_GRID: np.ndarray):
    """ヒートマップと境界線断面の2パネル図を出力する。"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("R=1 臨界境界：反応時間 × 車間距離", fontsize=13)

    # 10-A. ヒートマップ
    ax = axes[0]
    im = ax.imshow(
        R_GRID, aspect="auto", origin="lower",
        extent=[DIST_LIMITS[0], DIST_LIMITS[-1], TIME_WINDOWS[0], TIME_WINDOWS[-1]],
        cmap="jet_r", vmin=0, vmax=2,
    )
    plt.colorbar(im, ax=ax, label="R 値")
    cs = ax.contour(DIST_LIMITS, TIME_WINDOWS, R_GRID,
                    levels=[1.0], colors=["white"], linewidths=2.5, linestyles="--")
    ax.clabel(cs, fmt="R=1 臨界線", fontsize=9, colors="white")
    ax.scatter([FOLLOW_DIST_MAX], [RESPONSE_WINDOW], color="blue", s=80, zorder=5,
               label=f"デフォルト設定 ({FOLLOW_DIST_MAX}m, {RESPONSE_WINDOW}s)")
    ax.set_xlabel("追従とみなす距離 (m)", fontsize=11)
    ax.set_ylabel("反応時間上限 (秒)", fontsize=11)
    ax.set_title("R値ヒートマップ\n赤=R>1 自己増幅域、緑=R<1 収束域", fontsize=10)
    ax.legend(fontsize=9, loc="lower right")

    # 10-B. R=1 境界線断面
    ax2 = axes[1]
    boundary_dist = []
    for i, tw in enumerate(TIME_WINDOWS):
        row = R_GRID[i, :]
        crossing = np.where(row >= 1.0)[0]
        if len(crossing) > 0:
            j = crossing[0]
            if j > 0 and not np.isnan(row[j - 1]):
                r0, r1 = row[j - 1], row[j]
                d0, d1 = DIST_LIMITS[j - 1], DIST_LIMITS[j]
                d_cross = d0 + (1.0 - r0) / (r1 - r0) * (d1 - d0)
            else:
                d_cross = float(DIST_LIMITS[j])
            boundary_dist.append((tw, d_cross))

    if boundary_dist:
        b_tw, b_dist = zip(*boundary_dist)
        ax2.plot(b_dist, b_tw, color="#D85A30", linewidth=2.5,
                 marker="o", markersize=6, label="R=1 臨界境界")
        ax2.fill_betweenx(b_tw, b_dist, max(DIST_LIMITS),
                          alpha=0.15, color="#D85A30", label="自己増幅域 (R>1)")
        ax2.fill_betweenx(b_tw, 0, b_dist,
                          alpha=0.12, color="#378ADD", label="収束域 (R<1)")
        for tw, dist in list(boundary_dist)[::2]:
            ax2.annotate(f"{dist:.0f}m", xy=(dist, tw), xytext=(dist + 8, tw),
                         fontsize=8, color="#993C1D", va="center")

    ax2.set_xlabel("R=1 となる車間距離の境界 (m)", fontsize=11)
    ax2.set_ylabel("反応時間 (秒)", fontsize=11)
    ax2.set_title("R=1 臨界境界線\n右=危険域（R>1）　左=安全域（R<1）", fontsize=10)
    ax2.set_xlim(0, max(DIST_LIMITS) + 20)
    ax2.set_ylim(TIME_WINDOWS[0] - 0.5, TIME_WINDOWS[-1] + 0.5)
    ax2.legend(fontsize=9)
    ax2.grid(linewidth=0.3, alpha=0.4)

    plt.tight_layout()
    out = OUTPUT_DIR / "r1_boundary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"境界分析図保存: {out}")


# ════════════════════════════════════════════════════════
# 11. R値の時空間ヒートマップ（15分 × 100m）
# ════════════════════════════════════════════════════════
def plot_r_spatiotemporal(
    valid: pd.DataFrame,
    time_bin_min: float = 15.0,   # 時間ビン幅（分）
    kp_bin_m:     float = 100.0,  # キロポストビン幅（m）
):
    """
    急ブレーキイベントを「発生時刻（15分単位）× 発生位置（100m単位）」で集計し、
    各セルのR値（平均伝播件数）を2次元ヒートマップで可視化する。
 
    タイムスペース図と同じ軸配置（横=時刻、縦=キロポスト）にして
    並べて見比べやすくする。
 
    Parameters
    ----------
    valid        : calc_r_value() が返す、後続車ありのイベントDF
    time_bin_min : 時間ビン幅（分）。デフォルト 15分
    kp_bin_m     : キロポストビン幅（m）。デフォルト 100m
    """
    time_bin_sec = time_bin_min * 60
 
    df = valid.copy()
 
    # ── ビン列を作成 ──────────────────────────────────────
    df["t_bin"]  = (df["brake_t"]  // time_bin_sec).astype(int)
    df["kp_bin"] = (df["brake_kp"] // kp_bin_m    ).astype(int)
 
    # ── セルごとに集計 ────────────────────────────────────
    # R値（平均伝播件数）・急ブレーキ件数・伝播率
    agg = (
        df.groupby(["t_bin", "kp_bin"])
        .agg(
            R_mean      = ("n_infected", "mean"),
            n_events    = ("n_infected", "count"),
            n_infected  = ("n_infected", "sum"),
        )
        .reset_index()
    )
    agg["infect_rate"] = agg["n_infected"] / agg["n_events"]  # 伝播率（0〜1）
 
    # ── ピボット（行=kp_bin、列=t_bin）───────────────────
    t_vals  = sorted(agg["t_bin"].unique())
    kp_vals = sorted(agg["kp_bin"].unique())
 
    def to_matrix(value_col: str) -> np.ndarray:
        pivot = agg.pivot(index="kp_bin", columns="t_bin", values=value_col)
        pivot = pivot.reindex(index=kp_vals, columns=t_vals)
        return pivot.values.astype(float)
 
    R_mat     = to_matrix("R_mean")
    count_mat = to_matrix("n_events")
    rate_mat  = to_matrix("infect_rate")
 
    # ── 軸ラベル ─────────────────────────────────────────
    t_labels  = [f"{int(t * time_bin_min)}分" for t in t_vals]
    kp_labels = [f"{int(k * kp_bin_m)}"       for k in kp_vals]
 
    # ── 描画 ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"R値の時空間分布（時間ビン: {time_bin_min:.0f}分 / キロポストビン: {kp_bin_m:.0f}m）",
        fontsize=13,
    )
 
    def draw_heatmap(ax, mat, title, cmap, vmin, vmax, cbar_label, r1_contour=False):
        """共通ヒートマップ描画ヘルパー"""
        im = ax.imshow(
            mat,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        plt.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)
 
        # R=1 の等高線（R値マップのみ）
        if r1_contour and not np.all(np.isnan(mat)):
            try:
                cs = ax.contour(
                    mat,
                    levels=[1.0],
                    colors=["white"],
                    linewidths=2.0,
                    linestyles="--",
                )
                ax.clabel(cs, fmt="R=1", fontsize=8, colors="white")
            except Exception:
                pass  # データが少なくて等高線が引けない場合はスキップ
 
        # 軸ラベル（間引いて表示）
        t_step  = max(1, len(t_vals)  // 8)
        kp_step = max(1, len(kp_vals) // 8)
        ax.set_xticks(range(0, len(t_vals),  t_step))
        ax.set_xticklabels([t_labels[i]  for i in range(0, len(t_vals),  t_step)],
                           rotation=45, fontsize=8)
        ax.set_yticks(range(0, len(kp_vals), kp_step))
        ax.set_yticklabels([kp_labels[i] for i in range(0, len(kp_vals), kp_step)],
                           fontsize=8)
        ax.set_xlabel("時刻", fontsize=11)
        ax.set_ylabel("キロポスト (m)", fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.grid(False)
 
    # 11-A. R値ヒートマップ（メイン）
    draw_heatmap(
        axes[0], R_mat,
        title="R値（平均伝播件数）\n白破線: R=1 臨界線",
        cmap="RdYlGn_r",    # 赤=高R（危険）、緑=低R（安全）
        vmin=0, vmax=2,
        cbar_label="R 値",
        r1_contour=True,
    )
 
    # 11-B. 急ブレーキ件数ヒートマップ
    draw_heatmap(
        axes[1], count_mat,
        title="急ブレーキ発生件数\n（イベント数）",
        cmap="YlOrRd",
        vmin=0, vmax=np.nanpercentile(count_mat, 95),  # 外れ値で色が潰れないよう95%ile上限
        cbar_label="件数",
    )
 
    # 11-C. 伝播率ヒートマップ（伝播した割合 0〜1）
    draw_heatmap(
        axes[2], rate_mat,
        title="伝播率\n（R>0 イベントの割合）",
        cmap="RdYlGn_r",
        vmin=0, vmax=1,
        cbar_label="伝播率",
    )
 
    plt.tight_layout()
    out = OUTPUT_DIR / "r_spatiotemporal.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"時空間R値ヒートマップ保存: {out}")
 
    # ── セル別集計をCSVに保存 ────────────────────────────
    # 人間が読みやすい時刻・kp列を追加
    agg["t_label"]  = agg["t_bin"].apply(
        lambda b: f"{int(b * time_bin_min)}〜{int((b+1) * time_bin_min)}分")
    agg["kp_label"] = agg["kp_bin"].apply(
        lambda b: f"{int(b * kp_bin_m)}〜{int((b+1) * kp_bin_m)}m")
    agg = agg.sort_values(["t_bin", "kp_bin"])
    out_csv = OUTPUT_DIR / "r_spatiotemporal.csv"
    agg.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"時空間集計CSV保存: {out_csv}")
 
    # ── 上位セル（R値が高い危険な時空間）を表示 ────────────
    top = (
        agg[agg["n_events"] >= 2]           # 件数が少ないセルは除外
        .sort_values("R_mean", ascending=False)
        .head(10)
    )
    print(f"\n=== R値上位10セル（危険な時空間） ===")
    print(top[["t_label", "kp_label", "R_mean", "n_events", "infect_rate"]]
          .rename(columns={
              "t_label":     "時間帯",
              "kp_label":    "区間(kp)",
              "R_mean":      "R値",
              "n_events":    "急ブレーキ件数",
              "infect_rate": "伝播率",
          })
          .to_string(index=False))
 
    return agg
 

# ════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 各処理をここで ON/OFF できる
    df_ori                       = load_and_preprocess()           # 1. 読み込み
    df =  df_ori.iloc[:len(df_ori)//20]
    # plot_time_space(df)                                        # 2. タイムスペース図

    detected_only, brake_ev  = detect_brake_events(df)        # 3. 急ブレーキ検出
    results_df               = calc_propagation(              # 4. 伝播判定
                                   detected_only, brake_ev)
    R, valid                 = calc_r_value(results_df)       # 5. R値計算
    if not np.isnan(R):
        sensitivity_analysis(detected_only, brake_ev)         # 6. 感度分析
        kp_r = plot_r_value(R, valid)                         # 7. R値可視化
        save_summary(R, valid, results_df, kp_r)              # 8. サマリー
        R_GRID = scan_r1_boundary(detected_only, brake_ev)    # 9. 境界スキャン
        plot_r1_boundary(R_GRID)                              # 10. 境界可視化
        plot_r_spatiotemporal(valid, time_bin_min=15, kp_bin_m=100)  # 11. 時空間ヒートマップ

if __name__ == "__main__":
    main()