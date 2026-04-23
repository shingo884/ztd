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

def radom_data():
    input_file = r"T:\S-nakano\━━━2023━━━\ZTD\02_データ\02_データセット\a_阪神高速11号池田線（大阪方面）塚本合流付近\02_交通データセット\L001_F001_ALL\L001_F001_TRAJECTORY\L001_F001_trajectory.csv"
    df = pd.read_csv(input_file,encoding="shift-jis",names=["veicle_id","datetime","vehicle_type","velocity","trafic_lane","longitude","latitude","kilopost","vehicle_length","detected_flag"])
    df.columns 

    # 数値列
    num_cols = ['velocity', 'longitude', 'latitude', 'kilopost', 'vehicle_length']

    # 平均と分散
    means = df[num_cols].mean()
    stds = df[num_cols].std()

    # ランダムデータ作成（1000件）
    n = 1000
    random_num = pd.DataFrame({
        col: np.random.normal(means[col], stds[col], n)
        for col in num_cols
    })
    cat_cols = ['vehicle_type', 'trafic_lane', 'detected_flag']

    random_cat = pd.DataFrame({
        col: np.random.choice(
            df[col].value_counts().index,                 # ユニーク値
            size=n,
            p=df[col].value_counts(normalize=True).values # 確率
        )
        for col in cat_cols
    })
    random_time = pd.Series(
        pd.to_datetime(
            np.random.randint(
                df['datetime'].astype('int64').min(),
                df['datetime'].astype('int64').max(),
                n
            )
        ),
        name='datetime'
    )
    num_vehicles = 50  # 車両数（少なめにするのがポイント）

    random_id = pd.Series(
        np.random.choice(range(1, num_vehicles+1), size=n),
        name='veicle_id'
    )
    random_df = pd.concat(
                    [random_id, random_time.rename('datetime'), random_num, random_cat],
                    axis=1
                )
    
    random_df.to_csv(r"C:\Users\s-hayashi\Downloads\random.csv",encoding="shift-jis",index=False)


def process():
    input_file = r"T:\S-nakano\━━━2023━━━\ZTD\02_データ\02_データセット\a_阪神高速11号池田線（大阪方面）塚本合流付近\02_交通データセット\L001_F001_ALL\L001_F001_TRAJECTORY\L001_F001_trajectory.csv"
    OUTPUT_DIR = Path("output")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # タイムスペース図で表示するキロポスト範囲（Noneで全区間）
    KP_MIN, KP_MAX = None, None
    # 速度カラーマップ（0〜100 km/h）
    V_MIN, V_MAX = 0, 100
    
    # ── 1. データ読み込み・前処理 ─────────────────────────────
    
    df = pd.read_csv(input_file,encoding="shift-jis",names=["vehicle_id","datetime","vehicle_type","velocity","traffic_lane","longitude","latitude","kilopost","vehicle_length","detected_flag"])
    
    # カラム名の表記ゆれを吸収
    df.columns = df.columns.str.strip()

    # datetime を pandas Timestamp に変換（ナノ秒精度）
    df["datetime"] = pd.to_datetime(df["datetime"].astype(str),format="%H%M%S%f")
    # 分析用に「秒」の浮動小数点に変換
    df["t_sec"] = (df["datetime"] - df["datetime"].min()).dt.total_seconds()
    
    # キロポスト範囲フィルタ
    if KP_MIN is not None:
        df = df[df["kilopost"] >= KP_MIN]
    if KP_MAX is not None:
        df = df[df["kilopost"] <= KP_MAX]
    
    # vehicle_id × 時刻 でソート（軌跡線を正しく引くため必須）
    df = df.sort_values(["vehicle_id", "t_sec"]).reset_index(drop=True)
    
    # ── 2. データ概要の確認 ──────────────────────────────────
    print("=== データ概要 ===")
    print(f"  総レコード数   : {len(df):,}")
    print(f"  ユニーク車両数 : {df['vehicle_id'].nunique():,}")
    print(f"  時刻範囲       : {df['t_sec'].min():.3f}s 〜 {df['t_sec'].max():.3f}s")
    print(f"  キロポスト範囲 : {df['kilopost'].min():.0f}m 〜 {df['kilopost'].max():.0f}m")
    print(f"  速度範囲       : {df['velocity'].min():.1f} 〜 {df['velocity'].max():.1f} km/h")
    print(f"  補間レコード率 : {(df['detected_flag']==0).mean()*100:.1f}%")
    print(f"  大型車比率     : {(df['vehicle_type']==2).mean()*100:.1f}%")
    records_per_v = df.groupby("vehicle_id").size()
    print(f"  車両あたりレコード数: 平均 {records_per_v.mean():.1f}, 最大 {records_per_v.max()}")
    
    # ── 3. タイムスペース図 ───────────────────────────────────
    fig, axes = plt.subplots(
        1, 2, figsize=(16, 8),
        gridspec_kw={"width_ratios": [3, 1]},
        sharey=True,
    )
    fig.suptitle("ZTD タイムスペース図（塚本合流付近）", fontsize=14, y=1.01)
    
    ax = axes[0]
    cmap = cm.get_cmap("jet_r") # 赤(低速) → 黄 → 緑(高速)
    norm = mcolors.Normalize(vmin=V_MIN, vmax=V_MAX)
    
    lane_styles = {1: "-",  2: "--", 3: ":"}  # 車線ごとに線種を変える
    
    # 1車両 = 1軌跡線
    for vid, grp in tqdm(df.groupby("vehicle_id")):
        # detected_flag==0（補間）は破線で薄く表示
        detected = grp[grp["detected_flag"] == 1]
        interp   = grp[grp["detected_flag"] == 0]
    
        if len(detected) < 2:
            continue
    
        # 速度で色付け（区間ごとに色を変える）
        for i in range(len(detected) - 1):
            row0 = detected.iloc[i]
            row1 = detected.iloc[i + 1]
            v_avg = (row0["velocity"] + row1["velocity"]) / 2
            lane  = int(row0["traffic_lane"])
            ls    = lane_styles.get(lane, "-")
    
            ax.plot(
                [row0["t_sec"],    row1["t_sec"]],
                [row0["kilopost"], row1["kilopost"]],
                color=cmap(norm(v_avg)),
                linewidth=0.8,
                linestyle=ls,
                alpha=0.7,
            )
    
        # 補間区間は薄いグレーで
        if len(interp) >= 2:
            ax.plot( interp["t_sec"],interp["kilopost"],
                    color="gray", linewidth=0.4, alpha=0.3, linestyle=":")
    
    ax.set_xlabel("時刻 (s)", fontsize=12)
    ax.set_ylabel("キロポスト (m)", fontsize=12)
    ax.set_ylim(df['kilopost'].min(),df['kilopost'].max())
    ax.set_title("車両軌跡（時間→横）\n色: 速度、線種: 車線", fontsize=10)
    ax.grid(axis="both", linewidth=0.3, alpha=0.4)
    
    # カラーバー（速度スケール）
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="速度 (km/h)", fraction=0.03, pad=0.01)
    
    # ── 4. 右パネル：速度ヒストグラム ────────────────────────
    ax2 = axes[1]
    detected_df = df[df["detected_flag"] == 1]
    ax2.hist(detected_df["velocity"], bins=40, orientation="horizontal",
            color="#378ADD", alpha=0.7, edgecolor="white", linewidth=0.3)
    ax2.set_xlabel("車両数", fontsize=11)
    ax2.set_title("速度分布\n(detected only)", fontsize=10)
    ax2.axhline(detected_df["velocity"].mean(), color="#D85A30",
                linewidth=1.2, linestyle="--", label=f"平均 {detected_df['velocity'].mean():.1f} km/h")
    ax2.legend(fontsize=9)
    ax2.grid(axis="x", linewidth=0.3, alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "time_space_diagram.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n保存: {OUTPUT_DIR / 'time_space_diagram.png'}")
    
 
    # ── 5. 急ブレーキ検出（R値分析の前処理）─────────────────
    """
    次ステップ: 急ブレーキイベントの抽出
    detected_flag==1 のみで 0.1秒ごとの減速度を計算する。
    """
    
    detected_only = df[df["detected_flag"] == 1].copy()
    detected_only = detected_only.sort_values(["vehicle_id", "t_sec"])
    
    # 同一vehicle_id内の連続レコード間で減速度を計算
    detected_only["dv"] = detected_only.groupby("vehicle_id")["velocity"].diff()
    detected_only["dt"] = detected_only.groupby("vehicle_id")["t_sec"].diff()
    
    # 時刻差が 0.2秒以内のペアのみ有効（連続レコードの保証）
    detected_only = detected_only[detected_only["dt"].between(0.001, 0.2)]
    
    # 減速度 m/s²（km/h → m/s に変換）
    detected_only["decel_ms2"] = (detected_only["dv"] / 3.6) / detected_only["dt"]
    
    # 急ブレーキ閾値: -2.9 m/s² (-0.3G)
    BRAKE_THRESHOLD = -2.9
    brake_events = detected_only[detected_only["decel_ms2"] <= BRAKE_THRESHOLD]
    
    print(f"\n=== 急ブレーキイベント（閾値: {BRAKE_THRESHOLD} m/s²） ===")
    print(f"  検出件数    : {len(brake_events)}")
    print(f"  関与車両数  : {brake_events['vehicle_id'].nunique()}")
    if len(brake_events) > 0:
        print(f"  平均減速度  : {brake_events['decel_ms2'].mean():.2f} m/s²")
        print(f"  最大減速度  : {brake_events['decel_ms2'].min():.2f} m/s²")
        print(brake_events[["vehicle_id", "t_sec", "kilopost", "traffic_lane",
                            "velocity", "decel_ms2"]].head(10).to_string(index=False))
    
    brake_events.to_csv(OUTPUT_DIR / "brake_events.csv", index=False)
    print(f"\n急ブレーキCSV保存: {OUTPUT_DIR / 'brake_events.csv'}")
    
    # ── 6. 後続車両の特定 ────────────────────────────────────
    """
    急ブレーキ車（Patient Zero）の直後にいる後続車を特定する。
    条件:
    - 同一車線（traffic_lane が同じ）
    - 急ブレーキ発生時刻に、急ブレーキ車より後方（kilopost が小さい）にいる
    - 車間距離が FOLLOW_DIST_MAX m 以内
    """
    
    FOLLOW_DIST_MAX = 150   # 追従とみなす最大車間距離 (m)
    RESPONSE_WINDOW = 5.0   # 急ブレーキ後、後続車の反応を待つ時間窓 (秒)
    RESPONSE_THRESHOLD = -2.0  # 後続車の「有意な減速」閾値 (m/s²)  ← 急ブレーキより緩め
    
    # detected_only を vehicle_id × t_sec でインデックス化（後で高速検索するため）
    detected_only_indexed = detected_only.set_index(["vehicle_id", "t_sec"])
    
    results = []  # R値計算の素材
    
    for _, brake in brake_events.iterrows():
        b_vid  = brake["vehicle_id"]
        b_t    = brake["t_sec"]
        b_kp   = brake["kilopost"]
        b_lane = brake["traffic_lane"]
    
        # 急ブレーキ発生時刻に同じ車線にいる他車両のスナップショットを取得
        # → b_t ± 0.5秒以内のレコードで代用
        snapshot = detected_only[
            (detected_only["t_sec"].between(b_t - 0.5, b_t + 0.5)) &
            (detected_only["traffic_lane"] == b_lane) &
            (detected_only["vehicle_id"]   != b_vid)
        ]
    
        # 後方かつ FOLLOW_DIST_MAX 以内の車両を抽出
        # （キロポストが大きい方向に走行している前提。逆の場合は不等号を反転）
        followers = snapshot[
            (snapshot["kilopost"] < b_kp) &
            (snapshot["kilopost"] >= b_kp - FOLLOW_DIST_MAX)
        ]
    
        if followers.empty:
            results.append({
                "brake_vid": b_vid, "brake_t": b_t, "brake_kp": b_kp,
                "lane": b_lane, "brake_decel": brake["decel_ms2"],
                "n_followers": 0, "n_infected": 0,
            })
            continue
    
        # 各後続車について、RESPONSE_WINDOW 秒以内に有意な減速があるかチェック
        n_infected = 0
        for f_vid in followers["vehicle_id"].unique():
            future = detected_only[
                (detected_only["vehicle_id"] == f_vid) &
                (detected_only["t_sec"].between(b_t, b_t + RESPONSE_WINDOW))
            ]
            if future.empty:
                continue
            # 最大減速度を確認
            if future["decel_ms2"].min() <= RESPONSE_THRESHOLD:
                n_infected += 1
    
        results.append({
            "brake_vid":   b_vid,
            "brake_t":     b_t,
            "brake_kp":    b_kp,
            "lane":        b_lane,
            "brake_decel": brake["decel_ms2"],
            "n_followers": followers["vehicle_id"].nunique(),
            "n_infected":  n_infected,
        })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "r_value_raw.csv", index=False)
    print(f"\n伝播結果CSV保存: {OUTPUT_DIR / 'r_value_raw.csv'}")
    
    # ── 7. R値の計算 ─────────────────────────────────────────
    """
    R = 1件の急ブレーキが引き起こす有意な減速の平均件数
    = sum(n_infected) / 急ブレーキイベント件数（後続車がいたものに限定）
    
    感染症のR値と同様:
    R > 1 → 渋滞が自己増幅していく状態
    R < 1 → 渋滞が自然収束していく状態
    R = 1 → 臨界点
    """
    
    # 後続車がいたイベントのみ対象
    valid = results_df[results_df["n_followers"] > 0]
    
    if len(valid) == 0:
        print("\n後続車ありのイベントなし。FOLLOW_DIST_MAX や時刻窓を広げてください。")
    else:
        R = valid["n_infected"].sum() / len(valid)
        R_median = valid["n_infected"].median()
        R_std    = valid["n_infected"].std()
    
        print(f"\n=== 交通 R 値（実効再生産数） ===")
        print(f"  急ブレーキイベント総数   : {len(results_df)}")
        print(f"  後続車あり（有効）件数   : {len(valid)}")
        print(f"  R 値（平均）             : {R:.3f}")
        print(f"  R 値（中央値）           : {R_median:.1f}")
        print(f"  標準偏差                 : {R_std:.3f}")
        print(f"  → {'渋滞が自己増幅する状態' if R > 1 else '渋滞が収束する状態'} (R {'>' if R > 1 else '<'} 1)")
    
        # ── 8. 感度分析（閾値ごとのR値変化） ────────────────────
        """
        閾値の恣意性を下げるため、ブレーキ閾値・反応閾値を変えてもR値の
        大小関係が変わらないことを確認する。
        """
        sensitivity = []
        for b_thresh in [-2.0, -2.9, -4.0]:
            for r_thresh in [-1.0, -2.0, -3.0]:
                b_ev = detected_only[detected_only["decel_ms2"] <= b_thresh]
                if len(b_ev) == 0:
                    continue
    
                sub_results = []
                for _, brake in b_ev.iterrows():
                    b_vid  = brake["vehicle_id"]
                    b_t    = brake["t_sec"]
                    b_kp   = brake["kilopost"]
                    b_lane = brake["traffic_lane"]
    
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
    
                    infected = 0
                    for f_vid in followers["vehicle_id"].unique():
                        future = detected_only[
                            (detected_only["vehicle_id"] == f_vid) &
                            (detected_only["t_sec"].between(b_t, b_t + RESPONSE_WINDOW))
                        ]
                        if not future.empty and future["decel_ms2"].min() <= r_thresh:
                            infected += 1
                    sub_results.append(infected)
    
                sub_r = np.mean(sub_results) if sub_results else np.nan
                sensitivity.append({
                    "brake_thresh (m/s²)":    b_thresh,
                    "response_thresh (m/s²)": r_thresh,
                    "n_events":               len(b_ev),
                    "R":                      round(sub_r, 3) if not np.isnan(sub_r) else "—",
                })
    
        sens_df = pd.DataFrame(sensitivity)
        print(f"\n=== 感度分析（閾値ごとのR値） ===")
        print(sens_df.to_string(index=False))
        sens_df.to_csv(OUTPUT_DIR / "sensitivity.csv", index=False)
    
        # ── 9. R値の可視化 ───────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle("交通R値 分析結果", fontsize=13)
    
        # 9-A. 伝播件数ヒストグラム
        ax = axes[0]
        ax.hist(valid["n_infected"], bins=range(0, valid["n_infected"].max() + 2),
                color="#378ADD", edgecolor="white", linewidth=0.5, align="left")
        ax.axvline(R, color="#D85A30", linewidth=2, linestyle="--",
                label=f"R = {R:.2f}")
        ax.axvline(1.0, color="#639922", linewidth=1.5, linestyle=":",
                label="R = 1 （臨界点）")
        ax.set_xlabel("1イベントあたりの伝播件数", fontsize=11)
        ax.set_ylabel("急ブレーキイベント数", fontsize=11)
        ax.set_title("伝播件数の分布", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(axis="y", linewidth=0.3, alpha=0.4)
    
        # 9-B. キロポスト別R値（空間分布）
        ax = axes[1]
        bin_size = 200  # m
        valid["kp_bin"] = (valid["brake_kp"] // bin_size * bin_size).astype(int)
        kp_r = valid.groupby("kp_bin")["n_infected"].mean()
        bar_colors = ["#D85A30" if v > 1 else "#378ADD" for v in kp_r.values]
        ax.bar(kp_r.index, kp_r.values, width=bin_size * 0.8,
            color=bar_colors, edgecolor="white", linewidth=0.4)
        ax.axhline(1.0, color="#639922", linewidth=1.5, linestyle=":",
                label="R = 1 （臨界点）")
        ax.set_xlabel("キロポスト (m)", fontsize=11)
        ax.set_ylabel("区間平均 R 値", fontsize=11)
        ax.set_title("空間分布（赤=R>1 危険区間）", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(axis="y", linewidth=0.3, alpha=0.4)
    
        # 9-C. 時間帯別R値（時系列）
        ax = axes[2]
        t_min = valid["brake_t"].min()
        t_max = valid["brake_t"].max()
        time_window = (t_max - t_min) / 6  # 全体を6分割
        if time_window > 0:
            valid["t_bin"] = ((valid["brake_t"] - t_min) // time_window).astype(int)
            t_r = valid.groupby("t_bin")["n_infected"].mean()
            t_labels = [f"{t_min + i * time_window:.0f}s" for i in t_r.index]
            ax.plot(t_r.index, t_r.values, marker="o", color="#378ADD",
                    linewidth=1.5, markersize=5)
            ax.fill_between(t_r.index, t_r.values, alpha=0.15, color="#378ADD")
            ax.axhline(1.0, color="#639922", linewidth=1.5, linestyle=":",
                    label="R = 1 （臨界点）")
            ax.set_xticks(t_r.index)
            ax.set_xticklabels(t_labels, rotation=30, fontsize=8)
            ax.set_xlabel("時間帯", fontsize=11)
            ax.set_ylabel("R 値", fontsize=11)
            ax.set_title("時間的変化", fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(linewidth=0.3, alpha=0.4)
    
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "r_value_analysis.png", dpi=150, bbox_inches="tight")
        plt.show()
        print(f"\nR値分析図保存: {OUTPUT_DIR / 'r_value_analysis.png'}")
    
        # ── 10. サマリーレポート出力 ─────────────────────────────
        summary = {
            "R値（平均）":         round(R, 3),
            "R値（中央値）":       R_median,
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


    # ── 11. R=1 境界分析：反応時間 × 車間距離のパラメータスキャン ──────
    """
    RESPONSE_WINDOW（後続車の反応時間上限）と FOLLOW_DIST_MAX（追従とみなす距離）を
    グリッド状に変化させ、それぞれの組み合わせで R 値を計算する。
    
    R=1 の等高線（臨界境界）を描くことで：
    - 「反応時間が X 秒 かつ 車間距離が Y m 以内なら渋滞が自己増幅する」
    という具体的な閾値を実データから導出できる。
    """
    
    print("\n=== R=1 境界分析（パラメータスキャン） ===")
    print("  反応時間 x 車間距離 のグリッドでR値を計算中...")
    
    TIME_WINDOWS = np.arange(1.0, 12.0, 1.0)   # 反応時間 (秒): 1〜11s
    DIST_LIMITS  = np.arange(50,  351,  25)      # 追従距離 (m) : 50〜350m
    
    R_GRID = np.full((len(TIME_WINDOWS), len(DIST_LIMITS)), np.nan)
    
    for i, tw in enumerate(TIME_WINDOWS):
        for j, dl in enumerate(DIST_LIMITS):
            scan_results = []
            for _, brake in brake_events.iterrows():
                b_vid  = brake["vehicle_id"]
                b_t    = brake["t_sec"]
                b_kp   = brake["kilopost"]
                b_lane = brake["traffic_lane"]
    
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
    
                infected = 0
                for f_vid in followers["vehicle_id"].unique():
                    future = detected_only[
                        (detected_only["vehicle_id"] == f_vid) &
                        (detected_only["t_sec"].between(b_t, b_t + tw))
                    ]
                    if not future.empty and future["decel_ms2"].min() <= RESPONSE_THRESHOLD:
                        infected += 1
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
            mark = f"{val:.2f}" if not np.isnan(val) else "  -- "
            print(f"  {mark:>5}", end="")
        print()
    
    boundary_df = pd.DataFrame(
        R_GRID,
        index=pd.Index(TIME_WINDOWS,  name="reaction_time_s"),
        columns=pd.Index(DIST_LIMITS, name="follow_dist_m"),
    )
    boundary_df.to_csv(OUTPUT_DIR / "r1_boundary.csv")
    print(f"\n境界CSV保存: {OUTPUT_DIR / 'r1_boundary.csv'}")
    
    # ── 12. R=1 境界の可視化 ─────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("R=1 臨界境界：反応時間 x 車間距離", fontsize=13)
    
    # 12-A. ヒートマップ
    ax = axes[0]
    im = ax.imshow(
        R_GRID,
        aspect="auto", origin="lower",
        extent=[DIST_LIMITS[0], DIST_LIMITS[-1], TIME_WINDOWS[0], TIME_WINDOWS[-1]],
        cmap="jet_r", vmin=0, vmax=2,
    )
    plt.colorbar(im, ax=ax, label="R 値")
    
    cs = ax.contour(
        DIST_LIMITS, TIME_WINDOWS, R_GRID,
        levels=[1.0], colors=["white"], linewidths=2.5, linestyles="--",
    )
    ax.clabel(cs, fmt="R=1 臨界線", fontsize=9, colors="white")
    ax.scatter(
        [FOLLOW_DIST_MAX], [RESPONSE_WINDOW],
        color="blue", s=80, zorder=5,
        label=f"デフォルト設定 ({FOLLOW_DIST_MAX}m, {RESPONSE_WINDOW}s)",
    )
    ax.set_xlabel("追従とみなす距離 (m)", fontsize=11)
    ax.set_ylabel("反応時間上限 (秒)", fontsize=11)
    ax.set_title("R値ヒートマップ\n赤=R>1 自己増幅域、緑=R<1 収束域", fontsize=10)
    ax.legend(fontsize=9, loc="lower right")
    
    # 12-B. R=1 境界線の断面
    ax2 = axes[1]
    boundary_dist = []
    for i, tw in enumerate(TIME_WINDOWS):
        row = R_GRID[i, :]
        crossing = np.where(row >= 1.0)[0]
        if len(crossing) > 0:
            j = crossing[0]
            if j > 0 and not np.isnan(row[j-1]):
                r0, r1 = row[j-1], row[j]
                d0, d1 = DIST_LIMITS[j-1], DIST_LIMITS[j]
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
            ax2.annotate(
                f"{dist:.0f}m", xy=(dist, tw), xytext=(dist + 8, tw),
                fontsize=8, color="#993C1D", va="center",
            )
    
    ax2.set_xlabel("R=1 となる車間距離の境界 (m)", fontsize=11)
    ax2.set_ylabel("反応時間 (秒)", fontsize=11)
    ax2.set_title(
        "R=1 臨界境界線\n右上=危険域（距離長・時間短でも伝播）\n左下=安全域",
        fontsize=10,
    )
    ax2.set_xlim(0, max(DIST_LIMITS) + 20)
    ax2.set_ylim(TIME_WINDOWS[0] - 0.5, TIME_WINDOWS[-1] + 0.5)
    ax2.legend(fontsize=9)
    ax2.grid(linewidth=0.3, alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "r1_boundary.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"境界分析図保存: {OUTPUT_DIR / 'r1_boundary.png'}")