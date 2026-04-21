# -*- coding: utf-8 -*-
"""
批量评估：多病人(多文件)合并计算 Pearson / MAE + Bland-Altman(含95%一致性界限、误差趋势)
新增：真实值(CNAP)中 CO/SV 同时为空的时间段(5秒窗)剔除，预测值对应窗也自动剔除

修复重点（你遇到的 duration_ref 巨大问题）：
1) 时间解析不再“先 pd.to_datetime”，避免把纯数字/Excel小数误判成时间戳
2) 支持“Excel小数天(0~1)自动 *86400”“毫秒(ms)自动 /1000”
3) 起点 t0 不再用简单 min；自动识别并跳过少量异常小值(如 0)，从主时间簇开始归一化
4) pred/ref 都统一做归一化（从0开始），避免 bin_end 对不齐导致 merge 失败

✅ 本次按你的要求新增/修改（尽量只动这两处）：
A) 修复 “59分到 1:00.0” 跨小时回绕导致的时间算错：
   - 对冒号格式解析出的“秒序列”，若出现明显回绕（从接近3600突然变小），自动 +3600 纠正
B) 预测值/真实值：第一行起始时间从 5 秒开始，之后都是 5 的倍数（通过 bin_end 规则保证）
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress


# =========================
# 你只需要改这里（路径）
# =========================
REF_DIR = r"D:\心输出量项目\测试用例\测试用例-处理后的脉搏波数据和时间戳对齐的CNAP真实值"
PRED_DIR = r"D:\心输出量项目\测试用例\测试用例-用于计算皮尔逊系数和MAE"   # ←预测值文件夹
REF5S_OUT_DIR = r"D:\心输出量项目\测试用例\测试用例-真实值5秒均值"
RESULT_DIR = r"D:\心输出量项目\测试用例\测试用例-评估结果_全体合并"
# =========================


# =========================
# 参数
# =========================
STEP_SEC = 5
DURATION_TOL_SEC = 30.0   # 总时长允许差(秒)
DPI = 200
# =========================


def read_csv_robust(path: str) -> pd.DataFrame:
    encodings = ["utf-8-sig", "gbk", "utf-8", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, engine="python", on_bad_lines="skip")
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except Exception as e:
            last_err = e
    raise RuntimeError(f"读取失败: {path}\n最后错误: {last_err}")


def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    norm_map = {str(x).strip().lower(): x for x in df.columns}
    for c in candidates:
        key = c.strip().lower()
        if key in norm_map:
            return norm_map[key]
    return None


# -------------------------
# 1) 单值时间解析：更保守、更可控
# -------------------------
def parse_time_value(x) -> float:
    """
    返回：秒(float)，解析不了返回 np.nan
    支持：
    - "H:MM:SS" / "MM:SS(.ms)" / "SS(.ms)"(数字)
    - 带单位：ms/毫秒、s/秒
    - 日期时间：2025-01-01 12:00:00.123（返回 epoch 秒，后续会统一减去起点变成相对秒）
    - Excel 小数天：一整列都是 0~1 且跨度很小 -> 在序列级别再 *86400（这里先按数字返回）
    """
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "":
        return np.nan

    s2 = s.strip().strip("[](){}")
    low = s2.lower()
    is_ms = ("ms" in low) or ("毫秒" in s2)

    # 去掉单位字样（ms已提前判定）
    for token in ["秒", "sec", "s"]:
        s2 = s2.replace(token, "")
    s2 = s2.replace("毫秒", "").replace("ms", "")
    s2 = s2.strip()

    looks_like_datetime = (
        ("-" in s2 or "/" in s2 or "年" in s2) and (":" in s2)
    ) or ("T" in s2 and ":" in s2)

    if looks_like_datetime:
        dt = pd.to_datetime(s2, errors="coerce")
        if pd.isna(dt):
            return np.nan
        return float(dt.value / 1e9)

    if ":" in s2:
        parts = s2.split(":")
        try:
            if len(parts) == 3:
                h = float(parts[0]); m = float(parts[1]); sec = float(parts[2])
                val = h * 3600 + m * 60 + sec
            elif len(parts) == 2:
                m = float(parts[0]); sec = float(parts[1])
                val = m * 60 + sec
            else:
                return np.nan
            return float(val)
        except Exception:
            return np.nan

    try:
        v = float(s2)
        if is_ms:
            return float(v / 1000.0)
        return float(v)
    except Exception:
        return np.nan


# -------------------------
# ✅ A) 序列级：修复“跨小时回绕”(59:xx -> 1:00.x) 导致的时间倒退
# -------------------------
def fix_wraparound_seconds(t_sec: np.ndarray, wrap: float = 3600.0) -> np.ndarray:
    """
    对“本应单调递增”的秒序列，修复跨小时回绕造成的倒退：
    - 若出现 t[i] 明显小于 t[i-1]，并且差值接近 3600 的量级，则认为发生回绕
    - 对回绕后的所有点累计 +3600（可多次回绕）

    注意：这里只对“冒号解析出来的秒”（或其序列）生效也没关系；
    但我们在 normalize 前统一做一遍，不影响 epoch 秒（不会出现这种回绕特征）。
    """
    t = np.asarray(t_sec, dtype=float).copy()
    if t.size == 0:
        return t

    offset = 0.0
    prev = np.nan
    for i in range(t.size):
        if not np.isfinite(t[i]):
            continue
        cur = t[i] + offset
        if np.isfinite(prev):
            # 出现倒退：例如 3599.x -> 60.x
            if cur < prev - 1e-6:
                # 倒退幅度很大且接近一小时：就加一小时
                # （阈值取 wrap*0.5，是为了只捕获“跨小时”这种大跳变）
                if (prev - cur) > wrap * 0.5:
                    offset += wrap
                    cur = t[i] + offset
        t[i] = cur
        prev = cur
    return t


# -------------------------
# 2) 序列级时间归一化：原逻辑保留
# -------------------------
def normalize_time_series_to_seconds(t_raw: np.ndarray, step_sec: int, name: str = "") -> Tuple[np.ndarray, Dict[str, float]]:
    """
    输入：t_raw(秒 or epoch秒 or Excel小数天的“数字原样”)
    输出：t_norm（从0开始的秒），以及 info（包含scale、t0、剔除数量等）
    """
    t = np.asarray(t_raw, dtype=float)
    m = np.isfinite(t)
    info = {
        "scale": 1.0,
        "t0": np.nan,
        "excel_day_fraction": 0.0,
        "removed_negative": 0.0,
        "skipped_low_outlier": 0.0,
    }
    if m.sum() == 0:
        return t, info

    t_valid = t[m]
    t_min = float(np.min(t_valid))
    t_max = float(np.max(t_valid))
    t_range = float(t_max - t_min)

    # A) 识别 Excel 小数天
    if (t_min >= 0.0) and (t_max <= 2.0) and (t_range < 1.0) and (t_range * 86400.0 > max(10.0, 2 * step_sec)):
        t = t * 86400.0
        info["scale"] = 86400.0
        info["excel_day_fraction"] = 1.0
        m = np.isfinite(t)
        t_valid = t[m]

    # B) 稳健起点 t0
    t_sorted = np.sort(np.unique(t_valid))
    t0 = float(t_sorted[0])
    skipped = 0

    if t_sorted.size >= 2:
        gap01 = float(t_sorted[1] - t_sorted[0])
        gaps = np.diff(t_sorted)
        pos_gaps = gaps[gaps > 0]
        med_gap = float(np.median(pos_gaps)) if pos_gaps.size > 0 else 0.0

        big_thresh = max(60.0, 200.0 * med_gap) if med_gap > 0 else 60.0
        if gap01 > big_thresh:
            t0 = float(t_sorted[1])
            skipped = 1

        if skipped == 0 and t_sorted.size >= 3:
            check_k = min(5, len(gaps))
            for i in range(check_k):
                if gaps[i] > big_thresh:
                    if t_sorted[i] <= 1.0:
                        t0 = float(t_sorted[i + 1])
                        skipped = i + 1
                    break

    info["t0"] = t0
    info["skipped_low_outlier"] = float(skipped)

    t_norm = t - t0

    neg_mask = np.isfinite(t_norm) & (t_norm < -1e-6)
    info["removed_negative"] = float(np.sum(neg_mask))
    t_norm[neg_mask] = np.nan

    return t_norm, info


def sec_to_hms_str(sec: float) -> str:
    sec_i = int(round(sec))
    if sec_i < 0:
        sec_i = 0
    h = sec_i // 3600
    m = (sec_i % 3600) // 60
    s = sec_i % 60
    return f"{h}:{m:02d}:{s:02d}"


def to_bin_end_ceiling(t_sec: np.ndarray, step: int = 5) -> np.ndarray:
    t = np.asarray(t_sec, dtype=float)
    b = np.ceil(t / step) * step
    # ✅ B) 强制从 5 秒开始（第一行起始时间=5s；后面全是5的倍数）
    b[b < step] = step
    return b.astype(int)


def duration_seconds(t_sec: np.ndarray) -> float:
    t = np.asarray(t_sec, dtype=float)
    t = t[np.isfinite(t)]
    if t.size == 0:
        return float("nan")
    return float(np.max(t) - np.min(t))


def safe_pearson(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size < 2:
        return float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(pearsonr(x, y)[0])


def mae(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y[m] - x[m])))


def bland_altman_plot(ref: np.ndarray, pred: np.ndarray, title: str, unit: str, out_path: str) -> Dict[str, float]:
    ref = np.asarray(ref, dtype=float)
    pred = np.asarray(pred, dtype=float)
    m = np.isfinite(ref) & np.isfinite(pred)
    ref = ref[m]
    pred = pred[m]

    mean_val = (ref + pred) / 2.0
    diff = pred - ref

    bias = float(np.mean(diff)) if diff.size else float("nan")
    sd = float(np.std(diff, ddof=1)) if diff.size > 1 else float("nan")
    loa_low = bias - 1.96 * sd if np.isfinite(sd) else float("nan")
    loa_high = bias + 1.96 * sd if np.isfinite(sd) else float("nan")

    if diff.size >= 2:
        lr = linregress(mean_val, diff)
        slope = float(lr.slope)
        intercept = float(lr.intercept)
        r2 = float(lr.rvalue ** 2)
        pval = float(lr.pvalue)
    else:
        slope = intercept = r2 = pval = float("nan")

    plt.figure(figsize=(8, 5.5))
    plt.scatter(mean_val, diff, s=12, alpha=0.7)
    plt.axhline(bias, linestyle="--", linewidth=1)
    if np.isfinite(loa_low):
        plt.axhline(loa_low, linestyle="--", linewidth=1)
    if np.isfinite(loa_high):
        plt.axhline(loa_high, linestyle="--", linewidth=1)

    if diff.size >= 2 and np.isfinite(slope) and np.isfinite(intercept):
        x_line = np.linspace(np.min(mean_val), np.max(mean_val), 200)
        y_line = slope * x_line + intercept
        plt.plot(x_line, y_line, linewidth=1.5)

    plt.title(title)
    plt.xlabel(f"Mean of (Reference, Pred) [{unit}]")
    plt.ylabel(f"Pred - Reference [{unit}]")
    plt.grid(True, alpha=0.3)

    txt = (
        f"n={diff.size}\n"
        f"bias={bias:.4g} {unit}\n"
        f"SD={sd:.4g} {unit}\n"
        f"LoA=[{loa_low:.4g}, {loa_high:.4g}] {unit}\n"
        f"slope={slope:.4g}, R^2={r2:.4g}, p={pval:.4g}"
    )
    plt.gcf().text(0.73, 0.25, txt, fontsize=9, va="top")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI)
    plt.close()

    return {
        "n": float(diff.size),
        "bias": bias,
        "sd": sd,
        "loa_low": loa_low,
        "loa_high": loa_high,
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "pval": pval,
    }


def main():
    os.makedirs(REF5S_OUT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    ref_files = sorted(Path(REF_DIR).glob("*.csv"))
    pred_files = sorted(Path(PRED_DIR).glob("*.csv"))

    ref_map = {f.stem: str(f) for f in ref_files}
    pred_map = {f.stem: str(f) for f in pred_files}
    common_stems = sorted(set(ref_map.keys()) & set(pred_map.keys()))

    if not common_stems:
        print("[ERROR] 两个文件夹没有同名CSV(stem)可配对。请检查路径/文件名。")
        return

    all_matched_rows = []
    kept_pairs = 0
    dropped_pairs = 0

    for stem in common_stems:
        ref_path = ref_map[stem]
        pred_path = pred_map[stem]

        try:
            df_ref = read_csv_robust(ref_path)
            df_pred = read_csv_robust(pred_path)
        except Exception as e:
            print(f"[DROP] 读取失败，跳过: {stem}\n  {e}")
            dropped_pairs += 1
            continue

        # ---------- 预测值列 ----------
        pred_time_col = pick_col(df_pred, ["时间", "time"])
        pred_id_col = pick_col(df_pred, ["患者ID", "patient_id", "ID"])
        pred_co_col = pick_col(df_pred, ["心输出量(L/min)", "CO", "co", "心输出量"])
        pred_sv_col = pick_col(df_pred, ["每搏输出量(mL)", "SV", "sv", "每搏输出量"])
        pred_hr_col = pick_col(df_pred, ["心率(次/分)", "HR", "hr", "心率"])

        # ---------- 真实值列 ----------
        ref_time_col = pick_col(df_ref, ["起始时间", "时间", "start_time", "time"])
        ref_co_col = pick_col(df_ref, ["CO", "co", "心输出量", "心输出量(L/min)"])
        ref_sv_col = pick_col(df_ref, ["SV", "sv", "每搏输出量", "每搏输出量(mL)"])
        ref_hr_col = pick_col(df_ref, ["HR", "hr", "计算心率", "心率"])

        if pred_time_col is None or pred_co_col is None or pred_sv_col is None:
            print(f"[DROP] 预测值列名缺失，跳过: {stem}")
            print(f"  pred_time={pred_time_col}, pred_co={pred_co_col}, pred_sv={pred_sv_col}")
            dropped_pairs += 1
            continue

        if ref_time_col is None or ref_co_col is None or ref_sv_col is None:
            print(f"[DROP] 真实值列名缺失，跳过: {stem}")
            print(f"  ref_time={ref_time_col}, ref_co={ref_co_col}, ref_sv={ref_sv_col}")
            dropped_pairs += 1
            continue

        # ---------- 解析时间（序列级归一化） ----------
        # 原来：apply(parse_time_value) -> normalize
        # ✅ 新增：先做跨小时回绕修复（只会在出现明显倒退时起作用）
        pred_t_raw = df_pred[pred_time_col].apply(parse_time_value).astype(float).values
        pred_t_raw = fix_wraparound_seconds(pred_t_raw, wrap=3600.0)
        pred_t_norm, pred_info = normalize_time_series_to_seconds(pred_t_raw, STEP_SEC, name="pred")

        ref_t_raw = df_ref[ref_time_col].apply(parse_time_value).astype(float).values
        ref_t_raw = fix_wraparound_seconds(ref_t_raw, wrap=3600.0)
        ref_t_norm, ref_info = normalize_time_series_to_seconds(ref_t_raw, STEP_SEC, name="ref")

        dur_pred = duration_seconds(pred_t_norm)
        dur_ref = duration_seconds(ref_t_norm)

        if (not np.isfinite(dur_pred)) or (not np.isfinite(dur_ref)) or (abs(dur_pred - dur_ref) > DURATION_TOL_SEC):
            print(f"[DROP] 时间总长不一致(容差{DURATION_TOL_SEC}s): {stem}")
            print(f"  duration_pred={dur_pred:.3f}s, duration_ref={dur_ref:.3f}s, diff={abs(dur_pred-dur_ref):.3f}s")
            print(f"  pred_info: scale={pred_info['scale']}, t0={pred_info['t0']:.3f}, "
                  f"excel_day={int(pred_info['excel_day_fraction'])}, skipped_low={int(pred_info['skipped_low_outlier'])}, "
                  f"removed_neg={int(pred_info['removed_negative'])}")
            print(f"  ref_info : scale={ref_info['scale']}, t0={ref_info['t0']:.3f}, "
                  f"excel_day={int(ref_info['excel_day_fraction'])}, skipped_low={int(ref_info['skipped_low_outlier'])}, "
                  f"removed_neg={int(ref_info['removed_negative'])}")
            dropped_pairs += 1
            continue

        # ---------- 预测：5秒分箱 ----------
        dfp = df_pred.copy()
        dfp["_tsec_raw"] = dfp[pred_time_col].apply(parse_time_value).astype(float)
        # ✅ 同步做回绕修复（保证逐行一致）
        dfp["_tsec_raw"] = fix_wraparound_seconds(dfp["_tsec_raw"].values, wrap=3600.0)

        dfp["_tsec"] = dfp["_tsec_raw"] * float(pred_info["scale"]) - float(pred_info["t0"])
        dfp.loc[dfp["_tsec"] < 0, "_tsec"] = np.nan
        dfp = dfp.dropna(subset=["_tsec"])
        dfp["_bin_end"] = to_bin_end_ceiling(dfp["_tsec"].values, STEP_SEC)

        dfp[pred_co_col] = pd.to_numeric(dfp[pred_co_col], errors="coerce")
        dfp[pred_sv_col] = pd.to_numeric(dfp[pred_sv_col], errors="coerce")
        if pred_hr_col is not None:
            dfp[pred_hr_col] = pd.to_numeric(dfp[pred_hr_col], errors="coerce")

        agg_pred = {pred_co_col: "mean", pred_sv_col: "mean"}
        if pred_hr_col is not None:
            agg_pred[pred_hr_col] = "mean"
        if pred_id_col is not None:
            agg_pred[pred_id_col] = "first"

        pred5 = dfp.groupby("_bin_end", as_index=False).agg(agg_pred).sort_values("_bin_end")

        # ---------- 真实：5秒分箱 ----------
        dfr = df_ref.copy()
        dfr["_tsec_raw"] = dfr[ref_time_col].apply(parse_time_value).astype(float)
        # ✅ 同步做回绕修复（保证逐行一致）
        dfr["_tsec_raw"] = fix_wraparound_seconds(dfr["_tsec_raw"].values, wrap=3600.0)

        dfr["_tsec"] = dfr["_tsec_raw"] * float(ref_info["scale"]) - float(ref_info["t0"])
        dfr.loc[dfr["_tsec"] < 0, "_tsec"] = np.nan
        dfr = dfr.dropna(subset=["_tsec"])
        dfr["_bin_end"] = to_bin_end_ceiling(dfr["_tsec"].values, STEP_SEC)

        dfr[ref_co_col] = pd.to_numeric(dfr[ref_co_col], errors="coerce")
        dfr[ref_sv_col] = pd.to_numeric(dfr[ref_sv_col], errors="coerce")
        if ref_hr_col is not None:
            dfr[ref_hr_col] = pd.to_numeric(dfr[ref_hr_col], errors="coerce")

        agg_ref = {ref_co_col: "mean", ref_sv_col: "mean"}
        if ref_hr_col is not None:
            agg_ref[ref_hr_col] = "last"

        # ✅ 真实值：每 5 秒取均值（你要的“5秒取均值”就在这里）
        ref5 = dfr.groupby("_bin_end", as_index=False).agg(agg_ref).sort_values("_bin_end")

        # ✅ 剔除真实值中 CO 和 SV 同时为空的 5 秒窗
        before_n = len(ref5)
        ref5 = ref5.dropna(subset=[ref_co_col, ref_sv_col], how="all").copy()
        removed = before_n - len(ref5)

        if ref5.empty:
            print(f"[DROP] 真实值5秒分箱后 CO/SV 全为空（或被剔除为空白段），跳过: {stem}")
            dropped_pairs += 1
            continue

        # ---------- 合并：ref5 与 pred5 按 bin_end inner join ----------
        ref_cols = ["_bin_end", ref_co_col, ref_sv_col] + ([ref_hr_col] if ref_hr_col else [])
        pred_cols = ["_bin_end", pred_co_col, pred_sv_col] + ([pred_id_col] if pred_id_col else [])
        merged = pd.merge(ref5[ref_cols], pred5[pred_cols], on="_bin_end", how="inner")

        if merged.empty:
            print(f"[DROP] 合并后无匹配点(可能预测缺这些bin): {stem}")
            dropped_pairs += 1
            continue

        # 患者ID
        if pred_id_col is not None and merged[pred_id_col].notna().any():
            patient_id = str(merged[pred_id_col].dropna().iloc[0])
        else:
            patient_id = stem

        # ---------- 输出真实值5秒均值CSV（只输出剔除后的 ref5） ----------
        out_ref5 = pd.DataFrame({
            "数据类型": ["5秒均值"] * len(ref5),
            "患者ID": [patient_id] * len(ref5),
            # ✅ 第一行会是 0:00:05，后面都是 5 的倍数
            "时间": [sec_to_hms_str(x) for x in ref5["_bin_end"].values],
            "心输出量(L/min)": ref5[ref_co_col].values,
            "每搏输出量(mL)": ref5[ref_sv_col].values,
            "心率(次/分)": ref5[ref_hr_col].values if ref_hr_col else np.nan,
        })

        out_path = str(Path(REF5S_OUT_DIR) / f"{stem}.csv")
        out_ref5.to_csv(out_path, index=False, encoding="utf-8-sig")

        # ---------- 收集用于全体合并评估 ----------
        merged_eval = pd.DataFrame({
            "stem": stem,
            "患者ID": patient_id,
            "bin_end_sec": merged["_bin_end"].astype(int),
            "CO_ref": merged[ref_co_col].astype(float),
            "CO_pred": merged[pred_co_col].astype(float),
            "SV_ref": merged[ref_sv_col].astype(float),
            "SV_pred": merged[pred_sv_col].astype(float),
        })
        all_matched_rows.append(merged_eval)

        kept_pairs += 1
        print(f"[KEEP] {stem} | matched_points={len(merged_eval)} | removed_blank_bins={removed} | out_ref5={out_path}")
        print(f"       duration_pred={dur_pred:.3f}s, duration_ref={dur_ref:.3f}s | "
              f"pred_skipped_low={int(pred_info['skipped_low_outlier'])}, ref_skipped_low={int(ref_info['skipped_low_outlier'])}")

    print("\n======================")
    print(f"配对总数: {len(common_stems)}")
    print(f"保留(通过检查): {kept_pairs}")
    print(f"剔除: {dropped_pairs}")
    print("======================\n")

    if kept_pairs == 0:
        print("[ERROR] 没有任何文件通过检查，无法进行合并评估。")
        return

    big = pd.concat(all_matched_rows, ignore_index=True)
    big_path = str(Path(RESULT_DIR) / "ALL_MERGED_MATCHED_5S.csv")
    big.to_csv(big_path, index=False, encoding="utf-8-sig")
    print(f"[OK] 已保存全体合并匹配数据: {big_path}")

    # ---------- 指标 ----------
    co_r = safe_pearson(big["CO_ref"].values, big["CO_pred"].values)
    co_mae = mae(big["CO_ref"].values, big["CO_pred"].values)
    sv_r = safe_pearson(big["SV_ref"].values, big["SV_pred"].values)
    sv_mae = mae(big["SV_ref"].values, big["SV_pred"].values)

    print("========== 合并(所有人)指标 ==========")
    print(f"CO: n={np.sum(np.isfinite(big['CO_ref']) & np.isfinite(big['CO_pred']))} | r={co_r:.4f} | MAE={co_mae:.4f} L/min")
    print(f"SV: n={np.sum(np.isfinite(big['SV_ref']) & np.isfinite(big['SV_pred']))} | r={sv_r:.4f} | MAE={sv_mae:.4f} mL/beat")
    print("=====================================\n")

    # ---------- Bland-Altman ----------
    ba_co_path = str(Path(RESULT_DIR) / "BlandAltman_CO.png")
    ba_sv_path = str(Path(RESULT_DIR) / "BlandAltman_SV.png")

    co_stats = bland_altman_plot(
        ref=big["CO_ref"].values,
        pred=big["CO_pred"].values,
        title="Bland-Altman (CO): Pred vs CNAP Reference",
        unit="L/min",
        out_path=ba_co_path,
    )
    sv_stats = bland_altman_plot(
        ref=big["SV_ref"].values,
        pred=big["SV_pred"].values,
        title="Bland-Altman (SV): Pred vs CNAP Reference",
        unit="mL/beat",
        out_path=ba_sv_path,
    )

    print("[OK] Bland-Altman 图已保存：")
    print(f"  {ba_co_path}")
    print(f"  {ba_sv_path}\n")

    stats_df = pd.DataFrame([
        {"metric": "CO", **co_stats, "pearson_r": co_r, "mae": co_mae},
        {"metric": "SV", **sv_stats, "pearson_r": sv_r, "mae": sv_mae},
    ])
    stats_path = str(Path(RESULT_DIR) / "SUMMARY_STATS.csv")
    stats_df.to_csv(stats_path, index=False, encoding="utf-8-sig")
    print(f"[OK] 汇总统计已保存: {stats_path}")

    print("\n========== 95%一致性界限(LoA) & 误差趋势 ==========")
    print(f"CO LoA: [{co_stats['loa_low']:.4f}, {co_stats['loa_high']:.4f}] L/min | bias={co_stats['bias']:.4f}")
    print(f"CO error~mean: slope={co_stats['slope']:.6f}, R^2={co_stats['r2']:.4f}, p={co_stats['pval']:.4g}")
    print(f"SV LoA: [{sv_stats['loa_low']:.4f}, {sv_stats['loa_high']:.4f}] mL/beat | bias={sv_stats['bias']:.4f}")
    print(f"SV error~mean: slope={sv_stats['slope']:.6f}, R^2={sv_stats['r2']:.4f}, p={sv_stats['pval']:.4g}")
    print("=================================================\n")


if __name__ == "__main__":
    main()
