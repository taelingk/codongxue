# -*- coding: utf-8 -*-
"""
Bland-Altman 一致性分析（预测 vs CNAP真实值）
- 读取真实值 REF_IN：按 5 秒窗生成真实值 5秒均值（ref_5s）
- 读取预测值 PRED_IN：按时间对齐到 5 秒网格
- 计算 Pearson r / MAE（更鲁棒）
- 绘制 Bland-Altman 图（CO 与 SV）
- 分析“误差随测量值大小变化趋势”：对 diff ~ mean 做线性回归（给 slope / r / p）
- 计算 95% 一致性界限：bias ± 1.96*SD

你只需要改最上面的路径/列名即可。
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Tuple
from scipy.stats import linregress

# =========================
# 你只需要改这里（路径 / 列名）
# =========================
REF_IN  = r"D:\心输出量项目\测试用例\测试用例-处理后的脉搏波数据和时间戳对齐的CNAP真实值\测试用例27.csv"
PRED_IN = r"D:\心输出量项目\测试用例\测试用例-用于计算皮尔逊系数和MAE\测试用例27.csv"
REF_OUT = r"D:\心输出量项目\测试用例\测试用例-真实值5秒均值\测试用例27-1.csv"

# 真实值文件：时间列
TIME_COL_REF = "起始时间"

# 真实值文件：CO / SV / 心率 列名（按你真实值表里实际列名来）
REF_CO_COL = "CO"
REF_SV_COL = "SV"
REF_HR_CANDIDATES = ["HR", "计算心率", "心率", "心率(次/分)"]

# 真实值文件：患者ID列（真实值可能叫 ID 或 患者ID）
REF_ID_CANDIDATES = ["患者ID", "ID", "patient_id", "PatientID"]

# 预测文件：时间列 + CO/SV/HR 列名（按你预测表头）
TIME_COL_PRED = "时间"
PRED_CO_COL   = "心输出量(L/min)"
PRED_SV_COL   = "每搏输出量(mL)"
PRED_HR_COL   = "心率(次/分)"   # 没有就自动跳过

STEP_SEC   = 5
PRED_ALIGN = "round"   # "round" or "floor"

# Bland-Altman 输出图保存路径（会自动生成）
BA_OUT_DIR = r"D:\心输出量项目\测试用例\BA图输出"
BA_PREFIX  = "测试用例27"

# 输出列名（固定成你要的那一套）
OUT_TYPE_COL = "数据类型"
OUT_ID_COL   = "患者ID"
OUT_TIME_COL = "时间"
OUT_CO_COL   = "心输出量(L/min)"
OUT_SV_COL   = "每搏输出量(mL)"
OUT_HR_COL   = "心率(次/分)"

# =========================
# 工具函数
# =========================
def resolve_existing_file(path: str) -> str:
    p = Path(path)
    if p.exists():
        return str(p)
    parent = p.parent
    if parent.exists():
        cand = sorted(parent.glob(p.stem + "*.csv"))
        if cand:
            print(f"[WARN] 找不到指定文件：{p}")
            print(f"[WARN] 自动改用：{cand[0]}")
            return str(cand[0])
    raise FileNotFoundError(f"文件不存在：{p}")

def read_table_auto(path: str) -> pd.DataFrame:
    path = str(path)
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    last_err = None
    for enc in ["utf-8-sig", "utf-8", "gbk", "latin-1"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"读取失败：{path}\n最后一次错误：{last_err}")

def pick_first_existing_col(df: pd.DataFrame, candidates: List[str], tag: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"[{tag}] 在列名 {candidates} 中找不到任何一个。当前列名：{list(df.columns)}")

_date_prefix = re.compile(r"^\s*\d{4}[-/]\d{1,2}[-/]\d{1,2}")

def parse_time_to_ns(x) -> Optional[int]:
    """
    支持:
    - 'YYYY-MM-DD HH:MM:SS(.xxx)' -> datetime
    - 'HH:MM:SS(.xxx)' / 'MM:SS(.xxx)' -> 解析成秒
    - 数字 -> 秒
    返回纳秒 int；失败返回 None
    """
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s == "" or s.lower() in ["nan", "none"]:
        return None

    # 可能含日期
    if _date_prefix.match(s):
        ts = pd.to_datetime(s, errors="coerce")
        if pd.isna(ts):
            return None
        return int(ts.value)

    # 仅时间
    parts = s.split(":")
    try:
        if len(parts) == 3:
            hh = float(parts[0]); mm = float(parts[1]); ss = float(parts[2])
            sec = hh * 3600 + mm * 60 + ss
        elif len(parts) == 2:
            mm = float(parts[0]); ss = float(parts[1])
            sec = mm * 60 + ss
        else:
            sec = float(s)
        return int(round(sec * 1e9))
    except ValueError:
        return None

def format_hhmmss(sec_int: int) -> str:
    sec_int = int(sec_int)
    hh = sec_int // 3600
    mm = (sec_int % 3600) // 60
    ss = sec_int % 60
    return f"{hh}:{mm:02d}:{ss:02d}"

def align_to_5s(sec_series: pd.Series, step: int = 5, method: str = "round") -> pd.Series:
    s = sec_series.astype(float)
    if method == "round":
        return (np.round(s / step) * step).astype(int)
    elif method == "floor":
        return (np.floor(s / step) * step).astype(int)
    else:
        raise ValueError("PRED_ALIGN 只能是 'round' 或 'floor'")

# =========================
# 真实值：生成 5 秒窗均值，只保留输出需要的列
# =========================
def build_5s_reference_mean(ref_df: pd.DataFrame,
                            time_col: str,
                            co_col: str,
                            sv_col: str,
                            hr_col: str,
                            id_col: Optional[str],
                            step_sec: int = 5) -> pd.DataFrame:
    df = ref_df.copy()

    tns = df[time_col].apply(parse_time_to_ns)
    first_valid = tns.dropna()
    if first_valid.empty:
        raise ValueError(f"真实值时间列 {time_col} 没有任何可解析的时间。")
    t0_ns = int(first_valid.iloc[0])

    df["_tns"] = tns
    df = df.dropna(subset=["_tns"]).copy()

    df[co_col] = pd.to_numeric(df[co_col], errors="coerce")
    df[sv_col] = pd.to_numeric(df[sv_col], errors="coerce")
    df[hr_col] = pd.to_numeric(df[hr_col], errors="coerce")
    df = df.dropna(subset=[co_col, sv_col, hr_col]).copy()

    df = df.sort_values("_tns").reset_index(drop=True)
    df = df[df["_tns"] >= t0_ns].reset_index(drop=True)
    if df.empty:
        raise ValueError("从真实值第一行时间开始没有数据。")

    step_ns = int(step_sec * 1e9)
    t_end_ns = int(df["_tns"].iloc[-1])
    targets = np.arange(t0_ns + step_ns, t_end_ns + 1, step_ns, dtype=np.int64)

    pid_value = None
    if id_col is not None and id_col in df.columns:
        s = df[id_col].dropna()
        pid_value = s.iloc[0] if not s.empty else None

    out_rows = []
    for tg_ns in targets:
        win = df[(df["_tns"] >= tg_ns) & (df["_tns"] < tg_ns + step_ns)]
        if win.empty:
            continue

        co_mean = float(win[co_col].mean())
        sv_mean = float(win[sv_col].mean())
        hr_mean = float(win[hr_col].mean())

        rel_sec = int((tg_ns - t0_ns) // 1_000_000_000)

        out_rows.append({
            OUT_TYPE_COL: "5秒均值",
            OUT_ID_COL: pid_value,
            OUT_TIME_COL: format_hhmmss(rel_sec),
            OUT_CO_COL: round(co_mean, 3),
            OUT_SV_COL: round(sv_mean, 3),
            OUT_HR_COL: round(hr_mean, 3),
            "_rel_sec": rel_sec
        })

    out = pd.DataFrame(out_rows)
    if out.empty:
        raise ValueError("生成的5秒均值为空：请检查时间列是否正确、数据是否完整。")

    return out

# =========================
# 更鲁棒的 Pearson / MAE 计算
# =========================
def pearson_r_safe(x, y) -> float:
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy()
    y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy()

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask].ravel()
    y = y[mask].ravel()

    if x.size < 2:
        return np.nan
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])

def mae_safe(x, y) -> float:
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy()
    y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy()
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask].ravel()
    y = y[mask].ravel()
    if x.size == 0:
        return np.nan
    return float(np.mean(np.abs(x - y)))

def merge_on_grid(ref: pd.DataFrame, pred: pd.DataFrame, ref_col: str, pred_col: str) -> pd.DataFrame:
    """
    用 _rel_sec_5 合并，并确保结果里有 'ref_val' 和 'pred_val'
    """
    a = ref[["_rel_sec_5", ref_col]].rename(columns={ref_col: "ref_val"})
    b = pred[["_rel_sec_5", pred_col]].rename(columns={pred_col: "pred_val"})
    m = pd.merge(a, b, on="_rel_sec_5", how="inner")
    m["ref_val"] = pd.to_numeric(m["ref_val"], errors="coerce")
    m["pred_val"] = pd.to_numeric(m["pred_val"], errors="coerce")
    m = m.dropna(subset=["ref_val", "pred_val"]).reset_index(drop=True)
    return m

def eval_metrics(ref_5s: pd.DataFrame, pred_df: pd.DataFrame, step_sec: int = 5, pred_align: str = "round") -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    返回:
      metrics(dict),
      m_co(包含 _rel_sec_5/ref_val/pred_val),
      m_sv(同上; 若无sv列则为空df)
    """
    # ref_5s 必需列
    for c in ["_rel_sec", OUT_CO_COL, OUT_SV_COL, OUT_HR_COL]:
        if c not in ref_5s.columns:
            raise KeyError(f"ref_5s 缺少列：{c}")

    pred = pred_df.copy()
    if TIME_COL_PRED not in pred.columns:
        raise KeyError(f"预测文件缺少时间列 '{TIME_COL_PRED}'，当前列：{list(pred.columns)}")
    if PRED_CO_COL not in pred.columns:
        raise KeyError(f"预测文件缺少 CO 列 '{PRED_CO_COL}'，当前列：{list(pred.columns)}")

    pred["_tns"] = pred[TIME_COL_PRED].apply(parse_time_to_ns)
    pred = pred.dropna(subset=["_tns"]).copy()
    pred["_tns"] = pred["_tns"].astype(np.int64)

    # 预测时间是否带日期：带日期 -> 相对化到预测第一行
    first_pred_raw = str(pred_df[TIME_COL_PRED].dropna().iloc[0]).strip()
    pred_is_datetime = bool(_date_prefix.match(first_pred_raw))
    if pred_is_datetime:
        pred_t0 = int(pred["_tns"].iloc[0])
        pred["_tsec"] = (pred["_tns"] - pred_t0) / 1e9
    else:
        pred["_tsec"] = pred["_tns"] / 1e9

    pred["_tsec_int"] = np.floor(pred["_tsec"]).astype(int)
    pred["_rel_sec_5"] = align_to_5s(pred["_tsec_int"], step=step_sec, method=pred_align)

    ref = ref_5s.copy()
    ref["_rel_sec_5"] = ref["_rel_sec"].astype(int)

    out = {}

    # CO
    m_co = merge_on_grid(ref, pred, OUT_CO_COL, PRED_CO_COL)
    out["n_matched_CO"] = int(len(m_co))
    out["CO_r"] = pearson_r_safe(m_co["ref_val"], m_co["pred_val"])
    out["CO_MAE"] = mae_safe(m_co["ref_val"], m_co["pred_val"])

    # SV（可选）
    m_sv = pd.DataFrame(columns=["_rel_sec_5", "ref_val", "pred_val"])
    if PRED_SV_COL in pred.columns:
        m_sv = merge_on_grid(ref, pred, OUT_SV_COL, PRED_SV_COL)
        out["n_matched_SV"] = int(len(m_sv))
        out["SV_r"] = pearson_r_safe(m_sv["ref_val"], m_sv["pred_val"])
        out["SV_MAE"] = mae_safe(m_sv["ref_val"], m_sv["pred_val"])
    else:
        out.update({"n_matched_SV": 0, "SV_r": np.nan, "SV_MAE": np.nan,
                    "SV_note": f"预测文件没有列 '{PRED_SV_COL}'，跳过SV指标。"})

    # HR（可选）
    if PRED_HR_COL in pred.columns:
        m_hr = merge_on_grid(ref, pred, OUT_HR_COL, PRED_HR_COL)
        out["n_matched_HR"] = int(len(m_hr))
        out["HR_r"] = pearson_r_safe(m_hr["ref_val"], m_hr["pred_val"])
        out["HR_MAE"] = mae_safe(m_hr["ref_val"], m_hr["pred_val"])
    else:
        out.update({"n_matched_HR": 0, "HR_r": np.nan, "HR_MAE": np.nan,
                    "HR_note": f"预测文件没有列 '{PRED_HR_COL}'，跳过心率指标。"})

    return out, m_co, m_sv

# =========================
# Bland-Altman
# =========================
def bland_altman_stats(ref, pred) -> dict:
    """
    ref/pred: array-like
    返回: n, bias, sd, loa_low/high, slope/intercept/r/p (diff~mean)
    """
    ref = pd.to_numeric(pd.Series(ref), errors="coerce").to_numpy(dtype=float)
    pred = pd.to_numeric(pd.Series(pred), errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(ref) & np.isfinite(pred)
    ref = ref[mask]; pred = pred[mask]
    n = int(ref.size)
    if n < 3:
        return {"n": n, "bias": np.nan, "sd": np.nan, "loa_low": np.nan, "loa_high": np.nan,
                "slope": np.nan, "intercept": np.nan, "trend_r": np.nan, "trend_p": np.nan}

    diff = pred - ref
    mean = (pred + ref) / 2.0

    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    loa_low = bias - 1.96 * sd
    loa_high = bias + 1.96 * sd

    lr = linregress(mean, diff)
    return {
        "n": n,
        "bias": bias,
        "sd": sd,
        "loa_low": float(loa_low),
        "loa_high": float(loa_high),
        "slope": float(lr.slope),
        "intercept": float(lr.intercept),
        "trend_r": float(lr.rvalue),
        "trend_p": float(lr.pvalue),
        "mean": mean,
        "diff": diff,
    }

def plot_bland_altman(ref, pred, title, unit, save_path: Optional[str] = None):
    st = bland_altman_stats(ref, pred)

    fig = plt.figure(figsize=(9, 6), dpi=120)
    ax = fig.add_subplot(111)

    if st["n"] < 3 or not np.isfinite(st["bias"]):
        ax.text(0.5, 0.5, f"{title}\n有效配对样本不足 (n={st['n']})",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_axis_off()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        return st

    mean = st["mean"]
    diff = st["diff"]

    ax.scatter(mean, diff, alpha=0.75)
    ax.axhline(st["bias"], linestyle="--")
    ax.axhline(st["loa_low"], linestyle="--")
    ax.axhline(st["loa_high"], linestyle="--")

    # 趋势线：diff ~ mean
    xs = np.linspace(np.min(mean), np.max(mean), 200)
    ys = st["slope"] * xs + st["intercept"]
    ax.plot(xs, ys, linewidth=2)

    ax.set_title(title)
    ax.set_xlabel(f"均值 ((预测+CNAP)/2) [{unit}]")
    ax.set_ylabel(f"差值 (预测-CNAP) [{unit}]")
    ax.grid(True, linestyle="--", alpha=0.5)

    ann = (
        f"n={st['n']}\n"
        f"bias={st['bias']:.4f}\n"
        f"SD={st['sd']:.4f}\n"
        f"95%LoA=[{st['loa_low']:.4f}, {st['loa_high']:.4f}]\n"
        f"trend(diff~mean): slope={st['slope']:.5f}, r={st['trend_r']:.3f}, p={st['trend_p']:.3g}"
    )
    ax.text(0.02, 0.98, ann, transform=ax.transAxes,
            ha="left", va="top",
            bbox=dict(boxstyle="round", alpha=0.2))

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return st

# =========================
# 主流程
# =========================
if __name__ == "__main__":
    REF_IN = resolve_existing_file(REF_IN)
    PRED_IN = resolve_existing_file(PRED_IN)

    # 1) 读真实值
    ref_df = read_table_auto(REF_IN)
    if TIME_COL_REF not in ref_df.columns:
        raise KeyError(f"真实值缺少时间列 '{TIME_COL_REF}'，当前列：{list(ref_df.columns)}")

    # 2) 自动选择真实值心率列、ID列
    ref_hr_col = pick_first_existing_col(ref_df, REF_HR_CANDIDATES, "真实值-心率列")
    ref_id_col = None
    for c in REF_ID_CANDIDATES:
        if c in ref_df.columns:
            ref_id_col = c
            break

    for c in [REF_CO_COL, REF_SV_COL]:
        if c not in ref_df.columns:
            raise KeyError(f"真实值缺少列 '{c}'，当前列：{list(ref_df.columns)}")

    # 3) 生成真实值5秒均值（内部含 _rel_sec）
    ref_5s = build_5s_reference_mean(
        ref_df=ref_df,
        time_col=TIME_COL_REF,
        co_col=REF_CO_COL,
        sv_col=REF_SV_COL,
        hr_col=ref_hr_col,
        id_col=ref_id_col,
        step_sec=STEP_SEC
    )

    # 4) 输出：只保留你要的列
    out_cols = [OUT_TYPE_COL, OUT_ID_COL, OUT_TIME_COL, OUT_CO_COL, OUT_SV_COL, OUT_HR_COL]
    out_df = ref_5s[out_cols].copy()
    out_df.to_csv(REF_OUT, index=False, encoding="utf-8-sig")
    print(f"[OK] 输出真实值(5秒均值) CSV: {REF_OUT}，行数={len(out_df)}")

    # 5) 读预测值并计算指标（同时返回对齐后的 m_co/m_sv 用于 BA）
    pred_df = read_table_auto(PRED_IN)
    metrics, m_co, m_sv = eval_metrics(ref_5s, pred_df, step_sec=STEP_SEC, pred_align=PRED_ALIGN)

    print("\n[Metrics]")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # 6) Bland-Altman：CO & SV
    Path(BA_OUT_DIR).mkdir(parents=True, exist_ok=True)
    co_fig_path = str(Path(BA_OUT_DIR) / f"{BA_PREFIX}_BA_CO.png")
    sv_fig_path = str(Path(BA_OUT_DIR) / f"{BA_PREFIX}_BA_SV.png")

    # CO: ref_val=CNAP, pred_val=预测
    co_stats = plot_bland_altman(
        ref=m_co["ref_val"],
        pred=m_co["pred_val"],
        title="CO Bland-Altman（预测 vs CNAP）",
        unit="L/min",
        save_path=co_fig_path
    )

    # SV（如果预测文件没有SV列，会是空df）
    if len(m_sv) >= 3:
        sv_stats = plot_bland_altman(
            ref=m_sv["ref_val"],
            pred=m_sv["pred_val"],
            title="SV Bland-Altman（预测 vs CNAP）",
            unit="mL/beat",
            save_path=sv_fig_path
        )
    else:
        sv_stats = {"n": int(len(m_sv)), "bias": np.nan, "sd": np.nan, "loa_low": np.nan, "loa_high": np.nan,
                    "slope": np.nan, "intercept": np.nan, "trend_r": np.nan, "trend_p": np.nan}
        print("\n[WARN] SV 可用配对点不足（或预测文件缺少SV列），未生成SV Bland-Altman 图。")

    # 7) 打印 Bland-Altman 关键结论（含趋势）
    print("\n[Bland-Altman Summary]")
    if np.isfinite(co_stats.get("bias", np.nan)):
        print(f"  CO: n={co_stats['n']}, bias={co_stats['bias']:.4f} L/min, "
              f"95%LoA=[{co_stats['loa_low']:.4f}, {co_stats['loa_high']:.4f}] L/min, "
              f"trend slope={co_stats['slope']:.5f}, r={co_stats['trend_r']:.3f}, p={co_stats['trend_p']:.3g}")
        print(f"  CO图已保存: {co_fig_path}")

    if np.isfinite(sv_stats.get("bias", np.nan)):
        print(f"  SV: n={sv_stats['n']}, bias={sv_stats['bias']:.4f} mL/beat, "
              f"95%LoA=[{sv_stats['loa_low']:.4f}, {sv_stats['loa_high']:.4f}] mL/beat, "
              f"trend slope={sv_stats['slope']:.5f}, r={sv_stats['trend_r']:.3f}, p={sv_stats['trend_p']:.3g}")
        print(f"  SV图已保存: {sv_fig_path}")

    # 8) 可选：弹窗展示（需要时取消注释）
    # plt.show()
    plt.close("all")
