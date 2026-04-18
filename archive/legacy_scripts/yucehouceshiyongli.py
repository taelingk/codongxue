# -*- coding: utf-8 -*-
import os
from pathlib import Path
import pandas as pd

# ========= 你只需要改这里 =========
IN_DIR = r"D:\心输出量项目\测试用例\测试用例-处理后的脉搏波数据和时间戳对齐的CNAP真实值"
SHEET_NAME = "切片汇总"
# =================================

EXCEL_EXTS = {".xlsx", ".xls", ".xlsm", ".xlsb"}

def export_one_excel(excel_path: Path) -> tuple[bool, str]:
    """
    返回 (是否成功导出, 信息)
    """
    try:
        # 读取该工作簿的 sheet 列表（不直接读全文件，先确认sheet存在）
        xls = pd.ExcelFile(excel_path)
        if SHEET_NAME not in xls.sheet_names:
            return (False, f"[跳过] {excel_path.name}：未找到工作表“{SHEET_NAME}”，实际有：{xls.sheet_names}")

        # 只读目标sheet
        df = pd.read_excel(excel_path, sheet_name=SHEET_NAME)

        # 可选：去掉完全空的行/列（通常表格会更干净）
        df = df.dropna(how="all").dropna(axis=1, how="all")

        # 输出 csv：同名 .csv
        out_csv = excel_path.with_suffix(".csv")
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")

        return (True, f"[OK] {excel_path.name} -> {out_csv.name}  (rows={len(df)}, cols={df.shape[1]})")

    except Exception as e:
        return (False, f"[失败] {excel_path.name}：{type(e).__name__}: {e}")

def main():
    in_dir = Path(IN_DIR)
    if not in_dir.exists():
        print(f"[ERROR] 路径不存在：{in_dir}")
        return

    # 遍历目录下所有文件（不递归；如果你需要递归可改为 rglob）
    files = [p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in EXCEL_EXTS]

    if not files:
        print(f"[WARN] 目录下未找到 Excel 文件：{in_dir}")
        return

    ok_cnt = 0
    skip_fail_cnt = 0

    print(f"=== 开始批量导出：目录={in_dir}，目标sheet={SHEET_NAME}，文件数={len(files)} ===")
    for p in sorted(files):
        ok, msg = export_one_excel(p)
        print(msg)
        if ok:
            ok_cnt += 1
        else:
            skip_fail_cnt += 1

    print("\n=== 完成 ===")
    print(f"成功导出：{ok_cnt}")
    print(f"跳过/失败：{skip_fail_cnt}")
    print("提示：导出的 .csv 已生成在同目录下，文件名与原文件一致。")

if __name__ == "__main__":
    main()
