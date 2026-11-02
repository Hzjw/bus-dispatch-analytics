"""超大 CSV 按行数切分工具（流式处理，内存友好）。

项目背景：原始公交刷卡流水约 1.2 GB / 1,596 万行，一次性加载会占用数 GB 内存，
在普通笔记本上容易 OOM。本工具用 pandas 的 chunksize 流式读取，边读边写，
内存占用只与分块大小有关。

用法：
    python src/tools/split_large_csv.py --input data/raw/bus_card_records.csv --max-rows 1000000
    python src/tools/split_large_csv.py -i big.csv -o data/raw/parts -r 500000

输出：
    <output-dir>/part_001.csv, part_002.csv, ... （每个文件带表头）
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def split_csv(input_csv: Path, output_dir: Path, max_rows: int, chunk_size: int, encoding: str) -> None:
    if not input_csv.exists():
        raise SystemExit(f"[缺少输入文件] {input_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"输入：{input_csv}  ({input_csv.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"规则：每份不超过 {max_rows:,} 行，输出到 {output_dir}")

    part_index = 1
    rows_in_part = 0
    total_rows = 0
    writer = None
    out_path = None

    def close_part():
        nonlocal writer
        if writer is not None:
            writer.close()
            writer = None

    # chunksize 控制单次读入内存的行数，与每份文件的切分粒度解耦
    for chunk in pd.read_csv(input_csv, chunksize=chunk_size, low_memory=False):
        offset = 0
        while offset < len(chunk):
            if rows_in_part >= max_rows:
                close_part()
                print(f"  已写出 {out_path.name}（{rows_in_part:,} 行）")
                part_index += 1
                rows_in_part = 0

            take = min(max_rows - rows_in_part, len(chunk) - offset)
            piece = chunk.iloc[offset : offset + take]

            if writer is None:
                out_path = output_dir / f"part_{part_index:03d}.csv"
                writer = open(out_path, "w", encoding=encoding, newline="")
                piece.to_csv(writer, index=False, header=True)
            else:
                # 后续片段沿用第一份的表头，header=False 避免重复
                piece.to_csv(writer, index=False, header=False)

            offset += take
            rows_in_part += take
            total_rows += take

    close_part()
    if out_path is not None:
        print(f"  已写出 {out_path.name}（{rows_in_part:,} 行）")

    print(f"\n完成：共 {part_index} 个文件，合计 {total_rows:,} 行")


def main() -> None:
    parser = argparse.ArgumentParser(description="超大 CSV 流式切分工具")
    parser.add_argument("-i", "--input", required=True, type=Path, help="输入 CSV 路径")
    parser.add_argument("-o", "--output-dir", type=Path, default=None, help="输出目录（默认与输入同级的 parts/）")
    parser.add_argument("-r", "--max-rows", type=int, default=1_000_000, help="每个文件最大行数（默认 100 万）")
    parser.add_argument("-c", "--chunk-size", type=int, default=200_000, help="单次读入行数（默认 20 万）")
    parser.add_argument(
        "-e", "--encoding", default="utf-8",
        help="输出编码（默认 utf-8；若要用 Excel 直接打开，可传 utf-8-sig）",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or (args.input.parent / "parts")
    split_csv(args.input, output_dir, args.max_rows, args.chunk_size, args.encoding)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\n已中断")
