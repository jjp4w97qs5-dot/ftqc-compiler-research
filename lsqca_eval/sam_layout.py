from __future__ import annotations

"""SAM内の論理qubit配置に使う座標関数。"""

import math


def sam_coord(sam_type: str, local_index: int) -> tuple[int, int]:
    """bank内の0-based indexをLine-SAM/Point-SAM座標へ変換する。"""
    if local_index < 0:
        raise ValueError("local_index must be non-negative")
    if sam_type == "line-sam":
        row = used = 0
        while True:
            capacity = 2 * (row + 1)
            if local_index < used + capacity:
                return row, local_index - used
            used += capacity
            row += 1
    if sam_type == "point-sam":
        # 既存実験と同じ配置。port cell (0,0)を空けるため1つずらす。
        index = local_index + 1
        side = int(math.floor(math.sqrt(index)))
        rem = index - side * side
        return (side, rem) if rem < side else (rem - side, side)
    raise ValueError(f"Unknown sam_type: {sam_type}")
