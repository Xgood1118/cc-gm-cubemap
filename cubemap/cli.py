# -*- coding: utf-8 -*-
"""命令行入口：读取六面数据，对单个方向采样并写出结果。"""

import argparse
import json

import cubemap as m


def main():
    ap = argparse.ArgumentParser(description="立方体贴图接缝与过滤")
    ap.add_argument("data", help="六面数据 JSON 路径")
    ap.add_argument("--dir", required=True, help="方向向量 x,y,z")
    ap.add_argument("--out", required=True, help="采样结果输出 JSON 路径")
    a = ap.parse_args()

    with open(a.data, encoding="utf-8") as f:
        faces = json.load(f)
    cube = m.Cube(faces=faces)
    direction = tuple(float(s) for s in a.dir.split(","))

    result = m.sample(cube, direction)
    error = m.seam_error(cube, direction)

    payload = {
        "direction": list(direction),
        "face": result.face,
        "face_index": result.face_index,
        "uv": list(result.uv),
        "value": list(result.value),
        "cross_face_count": result.cross_face_count,
        "seam_error": error,
        "taps": [
            {"face": t.face, "uv": list(t.uv), "weight": t.weight}
            for t in result.taps
        ],
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(
        "face={face}({face_index}) uv=({u:.6f}, {v:.6f}) "
        "cross_face_taps={cross} seam_error={err:.6g}".format(
            face=result.face,
            face_index=result.face_index,
            u=result.uv[0],
            v=result.uv[1],
            cross=result.cross_face_count,
            err=error,
        )
    )


if __name__ == "__main__":
    main()
