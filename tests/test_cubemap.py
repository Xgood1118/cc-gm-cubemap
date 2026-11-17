# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cubemap as mod  # noqa: E402
from cubemap.core import FACE_ORDER, _direction_on_face  # noqa: E402

cubemap = mod


def read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def read_hex(path):
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


# ---------------------------------------------------------------- 夹具

N = 8


def _continuous_cube():
    """每个纹素值由其在立方体上的三维方向编码，保证跨面连续。"""
    faces = {}
    for name in FACE_ORDER:
        grid = []
        for r in range(N):
            row = []
            v = r / (N - 1)
            for c in range(N):
                u = c / (N - 1)
                d = _direction_on_face(name, u, v)
                row.append([d[0], d[1], d[2]])
            grid.append(row)
        faces[name] = grid
    return cubemap.Cube(faces=faces)


def _flat_cube():
    return cubemap.Cube(
        faces={name: [[[0, 0, 0]]] for name in FACE_ORDER}
    )


# ---------------------------------------------------------------- 坐标

def test_face_uv_covers_six():
    dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    faces = {cubemap.face_uv(d)[0] for d in dirs}
    assert len(faces) == 6
    assert set(FACE_ORDER) == faces


def test_face_uv_uv_range():
    import math
    for name in FACE_ORDER:
        # 每个面四个角方向都落在该面（按优先级规则）或其相邻面，
        # 但 UV 始终在 [0, 1] 内
        for u, v in ((0, 0), (1, 0), (0, 1), (1, 1)):
            d = _direction_on_face(name, u, v)
            face, (fu, fv) = cubemap.face_uv(d)
            assert face in FACE_ORDER
            assert 0.0 <= fu <= 1.0 and 0.0 <= fv <= 1.0
    # 面中心方向必然回到自身面且 UV 为 (0.5, 0.5)
    centers = {
        "px": (1, 0, 0), "nx": (-1, 0, 0),
        "py": (0, 1, 0), "ny": (0, -1, 0),
        "pz": (0, 0, 1), "nz": (0, 0, -1),
    }
    for name, d in centers.items():
        face, uv = cubemap.face_uv(d)
        assert face == name
        assert abs(uv[0] - 0.5) < 1e-12 and abs(uv[1] - 0.5) < 1e-12


def test_corner_directions_deterministic():
    # 面角方向：三分量绝对值相等，归属必须唯一且可复现
    corners = [
        (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
        (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1),
    ]
    results = [cubemap.face_uv(d) for d in corners]
    again = [cubemap.face_uv(d) for d in corners]
    assert results == again  # 确定性
    for d, (face, uv) in zip(corners, results):
        # x 优先级最高，面角方向恒归属 x 面
        assert face == ("px" if d[0] > 0 else "nx")
        assert all(0.0 <= t <= 1.0 for t in uv)
    # 面边方向（两分量绝对值相等）：优先级 x > y > z
    assert cubemap.face_uv((1, 1, 0))[0] == "px"
    assert cubemap.face_uv((0, 1, 1))[0] == "py"
    assert cubemap.face_uv((0, 0, 1))[0] == "pz"


def test_zero_vector():
    f, uv = cubemap.face_uv((0, 0, 0))
    assert f is not None and len(uv) == 2
    assert f == "px" and uv == (0.5, 0.5)
    # 零向量采样必须确定、不抛异常
    cube = _continuous_cube()
    r1 = cubemap.sample(cube, (0, 0, 0))
    r2 = cubemap.sample(cube, (0, 0, 0))
    assert r1.face == r2.face == "px"
    assert r1.value == r2.value
    assert cubemap.seam_error(cube, (0, 0, 0)) == 0.0


# ---------------------------------------------------------------- 过滤

def test_sample_returns_face_and_uv():
    cube = _flat_cube()
    r = cubemap.sample(cube, (0.6, 0.8, 0.0))
    assert r.face in FACE_ORDER
    assert r.face_index == FACE_ORDER.index(r.face)
    assert len(r.uv) == 2 and all(0.0 <= t <= 1.0 for t in r.uv)
    assert len(r.taps) == 9
    assert abs(sum(t.weight for t in r.taps) - 1.0) < 1e-12


def test_cross_face_taps_near_edge():
    cube = _continuous_cube()
    # 方向贴近 px / py 共享边（y 接近 x，z=0），核必然越界到相邻面
    r = cubemap.sample(cube, (1.0, 0.999, 0.0))
    assert r.face == "px"
    assert r.cross_face_count > 0
    assert any(t.face != r.face for t in r.taps)


def test_cross_face_consistency():
    # 连续数据上，同一方向从归属面与从相邻面（owner 覆盖）独立过滤，
    # 得到的值必须一致（方向空间核与面归属无关）
    cube = _continuous_cube()
    from cubemap.core import _filter

    directions = [
        (1.0, 0.999, 0.0),     # px / py 边
        (1.0, 0.0, 0.999),     # px / pz 边
        (-1.0, -0.999, 0.2),   # nx / ny 边附近
        (0.2, 1.0, 0.999),     # py / pz 边附近
        (0.6, 0.8, 0.0),       # CLI 同款方向
    ]
    for d in directions:
        owner, _ = cubemap.face_uv(d)
        ref = cubemap.sample(cube, d)
        for name in FACE_ORDER:
            if name == owner:
                continue
            other = _filter(cube, d, owner=name)
            for k in range(3):
                assert abs(ref.value[k] - other.value[k]) < 1e-12, d


def test_two_calls_identical():
    cube = _continuous_cube()
    for d in ((0.6, 0.8, 0.0), (1, 1, 1), (-0.3, 0.2, 0.9), (0, 0, 0)):
        r1 = cubemap.sample(cube, d)
        r2 = cubemap.sample(cube, d)
        assert r1 == r2
        assert cubemap.seam_error(cube, d) == cubemap.seam_error(cube, d)


def test_seam_error_detects_discontinuity():
    # 连续数据的接缝差为 0
    cube = _continuous_cube()
    edge_dir = (1.0, 1.0, 0.0)  # px/py 边
    assert cubemap.seam_error(cube, edge_dir) < 1e-12

    # 把 py 面整体染成不同常量，px/py 接缝应检出对应差值
    broken = {name: [row[:] for row in grid]
              for name, grid in cube.faces.items()}
    broken["py"] = [[[5.0, 5.0, 5.0]] * N for _ in range(N)]
    broken_cube = cubemap.Cube(faces=broken)
    err = cubemap.seam_error(broken_cube, edge_dir)
    assert err > 1.0
