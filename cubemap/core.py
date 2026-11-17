# -*- coding: utf-8 -*-
"""立方体贴图：方向空间无缝过滤采样器。

坐标约定（标准 OpenGL 立方体贴图方向）
--------------------------------------

方向向量 ``(x, y, z)`` 取绝对值最大的分量作为主轴，主轴符号决定正负面，
面内先算归一化坐标 ``sc, tc ∈ [-1, 1]``，再线性映射到 ``u, v ∈ [0, 1]``：

    u = (sc + 1) / 2,  v = (tc + 1) / 2

各面投影（ma 为该面主轴分量的绝对值）：

    px (+x): sc = -z / x,  tc = -y / x
    nx (-x): sc =  z / -x, tc = -y / -x
    py (+y): sc =  x / y,  tc =  z / y
    ny (-y): sc =  x / -y, tc = -z / -y
    pz (+z): sc =  x / z,  tc = -y / z
    nz (-z): sc = -x / -z, tc = -y / -z

网格按 ``grid[row][col]`` 存储：``row 0`` 对应 ``v = 0``（tc = -1），
``col 0`` 对应 ``u = 0``（sc = -1），每个元素是一个通道列表（典型为 RGB）。

边界归属
--------

- 两个分量绝对值相等（方向落在面边/面角）时，按固定轴优先级 ``x > y > z``
  归属，符号再决定正负面；因此任何方向都有唯一确定的面。
- 零向量 ``(0, 0, 0)`` 没有方向，约定归属 ``px`` 面中心 ``(u, v) = (0.5, 0.5)``，
  过滤时按 ``+x`` 方向处理，结果确定。

无缝过滤
--------

过滤核定义在三维方向空间而非面内：以单位方向 ``d`` 为中心构造正交切向量
``t1, t2``，在切平面上取 3×3 tent 核（权重 ``[1,2,1] ⊗ [1,2,1] / 16``），
步长为一个纹素（``1/cols``、``1/rows``）。每个 tap 归一化后独立经过
:func:`face_uv` 落到它所属的面，再做面内双线性插值。越过面边的 tap 会自动
落到相邻面上取真实纹素，而不是钳制在本面内。

因为同一方向生成的 tap 方向集合、权重完全相同，与中心方向归属哪个面无关，
所以同一方向在相邻两面的过滤结果严格一致（算法层面接缝误差上界为 0，
仅有浮点舍入差，量级不超过机器 epsilon）。
"""

from dataclasses import dataclass, field
from math import fabs, floor, sqrt

#: 面号 0..5 的固定顺序
FACE_ORDER = ("px", "nx", "py", "ny", "pz", "nz")
FACE_INDEX = {name: i for i, name in enumerate(FACE_ORDER)}

#: tent 核的偏移与单侧权重
_TAP_OFFSETS = (-1.0, 0.0, 1.0)
_TAP_WEIGHTS = (1.0, 2.0, 1.0)


@dataclass
class Cube:
    """六面数据，``faces`` 为 ``{面名: 二维通道网格}``。"""

    faces: dict


@dataclass(frozen=True)
class Tap:
    """单个过滤 tap 的落点信息。"""

    face: str
    uv: tuple
    weight: float


@dataclass(frozen=True)
class SampleResult:
    """采样结果。

    face        中心方向归属面名（见 :data:`FACE_ORDER`）
    face_index  面号 0..5
    uv          中心方向在归属面上的 (u, v)
    value       过滤后的通道值元组
    cross_face_count  落到归属面之外的 tap 数（跨面采样次数）
    taps        9 个 tap 的 (面, uv, 权重) 明细
    """

    face: str
    face_index: int
    uv: tuple
    value: tuple
    cross_face_count: int
    taps: tuple = field(default_factory=tuple)


# ---------------------------------------------------------------- 向量辅助

def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(a, k):
    return (a[0] * k, a[1] * k, a[2] * k)


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(a):
    return sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def _normalize(a):
    n = _norm(a)
    if n == 0.0:
        return a
    return (a[0] / n, a[1] / n, a[2] / n)


def _clamp01(x):
    return min(1.0, max(0.0, x))


# ---------------------------------------------------------------- 坐标投影

def _project(name, x, y, z):
    """按面约定把方向投影成 (sc, tc)，调用方需保证该面主轴分量非零。"""
    if name == "px":
        ma = x
        return -z / ma, -y / ma
    if name == "nx":
        ma = -x
        return z / ma, -y / ma
    if name == "py":
        ma = y
        return x / ma, z / ma
    if name == "ny":
        ma = -y
        return x / ma, -z / ma
    if name == "pz":
        ma = z
        return x / ma, -y / ma
    # nz
    ma = -z
    return -x / ma, -y / ma


def _try_project_uv(name, x, y, z):
    """投影成 (u, v)；主轴分量为 0 或符号与该面不符时返回 None。"""
    if name == "px":
        if x <= 0.0:
            return None
    elif name == "nx":
        if x >= 0.0:
            return None
    elif name == "py":
        if y <= 0.0:
            return None
    elif name == "ny":
        if y >= 0.0:
            return None
    elif name == "pz":
        if z <= 0.0:
            return None
    else:  # nz
        if z >= 0.0:
            return None
    sc, tc = _project(name, x, y, z)
    return _clamp01((sc + 1.0) * 0.5), _clamp01((tc + 1.0) * 0.5)


def _direction_on_face(name, u, v):
    """面内 (u, v) 反算单位立方面上的三维方向（未归一化）。"""
    sc = 2.0 * u - 1.0
    tc = 2.0 * v - 1.0
    if name == "px":
        return (1.0, -tc, -sc)
    if name == "nx":
        return (-1.0, -tc, sc)
    if name == "py":
        return (sc, 1.0, tc)
    if name == "ny":
        return (sc, -1.0, -tc)
    if name == "pz":
        return (sc, -tc, 1.0)
    # nz
    return (-sc, -tc, -1.0)


def face_uv(direction):
    """方向向量转 ``(面名, (u, v))``，覆盖全部六面，结果唯一确定。"""
    x, y, z = float(direction[0]), float(direction[1]), float(direction[2])

    # 零向量：约定 px 面中心
    if x == 0.0 and y == 0.0 and z == 0.0:
        return "px", (0.5, 0.5)

    ax, ay, az = fabs(x), fabs(y), fabs(z)
    # 平局按轴优先级 x > y > z 归属
    if ax >= ay and ax >= az:
        name = "px" if x >= 0.0 else "nx"
    elif ay >= az:
        name = "py" if y >= 0.0 else "ny"
    else:
        name = "pz" if z >= 0.0 else "nz"

    sc, tc = _project(name, x, y, z)
    u = _clamp01((sc + 1.0) * 0.5)
    v = _clamp01((tc + 1.0) * 0.5)
    return name, (u, v)


# ---------------------------------------------------------------- 面内插值

def _bilerp(grid, u, v):
    """面内双线性插值，返回通道元组；坐标钳制在 [0, 1]。

    网格标准形式为 ``grid[row][col] = [通道...]``；单纹素面允许简写为
    ``[[通道...]]``（一行即一个纹素，如 ``data/faces.json`` 的格式）。
    """
    rows = len(grid)
    first = grid[0]
    if first and not isinstance(first[0], (list, tuple)):
        # 单纹素面：grid = [[通道...]]，任意 UV 都取该纹素
        return tuple(float(c) for c in first)

    cols = len(first)
    fx = _clamp01(u) * (cols - 1)
    fy = _clamp01(v) * (rows - 1)
    c0 = int(floor(fx))
    r0 = int(floor(fy))
    c1 = min(c0 + 1, cols - 1)
    r1 = min(r0 + 1, rows - 1)
    wx = fx - c0
    wy = fy - r0
    p00 = grid[r0][c0]
    p10 = grid[r0][c1]
    p01 = grid[r1][c0]
    p11 = grid[r1][c1]
    w00 = (1.0 - wx) * (1.0 - wy)
    w10 = wx * (1.0 - wy)
    w01 = (1.0 - wx) * wy
    w11 = wx * wy
    return tuple(
        p00[k] * w00 + p10[k] * w10 + p01[k] * w01 + p11[k] * w11
        for k in range(len(p00))
    )


def _tangent_basis(d):
    """由单位方向确定唯一的正交切向量 (t1, t2)。"""
    up = (0.0, 1.0, 0.0) if fabs(d[1]) < 0.999999 else (1.0, 0.0, 0.0)
    t1 = _normalize(_cross(up, d))
    t2 = _cross(d, t1)
    return t1, t2


def _filter(cube, direction, owner=None):
    """方向空间 3×3 tent 过滤；owner 可指定归属面（测试用）。"""
    x, y, z = float(direction[0]), float(direction[1]), float(direction[2])
    if x == 0.0 and y == 0.0 and z == 0.0:
        # 零向量约定按 +x 方向过滤
        d = (1.0, 0.0, 0.0)
        if owner is None:
            owner = "px"
    else:
        d = _normalize((x, y, z))
        if owner is None:
            owner, _ = face_uv(direction)

    owner_grid = cube.faces[owner]
    owner_first = owner_grid[0]
    single_texel = bool(owner_first) and not isinstance(
        owner_first[0], (list, tuple)
    )
    if single_texel:
        step_u = step_v = 1.0
        channels = len(owner_first)
    else:
        step_u = 1.0 / len(owner_first)
        step_v = 1.0 / len(owner_grid)
        channels = len(owner_first[0])
    t1, t2 = _tangent_basis(d)

    taps = []
    acc = [0.0] * channels
    cross_face_count = 0

    for a, wa in zip(_TAP_OFFSETS, _TAP_WEIGHTS):
        for b, wb in zip(_TAP_OFFSETS, _TAP_WEIGHTS):
            weight = wa * wb / 16.0
            shifted = _add(
                d,
                _add(_scale(t1, a * step_u), _scale(t2, b * step_v)),
            )
            n = _norm(shifted)
            tap_dir = d if n < 1e-12 else _scale(shifted, 1.0 / n)
            tap_face, (tu, tv) = face_uv(tap_dir)
            value = _bilerp(cube.faces[tap_face], tu, tv)
            for k in range(channels):
                acc[k] += value[k] * weight
            if tap_face != owner:
                cross_face_count += 1
            taps.append(Tap(tap_face, (tu, tv), weight))

    _, center_uv = face_uv(direction)
    return SampleResult(
        face=owner,
        face_index=FACE_INDEX[owner],
        uv=center_uv,
        value=tuple(acc),
        cross_face_count=cross_face_count,
        taps=tuple(taps),
    )


def sample(cube, direction):
    """无缝过滤采样，返回 :class:`SampleResult`（含面号、UV、跨面次数）。"""
    return _filter(cube, direction)


def seam_error(cube, direction):
    """相邻面在最近接缝处的原始纹素不连续度（通道最大绝对差）。

    取方向归属面之外、投影越界距离最小的相邻面，把方向夹到该面边上得到
    接缝三维方向，再分别在两面的边上做面内双线性采样并比较。方向恰好落在
    面边上时，比较的就是该接缝本身；零向量返回 ``0.0``。

    注意这是*存储数据本身*的接缝差；无缝采样器在方向空间取 tap，同一方向
    从任一面计算都得到相同过滤值，不受该差异影响（一致性误差上界为 0）。
    """
    x, y, z = float(direction[0]), float(direction[1]), float(direction[2])
    if x == 0.0 and y == 0.0 and z == 0.0:
        return 0.0

    owner, _ = face_uv(direction)
    d = _normalize((x, y, z))

    best = None  # (越界距离, 相邻面, 夹到边上的 (u, v))
    for name in FACE_ORDER:
        if name == owner:
            continue
        uv = _try_project_uv(name, *d)
        if uv is None:
            excess, uv = float("inf"), (0.5, 0.5)
        else:
            u, v = uv
            excess = max(0.0, -u, u - 1.0, -v, v - 1.0)
        if best is None or excess < best[0]:
            best = (excess, name, uv)

    _, neighbor, (nu, nv) = best
    # 相邻面边上的点反算三维方向，再投影回归属面的对应边
    edge_direction = _direction_on_face(neighbor, nu, nv)
    owner_uv = _try_project_uv(owner, *edge_direction)
    if owner_uv is None:  # 数值兜底：理论上不会发生
        owner_uv = (nu, nv)

    value_owner = _bilerp(cube.faces[owner], owner_uv[0], owner_uv[1])
    value_neighbor = _bilerp(cube.faces[neighbor], nu, nv)
    return max(
        fabs(value_owner[k] - value_neighbor[k])
        for k in range(len(value_owner))
    )
