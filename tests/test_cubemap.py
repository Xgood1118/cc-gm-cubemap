# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cubemap as mod  # noqa: E402


def test_face_uv_covers_six():
    dirs = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    faces = {cubemap.face_uv(d)[0] for d in dirs}
    assert len(faces) == 6


def test_zero_vector():
    f, uv = cubemap.face_uv((0, 0, 0))
    assert f is not None and len(uv) == 2
