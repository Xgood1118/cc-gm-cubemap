# 立方体贴图接缝与过滤

一个能在六个面上做无缝过滤、并保证接缝采样一致的采样器

## 保留签名

- `Cube`
- `face_uv(direction)`
- `sample(cube, direction)`
- `seam_error(cube, direction)`

## 自测

```
python -m pytest tests -q
python -m cubemap.cli data/faces.json --dir 0.6,0.8,0 --out sample.json
```
