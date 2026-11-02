def _project_to_triangle(pts):
    y = pts[:, 1]
    x = pts[:, 0]
    y = np.clip(y, 0.0, S3H)
    x_min = y / S3
    x_max = 1 - y / S3
    x = np.clip(x, x_min, x_max)
    pts[:, 0] = x
    pts[:, 1] = y
    return pts
