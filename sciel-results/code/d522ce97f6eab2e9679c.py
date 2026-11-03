def compute_outer_side(centers, angles, ocx, ocy, otheta):
    """Compute minimum outer hexagon side for given configuration."""
    max_proj = 0.0
    for k in range(6):
        normal_angle = otheta + math.pi/6 + k * math.pi/3
        nx = math.cos(normal_angle)
        ny = math.sin(normal_angle)
        for i in range(len(centers)):
            cx, cy = centers[i]
            angle = angles[i]
            for vi in range(6):
                vert_angle = angle + vi * math.pi / 3
                vx = cx + math.cos(vert_angle)
                vy = cy + math.sin(vert_angle)
                proj = (vx - ocx) * nx + (vy - ocy) * ny
                if proj > max_proj:
                    max_proj = proj
    return 2.0 * max_proj / math.sqrt(3)
