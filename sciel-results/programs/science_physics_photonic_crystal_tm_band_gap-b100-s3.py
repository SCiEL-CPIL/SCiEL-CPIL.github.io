import math

def design(problem):
    size = problem["pixels"]
    cell = []
    # Define 4 air holes centered at (+/-0.25, +/-0.25)
    # Background is dielectric (1)
    r = 0.28
    centers = [
        (0.25, 0.25),
        (-0.25, 0.25),
        (0.25, -0.25),
        (-0.25, -0.25)
    ]
    for i in range(size):
        row = []
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            is_air = 0
            for cx, cy in centers:
                dx = x - cx
                dy = y - cy
                if dx * dx + dy * dy < r * r:
                    is_air = 1
                    break
            # 1 = Dielectric, 0 = Air
            # If is_air, pixel is 0. Else 1.
            row.append(0 if is_air else 1)
        cell.append(row)
    return cell
