import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    sqrt3 = np.sqrt(3)
    
    # Compact honeycomb cluster of 12 unit hexagons (axial coords for flat-top)
    axial_coords = [
        (-1, 0), (0, 0), (1, 0),
        (-2, 1), (-1, 1), (0, 1), (1, 1),
        (-1, 2), (0, 2), (1, 2),
        (-2, 3), (-1, 3), (0, 3)
    ]
    
    centers = np.array([[1.5 * q, sqrt3 * (r + q / 2)] for q, r in axial_coords])
    centers -= centers.mean(axis=0)
    
    # Precompute normals for the outer hexagon (flat-top)
    normals = np.array([[np.cos(k * np.pi / 3), np.sin(k * np.pi / 3)] for k in range(6)])
    
    def min_L_for_theta(theta):
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        rotated = centers @ R.T
        dots = rotated @ normals.T  # shape (12, 6)
        max_dot = dots.max()
        # Need max_dot <= L * sqrt(3) / 2  =>  L >= 2 * max_dot / sqrt(3)
        L = 2.0 * max_dot / sqrt3
        return L
    
    # Ternary search on theta in [0, pi/3)
    a, b = 0.0, np.pi / 3
    for _ in range(50):
        m1 = a + (b - a) / 3
        m2 = b - (b - a) / 3
        if min_L_for_theta(m1) < min_L_for_theta(m2):
            b = m2
        else:
            a = m1
    
    theta_opt = (a + b) / 2
    L_opt = min_L_for_theta(theta_opt)
    
    c, s = np.cos(theta_opt), np.sin(theta_opt)
    R = np.array([[c, -s], [s, c]])
    rotated = centers @ R.T
    
    inner = [[float(rotated[i, 0]), float(rotated[i, 1]), 0.0] for i in range(N)]
    
    return inner, (0.0, 0.0), float(L_opt), 0.0
# EVOLVE-BLOCK-END
