import numpy as np
from scipy.optimize import linprog
from itertools import combinations
import time

N = 12

def construct():
    start_time = time.time()
    deadline = 110.0 # 10s margin
    
    # Lattice generation
    def axial_to_cart(i, j):
        x = np.sqrt(3) * (i + j / 2.0)
        y = 1.5 * j
        return (x, y)

    core_axial = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
    ring2_axial = [
        (2, 0), (-2, 0), (0, 2), (0, -2),
        (2, -2), (-2, 2),
        (1, 1), (-1, -1),
        (2, -1), (-2, 1),
        (1, -2), (-1, 2)
    ]

    core_cart = [axial_to_cart(i, j) for i, j in core_axial]
    ring2_cart = [axial_to_cart(i, j) for i, j in ring2_axial]

    best_L = float('inf')
    best_result = None

    # Try inner orientations 0 and 30
    for inner_angle_deg in [0.0, 30.0]:
        inner_angle_rad = np.deg2rad(inner_angle_deg)
        
        # Pre-compute vertices for one hexagon at origin
        base_vertices = []
        for k in range(6):
            ang = inner_angle_rad + k * np.pi / 3
            base_vertices.append((np.cos(ang), np.sin(ang)))
        base_vertices = np.array(base_vertices)

        # Iterate over combinations of 5 from ring 2
        for combo in combinations(range(12), 5):
            if time.time() > deadline:
                break
                
            selected_ring2 = [ring2_cart[i] for i in combo]
            centers = core_cart + selected_ring2

            # Generate all vertices
            points = []
            for cx, cy in centers:
                for vx, vy in base_vertices:
                    points.append((cx + vx, cy + vy))
            points = np.array(points)

            # Optimize outer hex
            for theta_deg in np.arange(0, 60, 5):
                if time.time() > deadline:
                    break
                    
                theta = np.deg2rad(theta_deg)
                normals = []
                for i in range(6):
                    ang = theta + i * np.pi / 3
                    normals.append((np.cos(ang), np.sin(ang)))
                
                A_ub = []
                b_ub = []
                
                for nx, ny in normals:
                    for px, py in points:
                        A_ub.append([-nx, -ny, 1])
                        b_ub.append(-(nx * px + ny * py))
                
                A_ub.append([0, 0, -1])
                b_ub.append(0)

                A_ub = np.array(A_ub)
                b_ub = np.array(b_ub)
                c_obj = np.array([0, 0, 1])

                try:
                    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, method='highs')
                    if res.success:
                        r = res.x[2]
                        L = r * 2 / np.sqrt(3)
                        if L < best_L:
                            best_L = L
                            cx_out, cy_out = res.x[0], res.x[1]
                            inner = [[x, y, inner_angle_deg] for x, y in centers]
                            best_result = (inner, (cx_out, cy_out), L, theta_deg)
                except:
                    pass

    if best_result is None:
        # Fallback: return something valid, e.g., the seed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
        
    return best_result
