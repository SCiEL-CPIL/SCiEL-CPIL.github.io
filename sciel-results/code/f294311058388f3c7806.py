import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Initialization: Hexagonal Lattice
    def generate_hex_lattice(spacing):
        points = []
        k = int(1.0 / spacing) + 2
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = i * spacing + 0.5 * spacing * (j % 2)
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x*x + y*y <= 1.0:
                    points.append([x, y])
        return np.array(points) if points else None

    init_points = None
    # Try different spacings to find a good lattice density
    # The optimal radius is ~0.1258, so diameter ~0.2516.
    # Spacing should be around 0.25.
    for sp in [0.24, 0.25, 0.26, 0.23, 0.22]:
        pts = generate_hex_lattice(sp)
        if pts is not None and len(pts) >= N:
            # Select N points randomly from the lattice points inside the circle
            idx = np.random.choice(len(pts), N, replace=False)
            init_points = pts[idx]
            break
            
    if init_points is None:
        # Fallback to random
        theta = np.random.uniform(0, 2 * np.pi, N)
        r = np.sqrt(np.random.uniform(0, 1, N))
        init_points = np.column_stack((r * np.cos(theta), r * np.sin(theta)))

    points = init_points
    
    # Objective Function
    def compute_radius(pts):
        # Boundary
        d_o = np.linalg.norm(pts, axis=1)
        r_b = np.min(1.0 - d_o)
        
        # Pairwise
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        r_p = np.min(dists) / 2.0
        
        return min(r_b, r_p)

    curr_radius = compute_radius(points)
    best_points = points.copy()
    best_radius = curr_radius
    
    # SA Parameters
    T = 0.01
    T_min = 1e-8
    cooling_rate = 0.9999
    step_size = 0.02
    
    no_improve_count = 0
    max_no_improve = 10000
    
    while time.time() < deadline:
        k = np.random.randint(0, N)
        delta = np.random.normal(0, step_size, 2)
        
        old_pos = points[k].copy()
        points[k] += delta
        
        new_radius = compute_radius(points)
        
        delta_r = new_radius - curr_radius
        
        if delta_r > 0 or np.random.rand() < np.exp(delta_r / T):
            curr_radius = new_radius
            if curr_radius > best_radius:
                best_radius = curr_radius
                best_points = points.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            points[k] = old_pos
            no_improve_count += 1
            
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
        # Reheat if stuck
        if no_improve_count > max_no_improve:
            T = 0.005  # Reheat
            step_size = 0.04 # Larger steps to jump out of local minima
            no_improve_count = 0
            
    return best_points
