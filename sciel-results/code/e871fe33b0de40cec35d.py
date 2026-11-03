"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools
import time

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

def barycentric_to_cartesian(bary):
    """Convert barycentric coordinates (u, v) to Cartesian, where point = u*V1 + v*V2."""
    return bary[0] * V1 + bary[1] * V2

def cartesian_to_barycentric(p):
    """Convert Cartesian point to barycentric (u, v) where p = u*V1 + v*V2, u+v<=1."""
    # Solve: p = u*V1 + v*V2
    # p[0] = u * 1 + v * 0.5
    # p[1] = u * 0 + v * sqrt(3)/2
    v = p[1] / (np.sqrt(3) / 2)
    u = p[0] - v * 0.5
    return np.array([u, v])

def min_triangle_area(points):
    """Compute minimum triangle area among all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs((points[j][0] - points[i][0]) * (points[k][1] - points[i][1]) -
                                  (points[k][0] - points[i][0]) * (points[j][1] - points[i][1]))
                if area < min_area:
                    min_area = area
    return min_area

def negative_min_area_bary(x):
    """Objective: minimize negative min triangle area, using barycentric coords.
    x has shape (2*N,) representing N barycentric (u,v) pairs."""
    bary = x.reshape(-1, 2)
    # Clip to ensure valid barycentric: u>=0, v>=0, u+v<=1
    bary_clipped = np.clip(bary, 0, 1)
    # Ensure u+v<=1
    s = bary_clipped[:, 0] + bary_clipped[:, 1]
    mask = s > 1
    if np.any(mask):
        bary_clipped[mask] /= s[mask, np.newaxis]
    points = np.array([barycentric_to_cartesian(b) for b in bary_clipped])
    return -min_triangle_area(points)

def generate_grid_initialization(level=3, jitter_scale=0.01, seed=42):
    """Generate a structured grid of points in barycentric coordinates with jitter."""
    rng = np.random.RandomState(seed)
    bary_list = []
    # Generate grid points in barycentric space
    for i in range(level + 1):
        for j in range(level + 1 - i):
            u = i / level
            v = j / level
            if u + v <= 1.0:
                # Add small deterministic jitter
                ju = rng.uniform(-jitter_scale, jitter_scale)
                jv = rng.uniform(-jitter_scale, jitter_scale)
                bary_list.append([u + ju, v + jv])
    
    # If we have more than N points, randomly select N
    if len(bary_list) > N:
        indices = rng.choice(len(bary_list), size=N, replace=False)
        bary_list = [bary_list[i] for i in indices]
    elif len(bary_list) < N:
        # Pad with random points
        while len(bary_list) < N:
            u = rng.uniform(0, 1)
            v = rng.uniform(0, 1 - u)
            bary_list.append([u, v])
    
    return np.array(bary_list)

def generate_random_initialization(seed):
    """Generate random barycentric coordinates."""
    rng = np.random.RandomState(seed)
    bary_list = []
    for _ in range(N):
        u = rng.uniform(0, 1)
        v = rng.uniform(0, 1 - u)
        bary_list.append([u, v])
    return np.array(bary_list)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_points = None
    best_score = -float('inf')
    
    # Generate multiple diverse initializations
    initializations = []
    
    # Structured grid initializations with different levels and seeds
    for level in [2, 3, 4]:
        for seed in [42, 123, 7]:
            initializations.append(generate_grid_initialization(level=level, jitter_scale=0.01, seed=seed))
    
    # Random initializations
    for seed in [42, 123, 7, 999, 2024]:
        initializations.append(generate_random_initialization(seed))
    
    # Optimize each initialization
    for idx, bary_init in enumerate(initializations):
        if time.time() > deadline - 15.0:
            break
        
        x0 = bary_init.flatten()
        
        try:
            result = minimize(
                negative_min_area_bary,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True}
            )
        except Exception:
            continue
        
        if time.time() > deadline - 5.0:
            break
        
        # Convert back to Cartesian
        bary_opt = result.x.reshape(-1, 2)
        bary_opt = np.clip(bary_opt, 0, 1)
        s = bary_opt[:, 0] + bary_opt[:, 1]
        mask = s > 1
        if np.any(mask):
            bary_opt[mask] /= s[mask, np.newaxis]
        
        points = np.array([barycentric_to_cartesian(b) for b in bary_opt])
        
        score = min_triangle_area(points)
        if score > best_score:
            best_score = score
            best_points = points
    
    if best_points is None:
        # Fallback
        bary_fallback = generate_grid_initialization(level=3, jitter_scale=0.0, seed=42)
        best_points = np.array([barycentric_to_cartesian(b) for b in bary_fallback])
    
    return best_points
# EVOLVE-BLOCK-END
