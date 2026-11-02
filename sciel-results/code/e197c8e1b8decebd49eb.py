"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # area of the equilateral triangle with side 1

DEADLINE = 145  # safety margin from 150s limit

# Precompute all triplets for N=11 points (indices into the 11 points)
_TRIPLETS = list(itertools.combinations(range(N), 3))
_N_TRIPLETS = len(_TRIPLETS)  # 165

# Precompute index arrays for vectorized area computation
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])


def _vectorized_min_area(pts):
    """Vectorized computation of min triangle area using precomputed triplets."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return np.min(areas)


def _project_to_triangle(pts):
    """Project points back into the triangle if they are outside."""
    for i in range(len(pts)):
        x, y = pts[i]
        y = max(0.0, min(S3H, y))
        x_min = y / S3
        x_max = 1 - y / S3
        x = max(x_min, min(x_max, x))
        pts[i] = [x, y]
    return pts


def _is_inside_triangle(p):
    """Check if point p is inside or on the equilateral triangle."""
    x, y = p
    if y < -1e-9 or y > S3H + 1e-9:
        return False
    if x < -1e-9 or x > 1 + 1e-9:
        return False
    left_viol = y - S3 * x
    right_viol = y + S3 * x - S3
    if left_viol > 1e-9 or right_viol > 1e-9:
        return False
    return True


def _all_inside(pts):
    """Check if all points are inside the triangle."""
    for p in pts:
        if not _is_inside_triangle(p):
            return False
    return True


def _random_point_in_triangle(rng):
    """Generate a random point inside the triangle using barycentric coordinates."""
    u = rng.random()
    v = rng.random()
    if u + v > 1:
        u = 1 - u
        v = 1 - v
    a = 1 - u - v
    b = u
    c = v
    x = a * 0 + b * 1 + c * 0.5
    y = a * 0 + b * 0 + c * S3H
    return np.array([x, y])


def _perturb_points(pts, rng, scale):
    """Perturb points by adding Gaussian noise, then project back to triangle."""
    new_pts = pts.copy()
    noise = rng.normal(0, scale, size=(N, 2))
    new_pts += noise
    new_pts = _project_to_triangle(new_pts)
    return new_pts


def _simulated_annealing(rng, start_pts, max_time=120):
    """Run simulated annealing to maximize min triangle area."""
    start_time = time.time()
    
    # Initial configuration
    current_pts = start_pts.copy()
    current_area = _vectorized_min_area(current_pts)
    
    best_pts = current_pts.copy()
    best_area = current_area
    
    # Annealing schedule
    T_start = 0.01  # initial temperature (in area units)
    T_end = 1e-6    # final temperature
    alpha = 0.9995  # cooling rate
    
    T = T_start
    scale_start = 0.05  # initial perturbation scale
    scale_end = 0.001   # final perturbation scale
    
    step = 0
    while time.time() - start_time < max_time:
        step += 1
        
        # Generate perturbation
        scale = scale_start * (scale_end / scale_start) ** (step / max(1, int(120 / (T_start - T_end) * 100)))
        scale = max(scale, scale_end)
        
        new_pts = _perturb_points(current_pts, rng, scale)
        new_area = _vectorized_min_area(new_pts)
        
        # Metropolis acceptance criterion
        delta = new_area - current_area
        if delta > 0:
            # Always accept improvement
            current_pts = new_pts
            current_area = new_area
            if current_area > best_area:
                best_area = current_area
                best_pts = current_pts.copy()
        else:
            # Accept worse solution with probability exp(delta / T)
            if T > 1e-10:
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    current_pts = new_pts
                    current_area = new_area
        
        # Cool down
        T *= alpha
        
        # Periodic restart: if stuck, try a larger kick
        if step % 5000 == 0 and T > 0.001:
            # Large perturbation to escape local minimum
            kick_pts = _perturb_points(current_pts, rng, 0.1)
            kick_area = _vectorized_min_area(kick_pts)
            if kick_area > current_area:
                current_pts = kick_pts
                current_area = kick_area
    
    return best_pts, best_area


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_area = 0.0
    
    # Multiple independent SA runs with different initial configurations
    n_runs = 15
    for run in range(n_runs):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Different initial strategies
        if run == 0:
            # Start from a symmetric-ish configuration
            # Place points at vertices, edge midpoints, and centroid
            pts = np.array([
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, S3H],
                [0.5, 0.0],
                [0.25, S3H/2],
                [0.75, S3H/2],
                [0.5, S3H/3],
                [0.333, S3H/3],
                [0.667, S3H/3],
                [0.167, S3H/6],
                [0.833, S3H/6]
            ])
            pts = _project_to_triangle(pts)
        elif run == 1:
            # Grid-like initial configuration
            xs = np.linspace(0.1, 0.9, 3)
            ys = np.linspace(0.05, S3H - 0.05, 4)
            pts = []
            for i in range(N):
                x = xs[i % 3]
                y = ys[i // 3]
                pts.append([x, y])
            pts = np.array(pts)
            pts = _project_to_triangle(pts)
        else:
            # Random initial configuration
            pts = np.array([_random_point_in_triangle(rng) for _ in range(N)])
        
        # Run simulated annealing
        sa_pts, sa_area = _simulated_annealing(rng, pts, max_time=8)
        
        if sa_area > best_area:
            best_area = sa_area
            best_pts = sa_pts.copy()
    
    # Final polish: run one more SA from the best solution with longer time
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5 and best_pts is not None:
        sa_pts, sa_area = _simulated_annealing(rng, best_pts, max_time=DEADLINE - 5 - elapsed)
        if sa_area > best_area:
            best_area = sa_area
            best_pts = sa_pts.copy()
    
    # Ensure validity
    if best_pts is not None:
        best_pts = _project_to_triangle(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
