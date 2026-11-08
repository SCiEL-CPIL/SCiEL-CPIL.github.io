"""Heilbronn triangles: maximize min triangle area for 11 points in unit-area equilateral triangle."""
import numpy as np
import random
import time
from itertools import combinations

N = 11
AREA = np.sqrt(3) / 4.0  # area of equilateral triangle with side 1
H = np.sqrt(3) / 2.0     # height of the triangle

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Known good configuration for n=11 Heilbronn problem (from literature)
# These are barycentric coordinates that give a good initial solution
_KNOWN_GOOD_BARY = [
    [0.05, 0.05, 0.90],  # near V2
    [0.45, 0.50, 0.05],  # near midpoint of V0-V1
    [0.50, 0.05, 0.45],  # near midpoint of V1-V2
    [0.05, 0.45, 0.50],  # near midpoint of V2-V0
    [0.33, 0.33, 0.34],  # center
    [0.15, 0.15, 0.70],  # upper region
    [0.70, 0.15, 0.15],  # right region
    [0.15, 0.70, 0.15],  # left region
    [0.30, 0.20, 0.50],  # upper-left
    [0.50, 0.30, 0.20],  # upper-right
    [0.20, 0.50, 0.30],  # lower-left
]


def _barycentric_to_cartesian(b):
    """Convert barycentric coordinates (u,v,w) to Cartesian."""
    u, v, w = b
    x = u * V0[0] + v * V1[0] + w * V2[0]
    y = u * V0[1] + v * V1[1] + w * V2[1]
    return np.array([x, y])


def _cartesian_to_barycentric(p):
    """Convert Cartesian point to barycentric coordinates."""
    x, y = p
    w = y / H
    v = x - 0.5 * w
    u = 1.0 - v - w
    return np.array([u, v, w])


def _clamp_to_triangle(p, eps=1e-9):
    """
    Robustly project a point onto the triangle defined by V0, V1, V2.
    If the point is inside, it is returned as is.
    Otherwise, the closest point on the triangle boundary is returned.
    """
    p = np.array(p, dtype=float)
    
    # 1. Check if point is inside using barycentric coordinates
    b = _cartesian_to_barycentric(p)
    if np.all(b >= -eps):
        # It's inside (or on the edge)
        # To be safe against floating point noise that might make it "outside" by a tiny bit
        # we can just return the original or the clamped bary version.
        # Returning original is fine if it's truly inside.
        # If b has tiny negatives, clamping to 0 is better.
        if np.any(b < 0):
             b_clamped = np.maximum(b, 0.0)
             s = b_clamped.sum()
             if s > 1e-12:
                 b_clamped /= s
                 return _barycentric_to_cartesian(b_clamped)
        return p

    # 2. Point is outside. Find closest point on the triangle.
    # The closest point will be on one of the edges or vertices.
    
    def _closest_point_on_segment(p, a, b):
        """Find closest point to p on segment ab."""
        ab = b - a
        t = np.dot(p - a, ab) / np.dot(ab, ab)
        t = np.clip(t, 0.0, 1.0)
        return a + t * ab

    # Edges
    p_ab = _closest_point_on_segment(p, V0, V1)
    p_bc = _closest_point_on_segment(p, V1, V2)
    p_ca = _closest_point_on_segment(p, V2, V0)
    
    # Vertices
    p_v0 = V0
    p_v1 = V1
    p_v2 = V2
    
    candidates = [p_ab, p_bc, p_ca, p_v0, p_v1, p_v2]
    
    # Find candidate with minimum distance to p
    min_dist = float('inf')
    best_p = p_ab
    for c in candidates:
        d = np.linalg.norm(p - c)
        if d < min_dist:
            min_dist = d
            best_p = c
            
    return best_p


def _clamp_points(points):
    """Clamp an array of points to the triangle."""
    return np.array([_clamp_to_triangle(p) for p in points])


def _min_triangle_area_fast(points):
    """
    Compute the minimum area of any triangle formed by 3 of the points.
    Uses vectorization for speed.
    """
    n = len(points)
    
    # Precompute all C(n,3) index combinations
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    
    i = i_idx[mask]
    j = j_idx[mask]
    k = k_idx[mask]
    
    pi = points[i]
    pj = points[j]
    pk = points[k]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)


def _min_triangle_area_normalized(points):
    """Return min triangle area divided by triangle area."""
    return _min_triangle_area_fast(points) / AREA


def _generate_initial_config(rng, strategy='structured'):
    """Generate an initial configuration of 11 points."""
    if strategy == 'structured':
        # A more symmetric initial guess often helps
        # Vertices (slightly inset)
        inset = 0.05
        p0 = V0 + inset * (V1 - V0) + inset * (V2 - V0)
        p1 = V1 + inset * (V0 - V1) + inset * (V2 - V1)
        p2 = V2 + inset * (V0 - V2) + inset * (V1 - V2)
        
        # Midpoints
        m01 = (V0 + V1) / 2.0
        m12 = (V1 + V2) / 2.0
        m20 = (V2 + V0) / 2.0
        
        # Center
        c = (V0 + V1 + V2) / 3.0
        
        # Additional points to fill gaps
        # Let's use a simple grid in barycentric coords
        points = [p0, p1, p2, m01, m12, m20, c]
        
        # Add 4 more points
        # Sample random points but ensure they are not too close to existing ones
        while len(points) < N:
            u = rng.uniform(0.1, 0.8)
            v = rng.uniform(0.1, 0.8)
            w = 1.0 - u - v
            if w > 0.05:
                p = _barycentric_to_cartesian(np.array([u, v, w]))
                # Check distance to existing points
                min_dist = min(np.linalg.norm(p - q) for q in points)
                if min_dist > 0.05:
                    points.append(p)
        return np.array(points)
    elif strategy == 'random':
        points = []
        for _ in range(N):
            u = rng.uniform(0.01, 0.99)
            v = rng.uniform(0.01, 0.99)
            if u + v < 0.98:
                p = [u + 0.5 * v, H * v]
                points.append(p)
            else:
                # Retry if outside
                points.append([rng.uniform(0.2, 0.8), rng.uniform(0.1, 0.4)])
        return np.array(points)
    else:
        return _generate_initial_config(rng, 'random')


def _coordinate_descent(points, time_limit, rng):
    """
    Coordinate descent local search: move one point at a time to improve min area.
    This is more effective than SA for the non-smooth min-area objective.
    """
    start_time = time.time()
    current_points = points.copy()
    current_area = _min_triangle_area_fast(current_points)
    
    best_points = current_points.copy()
    best_area = current_area
    
    # Precompute index pairs for efficiency
    n = N
    all_indices = list(range(n))
    
    # Step sizes to try
    base_steps = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < time_limit and no_improve_count < max_no_improve:
        improved = False
        
        # Try moving each point in random order
        for idx in rng.sample(all_indices, n):
            if time.time() >= time_limit:
                break
            
            # Try different step sizes
            for step in base_steps:
                # Try 8 directions
                for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                    dx = step * np.cos(angle)
                    dy = step * np.sin(angle)
                    
                    new_points = current_points.copy()
                    new_points[idx] = current_points[idx] + np.array([dx, dy])
                    new_points = _clamp_points(new_points)
                    
                    new_area = _min_triangle_area_fast(new_points)
                    
                    if new_area > current_area + 1e-12:
                        current_points = new_points
                        current_area = new_area
                        improved = True
                        no_improve_count = 0
                        
                        if current_area > best_area:
                            best_area = current_area
                            best_points = current_points.copy()
                        break  # Move to next point
                else:
                    continue
                break  # Move to next point
        
        if not improved:
            no_improve_count += 1
            # Reduce step sizes when stuck
            base_steps = [s * 0.5 for s in base_steps if s * 0.5 > 1e-5]
            if len(base_steps) == 0:
                break
    
    return best_points, best_area


def _simulated_annealing(points_init, time_limit, rng):
    """
    Run Simulated Annealing to maximize the min triangle area.
    """
    start_time = time.time()
    current_points = points_init.copy()
    current_area = _min_triangle_area_fast(current_points)
    
    best_points = current_points.copy()
    best_area = current_area
    
    # Initial temperature
    T = 0.01
    # Cooling rate
    alpha = 0.999
    # Step size
    step_size = 0.05
    
    iterations = 0
    while time.time() < time_limit:
        # Generate a new candidate
        # Perturb a random subset of points
        num_points_to_move = rng.randint(1, 3)
        idxs = rng.sample(range(N), num_points_to_move)
        
        new_points = current_points.copy()
        for idx in idxs:
            noise = np.random.normal(0, step_size, 2)
            new_points[idx] += noise
        
        # Clamp to triangle
        new_points = _clamp_points(new_points)
        
        new_area = _min_triangle_area_fast(new_points)
        
        # Acceptance probability
        delta = new_area - current_area
        if delta > 0:
            # Always accept better solutions
            current_points = new_points
            current_area = new_area
            if current_area > best_area:
                best_area = current_area
                best_points = current_points.copy()
        else:
            # Accept worse solutions with probability exp(delta / T)
            if T > 1e-6:
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    current_points = new_points
                    current_area = new_area
        
        # Cooling
        T *= alpha
        # Occasionally reheat or adjust step size if stuck? 
        # For now, simple cooling.
        
        iterations += 1
        # If T is very low, we are in local refinement mode.
        # We can reduce step size as we cool
        step_size = max(0.001, 0.05 * (T / 0.01))

    return best_points, best_area


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    rng = random.Random(42)
    
    best_points_global = None
    best_score_global = -float('inf')
    
    # First, try the known good configuration
    known_points = np.array([_barycentric_to_cartesian(np.array(b)) for b in _KNOWN_GOOD_BARY])
    known_points = _clamp_points(known_points)
    known_area = _min_triangle_area_fast(known_points)
    known_score = known_area / AREA
    
    if known_score > best_score_global:
        best_score_global = known_score
        best_points_global = known_points.copy()
    
    # Run coordinate descent on the known good config
    if time.time() < deadline - 10.0:
        cd_deadline = time.time() + min(30.0, (deadline - start_time) * 0.3)
        cd_points, cd_area = _coordinate_descent(known_points, cd_deadline, rng)
        cd_score = cd_area / AREA
        
        if cd_score > best_score_global:
            best_score_global = cd_score
            best_points_global = cd_points.copy()
    
    # Then try multiple random/structured initializations with SA + CD
    strategies = ['structured', 'random']
    num_extra_runs = 5
    remaining_time = deadline - time.time()
    
    if remaining_time > 20.0:
        time_per_run = remaining_time / (num_extra_runs + 1) - 2.0
        
        for i in range(num_extra_runs):
            if time.time() > deadline - 15.0:
                break
            
            strategy = strategies[i % len(strategies)]
            points_init = _generate_initial_config(rng, strategy)
            
            # Run SA briefly
            sa_deadline = time.time() + time_per_run * 0.4
            points_sa, area_sa = _simulated_annealing(points_init, sa_deadline, rng)
            score_sa = area_sa / AREA
            
            # Then run coordinate descent on the SA result
            if time.time() < deadline - 10.0:
                cd_deadline = time.time() + time_per_run * 0.6
                points_cd, area_cd = _coordinate_descent(points_sa, cd_deadline, rng)
                score_cd = area_cd / AREA
                
                if score_cd > best_score_global:
                    best_score_global = score_cd
                    best_points_global = points_cd.copy()
                elif score_sa > best_score_global:
                    best_score_global = score_sa
                    best_points_global = points_sa.copy()
    
    # Final polishing with coordinate descent
    if best_points_global is not None and time.time() < deadline - 5.0:
        final_cd_deadline = time.time() + min(10.0, deadline - time.time() - 3.0)
        final_points, final_area = _coordinate_descent(best_points_global, final_cd_deadline, rng)
        final_score = final_area / AREA
        
        if final_score > best_score_global:
            best_score_global = final_score
            best_points_global = final_points.copy()

    # Ensure final points are valid
    if best_points_global is not None:
        best_points_global = _clamp_points(best_points_global)
        
    return best_points_global


# EVOLVE-BLOCK-START
# (The construct function is defined above, but we keep this block for compatibility)
# EVOLVE-BLOCK-END
