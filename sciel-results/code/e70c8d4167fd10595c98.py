import numpy as np
import time
import random

N = 50

def _score(positions):
    """
    Calculate the score for a given set of positions.
    Score is the maximum radius r such that circles of radius r
    centered at positions are disjoint and inside the unit circle.
    """
    # Pairwise distances
    # positions shape (N, 2)
    # dists shape (N, N)
    dists = np.sqrt(np.sum((positions[:, np.newaxis, :] - positions[np.newaxis, :, :]) ** 2, axis=2))
    
    # Minimum pairwise distance / 2
    # Mask diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Distance to boundary
    # min_i (1 - |c_i|)
    radii = np.sqrt(np.sum(positions ** 2, axis=1))
    min_bound = 1.0 - np.max(radii)
    
    return min(min_pair, min_bound)

def _hex_lattice_init():
    """
    Initialize positions in a hexagonal lattice pattern.
    """
    points = []
    # Generate a grid of hex points
    # We want roughly 50 points.
    # Hex spacing s.
    # Let's try a few spacings and pick the one that fits best.
    best_points = None
    best_count = 0
    for s in np.linspace(0.1, 0.5, 50):
        k = int(1.0 / s) + 2
        rows = [((i + 0.5 * (j % 2)) * s, j * s * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        pts = np.array(rows)
        # Filter points inside unit circle (with margin)
        r = np.hypot(pts[:, 0], pts[:, 1])
        pts = pts[r <= 1.0 - s / 2.0]
        if len(pts) >= N:
            # Keep the ones closest to center to ensure good packing
            pts = pts[np.argsort(r, kind="stable")[:N]]
            if len(pts) > best_count:
                best_count = len(pts)
                best_points = pts
    if best_points is not None:
        return best_points
    return np.random.rand(N, 2) * 2 - 1

def _random_init():
    """
    Initialize positions randomly inside the unit circle.
    """
    # Rejection sampling
    pts = []
    while len(pts) < N:
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
        if x*x + y*y <= 1:
            pts.append([x, y])
    return np.array(pts)

def _ring_init():
    """
    Initialize positions in concentric rings.
    """
    # 1 center, then rings of 6, 12, 18, 13 (total 50)
    pts = [[0, 0]]
    radii = [0.0]
    counts = [1]
    remaining = N - 1
    r = 0.2
    while remaining > 0:
        # Approximate number of points on a ring of radius r
        # Circumference 2*pi*r. Spacing approx 0.2*r? 
        # Let's just pick counts that sum to N.
        # 1, 6, 12, 18, 13
        c = min(remaining, int(2 * np.pi * r / 0.25))
        if c == 0: c = 1
        counts.append(c)
        radii.append(r)
        remaining -= c
        r += 0.25
        
    # Adjust radii to fit better
    # Scale radii so that outer ring is near boundary
    max_r = radii[-1]
    radii = [r * (0.9 / max_r) for r in radii]
    
    for r, c in zip(radii, counts):
        if c == 1 and r == 0:
            continue
        angles = np.linspace(0, 2 * np.pi, c, endpoint=False)
        for a in angles:
            pts.append([r * np.cos(a), r * np.sin(a)])
            
    return np.array(pts[:N])

def _simulated_annealing(init_pos, time_limit, seed):
    """
    Run Simulated Annealing to maximize the packing score.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    pos = init_pos.copy()
    cur_score = _score(pos)
    best_score = cur_score
    best_pos = pos.copy()
    
    # Parameters
    T = 0.01  # Initial temperature
    step_size = 0.1
    cooling_rate = 0.9995
    
    start_time = time.time()
    
    while time.time() < start_time + time_limit:
        # Pick a random circle
        i = random.randint(0, N - 1)
        
        # Generate random displacement
        delta = np.random.normal(0, step_size, 2)
        
        new_pos = pos.copy()
        new_pos[i] += delta
        
        # Constraint: Must be inside unit circle
        # If outside, reject move (or project, but reject is simpler for SA)
        if np.dot(new_pos[i], new_pos[i]) > 1.0:
            continue
            
        new_score = _score(new_pos)
        
        # Acceptance criterion
        if new_score > cur_score:
            pos = new_pos
            cur_score = new_score
            if cur_score > best_score:
                best_score = cur_score
                best_pos = pos.copy()
        else:
            # Calculate probability
            # Avoid division by zero or overflow
            delta_e = cur_score - new_score
            if delta_e < 0:
                prob = 1.0
            else:
                # exp(-delta_e / T)
                if T > 1e-9:
                    prob = np.exp(-delta_e / T)
                else:
                    prob = 0.0
            
            if random.random() < prob:
                pos = new_pos
                cur_score = new_score
                if cur_score > best_score:
                    best_score = cur_score
                    best_pos = pos.copy()
        
        # Cool down
        T *= cooling_rate
        # Optionally reduce step size
        # step_size *= 0.99999
        
    return best_pos, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for 120s limit
    
    best_pos = None
    best_score = 0.0
    
    # Initializations
    inits = [
        ("hex", _hex_lattice_init()),
        ("random", _random_init()),
        ("ring", _ring_init())
    ]
    
    # Run SA for each initialization
    # Allocate time based on number of inits
    # 3 inits, ~30s each
    time_per_run = 30.0
    seed = 42
    
    for name, init in inits:
        if time.time() > deadline - 10.0:
            break
            
        end_time = min(time.time() + time_per_run, deadline - 10.0)
        run_time = end_time - time.time()
        
        if run_time < 1.0:
            continue
            
        pos, score = _simulated_annealing(init, run_time, seed)
        
        if score > best_score:
            best_score = score
            best_pos = pos
            
        seed += 1
        
    # If we have time left, do a final refinement on the best pos found
    if best_pos is not None and time.time() < deadline - 5.0:
        remaining_time = deadline - 5.0 - time.time()
        if remaining_time > 5.0:
            pos, score = _simulated_annealing(best_pos, remaining_time, seed + 100)
            if score > best_score:
                best_score = score
                best_pos = pos
                
    if best_pos is None:
        # Fallback to hex lattice
        best_pos = _hex_lattice_init()
        
    return best_pos
