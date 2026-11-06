"""Parametric lattice + differential evolution + free-point refinement for 50 circles in a unit square."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import cKDTree

def _score_fast(centers):
    """Compute packing radius: min of half-pair-distances and boundary distances."""
    if centers.shape != (N, 2):
        return -1.0
    if not np.all(np.isfinite(centers)):
        return -1.0
    if np.any(np.abs(centers) > 0.5 + 1e-9):
        return -1.0
    # Boundary
    boundary = np.min(0.5 - np.abs(centers), axis=1)
    min_boundary = np.min(boundary)
    if min_boundary < -1e-9:
        return -1.0
    # Pair distances
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1]) / 2.0
    return min(min_pair, min_boundary)

def _score_vectorized(centers):
    """Vectorized score for batch evaluation (used by DE)."""
    n = centers.shape[0]
    # Boundary
    boundary = np.min(0.5 - np.abs(centers), axis=1)
    min_boundary = np.min(boundary)
    # Pair distances - use cKDTree
    try:
        tree = cKDTree(centers)
        dists, _ = tree.query(centers, k=2)
        min_pair = np.min(dists[:, 1]) / 2.0
    except:
        return -1.0
    return min(min_pair, min_boundary)

def _build_lattice(row_lengths, dx, dy, x_offset, y_offset, stagger_phase=0.5):
    """
    Build a lattice from row-length pattern.
    row_lengths: list of ints, sum must equal N
    dx: horizontal spacing
    dy: vertical spacing  
    x_offset, y_offset: global offsets
    stagger_phase: fraction of dx to shift odd rows (0.5 = standard hex)
    Returns (N,2) array or None if invalid.
    """
    total = sum(row_lengths)
    if total != N:
        return None
    
    centers = []
    y = y_offset
    for i, n in enumerate(row_lengths):
        # Center this row horizontally
        row_width = (n - 1) * dx if n > 1 else 0.0
        x_start = x_offset - row_width / 2.0 + (i % 2) * dx * stagger_phase
        for j in range(n):
            centers.append([x_start + j * dx, y])
        y += dy
    
    centers = np.array(centers)
    if centers.shape != (N, 2):
        return None
    if np.any(np.abs(centers) > 0.5 + 1e-6):
        return None
    return centers

def _lattice_params_to_centers(params):
    """Convert parameter vector to center positions."""
    # params: [dx, dy, x_offset, y_offset, stagger_phase, row_lengths...]
    # We'll encode row_lengths as integers from the last elements
    # For simplicity, use a fixed pattern index encoded in first float
    
    dx = params[0]
    dy = params[1]
    x_offset = params[2]
    y_offset = params[3]
    stagger_phase = params[4]
    
    # Pattern index: use params[5] to select from predefined patterns
    pattern_idx = int(np.round(params[5]))
    pattern_idx = max(0, min(pattern_idx, len(ROW_PATTERNS) - 1))
    row_lengths = ROW_PATTERNS[pattern_idx]
    
    centers = _build_lattice(row_lengths, dx, dy, x_offset, y_offset, stagger_phase)
    if centers is None:
        return None
    return centers

def _lattice_objective(params):
    """Objective for differential evolution (minimize negative score)."""
    centers = _lattice_params_to_centers(params)
    if centers is None:
        return 1.0  # bad
    s = _score_fast(centers)
    if s < 0:
        return 1.0
    return -s

# Predefined row-length patterns that sum to 50
ROW_PATTERNS = [
    [8, 7, 7, 7, 7, 7, 7],      # 7 rows
    [7, 6, 7, 6, 7, 6, 7, 4],   # 8 rows
    [6, 7, 6, 7, 6, 7, 6, 5],   # 8 rows
    [5, 6, 5, 6, 5, 6, 5, 6, 6],# 9 rows
    [10, 9, 9, 9, 9, 4],        # 6 rows
    [8, 8, 8, 8, 8],            # 5 rows (rectangular-ish)
    [7, 7, 7, 7, 7, 7, 7, 1],   # 8 rows
    [9, 8, 8, 8, 8, 9],         # 6 rows
    [6, 6, 6, 6, 6, 6, 6, 6, 2],# 9 rows
    [5, 5, 5, 5, 5, 5, 5, 5, 5],# 9 rows
    [8, 7, 8, 7, 8, 7, 5],      # 7 rows
    [7, 8, 7, 8, 7, 8, 5],      # 7 rows
    [9, 7, 9, 7, 9, 9],         # 6 rows
    [6, 8, 6, 8, 6, 8, 8],      # 7 rows
    [8, 6, 8, 6, 8, 6, 8],      # 7 rows
]

def _free_point_refinement(centers, start_time, deadline, max_iter=2000):
    """
    Refine all points freely using a smooth objective.
    Uses gradient-free optimization on the soft-min score.
    """
    best_centers = centers.copy()
    best_score = _score_fast(centers)
    
    def soft_min_score(c, beta=30.0):
        boundary = np.min(0.5 - np.abs(c), axis=1)
        tree = cKDTree(c)
        dists, _ = tree.query(c, k=2)
        pair_dists = dists[:, 1] / 2.0
        all_d = np.concatenate([pair_dists, boundary])
        # Soft min
        neg_bx = -beta * all_d
        m = np.max(neg_bx)
        log_sum_exp = m + np.log(np.sum(np.exp(neg_bx - m)))
        return -log_sum_exp / beta
    
    def objective_flat(x):
        c = x.reshape(N, 2)
        c = np.clip(c, -0.4999, 0.4999)
        return -soft_min_score(c)
    
    x0 = best_centers.flatten()
    
    # Use Nelder-Mead with restarts
    for restart in range(5):
        if time.time() - start_time > deadline:
            break
        
        # Perturb for restart
        if restart == 0:
            x_init = x0.copy()
        else:
            x_init = x0 + np.random.randn(N * 2) * 0.005
            x_init = np.clip(x_init, -0.4999, 0.4999)
        
        res = minimize(objective_flat, x_init, method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12,
                                'adaptive': True})
        
        c = np.clip(res.x.reshape(N, 2), -0.5, 0.5)
        s = _score_fast(c)
        if s > best_score:
            best_score = s
            best_centers = c.copy()
            x0 = c.flatten()
    
    return best_centers, best_score

def _coordinate_descent_polish(centers, start_time, deadline, max_sweeps=100):
    """Fine coordinate descent polish."""
    best_centers = centers.copy()
    best_score = _score_fast(centers)
    step = 1e-3
    
    for sweep in range(max_sweeps):
        if time.time() - start_time > deadline:
            break
        improved = False
        for i in range(N):
            for dim in range(2):
                base = best_centers[i, dim]
                best_val = base
                best_s = best_score
                # Try offsets
                for delta in np.linspace(-step, step, 13):
                    v = base + delta
                    if v < -0.5 or v > 0.5:
                        continue
                    trial = best_centers.copy()
                    trial[i, dim] = v
                    s = _score_fast(trial)
                    if s > best_s + 1e-14:
                        best_s = s
                        best_val = v
                if best_val != base:
                    best_centers[i, dim] = best_val
                    improved = True
        if best_score < best_s:
            best_score = best_s
        if sweep % 5 == 4:
            step *= 0.7
        if not improved and step < 1e-9:
            break
    return best_centers, best_score

def _generate_hex_initial(n_cols, n_rows):
    """Generate hex grid initial config."""
    if n_cols * n_rows < N:
        return None
    dx = 1.0 / (n_cols - 0.5)
    dy = dx * np.sqrt(3) / 2.0
    total_w = (n_cols - 1) * dx + dx / 2.0
    total_h = (n_rows - 1) * dy
    ox = -total_w / 2.0
    oy = -total_h / 2.0
    centers = []
    for i in range(n_rows):
        for j in range(n_cols):
            x = ox + j * dx + (i % 2) * dx / 2.0
            y = oy + i * dy
            centers.append([x, y])
    centers = np.array(centers[:N])
    if np.any(np.abs(centers) > 0.5):
        return None
    return centers

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    np.random.seed(42)
    
    best_centers = None
    best_score = -np.inf
    
    # Phase 1: Differential evolution over lattice parameters
    # Parameters: [dx, dy, x_offset, y_offset, stagger_phase, pattern_idx]
    # Ranges
    bounds = [
        (0.04, 0.12),   # dx
        (0.03, 0.10),   # dy
        (-0.15, 0.15),  # x_offset
        (-0.15, 0.15),  # y_offset
        (0.3, 0.7),     # stagger_phase
        (0, float(len(ROW_PATTERNS) - 1)),  # pattern_idx
    ]
    
    de_start = time.time()
    de_deadline = min(deadline - 40, de_start + 50)
    
    try:
        result = differential_evolution(
            _lattice_objective, bounds,
            maxiter=100, popsize=15, tol=1e-8,
            mutation=(0.3, 1.0), recombination=0.8,
            seed=42, polish=True,
            callback=lambda x, convergence: None
        )
        # Check time
        if time.time() - start_time < de_deadline:
            centers = _lattice_params_to_centers(result.x)
            if centers is not None:
                s = _score_fast(centers)
                if s > best_score:
                    best_score = s
                    best_centers = centers.copy()
    except Exception as e:
        pass
    
    # Phase 2: Try multiple lattice patterns with optimized geometry
    if time.time() - start_time < deadline - 35:
        for pat_idx, row_lengths in enumerate(ROW_PATTERNS):
            if time.time() - start_time > deadline - 30:
                break
            
            # For each pattern, optimize dx, dy, offsets
            n_rows = len(row_lengths)
            max_cols = max(row_lengths)
            
            # Estimate good dx, dy
            # Width constraint: (max_cols-1)*dx + dx*0.5 <= 1.0
            # Height constraint: (n_rows-1)*dy <= 1.0
            dx_max = 1.0 / (max_cols - 0.5)
            dy_max = 1.0 / (n_rows - 1)
            
            # Grid search over dx, dy, offsets
            dx_range = np.linspace(0.5 * dx_max, dx_max, 8)
            dy_range = np.linspace(0.6 * dy_max, dy_max, 8)
            xo_range = np.linspace(-0.05, 0.05, 5)
            yo_range = np.linspace(-0.05, 0.05, 5)
            sp_range = [0.4, 0.5, 0.6]
            
            for dx in dx_range:
                for dy in dy_range:
                    for xo in xo_range:
                        for yo in yo_range:
                            for sp in sp_range:
                                c = _build_lattice(row_lengths, dx, dy, xo, yo, sp)
                                if c is None:
                                    continue
                                s = _score_fast(c)
                                if s > best_score:
                                    best_score = s
                                    best_centers = c.copy()
    
    # Phase 3: Free-point refinement from best lattice
    if best_centers is not None and time.time() - start_time < deadline - 10:
        refine_deadline = deadline - 5
        refined, refine_score = _free_point_refinement(
            best_centers, start_time, refine_deadline, max_iter=1000
        )
        if refine_score > best_score:
            best_score = refine_score
            best_centers = refined.copy()
    
    # Phase 4: Coordinate descent polish
    if best_centers is not None and time.time() - start_time < deadline - 3:
        polished, polish_score = _coordinate_descent_polish(
            best_centers, start_time, deadline - 1
        )
        if polish_score > best_score:
            best_score = polish_score
            best_centers = polished.copy()
    
    # Fallback
    if best_centers is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    # Final validation
    final_score = _score_fast(best_centers)
    if final_score <= 0:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    return best_centers
# EVOLVE-BLOCK-END
