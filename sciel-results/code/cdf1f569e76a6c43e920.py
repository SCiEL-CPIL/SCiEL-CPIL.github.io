"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

# Precompute triangle indices
TRI_INDICES = list(combinations(range(N), 3))
NUM_TRIS = len(TRI_INDICES)  # 286

# Precompute index arrays for vectorized area computation
_I = np.array([t[0] for t in TRI_INDICES])
_J = np.array([t[1] for t in TRI_INDICES])
_K = np.array([t[2] for t in TRI_INDICES])

def _tri_areas(pts):
    """Vectorized triangle areas for all 286 triples."""
    pts = np.asarray(pts, dtype=np.float64)
    pi = pts[_I]
    pj = pts[_J]
    pk = pts[_K]
    v1 = pj - pi
    v2 = pk - pi
    return 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])

def _hull_area(pts):
    """Compute area of convex hull."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return 0.0
    if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
        return 0.0
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 0.0
    min_area = np.min(_tri_areas(pts))
    return min_area / ha

def _spectral_init(rng, n_points=13):
    """
    Generate points using spectral graph theory concept:
    Create a random SPD matrix, find its eigenvectors, and use the first 2 components
    of the top N eigenvectors as coordinates? No, that gives N points in 2D if we take
    the first 2 eigenvectors' rows?
    
    Actually, let's interpret "points as eigenvectors" as:
    We want N points in 2D.
    Let M be a random symmetric matrix.
    Let V be the matrix of eigenvectors of M.
    We can take the first N rows of V[:, :2] (first 2 eigenvectors) as our points.
    This ensures the points are orthogonal-ish and spread out.
    """
    M = rng.randn(n_points, n_points)
    M = M + M.T
    # Ensure SPD
    M = M @ M.T + np.eye(n_points) * 0.1
    eigvals, eigvecs = np.linalg.eigh(M)
    # Take the first 2 eigenvectors (columns) and the first N rows
    # This gives an N x 2 matrix
    pts = eigvecs[:n_points, :2]
    # Normalize to unit disk roughly
    norms = np.linalg.norm(pts, axis=1)
    norms[norms == 0] = 1
    pts = pts / norms[:, np.newaxis]
    return pts

def _regular_polygon(n, radius=1.0, center=(0,0)):
    """Generate regular polygon vertices."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([center[0] + radius*np.cos(angles), 
                            center[1] + radius*np.sin(angles)])

def _simulated_annealing(pts, time_budget, rng):
    """
    Perform Simulated Annealing on the exact objective.
    This is a structural change from gradient-based methods to stochastic global search.
    """
    start_time = time.time()
    current_pts = pts.copy()
    current_score = _score(current_pts)
    best_pts = current_pts.copy()
    best_score = current_score
    
    # Initial temperature
    T = 0.01
    cooling_rate = 0.95
    min_T = 1e-6
    step_size = 0.05
    
    while time.time() - start_time < time_budget - 1 and T > min_T:
        # Generate a perturbation
        # Move one point randomly
        idx = rng.randint(N)
        delta = rng.normal(0, step_size, 2)
        new_pts = current_pts.copy()
        new_pts[idx] += delta
        
        new_score = _score(new_pts)
        
        # Acceptance probability
        if new_score > current_score:
            current_pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts.copy()
        else:
            # Metropolis criterion
            # Delta is negative
            delta_score = current_score - new_score
            if delta_score < 1e-12:
                accept = True
            else:
                # P = exp(-delta / T)
                # We want to maximize, so if new is worse, delta_score > 0
                # We accept with prob exp(-delta_score / T)
                try:
                    prob = np.exp(-delta_score / T)
                except OverflowError:
                    prob = 0.0
                accept = rng.rand() < prob
            
            if accept:
                current_pts = new_pts
                current_score = new_score
        
        T *= cooling_rate
        
        # Adapt step size if stuck
        if rng.rand() < 0.05:
            step_size *= 0.9

    return best_pts, best_score

def _nelder_mead_polish(pts, time_budget):
    """
    Final polish using Nelder-Mead on exact objective.
    """
    def obj(x):
        return -_score(x.reshape(N, 2))
    
    try:
        res = minimize(obj, pts.flatten(), method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10,
                                'initial_simplex': 1e-2})
        cand = res.x.reshape(N, 2)
        s = _score(cand)
        return cand, s
    except Exception:
        return pts, _score(pts)

def construct():
    """Main entry point: find optimal 13-point configuration."""
    start_time = time.time()
    total_budget = 145.0  # Safety margin for 150s limit
    
    rng = np.random.RandomState(42)
    
    best_points = None
    best_ratio = -1e9
    
    # Generate diverse starting points
    starts = []
    
    # 1. Regular 13-gon
    starts.append(_regular_polygon(N, radius=1.0))
    
    # 2-5: Spectral initializations
    for _ in range(4):
        starts.append(_spectral_init(rng, N))
        
    # 6-8: Random points on unit disk
    for _ in range(3):
        r = np.sqrt(rng.uniform(0, 1, N))
        theta = rng.uniform(0, 2*np.pi, N)
        starts.append(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))

    # Evaluate initial candidates
    for cand in starts:
        if time.time() - start_time > total_budget:
            break
        try:
            s = _score(cand)
            if s > best_ratio:
                best_ratio = s
                best_points = cand.copy()
        except Exception:
            pass
    
    # Sort starts by initial score (best first)
    scored_starts = []
    for start in starts:
        try:
            s = _score(start)
            scored_starts.append((s, start))
        except Exception:
            scored_starts.append((0.0, start))
    scored_starts.sort(key=lambda x: -x[0])
    
    # Optimize from each starting point
    n_starts = len(scored_starts)
    # Allocate time: 60% to SA, 40% to NM
    base_time = (total_budget - 5.0) / n_starts
    
    for i, (init_score, start) in enumerate(scored_starts):
        if time.time() - start_time > total_budget - 2:
            break
        
        remaining_time = total_budget - (time.time() - start_time)
        # Give more time to better starts
        quality_factor = 1.0 + 0.5 * (init_score / max(best_ratio, 1e-9))
        alloc_time = min(base_time * quality_factor, remaining_time - 1.0)
        if alloc_time < 2.0:
            continue
        
        try:
            # Stage 1: Simulated Annealing
            sa_time = alloc_time * 0.6
            sa_pts, sa_score = _simulated_annealing(start, sa_time, rng)
            
            # Stage 2: Nelder-Mead Polish
            nm_time = alloc_time * 0.4
            nm_pts, nm_score = _nelder_mead_polish(sa_pts, nm_time)
            
            if nm_score > best_ratio:
                best_ratio = nm_score
                best_points = nm_pts.copy()
        except Exception:
            pass
    
    # Final refinement on the best found
    if best_points is not None and time.time() - start_time < total_budget - 3:
        final_time = total_budget - (time.time() - start_time) - 1.0
        if final_time > 5.0:
            try:
                # Quick SA + NM
                final_sa_pts, _ = _simulated_annealing(best_points, final_time * 0.5, rng)
                final_pts, final_score = _nelder_mead_polish(final_sa_pts, final_time * 0.5)
                if final_score > best_ratio:
                    best_ratio = final_score
                    best_points = final_pts.copy()
            except Exception:
                pass
    
    if best_points is None:
        # Fallback: regular polygon
        best_points = _regular_polygon(N, radius=1.0)
    
    return best_points
# EVOLVE-BLOCK-END
