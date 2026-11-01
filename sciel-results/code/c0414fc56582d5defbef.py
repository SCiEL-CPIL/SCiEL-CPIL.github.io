"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32
import numpy as np
from scipy.optimize import minimize
import time

# EVOLVE-BLOCK-START

def _soft_min(vals, beta=10.0):
    """
    Soft minimum using log-sum-exp trick for numerical stability.
    """
    beta = max(beta, 1.0)
    # For soft min: -logsumexp(-vals/beta) * beta
    # But we want a smooth approximation to min
    # min(x) ≈ -(1/beta) * log(sum(exp(-beta*x)))
    # To avoid overflow, use the max as reference
    v = np.asarray(vals, dtype=np.float64)
    if v.size == 0:
        return np.array([], dtype=np.float64)
    
    # For each row, compute soft min
    # If vals is 1D, treat as single row
    if v.ndim == 1:
        v = v.reshape(1, -1)
    
    out = np.empty(v.shape[0], dtype=np.float64)
    for i in range(v.shape[0]):
        row = v[i]
        if row.size == 0:
            out[i] = 0.0
            continue
        # soft min = -(1/beta) * log(sum(exp(-beta * row)))
        # Use max for stability
        m = np.min(row)
        # exp(-beta*(row - m)) is bounded by exp(0)=1 for the min element
        # and exp(-beta*(row-m)) < 1 for others
        shifted = -beta * (row - m)
        # shifted >= 0, max is beta*(max(row)-m)
        # sum(exp(shifted)) could be large if many elements close to m
        # But exp(shifted) <= exp(beta*(max(row)-m)) which could be huge
        # Better: use the standard logsumexp with the min
        # logsumexp(shifted) = m_shifted + log(sum(exp(shifted - m_shifted))) where m_shifted = max(shifted)
        m_shifted = np.max(shifted)
        if m_shifted > 700:  # overflow protection
            # All elements are very close to the min
            out[i] = m
        else:
            log_sum = m_shifted + np.log(np.sum(np.exp(shifted - m_shifted)))
            out[i] = -(1.0 / beta) * log_sum
    return out

def _clearance_sum(pos, beta=10.0):
    """
    Objective function for optimization.
    Minimizes negative sum of soft clearances.
    pos: shape (2*N,)
    """
    p = pos.reshape(N, 2)
    # Calculate pairwise distances
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    
    # Minimum distance to any other point
    min_dist_to_others = np.min(dist, axis=1)
    
    # Distance to box boundaries
    dist_to_box = np.min(np.stack([p[:, 0], 1.0 - p[:, 0], p[:, 1], 1.0 - p[:, 1]], axis=1), axis=1)
    
    # Clearance for each point
    # r_i <= 0.5 * dist(i, j) ensures r_i + r_j <= dist(i, j)
    # Use soft min for smoothness
    clearances = np.minimum(dist_to_box, 0.5 * min_dist_to_others)
    
    return -np.sum(clearances)

def _clearance_sum_smooth(pos, beta=10.0):
    """
    Smooth version using soft-min for the clearance calculation.
    """
    p = pos.reshape(N, 2)
    # Calculate pairwise distances
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    
    # For each point i, we want soft_min over j != i of 0.5*dist(i,j)
    # and soft_min of dist_to_box
    # Actually, clearance_i = min(dist_to_box_i, min_j 0.5*dist(i,j))
    # = min(dist_to_box_i, min_j 0.5*dist(i,j))
    # We can compute soft_min of the combined set for each i
    
    # dist_to_box for each point
    dist_to_box = np.min(np.stack([p[:, 0], 1.0 - p[:, 0], p[:, 1], 1.0 - p[:, 1]], axis=1), axis=1)
    
    # For each i, the candidates are: dist_to_box[i] and 0.5*dist[i,j] for j != i
    # Build a (N, N+1) matrix where column 0 is dist_to_box and columns 1..N are 0.5*dist
    candidates = np.zeros((N, N + 1), dtype=np.float64)
    candidates[:, 0] = dist_to_box
    candidates[:, 1:] = 0.5 * dist
    np.fill_diagonal(candidates[:, 1:], np.inf)
    # Actually, we need to set diagonal to inf so it's not considered
    # candidates[i, i+1] corresponds to dist[i,i] which is inf already
    
    # Compute soft min for each row
    clearances = _soft_min(candidates, beta=beta)
    
    return -np.sum(clearances)

def _get_valid_packing(pos):
    """
    Convert optimized positions to a valid packing with radii.
    Returns (positions (N,2), radii (N,))
    """
    p = pos.reshape(N, 2).copy()
    # Clamp to unit square
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    
    # Calculate radii based on clearances
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    
    min_dist_to_others = np.min(dist, axis=1)
    dist_to_box = np.min(np.stack([p[:, 0], 1.0 - p[:, 0], p[:, 1], 1.0 - p[:, 1]], axis=1), axis=1)
    
    radii = np.minimum(dist_to_box, 0.5 * min_dist_to_others)
    
    # Apply safety factor to ensure strict non-overlap
    safety = 0.9999
    radii *= safety
    
    # Ensure radii are non-negative
    radii = np.maximum(radii, 0.0)
    
    # Adjust positions to be at least radii away from boundaries
    # If p[i,0] < radii[i], set p[i,0] = radii[i]
    # But this might cause overlap with other circles. For now, just clamp.
    p[:, 0] = np.clip(p[:, 0], radii, 1.0 - radii)
    p[:, 1] = np.clip(p[:, 1], radii, 1.0 - radii)
    
    return p, radii

def _hex_initialization():
    """Generate a hexagonal grid initialization."""
    pts = []
    # Use a grid density that provides at least N points
    # 6x6 = 36 points
    for i in range(6):
        for j in range(6):
            x = (i + 0.5) / 6.0
            y = (j + 0.5) / 6.0
            # Stagger rows
            if i % 2 == 0:
                y += 0.5 / 6.0 * 0.8
            # Ensure within bounds
            if 0.0 < x < 1.0 and 0.0 < y < 1.0:
                pts.append([x, y])
    
    # If we have more than N, take the first N
    if len(pts) < N:
        while len(pts) < N:
            pts.append([np.random.rand(), np.random.rand()])
            
    return np.array(pts[:N]).flatten()

def _physics_preprocess(pos, n_iters=100):
    """
    Simple physics-based push-apart to get better starting points.
    """
    p = pos.reshape(N, 2).copy()
    for _ in range(n_iters):
        # Compute forces
        forces = np.zeros((N, 2), dtype=np.float64)
        for i in range(N):
            for j in range(i + 1, N):
                d = p[i] - p[j]
                dist = np.sqrt(np.sum(d ** 2))
                if dist < 0.1:  # Only push if close
                    if dist > 1e-10:
                        f = 0.5 * (0.1 - dist) * d / dist
                    else:
                        f = np.array([0.01, 0.01])
                    forces[i] += f
                    forces[j] -= f
        p += 0.1 * forces
        # Clamp to unit square
        p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
        p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    return p.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    best_score = -np.inf
    best_result = None
    
    # List of initializations
    inits = [
        _hex_initialization(),
        np.random.rand(2 * N),
        np.random.rand(2 * N),
        np.random.rand(2 * N),
        np.random.rand(2 * N)
    ]
    
    # Preprocess some initializations with physics
    for idx in range(len(inits)):
        if time.time() > deadline - 30.0:
            break
        inits[idx] = _physics_preprocess(inits[idx], n_iters=50)
    
    # Try multiple beta values for the smooth objective
    betas = [5.0, 10.0, 20.0]
    
    for init_pos in inits:
        if time.time() > deadline - 20.0:
            break
            
        for beta in betas:
            if time.time() > deadline - 10.0:
                break
                
            try:
                # Use L-BFGS-B for local optimization with smooth objective
                res = minimize(_clearance_sum_smooth, init_pos, method='L-BFGS-B', 
                               args=(beta,),
                               options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
                
                # Evaluate the result with the exact (non-smooth) objective
                if res.fun < best_score or best_result is None:
                    pos, radii = _get_valid_packing(res.x)
                    score = np.sum(radii)
                    
                    if score > best_score:
                        best_score = score
                        best_result = np.column_stack((pos, radii))
                        
            except Exception:
                continue
            
            # Also try with the non-smooth objective from the same start
            if time.time() > deadline - 10.0:
                break
            try:
                res2 = minimize(_clearance_sum, init_pos, method='L-BFGS-B', 
                                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
                
                pos2, radii2 = _get_valid_packing(res2.x)
                score2 = np.sum(radii2)
                
                if score2 > best_score:
                    best_score = score2
                    best_result = np.column_stack((pos2, radii2))
                    
            except Exception:
                continue
    
    # Final polish: take the best result and run a high-precision optimization
    if best_result is not None and time.time() < deadline - 5.0:
        try:
            start_pos = best_result[:, :2].flatten()
            res_final = minimize(_clearance_sum, start_pos, method='L-BFGS-B', 
                                 options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-12})
            
            pos_final, radii_final = _get_valid_packing(res_final.x)
            score_final = np.sum(radii_final)
            
            if score_final > best_score:
                best_score = score_final
                best_result = np.column_stack((pos_final, radii_final))
                
        except Exception:
            pass
    
    # Fallback if optimization failed
    if best_result is None:
        pos, radii = _get_valid_packing(inits[0])
        best_result = np.column_stack((pos, radii))
        
    return best_result

# EVOLVE-BLOCK-END
