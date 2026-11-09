import numpy as np
from scipy.optimize import minimize

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    
    def calc_min_score(angles):
        # angles: (n,) array
        x = np.cos(angles)
        y = np.sin(angles)
        poly = np.stack([x, y], axis=1)
        
        # Distances: (n, n)
        dists = np.linalg.norm(poly[:, np.newaxis, :] - poly[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Sort distances for each vertex
        sorted_dists = np.sort(dists, axis=1)
        
        # Check min dist
        min_dists = sorted_dists[:, 0]
        valid_mask = min_dists >= 0.05
        
        # Gaps
        gaps = np.diff(sorted_dists, axis=1)
        k_indices = np.argmin(gaps, axis=1)
        
        # D calculation
        d_k = sorted_dists[np.arange(n), k_indices]
        d_k1 = sorted_dists[np.arange(n), k_indices + 1]
        D = (d_k + d_k1) / 2.0
        
        valid_mask &= (D >= 0.1)
        
        # Select 4 distances
        # Create indices for the 4 distances
        # If k=0: 0,1,2,3
        # If k=17: 15,16,17,18
        # Else: k-1, k, k+1, k+2
        
        idx0 = np.where(k_indices == 0, 0, np.where(k_indices == 17, 15, k_indices - 1))
        idx1 = np.where(k_indices == 0, 1, np.where(k_indices == 17, 16, k_indices))
        idx2 = np.where(k_indices == 0, 2, np.where(k_indices == 17, 17, k_indices + 1))
        idx3 = np.where(k_indices == 0, 3, np.where(k_indices == 17, 18, k_indices + 2))
        
        d0 = sorted_dists[np.arange(n), idx0]
        d1 = sorted_dists[np.arange(n), idx1]
        d2 = sorted_dists[np.arange(n), idx2]
        d3 = sorted_dists[np.arange(n), idx3]
        
        # Ratios
        r0 = np.maximum(d0 / D, D / d0)
        r1 = np.maximum(d1 / D, D / d1)
        r2 = np.maximum(d2 / D, D / d2)
        r3 = np.maximum(d3 / D, D / d3)
        
        raw = (r0 + r1 + r2 + r3) / 4.0
        scores = np.where(valid_mask, 1.0 / raw, 0.0)
        scores = np.minimum(scores, 1.0)
        
        return np.min(scores)

    def objective(deltas):
        # deltas: (n-1,)
        d = np.abs(deltas)
        total = np.sum(d)
        last = 2.0 * np.pi - total
        if last <= 0:
            return 0.0 # Invalid
            
        all_deltas = np.append(d, last)
        angles = np.cumsum(all_deltas)
        angles[0] = 0.0
        # Normalize angles to 2pi
        angles = angles / angles[-1] * 2.0 * np.pi
        
        return -calc_min_score(angles)

    # Initial guess
    x0 = np.ones(n - 1) * (2.0 * np.pi / n)
    
    # Optimization
    res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-7})
    
    best_deltas = res.x
    if not np.all(np.isfinite(best_deltas)):
        best_deltas = x0
        
    d = np.abs(best_deltas)
    total = np.sum(d)
    last = 2.0 * np.pi - total
    if last <= 0:
        # Fallback to regular
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    else:
        all_deltas = np.append(d, last)
        angles = np.cumsum(all_deltas)
        angles[0] = 0.0
        angles = angles / angles[-1] * 2.0 * np.pi
        
    x = np.cos(angles)
    y = np.sin(angles)
    poly = np.stack([x, y], axis=1)
    
    centroid = np.mean(poly, axis=0)
    poly = poly - centroid
    mean_dist = np.mean(np.linalg.norm(poly, axis=1))
    if mean_dist > 0.01:
        poly = poly / mean_dist
        
    return poly
