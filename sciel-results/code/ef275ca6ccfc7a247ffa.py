import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, basinhopping
import time

def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 560
    n = 14

    def ratio(pts):
        # Normalize to unit sphere to reduce search space
        norms = np.linalg.norm(pts, axis=1)
        mask = norms < 1e-12
        if np.any(mask):
            norms[mask] = 1.0
            pts = pts.copy()
            pts[mask] = 0.0
            pts[mask, 0] = 1.0
        pts = pts / norms[:, np.newaxis]
        
        d = pdist(pts)
        d_min = float(d.min())
        d_max = float(d.max())
        if d_max <= 1e-15:
            return 0.0
        return d_min / d_max

    def neg_ratio(x):
        pts = x.reshape(n, 3)
        return -ratio(pts)

    # ... initializations ...
    
    # Use basinhopping on the best candidate
    # ...
