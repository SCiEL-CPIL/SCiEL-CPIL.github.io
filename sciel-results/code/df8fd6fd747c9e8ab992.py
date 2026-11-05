"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from collections import Counter

def _score_numpy(points):
    """
    Compute the max multiplicity of a positive doubled area.
    points: list of (x, y) tuples or numpy array of shape (N, 2)
    """
    if isinstance(points, list):
        pts = np.array(points)
    else:
        pts = points
        
    n = len(pts)
    if n < 3:
        return 0
        
    # Compute all pairwise differences? No, compute determinants for all triples.
    # Vectorized approach:
    # For each i, j, k, det = abs((x_j - x_i)*(y_k - y_i) - (x_k - x_i)*(y_j - y_i))
    # This is O(N^3) memory if we do it all at once, which is 30^3 = 27000, very small.
    
    # We can use broadcasting to compute all determinants at once.
    # Shape (N, N, N)
    # p_i = pts[i], p_j = pts[j], p_k = pts[k]
    # det = (p_j - p_i) x (p_k - p_i)
    
    # Let's do it in a way that avoids huge memory if N was larger, but for N=30 it's fine.
    # Actually, we can just loop over i and use vectorized ops for j, k.
    
    max_mult = 0
    # Pre-compute area for all triples
    # Using itertools.combinations is slow in Python loop, but N=30 is small.
    # Let's try to vectorize the whole thing.
    
    # Create index arrays
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    # Mask for i < j < k to avoid duplicates and self-loops
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    
    # Extract coordinates
    x_i = pts[i_idx[mask], 0]
    y_i = pts[i_idx[mask], 1]
    x_j = pts[j_idx[mask], 0]
    y_j = pts[j_idx[mask], 1]
    x_k = pts[k_idx[mask], 0]
    y_k = pts[k_idx[mask], 1]
    
    # Calculate doubled area (determinant)
    dets = np.abs((x_j - x_i) * (y_k - y_i) - (x_k - x_i) * (y_j - y_i))
    
    # Filter out zero areas
    dets = dets[dets > 0]
    
    if len(dets) == 0:
        return 0
        
    # Count frequencies
    # np.unique with return_counts is fast
    vals, counts = np.unique(dets, return_counts=True)
    max_mult = np.max(counts)
    
    return int(max_mult)

def construct():
    """
    Find 30 distinct integer points in [-30, 30]^2 that maximize the
    largest multiplicity of a single positive doubled area.
    """
    # Start with a good initial guess: a grid
    # 5x6 grid centered at 0
    best_points = None
    best_score = 0
    
    # Initialize with a grid
    # Try a few grid offsets
    for ox in range(-2, 3):
        for oy in range(-2, 3):
            pts = []
            for i in range(5):
                for j in range(6):
                    pts.append((ox + j, oy + i))
            # Check bounds
            if all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in pts) and len(set(pts)) == 30:
                s = _score_numpy(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts[:]
                    
    if best_points is None:
        # Fallback
        best_points = [(x, y) for x in range(6) for y in range(5)]
        best_score = _score_numpy(best_points)
        
    # Local Search
    # Time limit: 55 seconds
    start_time = time.time()
    time_limit = 55.0
    
    # Convert to list for easy mutation
    current_points = best_points[:]
    current_score = best_score
    
    # Randomize a bit to escape local minima? 
    # Or just start from the grid.
    
    while time.time() - start_time < time_limit:
        # Pick a random point to move
        idx = np.random.randint(0, 30)
        x, y = current_points[idx]
        
        # Pick a random move: dx, dy in {-1, 0, 1}, not both 0
        dx = np.random.randint(-1, 2)
        dy = np.random.randint(-1, 2)
        if dx == 0 and dy == 0:
            continue
            
        new_x = x + dx
        new_y = y + dy
        
        # Check bounds
        if new_x < -30 or new_x > 30 or new_y < -30 or new_y > 30:
            continue
            
        # Check collision with other points
        # This is O(N)
        new_pt = (new_x, new_y)
        collision = False
        for i, p in enumerate(current_points):
            if i != idx and p == new_pt:
                collision = True
                break
        if collision:
            continue
            
        # Apply move
        old_pt = current_points[idx]
        current_points[idx] = new_pt
        
        # Score
        new_score = _score_numpy(current_points)
        
        # Accept if better, or accept with some probability if equal (to explore)
        # Strict improvement for simplicity first
        if new_score > current_score:
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points[:]
        else:
            # Revert
            current_points[idx] = old_pt
            
    return best_points
# EVOLVE-BLOCK-END
