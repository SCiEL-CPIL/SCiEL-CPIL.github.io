<<<<<<< SEARCH
import time
import random
from itertools import combinations

def _score(points):
    """Compute the maximum multiplicity of any single positive doubled area."""
    n = len(points)
    if n < 3:
        return 0
    # Use a dictionary to count areas
    area_counts = {}
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            for k in range(j + 1, n):
                xk, yk = points[k]
                # Doubled area = |xi(yj - yk) + xj(yk - yi) + xk(yi - yj)|
                d = abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
                if d > 0:
                    if d in area_counts:
                        area_counts[d] += 1
                    else:
                        area_counts[d] = 1
    if not area_counts:
        return 0
    return max(area_counts.values())
=======
import time
import random
from itertools import combinations
from collections import Counter
import numpy as np

def _score(points):
    """Compute the maximum multiplicity of any single positive doubled area using vectorized ops."""
    n = len(points)
    if n < 3:
        return 0
    pts = np.array(points, dtype=np.int64)
    # Compute all C(n,3) doubled areas vectorized
    # For each triple (i,j,k), doubled area = |xi(yj-yk) + xj(yk-yi) + xk(yi-yj)|
    # This equals |det([xi, yi, 1; xj, yj, 1; xk, yk, 1])|
    # More efficiently: cross product of (pj-pi) and (pk-pi)
    
    # Generate all index triples
    idx_i, idx_j, idx_k = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                idx_i.append(i)
                idx_j.append(j)
                idx_k.append(k)
    
    idx_i = np.array(idx_i)
    idx_j = np.array(idx_j)
    idx_k = np.array(idx_k)
    
    # Vectorized computation
    pi = pts[idx_i]
    pj = pts[idx_j]
    pk = pts[idx_k]
    
    # Cross product of (pj-pi) and (pk-pi) gives 2x area
    v1 = pj - pi
    v2 = pk - pi
    areas = np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    
    # Filter out zero areas
    areas = areas[areas > 0]
    if len(areas) == 0:
        return 0
    
    # Count occurrences
    counts = Counter(areas.tolist())
    return max(counts.values())

def _score_incremental(points, changed_idx, old_pt, new_pt):
    """
    Fast incremental score update: only recompute triangles involving the changed point.
    Returns the new max multiplicity.
    """
    n = len(points)
    pts = np.array(points, dtype=np.int64)
    
    # We need to count areas for all triples
    # Triangles NOT involving changed_idx keep their areas
    # Triangles involving changed_idx need recomputation
    
    # For efficiency, let's just do full recompute but optimized
    # Actually, let's compute all areas and count
    
    idx_i, idx_j, idx_k = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                idx_i.append(i)
                idx_j.append(j)
                idx_k.append(k)
    
    idx_i = np.array(idx_i)
    idx_j = np.array(idx_j)
    idx_k = np.array(idx_k)
    
    pi = pts[idx_i]
    pj = pts[idx_j]
    pk = pts[idx_k]
    
    v1 = pj - pi
    v2 = pk - pi
    areas = np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    
    areas = areas[areas > 0]
    if len(areas) == 0:
        return 0
    
    counts = Counter(areas.tolist())
    return max(counts.values())
>>>>>>> REPLACE

