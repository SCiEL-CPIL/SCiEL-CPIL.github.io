"""Improved program: two-lattice structure with simulated annealing for doubled triangle area multiplicity."""
import random
import time
import numpy as np
from collections import Counter

def _score(points):
    """Compute the largest multiplicity of a positive doubled area among all triples using NumPy."""
    pts = np.array(points)
    n = len(pts)
    
    # Generate indices for all triples i < j < k
    i_idx, j_idx, k_idx = np.indices((n, n, n)).reshape(3, -1)
    
    # Filter for i < j < k
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    k_idx = k_idx[mask]
    
    # Get coordinates
    pi = pts[i_idx]
    pj = pts[j_idx]
    pk = pts[k_idx]
    
    # Calculate determinants (doubled areas)
    dx_ij = pj[:, 0] - pi[:, 0]
    dy_ij = pj[:, 1] - pi[:, 1]
    dx_ik = pk[:, 0] - pi[:, 0]
    dy_ik = pk[:, 1] - pi[:, 1]
    
    dets = dx_ij * dy_ik - dy_ij * dx_ik
    
    # Filter out zero areas
    non_zero_dets = dets[dets != 0]
    
    if len(non_zero_dets) == 0:
        return 0
        
    # Count multiplicities
    values, counts = np.unique(non_zero_dets, return_counts=True)
    
    return int(np.max(counts))

def _generate_triangular_lattice(center, size, scale):
    """Generate a triangular lattice of 'size' points centered at 'center' with given scale."""
    pts = []
    # Triangular lattice: basis vectors (1, 0) and (1/2, sqrt(3)/2)
    # For integer coords, use (2, 0) and (1, 1) as basis for a skewed grid
    basis1 = (scale, 0)
    basis2 = (0, scale)
    
    # Generate points in a diamond/rhombus shape to get approximately 'size' points
    max_coord = int(np.sqrt(size)) + 1
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            x = center[0] + i * basis1[0] + j * basis2[0]
            y = center[1] + i * basis1[1] + j * basis2[1]
            if -30 <= x <= 30 and -30 <= y <= 30:
                pts.append((x, y))
    
    # If we have too many, randomly sample; if too few, add random points
    if len(pts) > size:
        pts = random.sample(pts, size)
    elif len(pts) < size:
        while len(pts) < size:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            if (x, y) not in pts:
                pts.append((x, y))
    
    return pts

def _generate_two_lattice_config(shift_x, shift_y, lattice_scale, seed_offset=0):
    """Generate 30 points as union of two 15-point triangular lattices."""
    rng = random.Random(42 + seed_offset)
    
    # First lattice centered at origin area
    center1 = (-shift_x // 2, -shift_y // 2)
    pts1 = _generate_triangular_lattice(center1, 15, lattice_scale)
    
    # Second lattice shifted
    center2 = (shift_x - shift_x // 2, shift_y - shift_y // 2)
    pts2 = _generate_triangular_lattice(center2, 15, lattice_scale)
    
    # Combine and ensure all are distinct and in range
    all_pts = list(set(pts1 + pts2))
    
    # If duplicates or out of range, fill with random points
    while len(all_pts) < 30:
        x = rng.randint(-30, 30)
        y = rng.randint(-30, 30)
        if (x, y) not in all_pts and -30 <= x <= 30 and -30 <= y <= 30:
            all_pts.append((x, y))
    
    # Take first 30 if we have more
    all_pts = all_pts[:30]
    
    return all_pts

def _score_two_lattice(shift_x, shift_y, lattice_scale, seed_offset=0):
    """Score a two-lattice configuration."""
    points = _generate_two_lattice_config(shift_x, shift_y, lattice_scale, seed_offset)
    return _score(points)

def construct():
    # Set deadline to 50s to be safe against the 90s kill
    deadline = time.time() + 50
    
    rng = random.Random(42)
    
    # Strategy 1: Try various two-lattice configurations
    best_score = 0
    best_points = None
    
    # Try different shift vectors and scales
    for shift_x in range(-10, 11, 2):
        for shift_y in range(-10, 11, 2):
            for lattice_scale in [1, 2, 3]:
                points = _generate_two_lattice_config(shift_x, shift_y, lattice_scale)
                score = _score(points)
                if score > best_score:
                    best_score = score
                    best_points = points[:]
                
                if time.time() > deadline - 5:
                    return best_points if best_points else [(0,0)]
    
    # Strategy 2: Simulated annealing on the shift vector
    if best_points is None:
        best_points = _generate_two_lattice_config(0, 0, 1)
        best_score = _score(best_points)
    
    current_points = best_points[:]
    current_score = best_score
    temp = 10.0
    cooling = 0.9995
    
    while time.time() < deadline - 5:
        # Perturb: randomly move one point
        idx = rng.randint(0, 29)
        old_pt = current_points[idx]
        new_pt = (rng.randint(-30, 30), rng.randint(-30, 30))
        
        current_points[idx] = new_pt
        
        # Check validity
        valid = True
        seen = set()
        for p in current_points:
            if p in seen:
                valid = False
                break
            seen.add(p)
            if not (-30 <= p[0] <= 30 and -30 <= p[1] <= 30):
                valid = False
                break
        
        if not valid:
            current_points[idx] = old_pt
            continue
        
        new_score = _score(current_points)
        
        if new_score >= current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = current_points[:]
        else:
            # Accept with annealing probability
            if rng.random() < temp:
                pass
            else:
                current_points[idx] = old_pt
        
        temp *= cooling
        if temp < 0.01:
            temp = 5.0
            current_points = best_points[:]
            current_score = best_score
    
    return best_points
