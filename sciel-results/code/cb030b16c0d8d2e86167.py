"""Improved program: triangular lattice structure with simulated annealing."""
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

def _generate_triangular_lattice_subset(scale, rotation, offset, count):
    """Generate a subset of points from a triangular/hexagonal lattice."""
    # Triangular lattice basis vectors
    a1 = np.array([scale, 0])
    a2 = np.array([scale * 0.5, scale * np.sqrt(3) / 2])
    
    # Apply rotation
    if rotation != 0:
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)
        rot = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        a1 = rot @ a1
        a2 = rot @ a2
    
    # Generate lattice points in a range
    pts = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            p = i * a1 + j * a2 + np.array(offset)
            pts.append((round(p[0], 6), round(p[1], 6)))
    
    # Sort by distance from origin for deterministic selection
    pts.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    return pts[:count]

def _lattice_to_integer_points(points_float, target_count=30):
    """Convert float lattice points to distinct integer points in [-30,30]."""
    # Round to nearest integers
    int_pts = [(int(round(x)), int(round(y))) for x, y in points_float]
    
    # Remove duplicates and out-of-range points
    seen = set()
    valid_pts = []
    for p in int_pts:
        if p not in seen and -30 <= p[0] <= 30 and -30 <= p[1] <= 30:
            seen.add(p)
            valid_pts.append(p)
    
    return valid_pts

def _build_configuration(scale, rotation, shift_x, shift_y, subset1_count=15, subset2_count=15):
    """Build a configuration from two shifted triangular lattice subsets."""
    # Generate first subset
    subset1 = _generate_triangular_lattice_subset(scale, rotation, (0, 0), subset1_count)
    
    # Generate second subset with shift
    subset2 = _generate_triangular_lattice_subset(scale, rotation, (shift_x, shift_y), subset2_count)
    
    # Combine and convert to integer points
    all_pts = subset1 + subset2
    int_pts = _lattice_to_integer_points(all_pts, 30)
    
    # If we don't have enough points, fill with random points
    while len(int_pts) < 30:
        x, y = random.randint(-30, 30), random.randint(-30, 30)
        if (x, y) not in int_pts:
            int_pts.append((x, y))
    
    return int_pts[:30]

def construct():
    # Set deadline to 80s to be safe against the 90s kill
    deadline = time.time() + 80
    
    # Initial configuration: two shifted triangular lattice subsets
    best_params = (2.0, 0.0, 5.0, 0.0)  # scale, rotation, shift_x, shift_y
    best_points = _build_configuration(*best_params)
    best_score = _score(best_points)
    
    # Simulated annealing to optimize lattice parameters
    rng = random.Random(42)
    current_params = best_params[:]
    current_points = best_points[:]
    current_score = best_score
    temp = 1.0
    cooling = 0.9995
    
    def _score_config(params):
        """Score a configuration defined by lattice parameters."""
        pts = _build_configuration(*params)
        if len(pts) < 30 or len(set(pts)) < 30:
            return 0
        return _score(pts)
    
    while time.time() < deadline - 5:
        # Perturb parameters
        scale, rotation, shift_x, shift_y = current_params
        
        # Random perturbation
        if rng.random() < 0.3:
            scale += rng.gauss(0, 0.5)
            scale = max(1.0, min(10.0, scale))
        if rng.random() < 0.3:
            rotation += rng.gauss(0, 0.1)
        if rng.random() < 0.3:
            shift_x += rng.gauss(0, 1.0)
            shift_x = max(-20.0, min(20.0, shift_x))
        if rng.random() < 0.3:
            shift_y += rng.gauss(0, 1.0)
            shift_y = max(-20.0, min(20.0, shift_y))
        
        new_params = (scale, rotation, shift_x, shift_y)
        new_score = _score_config(new_params)
        
        # Accept/reject based on simulated annealing
        if new_score >= current_score:
            current_params = new_params
            current_points = _build_configuration(*new_params)
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = current_points[:]
                best_params = new_params
        else:
            # Accept with probability exp(-delta/temp)
            delta = current_score - new_score
            if rng.random() < np.exp(-delta / max(temp, 1e-6)):
                current_params = new_params
                current_points = _build_configuration(*new_params)
                current_score = new_score
        
        temp *= cooling
        if temp < 1e-4:
            # Reset temperature and restart from best
            temp = 1.0
            current_params = best_params[:]
            current_points = best_points[:]
            current_score = best_score
    
    return best_points
