"""Hexagon packing: place 11 unit hexagons inside a regular hexagon, minimize outer side."""
import numpy as np
import time
import random

N = 11
SQRT3 = np.sqrt(3.0)
HF = SQRT3 / 2.0  # inradius of unit hexagon (circumradius=1)
FW = SQRT3       # across-flats distance of unit hexagon


def _hex_verts(cx, cy, r, angle_deg):
    """Vertices of regular hexagon: center (cx,cy), circumradius r, rotation angle_deg."""
    a = np.deg2rad(angle_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + r * np.cos(angles), cy + r * np.sin(angles)])


def _sat_min_distance(va, vb, ra, rb):
    """
    Minimum distance between two convex polygons va, vb (each Nx2),
    with circumradii ra, rb (for scaling the distance).
    Actually we just compute the geometric minimum distance between the two polygons.
    Returns 0 if overlapping, positive if separated.
    """
    # For each edge of each polygon, project all vertices of the other polygon
    # onto the edge's normal. Track the maximum gap.
    min_gap = 1e18
    
    for (verts, other) in [(va, vb), (vb, va)]:
        for i in range(len(verts)):
            p1 = verts[i]
            p2 = verts[(i + 1) % len(verts)]
            edge = p2 - p1
            # Normal to edge (pointing outward for a CCW polygon)
            normal = np.array([-edge[1], edge[0]])
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-12:
                continue
            normal /= norm_len
            
            # Project all vertices of 'other' and 'verts' onto this normal
            proj_other = other @ normal
            proj_self = verts @ normal
            
            min_other = np.min(proj_other)
            max_self = np.max(proj_self)
            
            # Gap: if min_other > max_self, they're separated in this direction
            gap = min_other - max_self
            if gap > min_gap:
                min_gap = gap
    
    if min_gap < 0:
        return 0.0  # Overlapping
    return min_gap


def _check_inner_overlap(pos, angle, i, j):
    """Check if inner hexagons i and j overlap. Returns min distance (0 if overlap)."""
    va = _hex_verts(pos[i, 0], pos[i, 1], 1.0, pos[i, 2])
    vb = _hex_verts(pos[j, 0], pos[j, 1], 1.0, pos[j, 2])
    return _sat_min_distance(va, vb, 1.0, 1.0)


def _check_outer_constraint(pos, angle, idx, outer_side, outer_angle, outer_cx=0.0, outer_cy=0.0):
    """
    Check if inner hexagon at idx fits inside outer hexagon.
    Returns minimum margin (negative if outside).
    """
    va = _hex_verts(pos[idx, 0], pos[idx, 1], 1.0, pos[idx, 2])
    vo = _hex_verts(outer_cx, outer_cy, outer_side, outer_angle)
    
    # Check if all vertices of va are inside vo
    # For convex polygons, check each edge of vo: all vertices of va must be on the inner side
    min_margin = 1e18
    for i in range(6):
        p1 = vo[i]
        p2 = vo[(i + 1) % 6]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-12:
            continue
        normal /= norm_len
        
        # For a CCW polygon, the interior is to the left of each edge
        # So the inward normal is actually -normal if we define normal as CCW outward
        # Let's be careful: for a regular hexagon with vertices in CCW order,
        # the outward normal of edge (p1->p2) is (edge_y, -edge_x) / |edge| ... 
        # Actually let's just check: a point is inside if it's on the same side as the center.
        
        # Project center of outer hexagon and vertices of va
        c_proj = (np.array([outer_cx, outer_cy]) - p1) @ normal
        va_proj = (va - p1) @ normal
        
        # If center is on the positive side, all va vertices should also be on positive side
        # If center is on negative side, all should be on negative side
        # The margin is the minimum distance to the boundary in the direction away from center
        
        if c_proj >= 0:
            # Interior is on the positive side
            min_va = np.min(va_proj)
            margin = min_va - 0  # distance from boundary (at 0) to closest va vertex
        else:
            # Interior is on the negative side
            max_va = np.max(va_proj)
            margin = 0 - max_va
        
        if margin < min_margin:
            min_margin = margin
    
    return min_margin


def _evaluate(pos, angle, outer_side, outer_angle=0.0, outer_cx=0.0, outer_cy=0.0):
    """
    Evaluate a configuration.
    Returns (score, penalty, is_valid)
    score = outer_side (to minimize)
    penalty = sum of overlap violations (to minimize, should be 0 for valid)
    is_valid = True if no overlaps and all inside outer
    """
    penalty = 0.0
    valid = True
    
    # Check inner-inner overlaps
    for i in range(N):
        for j in range(i + 1, N):
            d = _check_inner_overlap(pos, angle, i, j)
            if d < 0.001:  # Overlapping or touching
                if d < 0:
                    valid = False
                    penalty += 10.0 * (0.001 - d) ** 2
                else:
                    # Very close, small penalty
                    penalty += 0.1 * (0.001 - d) ** 2
    
    # Check inner-outer containment
    for i in range(N):
        margin = _check_outer_constraint(pos, angle, i, outer_side, outer_angle, outer_cx, outer_cy)
        if margin < 0.001:
            if margin < 0:
                valid = False
                penalty += 10.0 * (0.001 - margin) ** 2
            else:
                penalty += 0.1 * (0.001 - margin) ** 2
    
    score = outer_side
    return score, penalty, valid


def _initial_positions():
    """Generate initial hexagon lattice positions for N hexagons."""
    pos = np.zeros((N, 3))
    angle = np.zeros(N)
    
    # Hexagonal lattice: rows of hexagons
    # For N=11, use a pattern like:
    #   Row 0: 3 hexagons
    #   Row 1: 4 hexagons  
    #   Row 2: 3 hexagons
    #   Row 3: 1 hexagon
    # Or a more compact arrangement.
    
    # Let's use a standard hex packing: 
    # Centers at distance 2*HF = SQRT3 in vertical, 1.5 in horizontal for adjacent rows
    
    idx = 0
    rows = [3, 4, 3, 1]  # 11 total
    y_offset = 0
    for row_idx, count in enumerate(rows):
        x_start = -(count - 1) * 0.75  # Half spacing
        for c in range(count):
            pos[idx, 0] = x_start + c * 1.5
            pos[idx, 1] = y_offset
            angle[idx] = 0.0
            idx += 1
        y_offset += SQRT3 / 2.0 * 2  # Vertical spacing between rows
    
    # Center the configuration
    pos[:, 0] -= np.mean(pos[:, 0])
    pos[:, 1] -= np.mean(pos[:, 1])
    
    return pos, angle


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_score = 1e18
    best_config = None
    best_penalty = 1e18
    
    # Try multiple initial configurations
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        # Initialize
        if restart == 0:
            pos, angle = _initial_positions()
        else:
            # Perturbed initialization
            pos, angle = _initial_positions()
            pos += np.random.normal(0, 0.5, size=pos.shape)
            angle += np.random.normal(0, 15, size=angle.shape)
        
        # Estimate initial outer side
        max_r = 0
        for i in range(N):
            verts = _hex_verts(pos[i, 0], pos[i, 1], 1.0, angle[i])
            r = np.max(np.sqrt(verts[:, 0]**2 + verts[:, 1]**2))
            max_r = max(max_r, r)
        outer_side = max_r + 1.0
        outer_angle = 0.0
        
        # Simulated annealing
        T = 1.0
        T_min = 0.001
        cooling = 0.995
        
        current_score, current_penalty, current_valid = evaluate(pos, angle, outer_side, outer_angle)
        best_local_score = current_score + current_penalty
        best_local_pos = pos.copy()
        best_local_angle = angle.copy()
        best_local_side = outer_side
        
        max_steps = 5000
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            
            # Random perturbation
            new_pos = pos.copy()
            new_angle = angle.copy()
            new_side = outer_side
            
            # Perturb positions
            new_pos[:, 0] += np.random.normal(0, 0.3, size=N)
            new_pos[:, 1] += np.random.normal(0, 0.3, size=N)
            new_angle += np.random.normal(0, 5, size=N)
            
            # Occasionally perturb outer side
            if random.random() < 0.1:
                new_side *= (1 + np.random.normal(0, 0.05))
                new_side = max(new_side, 3.0)
            
            # Evaluate
            new_score, new_penalty, new_valid = evaluate(new_pos, new_angle, new_side, outer_angle)
            new_total = new_score + new_penalty
            current_total = current_score + current_penalty
            
            # Accept?
            if new_total < current_total or random.random() < np.exp(-(new_total - current_total) / T):
                pos = new_pos
                angle = new_angle
                outer_side = new_side
                current_score, current_penalty, current_valid = new_score, new_penalty, new_valid
                
                if current_total < best_local_score:
                    best_local_score = current_total
                    best_local_pos = pos.copy()
                    best_local_angle = angle.copy()
                    best_local_side = outer_side
            
            # Cool down
            T = max(T * cooling, T_min)
        
        # Check best local
        if best_local_penalty < 1e-6:
            score, penalty, valid = evaluate(best_local_pos, best_local_angle, best_local_side, outer_angle)
            if valid and score < best_score:
                best_score = score
                best_config = (best_local_pos.copy(), best_local_angle.copy(), best_local_side)
    
    # If no valid solution found, use a fallback
    if best_config is None:
        # Fallback: linear chain
        pos = np.zeros((N, 3))
        angle = np.zeros(N)
        for i in range(N):
            pos[i, 0] = (i - N // 2) * 2.0
            pos[i, 1] = 0.0
            angle[i] = 0.0
        
        max_r = 0
        for i in range(N):
            verts = _hex_verts(pos[i, 0], pos[i, 1], 1.0, angle[i])
            r = np.max(np.sqrt(verts[:, 0]**2 + verts[:, 1]**2))
            max_r = max(max_r, r)
        
        best_score = max_r + 0.5
        best_config = (pos, angle, best_score)
    
    pos, angle, side = best_config
    outer_cx, outer_cy = 0.0, 0.0
    
    return [
        [float(pos[i, 0]), float(pos[i, 1]), float(angle[i])]
        for i in range(N)
    ], (outer_cx, outer_cy), float(side), float(0.0)
