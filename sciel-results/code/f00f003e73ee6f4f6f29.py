"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

def _dist_sq(a1, b1, a2, b2):
    da, db = a1 - a2, b1 - b2
    return da * da + da * db + db * db

def _count_edges(points):
    """Count pairs at distance exactly 1 in the triangular lattice."""
    pts = set(points)
    count = 0
    for a, b in pts:
        for da, db in [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]:
            if (a+da, b+db) in pts:
                count += 1
    return count // 2

def _hex_spiral_points(n):
    """Generate n points in a hexagonal spiral on the triangular lattice.
    
    The hexagonal spiral fills concentric hexagonal rings. For n=32, this
    should achieve the Harborth optimum of 76 edges.
    
    Ring r (r>=1) has 6*r points. Total points up to ring R: 1 + 3*R*(R+1).
    For n=32: ring 0 = 1 pt, ring 1 = 6 pts (total 7), ring 2 = 12 pts (total 19),
    ring 3 = 18 pts (total 37 >= 32). So we need rings 0,1,2 fully and 13 from ring 3.
    """
    if n <= 1:
        return [(0, 0)]
    
    points = [(0, 0)]
    
    # Generate hexagonal rings in order
    ring = 1
    while len(points) < n:
        # Generate all 6*ring points on this ring
        ring_points = []
        
        # The six directions for a hexagonal ring in axial coordinates
        # Starting from corner (ring, 0) and going around
        # Side 1: (ring, 0) to (0, ring): delta = (-1, 1), length ring
        # Side 2: (0, ring) to (-ring, ring): delta = (-1, 0), length ring
        # Side 3: (-ring, ring) to (-ring, 0): delta = (0, -1), length ring
        # Side 4: (-ring, 0) to (0, -ring): delta = (1, -1), length ring
        # Side 5: (0, -ring) to (ring, -ring): delta = (1, 0), length ring
        # Side 6: (ring, -ring) to (ring, 0): delta = (0, 1), length ring
        
        sides = [
            ((ring, 0), (-1, 1)),
            ((0, ring), (-1, 0)),
            ((-ring, ring), (0, -1)),
            ((-ring, 0), (1, -1)),
            ((0, -ring), (1, 0)),
            ((ring, -ring), (0, 1)),
        ]
        
        for start, delta in sides:
            for t in range(ring):
                pt = (start[0] + delta[0] * t, start[1] + delta[1] * t)
                ring_points.append(pt)
        
        # Add ring points until we have n total
        for pt in ring_points:
            if len(points) >= n:
                break
            points.append(pt)
        
        ring += 1
    
    return points[:n]

def _generate_candidates(max_coord=20):
    """Generate candidate points in a bounding box."""
    candidates = []
    for a in range(-max_coord, max_coord + 1):
        for b in range(-max_coord, max_coord + 1):
            candidates.append((a, b))
    return candidates

def _greedy_construct(candidates, n_points=32, seed=42):
    """Greedy construction: add points one at a time, maximizing new edges."""
    random.seed(seed)
    
    # Start with a random point near origin
    placed = []
    placed_set = set()
    
    # First point: origin
    first = (0, 0)
    placed.append(first)
    placed_set.add(first)
    
    for _ in range(n_points - 1):
        best_point = None
        best_new_edges = -1
        best_score = -1
        
        for pt in candidates:
            if pt in placed_set:
                continue
            
            # Check minimum distance constraint: dist >= 1 from all placed points
            valid = True
            for pa, pb in placed:
                if _dist_sq(pt[0], pt[1], pa, pb) < 1:
                    valid = False
                    break
            if not valid:
                continue
            
            # Count new edges formed with already-placed points
            new_edges = 0
            for pa, pb in placed:
                if _dist_sq(pt[0], pt[1], pa, pb) == 1:
                    new_edges += 1
            
            # Score: primarily new edges, secondary: prefer points closer to cluster center
            # Compute distance from centroid of placed points
            if placed:
                cx = sum(p[0] for p in placed) / len(placed)
                cy = sum(p[1] for p in placed) / len(placed)
                # Approximate distance from centroid
                dx = pt[0] - cx
                dy = pt[1] - cy
                # Use lattice distance approximation
                dist_to_center = abs(dx) + abs(dy) + abs(dx + dy)
            else:
                dist_to_center = 0
            
            score = (new_edges, -dist_to_center)
            
            if score > (best_new_edges, -best_score):
                best_new_edges = new_edges
                best_score = dist_to_center
                best_point = pt
        
        if best_point is None:
            # No valid point found, break
            break
        
        placed.append(best_point)
        placed_set.add(best_point)
    
    return placed

def _local_improve(points, candidates, max_iterations=500):
    """Local improvement: try swapping points to increase edge count."""
    best_points = list(points)
    best_set = set(best_points)
    best_count = _count_edges(best_points)
    
    # Precompute candidate list excluding current points
    candidate_list = [c for c in candidates if c not in best_set]
    
    improved = True
    iteration = 0
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        for i in range(len(best_points)):
            old_pt = best_points[i]
            
            for new_pt in candidate_list:
                if new_pt == old_pt or new_pt in best_set:
                    continue
                
                # Check minimum distance constraint
                valid = True
                for j in range(len(best_points)):
                    if i == j:
                        continue
                    if _dist_sq(new_pt[0], new_pt[1], best_points[j][0], best_points[j][1]) < 1:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Compute new edge count efficiently
                # Remove old_pt, add new_pt
                new_set = best_set - {old_pt} | {new_pt}
                
                # Count edges: for each point in new_set, count its neighbors in new_set
                new_count = 0
                for a, b in new_set:
                    for da, db in [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]:
                        if (a+da, b+db) in new_set:
                            new_count += 1
                new_count //= 2
                
                if new_count > best_count:
                    best_points[i] = new_pt
                    best_set = new_set
                    best_count = new_count
                    # Update candidate list
                    candidate_list = [c for c in candidates if c not in best_set]
                    improved = True
                    break
            
            if improved:
                break
    
    return best_points, best_count

def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    # Primary method: hexagonal spiral (should achieve optimal 76 edges)
    spiral_points = _hex_spiral_points(32)
    spiral_count = _count_edges(spiral_points)
    
    best_points = list(spiral_points)
    best_count = spiral_count
    
    # If we already have the optimum, return immediately
    if best_count >= 76:
        return [[a, b] for a, b in best_points]
    
    # Secondary: try greedy with multiple seeds
    candidates = _generate_candidates(max_coord=15)
    
    for seed in range(20):
        if time.time() > deadline:
            break
        
        points = _greedy_construct(candidates, n_points=32, seed=seed)
        if len(points) < 32:
            continue
        
        count = _count_edges(points)
        if count > best_count:
            best_count = count
            best_points = list(points)
            if best_count >= 76:
                return [[a, b] for a, b in best_points]
    
    # Local improvement on the best found
    if time.time() < deadline - 10:
        best_points, best_count = _local_improve(best_points, candidates, max_iterations=200)
    
    return [[a, b] for a, b in best_points]
# EVOLVE-BLOCK-END
