"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
from collections import defaultdict

def _doubled_area(p1, p2, p3):
    """Compute absolute doubled area of triangle with vertices p1, p2, p3."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

def _evaluate(points):
    """Find the maximum multiplicity of any single positive doubled area."""
    n = len(points)
    area_counts = defaultdict(int)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = _doubled_area(points[i], points[j], points[k])
                if a > 0:
                    area_counts[a] += 1
    if not area_counts:
        return 0
    return max(area_counts.values())

def _count_area_for_points(points, target_area):
    """Count how many triples in points have doubled area equal to target_area."""
    count = 0
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _doubled_area(points[i], points[j], points[k]) == target_area:
                    count += 1
    return count

def construct():
    """
    Construct 30 distinct integer points in [-30,30]^2 that maximize
    the largest multiplicity of a single positive doubled area.
    
    Uses graph/set-cover approach: for each candidate target area A,
    pre-compute which pairs of points form triangles of area A with a
    third point, then greedily select points to maximize count of
    triangles with area A.
    """
    all_points = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    n_total = len(all_points)
    
    # Candidate target areas to try
    target_areas = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    
    best_score = 0
    best_points = None
    
    # For each target area, try greedy selection
    for target_area in target_areas:
        # Pre-compute for each pair (i,j), which third points k give area = target_area
        # This is expensive, so we'll use a smarter approach:
        # For a fixed target area A, we want to select 30 points such that
        # many triples have doubled area A.
        
        # Greedy approach: start with a seed point, then iteratively add
        # the point that creates the most new triangles of area A
        # with already-selected points.
        
        # Try multiple starting seeds
        for seed_idx in range(0, n_total, 100):  # Sample seeds
            seed = all_points[seed_idx]
            selected = [seed]
            
            # Iteratively add points that maximize new triangles of area A
            for _ in range(29):
                best_new_point = None
                best_new_count = -1
                
                # Sample candidate points (not all 3600)
                candidates = all_points[seed_idx-50:seed_idx+50] if seed_idx+50 < n_total else all_points[:100]
                if len(candidates) < 50:
                    candidates = all_points[:100]
                
                for candidate in candidates:
                    if candidate in selected:
                        continue
                    
                    # Count how many triangles with area A would be formed
                    # by adding this candidate to selected
                    new_count = 0
                    for i in range(len(selected)):
                        for j in range(i+1, len(selected)):
                            if _doubled_area(selected[i], selected[j], candidate) == target_area:
                                new_count += 1
                    
                    if new_count > best_new_count:
                        best_new_count = new_count
                        best_new_point = candidate
                
                if best_new_point is None:
                    break
                selected.append(best_new_point)
            
            if len(selected) == 30 and len(set(selected)) == 30:
                score = _count_area_for_points(selected, target_area)
                if score > best_score:
                    best_score = score
                    best_points = selected[:]
    
    # Also try some structured configurations as fallback
    if best_points is None or best_score < 10:
        # 6x5 grid
        points = [(x, y) for x in range(6) for y in range(5)]
        score = _evaluate(points)
        if score > best_score:
            best_score = score
            best_points = points
    
    # 15x2 grid
    points = [(x, y) for x in range(15) for y in range(2)]
    score = _evaluate(points)
    if score > best_score:
        best_score = score
        best_points = points
    
    # 10x3 grid
    points = [(x, y) for x in range(10) for y in range(3)]
    score = _evaluate(points)
    if score > best_score:
        best_score = score
        best_points = points
    
    if best_points is None:
        best_points = [(x, y) for x in range(6) for y in range(5)]
    
    return best_points
# EVOLVE-BLOCK-END
