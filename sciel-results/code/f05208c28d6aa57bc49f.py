import time
import math
import random

def construct():
    """
    Construct 32 points on the triangular lattice maximizing unit-distance pairs.
    Optimum is 76 edges (Harborth's theorem for n=32).
    
    Strategy: Build a hexagonal cluster (center + rings 0-2 = 19 points, plus 13 from ring 3).
    Use local search to optimize which 13 points from ring 3 to include.
    """
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Neighbor offsets for unit distance on triangular lattice
    # Points a + b*omega, omega = exp(i*pi/3)
    # Unit distance: a^2 + ab + b^2 == 1
    neighbors = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    def dist_sq(p1, p2):
        da = p1[0] - p2[0]
        db = p1[1] - p2[1]
        return da * da + da * db + db * db
    
    # Generate hexagonal rings
    # Ring 0: just (0,0)
    # Ring k: 6k points at hexagonal distance k
    
    def hex_ring(k):
        """Generate all points at hexagonal distance k from origin."""
        if k == 0:
            return [(0, 0)]
        points = []
        # The hexagonal ring of radius k consists of 6 sides of k points each
        # Starting from (k, 0) and going around
        # Directions: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
        dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        for d in range(6):
            dx, dy = dirs[d]
            for i in range(k):
                # Start at corner and move along the side
                # Corner d is at position: sum of first d directions * k
                cx, cy = 0, 0
                for j in range(d):
                    cx += dirs[j][0] * k
                    cy += dirs[j][1] * k
                # Now move i steps in direction d
                px = cx + dx * i
                py = cy + dy * i
                points.append((px, py))
        return points
    
    # Build the hexagonal cluster
    # Ring 0: 1 point
    # Ring 1: 6 points
    # Ring 2: 12 points
    # Ring 3: 18 points (need 13 of these)
    
    ring0 = hex_ring(0)
    ring1 = hex_ring(1)
    ring2 = hex_ring(2)
    ring3 = hex_ring(3)
    
    # Verify counts
    assert len(ring0) == 1, f"ring0 has {len(ring0)}"
    assert len(ring1) == 6, f"ring1 has {len(ring1)}"
    assert len(ring2) == 12, f"ring2 has {len(ring2)}"
    assert len(ring3) == 18, f"ring3 has {len(ring3)}"
    
    # Fixed points: rings 0, 1, 2 (19 points)
    fixed_points = set(ring0 + ring1 + ring2)
    # Variable points: ring 3 (18 points, choose 13)
    variable_points = ring3
    
    # Build adjacency for all points in rings 0-3
    all_points = list(fixed_points) + variable_points
    point_to_idx = {p: i for i, p in enumerate(all_points)}
    n = len(all_points)
    
    # Precompute adjacency: which points are at unit distance
    adj = [set() for _ in range(n)]
    for i, p in enumerate(all_points):
        for dx, dy in neighbors:
            neighbor = (p[0] + dx, p[1] + dy)
            if neighbor in point_to_idx:
                j = point_to_idx[neighbor]
                adj[i].add(j)
    
    # Count edges in a set of points
    def count_edges(selected_indices):
        """Count unit-distance pairs among selected indices."""
        selected_set = set(selected_indices)
        edges = 0
        for i in selected_set:
            for j in adj[i]:
                if j in selected_set and i < j:
                    edges += 1
        return edges
    
    # Initial selection: first 13 points from ring 3
    # Ring 3 indices: 19 to 36
    ring3_indices = list(range(19, 37))
    
    # Start with a reasonable initial selection
    # Pick points that are "closest" to the fixed points in terms of connectivity
    # For now, just take the first 13
    initial_selection = set(list(range(n)))  # all points
    # Remove 5 from ring 3
    # Let's start by removing the last 5
    initial_selection = set(range(19)) | set(ring3_indices[:13])
    
    best_edges = count_edges(initial_selection)
    best_selection = set(initial_selection)
    
    # Local search: try swapping a point in ring 3 with one not in ring 3
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        in_ring3 = list(best_selection & set(ring3_indices))
        out_ring3 = [i for i in ring3_indices if i not in best_selection]
        
        for i in in_ring3:
            if time.time() >= deadline - 5:
                break
            for j in out_ring3:
                new_selection = best_selection - {i} | {j}
                edges = count_edges(new_selection)
                if edges > best_edges:
                    best_edges = edges
                    best_selection = new_selection
                    improved = True
                    break
            if improved:
                break
    
    # Also try 2-swap moves for better optimization
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        in_ring3 = list(best_selection & set(ring3_indices))
        out_ring3 = [i for i in ring3_indices if i not in best_selection]
        
        for i in range(len(in_ring3)):
            if time.time() >= deadline - 5:
                break
            for j in range(i + 1, len(in_ring3)):
                if time.time() >= deadline - 5:
                    break
                for k in out_ring3:
                    for l in out_ring3:
                        if k >= l:
                            continue
                        new_selection = best_selection - {in_ring3[i], in_ring3[j]} | {k, l}
                        edges = count_edges(new_selection)
                        if edges > best_edges:
                            best_edges = edges
                            best_selection = new_selection
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    
    # Convert selection to point coordinates
    result = [all_points[i] for i in sorted(best_selection)]
    
    # Verify we have exactly 32 distinct points
    assert len(result) == 32, f"Got {len(result)} points"
    assert len(set(result)) == 32, "Points not distinct"
    
    # Verify all points are within bounds
    for a, b in result:
        assert abs(a) <= 64 and abs(b) <= 64, f"Point ({a},{b}) out of bounds"
    
    # Verify minimum distance is at least 1
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            d = dist_sq(result[i], result[j])
            assert d >= 1, f"Points {result[i]} and {result[j]} too close: d^2={d}"
    
    # Count final edges
    final_edges = count_edges(best_selection)
    
    return [[a, b] for a, b in result]
