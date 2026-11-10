"""Improved program: hexagonal construction + local search for 32 points on triangular lattice."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _dist2(a1, b1, a2, b2):
    """Squared distance on triangular lattice."""
    da = a1 - a2
    db = b1 - b2
    return da*da + da*db + db*db

def _count_edges(points):
    """Count unit-distance pairs among given (a,b) points."""
    point_set = set(tuple(p) for p in points)
    neighbors = [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]
    edge_count = 0
    for a, b in point_set:
        for da, db in neighbors:
            if (a+da, b+db) in point_set:
                edge_count += 1
    return edge_count // 2

def _edge_count_delta(point_set, add_pt, remove_pt):
    """Compute edge count change when adding add_pt and removing remove_pt.
    Returns the delta in edge count."""
    neighbors = [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]
    
    # Edges lost by removing remove_pt
    lost = 0
    for da, db in neighbors:
        if (remove_pt[0]+da, remove_pt[1]+db) in point_set:
            lost += 1
    
    # Edges gained by adding add_pt (but not counting edge to remove_pt twice)
    gained = 0
    for da, db in neighbors:
        np = (add_pt[0]+da, add_pt[1]+db)
        if np in point_set:
            gained += 1
        elif np == remove_pt:
            gained += 1  # This edge is restored
    
    return gained - lost

def _hexagon_points(r):
    """All points with hex distance <= r from origin: max(|a|,|b|,|a+b|) <= r."""
    pts = []
    for a in range(-r, r+1):
        for b in range(-r, r+1):
            if max(abs(a), abs(b), abs(a+b)) <= r:
                pts.append((a, b))
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # safety margin before 150s kill
    
    # Hexagon of radius 2 has 1 + 3*2*3 = 19 points
    core = _hexagon_points(2)
    # Hexagon of radius 3 has 1 + 3*3*4 = 37 points
    all_r3 = _hexagon_points(3)
    ring3 = [p for p in all_r3 if max(abs(p[0]), abs(p[1]), abs(p[0]+p[1])) == 3]
    
    n_need = 32 - len(core)  # 32 - 19 = 13
    n_ring = len(ring3)  # 37 - 19 = 18
    
    # Greedy initial selection from ring3
    selected = set()
    core_set = set(core)
    for _ in range(n_need):
        best_gain = -1
        best_idx = -1
        for i, p in enumerate(ring3):
            if i in selected:
                continue
            gain = 0
            neighbors = [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]
            for da, db in neighbors:
                np = (p[0]+da, p[1]+db)
                if np in core_set:
                    gain += 1
                elif any(np == ring3[j] for j in selected):
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_idx >= 0:
            selected.add(best_idx)
    
    best_selection = sorted(selected)
    current_points = core + [ring3[i] for i in best_selection]
    best_edges = _count_edges(current_points)
    
    # Local search: single swaps with incremental edge counting
    if time.time() < deadline - 5.0:
        current_selection = set(best_selection)
        current_edges = best_edges
        # Build the full point set for quick lookup
        current_point_set = set(core + [ring3[i] for i in current_selection])
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for keep in list(current_selection):
                if time.time() >= deadline - 5.0:
                    break
                keep_pt = ring3[keep]
                for add in range(n_ring):
                    if add in current_selection:
                        continue
                    add_pt = ring3[add]
                    # Compute delta using incremental method
                    delta = _edge_count_delta(current_point_set, add_pt, keep_pt)
                    e = current_edges + delta
                    if e > current_edges:
                        # Update the point set
                        current_point_set.remove(keep_pt)
                        current_point_set.add(add_pt)
                        current_edges = e
                        current_selection = current_selection - {keep} | {add}
                        improved = True
                        if current_edges >= 76:
                            break
                if current_edges >= 76:
                    break
            if current_edges >= 76:
                break
        
        if current_edges > best_edges:
            best_edges = current_edges
            best_selection = sorted(current_selection)
    
    # Final local search with 2-swaps if time permits, using incremental counting
    if time.time() < deadline - 10.0 and best_edges < 76:
        current_selection = set(best_selection)
        current_edges = best_edges
        current_point_set = set(core + [ring3[i] for i in current_selection])
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            sel_list = sorted(current_selection)
            for i in range(len(sel_list)):
                for j in range(i+1, len(sel_list)):
                    if time.time() >= deadline - 10.0:
                        break
                    keep1, keep2 = sel_list[i], sel_list[j]
                    keep1_pt = ring3[keep1]
                    keep2_pt = ring3[keep2]
                    for add1 in range(n_ring):
                        if add1 in current_selection:
                            continue
                        add1_pt = ring3[add1]
                        for add2 in range(n_ring):
                            if add2 in current_selection or add2 == add1:
                                continue
                            add2_pt = ring3[add2]
                            # Compute delta for removing keep1, keep2 and adding add1, add2
                            # First compute edges lost
                            lost = 0
                            neighbors = [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]
                            for da, db in neighbors:
                                np1 = (keep1_pt[0]+da, keep1_pt[1]+db)
                                np2 = (keep2_pt[0]+da, keep2_pt[1]+db)
                                if np1 in current_point_set and np1 != keep2_pt:
                                    lost += 1
                                if np2 in current_point_set and np2 != keep1_pt:
                                    lost += 1
                            
                            # Edges gained
                            gained = 0
                            for da, db in neighbors:
                                np1 = (add1_pt[0]+da, add1_pt[1]+db)
                                np2 = (add2_pt[0]+da, add2_pt[1]+db)
                                if np1 in current_point_set and np1 != keep1_pt and np1 != keep2_pt:
                                    gained += 1
                                elif np1 == keep1_pt or np1 == keep2_pt:
                                    gained += 1
                                if np2 in current_point_set and np2 != keep1_pt and np2 != keep2_pt:
                                    gained += 1
                                elif np2 == keep1_pt or np2 == keep2_pt:
                                    gained += 1
                            
                            # Edge between add1 and add2
                            if (add1_pt[0]-add2_pt[0])**2 + (add1_pt[0]-add2_pt[0])*(add1_pt[1]-add2_pt[1]) + (add1_pt[1]-add2_pt[1])**2 == 1:
                                gained += 1
                            
                            # Edge between keep1 and keep2 (already counted in lost if they were neighbors)
                            if (keep1_pt[0]-keep2_pt[0])**2 + (keep1_pt[0]-keep2_pt[0])*(keep1_pt[1]-keep2_pt[1]) + (keep1_pt[1]-keep2_pt[1])**2 == 1:
                                lost += 1
                            
                            delta = gained - lost
                            e = current_edges + delta
                            if e > current_edges:
                                # Update point set
                                current_point_set.remove(keep1_pt)
                                current_point_set.remove(keep2_pt)
                                current_point_set.add(add1_pt)
                                current_point_set.add(add2_pt)
                                current_edges = e
                                current_selection = current_selection - {keep1, keep2} | {add1, add2}
                                improved = True
                                if current_edges >= 76:
                                    break
                        if current_edges >= 76:
                            break
                    if current_edges >= 76:
                        break
                if current_edges >= 76:
                    break
            if current_edges >= 76:
                break
        
        if current_edges > best_edges:
            best_edges = current_edges
            best_selection = sorted(current_selection)
    
    # Build final result
    final_points = core + [ring3[i] for i in best_selection]
    
    # Validation
    assert len(final_points) == 32, f"Got {len(final_points)} points"
    assert len(set(tuple(p) for p in final_points)) == 32, "Points not distinct"
    for a, b in final_points:
        assert abs(a) <= 64 and abs(b) <= 64, f"Point ({a},{b}) out of bounds"
    
    result = [[a, b] for a, b in final_points]
    return result
# EVOLVE-BLOCK-END
