import math
import time

_NEIGH = [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)]

def _dist_sq(a1, b1, a2, b2):
    da, db = a1 - a2, b1 - b2
    return da * da + da * db + db * db

def _hexdist(a,b):
    return (abs(a)+abs(b)+abs(a+b))//2

def _count_edges(pts):
    s = set(pts)
    cnt = 0
    for (a,b) in s:
        for da,db in _NEIGH:
            if (a+da, b+db) in s:
                cnt += 1
    return cnt // 2

def _greedy_build(n=32, max_coord=8):
    """Greedy construction: build a dense cluster minimizing perimeter growth."""
    placed = [(0, 0)]
    placed_set = {(0, 0)}
    
    # Generate candidates in a bounding box
    candidates = []
    for a in range(-max_coord, max_coord + 1):
        for b in range(-max_coord, max_coord + 1):
            if (a, b) not in placed_set:
                candidates.append((a, b))
    
    for _ in range(n - 1):
        best_pt = None
        best_score = None  # (new_edges, -perimeter_growth, -dist_to_centroid)
        
        # Compute centroid
        cx = sum(p[0] for p in placed) / len(placed)
        cy = sum(p[1] for p in placed) / len(placed)
        
        # Compute current perimeter (approximate: number of boundary edges)
        # Actually, let's use a simpler metric: prefer points that maximize new edges
        # and minimize distance from centroid
        
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
            
            # Distance from centroid (approximate lattice distance)
            dx = pt[0] - cx
            dy = pt[1] - cy
            dist_to_center = abs(dx) + abs(dy) + abs(dx + dy)
            
            # Score: primarily new edges, secondarily closeness to centroid
            # This minimizes perimeter growth
            score = (new_edges, -dist_to_center)
            
            if best_score is None or score > best_score:
                best_score = score
                best_pt = pt
        
        if best_pt is None:
            break
        
        placed.append(best_pt)
        placed_set.add(best_pt)
        # Remove from candidates for efficiency
        # (we'll just check membership in placed_set)
    
    return placed

def _local_improve(points, candidates, deadline):
    """Local improvement: try swapping points to increase edge count."""
    best = list(points)
    best_set = set(best)
    current_edges = _count_edges(best)
    
    improved = True
    iteration = 0
    max_iterations = 200
    
    while improved and iteration < max_iterations:
        if time.time() > deadline:
            break
        improved = False
        iteration += 1
        
        best_delta = 0
        best_swap_i = -1
        best_swap_new = None
        
        for i in range(len(best)):
            if time.time() > deadline:
                break
            old_pt = best[i]
            
            for new_pt in candidates:
                if new_pt == old_pt or new_pt in best_set:
                    continue
                
                # Check minimum distance constraint: dist >= 1 from all other placed points
                valid = True
                for j in range(len(best)):
                    if i == j:
                        continue
                    if _dist_sq(new_pt[0], new_pt[1], best[j][0], best[j][1]) < 1:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Calculate edge change
                edges_lost = 0
                for da, db in _NEIGH:
                    if (old_pt[0]+da, old_pt[1]+db) in best_set:
                        edges_lost += 1
                        
                edges_gained = 0
                for da, db in _NEIGH:
                    nb = (new_pt[0]+da, new_pt[1]+db)
                    if nb in best_set and nb != old_pt:
                        edges_gained += 1
                        
                delta = edges_gained - edges_lost
                if delta > best_delta:
                    best_delta = delta
                    best_swap_i = i
                    best_swap_new = new_pt
                    
        if best_delta > 0:
            # Perform the best swap found
            old_pt = best[best_swap_i]
            best[best_swap_i] = best_swap_new
            best_set.remove(old_pt)
            best_set.add(best_swap_new)
            current_edges += best_delta
            improved = True
            continue
        
        # Try 2-point swaps if single swaps didn't help
        if time.time() > deadline:
            break
        
        best_2delta = 0
        best_2i = -1
        best_2j = -1
        best_2new_i = None
        best_2new_j = None
        
        n_pts = len(best)
        for i in range(n_pts):
            if time.time() > deadline:
                break
            for j in range(i+1, n_pts):
                if time.time() > deadline:
                    break
                old_i = best[i]
                old_j = best[j]
                
                for new_i in candidates:
                    if time.time() > deadline:
                        break
                    if new_i == old_i or new_i == old_j or new_i in best_set:
                        continue
                    
                    # Check new_i against all others
                    valid_i = True
                    for k in range(n_pts):
                        if k == i or k == j:
                            continue
                        if _dist_sq(new_i[0], new_i[1], best[k][0], best[k][1]) < 1:
                            valid_i = False
                            break
                    if not valid_i:
                        continue
                    
                    for new_j in candidates:
                        if time.time() > deadline:
                            break
                        if new_j == old_i or new_j == old_j or new_j == new_i or new_j in best_set:
                            continue
                        
                        # Check new_j against all others (excluding i,j positions)
                        valid_j = True
                        for k in range(n_pts):
                            if k == i or k == j:
                                continue
                            if _dist_sq(new_j[0], new_j[1], best[k][0], best[k][1]) < 1:
                                valid_j = False
                                break
                        if not valid_j:
                            continue
                        
                        # Check new_i and new_j against each other
                        if _dist_sq(new_i[0], new_i[1], new_j[0], new_j[1]) < 1:
                            continue
                        
                        # Calculate edge changes
                        edges_lost = 0
                        for da, db in _NEIGH:
                            if (old_i[0]+da, old_i[1]+db) in best_set:
                                edges_lost += 1
                            if (old_j[0]+da, old_j[1]+db) in best_set:
                                edges_lost += 1
                        # Subtract edges between old_i and old_j (counted twice)
                        if _dist_sq(old_i[0], old_i[1], old_j[0], old_j[1]) == 1:
                            edges_lost -= 2
                        
                        edges_gained = 0
                        for da, db in _NEIGH:
                            nb = (new_i[0]+da, new_i[1]+db)
                            if nb in best_set and nb != old_i and nb != old_j:
                                edges_gained += 1
                            nb = (new_j[0]+da, new_j[1]+db)
                            if nb in best_set and nb != old_i and nb != old_j:
                                edges_gained += 1
                        # Add edge between new_i and new_j if they're adjacent
                        if _dist_sq(new_i[0], new_i[1], new_j[0], new_j[1]) == 1:
                            edges_gained += 1
                        
                        delta = edges_gained - edges_lost
                        if delta > best_2delta:
                            best_2delta = delta
                            best_2i = i
                            best_2j = j
                            best_2new_i = new_i
                            best_2new_j = new_j
        
        if best_2delta > 0:
            old_i = best[best_2i]
            old_j = best[best_2j]
            best[best_2i] = best_2new_i
            best[best_2j] = best_2new_j
            best_set.remove(old_i)
            best_set.remove(old_j)
            best_set.add(best_2new_i)
            best_set.add(best_2new_j)
            current_edges += best_2delta
            improved = True
    
    return best, current_edges

def construct():
    n = 32
    max_r = 3 # 1 + 6 + 12 + 18 = 37 >= 32
    
    pts = []
    for a in range(-max_r, max_r+1):
        for b in range(-max_r, max_r+1):
            d = _hexdist(a,b)
            if d <= max_r:
                # Calculate angle for sorting to create a spiral order
                x = a + b/2.0
                y = b * 1.7320508
                angle = math.atan2(y, x)
                pts.append((d, angle, a, b))
                
    # Sort by distance, then angle
    pts.sort(key=lambda x: (x[0], x[1]))
    
    # Take first n
    result = [(a,b) for d, ang, a, b in pts[:n]]
    
    return [[a, b] for a, b in result]
