import time
import random

def construct():
    t0 = time.time()
    deadline = 145.0
    
    # Neighbors in triangular lattice (axial coordinates)
    # (a, b) -> a + b*omega
    # d^2 = da^2 + da*db + db^2
    # Unit distance if d^2 == 1
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    neighbors = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    def count_edges(pts):
        pset = set(pts)
        edges = 0
        for a, b in pts:
            for da, db in neighbors:
                if (a + da, b + db) in pset:
                    edges += 1
        return edges // 2

    def edge_delta_remove_add(pts_set, remove_pt, add_pt):
        lost = 0
        for da, db in neighbors:
            if (remove_pt[0] + da, remove_pt[1] + db) in pts_set:
                lost += 1
        
        gained = 0
        for da, db in neighbors:
            nb = (add_pt[0] + da, add_pt[1] + db)
            if nb in pts_set and nb != remove_pt:
                gained += 1
        
        return gained - lost

    # 1. Initial Configuration: Hexagonal core
    # Points with hex distance <= 2
    # Hex distance of (a,b) is max(|a|, |b|, |a+b|) ? 
    # Let's verify.
    # (1,0) -> max(1,0,1)=1. dist to (0,0) is 1. Correct.
    # (1,1) -> max(1,1,2)=2. dist to (0,0) is sqrt(1+1+1)=sqrt(3). 
    # Wait, axial coords (q,r). 
    # Distance is (|q| + |r| + |q+r|) / 2.
    # Let's use this formula.
    
    def hex_dist(a, b):
        return (abs(a) + abs(b) + abs(a + b)) // 2
        
    # Generate points by hex distance
    # dist 0: 1 pt
    # dist 1: 6 pts
    # dist 2: 12 pts
    # Total 19 pts.
    
    core = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if hex_dist(a, b) <= 2:
                core.append((a, b))
                
    # We need 32 points. We have 19. Need 13 more.
    # Candidates are points with hex_dist 3 or 4.
    # Let's collect candidates with hex_dist 3 first.
    candidates = []
    for a in range(-5, 6):
        for b in range(-5, 6):
            d = hex_dist(a, b)
            if d >= 3 and d <= 4: # Look a bit further just in case
                candidates.append((a, b))
                
    # Greedy selection of 13 candidates to add to core
    current = list(core)
    current_set = set(current)
    
    # For each candidate, calculate how many edges it adds to current_set
    # Sort by number of added edges (descending), break ties by hex_dist (ascending)
    
    def get_score(pt, pset):
        score = 0
        for da, db in neighbors:
            if (pt[0] + da, pt[1] + db) in pset:
                score += 1
        return score

    # We will add 13 points one by one
    # This is a greedy heuristic, might not be optimal, but good start.
    # Then we polish.
    
    remaining_needed = 32 - len(current)
    available = list(candidates)
    
    # To make greedy more robust, we can do a few passes or just sort once.
    # Sorting once might be okay if we update the set.
    
    for _ in range(remaining_needed):
        if not available:
            break
        best_pt = None
        best_score = -1
        best_dist = 999
        for pt in available:
            if pt in current_set:
                continue
            s = get_score(pt, current_set)
            d = hex_dist(pt[0], pt[1])
            if s > best_score or (s == best_score and d < best_dist):
                best_score = s
                best_dist = d
                best_pt = pt
        
        if best_pt:
            current.append(best_pt)
            current_set.add(best_pt)
            available.remove(best_pt)
            
    # Now we have a set of 32 points.
    # Polish with local search.
    
    best_points = list(current)
    best_edges = count_edges(best_points)
    
    if best_edges >= 76:
        return [list(p) for p in best_points]
        
    # Local Search: 1-swap
    # We will try to swap any point in current with any point in a larger pool.
    # Pool: points with hex_dist <= 5 that are not in current.
    
    pool = []
    for a in range(-6, 7):
        for b in range(-6, 7):
            if (a, b) not in current_set:
                pool.append((a, b))
                
    random.seed(42)
    current = list(best_points)
    current_set = set(current)
    cur_edges = best_edges
    
    stall = 0
    stall_limit = 1000
    
    while (time.time() - t0) < deadline:
        if stall > stall_limit:
            # Restart or perturb
            # Perturb: random swap
            i = random.randint(0, 31)
            j = random.randint(0, len(pool)-1)
            # Ensure distinct
            if pool[j] in current_set:
                continue
            old = current[i]
            new = pool[j]
            
            # Calculate delta
            temp_set = current_set - {old}
            delta = edge_delta_remove_add(temp_set, old, new)
            
            # Always accept if it improves, sometimes if it doesn't?
            # For speed, let's just do hill climbing with restarts.
            # If we stall, we break and restart with a new random seed or perturbation.
            
            # Actually, let's just do random restarts if we stall.
            break
            
        # Pick random point to swap
        i = random.randint(0, 31)
        old = current[i]
        
        # Pick random point from pool
        j = random.randint(0, len(pool) - 1)
        new = pool[j]
        
        if new in current_set:
            stall += 1
            continue
            
        temp_set = current_set - {old}
        delta = edge_delta_remove_add(temp_set, old, new)
        
        if delta > 0:
            current[i] = new
            current_set.discard(old)
            current_set.add(new)
            cur_edges += delta
            stall = 0
            
            # Update pool: swap old into pool
            pool[j] = old
            
            if cur_edges > best_edges:
                best_edges = cur_edges
                best_points = list(current)
                if best_edges >= 76:
                    break
        else:
            stall += 1
            
    # If we didn't reach 76, try 2-opt or more restarts?
    # Given the small size, 1-swap hill climbing from a good start (greedy hex) should reach 76.
    # If not, we might need to try a few more restarts.
    
    if best_edges < 76:
        # Try a few more restarts with different random seeds
        for seed in range(10):
            if (time.time() - t0) > deadline:
                break
            random.seed(seed)
            current = list(best_points)
            current_set = set(current)
            cur_edges = best_edges
            stall = 0
            
            while (time.time() - t0) < deadline and stall < 2000:
                i = random.randint(0, 31)
                old = current[i]
                j = random.randint(0, len(pool) - 1)
                new = pool[j]
                
                if new in current_set:
                    stall += 1
                    continue
                    
                temp_set = current_set - {old}
                delta = edge_delta_remove_add(temp_set, old, new)
                
                if delta > 0:
                    current[i] = new
                    current_set.discard(old)
                    current_set.add(new)
                    cur_edges += delta
                    stall = 0
                    pool[j] = old
                    
                    if cur_edges > best_edges:
                        best_edges = cur_edges
                        best_points = list(current)
                        if best_edges >= 76:
                            break
                else:
                    stall += 1
                    
            if best_edges >= 76:
                break

    return [list(p) for p in best_points]
