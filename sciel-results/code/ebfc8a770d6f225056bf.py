"""Construct 32 triangular-lattice points maximizing unit-distance edges."""
import time
import random
from itertools import combinations

def construct():
    # Deadline guard
    t0 = time.time()
    deadline = 145.0  # leave safety margin before 150s kill
    
    # Triangular lattice with omega = exp(i*pi/3) = 1/2 + i*sqrt(3)/2
    # Point a + b*omega has Cartesian coords (a + b/2, b*sqrt(3)/2)
    # d^2 between (a1,b1) and (a2,b2) = da^2 + da*db + db^2 where da=a1-a2, db=b1-b2
    # Unit distance: d^2 = 1 => (da,db) in {(1,0),(0,1),(1,-1),(-1,0),(0,-1),(-1,1)}
    
    neighbors = [(1,0),(0,1),(1,-1),(-1,0),(0,-1),(-1,1)]
    
    def count_edges(pts):
        pset = set(pts)
        edges = 0
        for a, b in pts:
            for da, db in neighbors:
                if (a+da, b+db) in pset:
                    edges += 1
        return edges // 2
    
    def edge_delta_remove_add(pts_set, remove_pt, add_pt):
        """Compute change in edge count when removing remove_pt and adding add_pt."""
        # Edges lost from remove_pt
        lost = 0
        for da, db in neighbors:
            if (remove_pt[0]+da, remove_pt[1]+db) in pts_set:
                lost += 1
        
        # Edges gained from add_pt (to all points except remove_pt)
        gained = 0
        for da, db in neighbors:
            nb = (add_pt[0]+da, add_pt[1]+db)
            if nb in pts_set and nb != remove_pt:
                gained += 1
        
        return gained - lost
    
    # Generate candidate points by increasing distance from origin
    # d^2 = a^2 + a*b + b^2
    candidates = []
    for a in range(-64, 65):
        for b in range(-64, 65):
            d2 = a*a + a*b + b*b
            candidates.append((d2, a, b))
    
    candidates.sort()
    
    # Take the 32 closest to origin as initial set
    points = [(a, b) for d2, a, b in candidates[:32]]
    edges = count_edges(points)
    
    # If we already have 76, return immediately
    if edges >= 76:
        return [list(p) for p in points]
    
    # Local search with swaps
    best_points = list(points)
    best_edges = edges
    
    # Pool of candidate points to swap in (next 100 after the initial 32)
    pool_size = 100
    swap_pool = [(a, b) for d2, a, b in candidates[32:32+pool_size]]
    
    random.seed(42)
    
    current = list(points)
    current_set = set(current)
    cur_edges = edges
    
    max_iters = 200000
    stall_counter = 0
    stall_limit = 5000
    
    for iteration in range(max_iters):
        if (time.time() - t0) > deadline:
            break
        
        # Pick random point to remove
        idx = random.randint(0, len(current) - 1)
        removed = current[idx]
        
        # Pick random point to add from swap pool
        sidx = random.randint(0, len(swap_pool) - 1)
        add_pt = swap_pool[sidx]
        
        # Check if add_pt is already in current (shouldn't be, but just in case)
        if add_pt in current_set:
            continue
        
        # Compute delta
        temp_set = current_set - {removed}
        delta = edge_delta_remove_add(temp_set, removed, add_pt)
        
        if delta > 0:
            current[idx] = add_pt
            current_set.discard(removed)
            current_set.add(add_pt)
            cur_edges += delta
            stall_counter = 0
            
            if cur_edges > best_edges:
                best_edges = cur_edges
                best_points = list(current)
                if best_edges >= 76:
                    break
            
            # Update swap pool: swap the removed point into the pool
            swap_pool[sidx] = removed
        else:
            stall_counter += 1
            if stall_counter > stall_limit:
                # Reset stall counter and try a different random seed
                stall_counter = 0
                random.seed(random.randint(0, 1000000))
    
    # Final improvement: try 2-opt style swaps (remove 2, add 2)
    if best_edges < 76 and (time.time() - t0) < deadline - 5:
        current = list(best_points)
        current_set = set(current)
        cur_edges = best_edges
        
        # Get a larger pool for 2-opt
        pool2 = [(a, b) for d2, a, b in candidates[32:32+200]]
        
        stall2 = 0
        stall2_limit = 10000
        
        for iteration in range(100000):
            if (time.time() - t0) > deadline:
                break
            
            if stall2 > stall2_limit:
                break
            
            # Pick 2 points to remove
            idx1, idx2 = random.sample(range(len(current)), 2)
            rem1, rem2 = current[idx1], current[idx2]
            
            # Pick 2 points to add from pool
            sidx1, sidx2 = random.sample(range(len(pool2)), 2)
            add1, add2 = pool2[sidx1], pool2[sidx2]
            
            if add1 in current_set or add2 in current_set:
                stall2 += 1
                continue
            
            # Compute delta for 2-opt
            temp_set = current_set - {rem1, rem2}
            
            # Edges lost
            lost = 0
            for pt in [rem1, rem2]:
                for da, db in neighbors:
                    nb = (pt[0]+da, pt[1]+db)
                    if nb in temp_set:
                        lost += 1
                # Edge between rem1 and rem2 (if they were connected)
            
            # Check if rem1 and rem2 are connected
            da = rem1[0] - rem2[0]
            db = rem1[1] - rem2[1]
            if da*da + da*db + db*db == 1:
                lost += 1
            
            # Edges gained
            gained = 0
            for pt in [add1, add2]:
                for da, db in neighbors:
                    nb = (pt[0]+da, pt[1]+db)
                    if nb in temp_set:
                        gained += 1
            
            # Edge between add1 and add2
            da = add1[0] - add2[0]
            db = add1[1] - add2[1]
            if da*da + da*db + db*db == 1:
                gained += 1
            
            delta = gained - lost
            
            if delta > 0:
                current[idx1] = add1
                current[idx2] = add2
                current_set.discard(rem1)
                current_set.discard(rem2)
                current_set.add(add1)
                current_set.add(add2)
                cur_edges += delta
                stall2 = 0
                
                if cur_edges > best_edges:
                    best_edges = cur_edges
                    best_points = list(current)
                    if best_edges >= 76:
                        break
                
                # Update pool
                pool2[sidx1] = rem1
                pool2[sidx2] = rem2
            else:
                stall2 += 1
    
    points = best_points
    return [list(p) for p in points]
