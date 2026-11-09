import random
import time

def construct():
    # Deadline guard
    start_time = time.time()
    max_time = 10.0  # 10 seconds is plenty for this small problem
    
    # 1. Generate candidate points
    # We need 32 points. A hexagon of radius 4 has 61 points.
    # A hexagon of radius 5 has 91 points.
    # Let's use radius 5 to allow flexibility for the optimizer to shape the boundary.
    R = 5
    candidates = []
    for a in range(-R, R + 1):
        for b in range(-R, R + 1):
            if abs(a) <= R and abs(b) <= R and abs(a - b) <= R:
                candidates.append((a, b))
    
    # Map point to index for fast lookup
    point_to_idx = {p: i for i, p in enumerate(candidates)}
    
    # Precompute neighbors for each candidate
    # Neighbors of (a,b) are (a+1,b), (a-1,b), (a,b+1), (a,b-1), (a+1,b-1), (a-1,b+1)
    # Wait, standard axial coords for triangular lattice:
    # If omega = exp(i*pi/3), points are a + b*omega.
    # Distance squared between (a,b) and (c,d):
    # |a + b*omega - c - d*omega|^2 = |(a-c) + (b-d)*omega|^2
    # Let x = a-c, y = b-d.
    # |x + y*e^{i*pi/3}|^2 = x^2 + y^2 + 2xy cos(pi/3) = x^2 + y^2 + xy.
    # Distance is 1 if x^2 + y^2 + xy = 1.
    # Solutions: (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1).
    # So neighbors are:
    # (a+1, b), (a-1, b), (a, b+1), (a, b-1), (a+1, b-1), (a-1, b+1).
    
    neighbors = {}
    for p in candidates:
        a, b = p
        nbs = [
            (a+1, b), (a-1, b), 
            (a, b+1), (a, b-1), 
            (a+1, b-1), (a-1, b+1)
        ]
        neighbors[p] = [n for n in nbs if n in point_to_idx]
        
    # 2. Initialize with a "hexagonal" selection of 32 points
    # Sort candidates by hexagonal distance from origin: max(|a|, |b|, |a-b|)
    def hex_dist(p):
        a, b = p
        return max(abs(a), abs(b), abs(a-b))
        
    sorted_candidates = sorted(candidates, key=hex_dist)
    
    # Take the first 32
    initial_set = set(sorted_candidates[:32])
    
    def count_edges(S):
        count = 0
        # To avoid double counting, only count if neighbor index > current index?
        # Or just sum degrees and divide by 2.
        total_deg = 0
        for p in S:
            for nb in neighbors[p]:
                if nb in S:
                    total_deg += 1
        return total_deg // 2

    current_edges = count_edges(initial_set)
    best_set = initial_set.copy()
    best_edges = current_edges
    
    # If we already hit 76, return
    if best_edges >= 76:
        return [[a, b] for a, b in best_set]
        
    # 3. Local Search (Hill Climbing with Random Restarts)
    # We will perform random swaps.
    
    # Helper to calculate delta edges for swapping u (out) and v (in)
    # S is the current set.
    # New set S' = S - {u} + {v}
    # Edges lost: edges from u to S - {u}.
    # Edges gained: edges from v to S - {u}.
    # Note: if u and v are neighbors, they don't form an edge in S' anyway.
    
    def get_delta(S, u, v):
        # u is in S, v is not in S
        # Count neighbors of u in S (excluding u itself, which is not in neighbors)
        # But we are removing u, so we lose edges (u, w) where w in S and w != u.
        deg_u = 0
        for nb in neighbors[u]:
            if nb in S:
                deg_u += 1
                
        # Count neighbors of v in S (excluding u, because u is removed)
        # Actually, v is not in S yet.
        # We gain edges (v, w) where w in S and w != u.
        deg_v = 0
        for nb in neighbors[v]:
            if nb in S and nb != u:
                deg_v += 1
                
        return deg_v - deg_u

    n_points = len(candidates)
    num_selected = 32
    
    # Shuffle candidates for random restarts/swaps
    # We maintain a list of selected and unselected indices for fast sampling
    
    for restart in range(20):
        if time.time() - start_time > max_time:
            break
            
        if restart == 0:
            current_set = best_set.copy()
        else:
            # Random restart: Pick 32 random points from candidates
            # Or perturb the best set
            current_set = best_set.copy()
            # Perform some random swaps to escape local optimum
            for _ in range(50):
                # Pick random u in current_set
                u = random.choice(list(current_set))
                # Pick random v not in current_set
                # To make this efficient, we can just pick a random candidate and check
                # Since N is small (91), this is fine
                v = random.choice(candidates)
                if v in current_set:
                    continue
                current_set.remove(u)
                current_set.add(v)
                
        current_edges = count_edges(current_set)
        
        # Hill climb
        improved = True
        while improved and (time.time() - start_time < max_time):
            improved = False
            # Try all possible swaps? 32 * (91-32) = 32 * 59 = 1888 swaps.
            # Each swap check is fast (iterating over ~6 neighbors).
            # This is very fast.
            
            # To be safe, iterate over all u in current_set and v in candidates \ current_set
            # We can optimize by pre-computing lists, but N is small.
            
            for u in list(current_set):
                if time.time() - start_time > max_time:
                    break
                for v in candidates:
                    if v in current_set:
                        continue
                    
                    delta = get_delta(current_set, u, v)
                    if delta > 0:
                        # Perform swap
                        current_set.remove(u)
                        current_set.add(v)
                        current_edges += delta
                        improved = True
                        # If we reach 76, we can stop early
                        if current_edges >= 76:
                            best_set = current_set.copy()
                            best_edges = current_edges
                            return [[a, b] for a, b in best_set]
                        # Restart the inner loop to pick a new u? 
                        # Or just continue? Continuing might miss better swaps involving the new v.
                        # But hill climbing usually just takes the first improvement or best improvement.
                        # Taking the first improvement is faster. Let's restart the u loop.
                        break
                if improved:
                    # If we broke the v loop, we should restart the u loop to ensure we don't miss anything
                    # due to the set changing.
                    if current_edges >= 76:
                        break
                    # Restart u loop
                    # The 'break' above breaks the v loop. We need to break the u loop too to restart.
                    # But we want to restart the whole while loop iteration effectively.
                    # Let's just break both and let the while loop continue.
                    # Actually, if we break the v loop, we should break the u loop to re-evaluate from scratch.
                    # But 'improved' is True, so while continues.
                    # We need to break the u loop.
                    # The structure:
                    # for u:
                    #   for v:
                    #     if delta > 0:
                    #       swap
                    #       improved = True
                    #       break v
                    #   if improved:
                    #     break u  # <-- Add this
                    
            if best_edges < current_edges:
                best_set = current_set.copy()
                best_edges = current_edges
                
            if best_edges >= 76:
                break

    # Return the best found
    return [[a, b] for a, b in best_set]
