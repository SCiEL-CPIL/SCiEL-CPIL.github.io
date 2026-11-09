import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Target n
    n = 35
    
    # Initialize random graph
    # Start with p=0.5
    edges = set()
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.5:
                edges.add((i, j))
    
    def get_neighbors(v):
        return {u for (u, w) in edges if u == v} | {w for (u, w) in edges if w == v}
    
    def has_edge(u, v):
        if u > v: u, v = v, u
        return (u, v) in edges
    
    def count_k4():
        count = 0
        verts = list(range(n))
        # Optimize: Iterate over edges (u,v), then common neighbors
        # For each pair of common neighbors (w,x) that are connected, it's a K4
        # But this counts each K4 multiple times.
        # A K4 has 6 edges.
        # Let's just iterate over combinations of 4 vertices.
        # For n=35, C(35,4) = 52395. This is fast.
        for combo in itertools.combinations(verts, 4):
            u, v, w, x = combo
            if has_edge(u, v) and has_edge(u, w) and has_edge(u, x) and \
               has_edge(v, w) and has_edge(v, x) and has_edge(w, x):
                count += 1
        return count

    def count_ind6():
        count = 0
        verts = list(range(n))
        # C(35, 6) = 1623160. This might be slow if called often.
        # Optimization: Only check if the subgraph induced by the 6 vertices is empty.
        # We can prune: if any edge exists, skip.
        for combo in itertools.combinations(verts, 6):
            is_indep = True
            # Check all 15 pairs
            for i in range(6):
                for j in range(i+1, 6):
                    if has_edge(combo[i], combo[j]):
                        is_indep = False
                        break
                if not is_indep:
                    break
            if is_indep:
                count += 1
        return count

    def total_cost():
        return count_k4() + count_ind6()
    
    # Initial cost
    cost = total_cost()
    
    best_n = 0
    best_edges = set()
    
    # If initial is valid, return
    if cost == 0:
        best_n = n
        best_edges = edges
    else:
        # Local Search
        # To speed up, we might need a faster violation finder or cost estimator.
        # Given time constraints, full count is risky.
        # Let's try a few steps of full count.
        # If it's too slow, we fallback to smaller n.
        
        steps = 0
        max_steps = 2000 # Arbitrary limit
        
        while time.time() < deadline and cost > 0 and steps < max_steps:
            # Pick a random edge
            u, v = random.randint(0, n-1), random.randint(0, n-1)
            if u == v: continue
            if u > v: u, v = v, u
            
            # Try flipping
            old_cost = cost
            if (u, v) in edges:
                edges.remove((u, v))
            else:
                edges.add((u, v))
            
            # Recalculate cost
            # This is the bottleneck.
            # If this takes > 0.5s per step, we can only do 200 steps in 100s.
            # C(35,6) is 1.6M. In Python, 1.6M iterations with inner loops might take ~1s.
            # So we are limited to ~100 steps.
            # This might not be enough to reach cost 0 from random.
            
            new_cost = total_cost()
            steps += 1
            
            if new_cost < old_cost:
                cost = new_cost
                if cost == 0:
                    best_n = n
                    best_edges = edges
                    break
            else:
                # Revert
                if (u, v) in edges:
                    edges.remove((u, v))
                else:
                    edges.add((u, v))
                # With small probability, accept worse (annealing)
                if random.random() < 0.1:
                    cost = new_cost
                    # Keep the change
                    pass
                else:
                    # Revert is already done if we didn't keep it?
                    # Wait, if new_cost > old_cost, we revert.
                    # The code above reverts immediately if new_cost >= old_cost?
                    # No, I wrote: if new_cost < old_cost: ... else: revert.
                    # So it's a greedy hill climbing.
                    pass
        
        # If we didn't find a valid graph for n=35, try n=34, etc.
        if best_n == 0:
            for target_n in range(34, 16, -1):
                if time.time() > deadline - 5: break
                n = target_n
                edges = set()
                for i in range(n):
                    for j in range(i+1, n):
                        if random.random() < 0.5:
                            edges.add((i, j))
                
                # Quick check for validity
                # We can use a simpler check for smaller n
                valid = True
                # Check K4
                for combo in itertools.combinations(range(n), 4):
                    u, v, w, x = combo
                    if has_edge(u, v) and has_edge(u, w) and has_edge(u, x) and \
                       has_edge(v, w) and has_edge(v, x) and has_edge(w, x):
                        valid = False
                        break
                if valid:
                    # Check Ind 6
                    for combo in itertools.combinations(range(n), 6):
                        is_indep = True
                        for i in range(6):
                            for j in range(i+1, 6):
                                if has_edge(combo[i], combo[j]):
                                    is_indep = False
                                    break
                            if not is_indep: break
                        if is_indep:
                            valid = False
                            break
                    if not valid: break
                
                if valid:
                    best_n = n
                    best_edges = edges
                    break
                # If not valid immediately, try a few local search steps
                # (Omitted for brevity in thought, but included in code below)
                # Actually, for smaller n, random might hit it or local search is faster.
                # Let's just do a few random restarts for smaller n if time permits.
                
                # Simple local search for this n
                cost = count_k4() + count_ind6()
                for _ in range(50): # Fewer steps for smaller n
                    if time.time() > deadline - 5: break
                    if cost == 0: break
                    u, v = random.randint(0, n-1), random.randint(0, n-1)
                    if u == v: continue
                    if u > v: u, v = v, u
                    
                    if (u, v) in edges:
                        edges.remove((u, v))
                    else:
                        edges.add((u, v))
                    
                    new_cost = count_k4() + count_ind6()
                    if new_cost < cost:
                        cost = new_cost
                    else:
                        # Revert
                        if (u, v) in edges:
                            edges.remove((u, v))
                        else:
                            edges.add((u, v))
                
                if cost == 0:
                    best_n = n
                    best_edges = edges
                    break

    # Fallback to seed if nothing found
    if best_n == 0:
        squares = {x * x % 17 for x in range(1, 17)}
        return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}

    return {'n': best_n, 'edges': [list(e) for e in best_edges]}
