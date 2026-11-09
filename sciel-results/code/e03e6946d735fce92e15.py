import time
import random
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 40
    target_edges = 401
    
    # Partition
    a_size = 9
    b_size = 31
    A = list(range(a_size))
    B = list(range(a_size, n))
    
    # Split B
    B_special = B[:9]
    B_other = B[9:]
    
    edges = set()
    
    # 1. Add all edges between A and B, EXCEPT (A[i], B_special[i])
    for a in A:
        for b in B:
            # Check if this is a removed edge
            # A[i] corresponds to B_special[i]
            # We need to map a to its index in A
            # A is 0..8. B_special is 9..17.
            # Removed: (0, 9), (1, 10), ..., (8, 17)
            if b in B_special:
                idx = B_special.index(b)
                if a == idx:
                    continue
            edges.add((a, b))
            
    # Current edges: 279 - 9 = 270
    needed_in_B = target_edges - 270 # 131
    
    # Construct G_B on B
    # Constraints:
    # 1. No edges within B_other
    # 2. Triangle-free
    # 3. Max degree 8
    # 4. 131 edges
    
    # Potential edges in B:
    # - Between B_special and B_other
    # - Within B_special
    
    potential_edges = []
    # B_special to B_other
    for bs in B_special:
        for bo in B_other:
            potential_edges.append((bs, bo))
    # Within B_special
    for i in range(len(B_special)):
        for j in range(i+1, len(B_special)):
            potential_edges.append((B_special[i], B_special[j]))
            
    # Shuffle
    random.shuffle(potential_edges)
    
    selected_edges = []
    degree = {v: 0 for v in B}
    neighbors = {v: set() for v in B}
    
    # We need to be careful about triangles.
    # If we add (u, v), we must ensure no common neighbor w exists.
    # i.e., neighbors[u] & neighbors[v] must be empty.
    
    # Greedy approach with backtracking/restart if stuck
    found = False
    for attempt in range(100):
        if time.time() > deadline - 10:
            break
        random.shuffle(potential_edges)
        selected_edges = []
        degree = {v: 0 for v in B}
        neighbors = {v: set() for v in B}
        
        for u, v in potential_edges:
            if len(selected_edges) == needed_in_B:
                break
            if degree[u] < 8 and degree[v] < 8:
                # Check for triangles
                if not (neighbors[u] & neighbors[v]):
                    selected_edges.append((u, v))
                    degree[u] += 1
                    degree[v] += 1
                    neighbors[u].add(v)
                    neighbors[v].add(u)
                    
        if len(selected_edges) == needed_in_B:
            found = True
            break
            
    if not found:
        # Fallback: relax degree to 9?
        # If we can't find it, return the best we have or a valid graph with higher score
        # For now, just return what we have (might be < 131 edges, invalid)
        # Or use the old method
        pass
        
    for u, v in selected_edges:
        edges.add((u, v))
        
    # Verify edge count
    # If not enough, add any valid edges?
    # The problem requires EXACTLY 401 edges.
    # If the greedy fails, we might have fewer.
    # In that case, we should fall back to a known valid construction (e.g. the old one)
    # to ensure validity, even if score is lower.
    
    if len(edges) != 401:
        # Fallback to the previous deterministic construction
        edges = set()
        A = list(range(9))
        B = list(range(9, 40))
        for a in A:
            for b in B:
                edges.add((a, b))
        needed = 401 - 279
        B1 = B[:15]
        B2 = B[15:]
        pot = list(itertools.product(B1, B2))
        random.shuffle(pot)
        deg = {v:0 for v in B}
        for u, v in pot:
            if len(edges) == 401: break
            if deg[u] < 8 and deg[v] < 8:
                edges.add((u, v))
                deg[u]+=1
                deg[v]+=1
        # If still not 401, force add (might increase score)
        all_pot = list(itertools.combinations(B, 2))
        random.shuffle(all_pot)
        for u, v in all_pot:
            if len(edges) == 401: break
            if (u, v) not in edges and (v, u) not in edges:
                edges.add((u, v))
                
    result = [[u, v] for u, v in edges]
    return result
