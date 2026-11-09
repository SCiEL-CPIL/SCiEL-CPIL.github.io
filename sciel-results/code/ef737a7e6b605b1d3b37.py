import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    best_n = 0
    best_edges = []

    # Try n=47 first, then 46, then 45...
    for target_n in [47, 46, 45, 44, 43, 42, 41, 40, 39]:
        if time.time() > deadline:
            break
            
        min_deg_needed = target_n - 39
        
        if min_deg_needed < 0:
            min_deg_needed = 0
            
        found = False
        
        # Number of restarts depends on time available
        # For larger n, fewer restarts might be needed if greedy works well, 
        # but it's stochastic. Let's try 50 restarts.
        max_restarts = 50
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            random.seed(restart * 100 + target_n)
            
            A = np.zeros((target_n, target_n), dtype=bool)
            deg = np.zeros(target_n, dtype=int)
            
            # Pre-compute common neighbors? 
            # We will update a 2D array `common` where common[u][v] = |N(u) cap N(v)|.
            common = np.zeros((target_n, target_n), dtype=int)
            
            # Helper to check if we can add edge (u, v)
            # Returns True if safe
            def can_add(u, v):
                # 1. u and v must not be connected
                if A[u, v]:
                    return False
                # 2. common[u][v] must be <= 1
                if common[u, v] > 1:
                    return False
                # 3. For all w in N(v) (w != u), common[u][w] must be 0
                #    Because if w is neighbor of v, and we add u-v, 
                #    v becomes a common neighbor of u and w.
                #    So common[u][w] increases by 1.
                #    We need new common[u][w] <= 1 => old common[u][w] <= 0.
                #    Since common counts are non-negative, it must be 0.
                for w in range(target_n):
                    if w == u or w == v:
                        continue
                    if A[v, w]:
                        if common[u, w] > 0:
                            return False
                # 4. For all w in N(u) (w != v), common[v][w] must be 0
                for w in range(target_n):
                    if w == u or w == v:
                        continue
                    if A[u, w]:
                        if common[v, w] > 0:
                            return False
                return True

            def add_edge(u, v):
                A[u, v] = True
                A[v, u] = True
                deg[u] += 1
                deg[v] += 1
                
                # Update common neighbors
                # common[x][y] increases if x is neighbor of v and y is neighbor of u?
                # No.
                # Adding (u,v) means:
                # For any x, y:
                # common[x][y] increases by 1 if:
                #   (x==u and v in N(y)) OR (y==u and v in N(x)) 
                #   OR (x==v and u in N(y)) OR (y==v and u in N(x))
                # Basically, v becomes a common neighbor of u and N(v).
                # And u becomes a common neighbor of v and N(u).
                
                # Update common[u][w] for w in N(v)
                for w in range(target_n):
                    if w != u and A[v, w]:
                        common[u, w] += 1
                        common[w, u] += 1
                        
                # Update common[v][w] for w in N(u)
                for w in range(target_n):
                    if w != v and A[u, w]:
                        common[v, w] += 1
                        common[w, v] += 1
                        
                # Note: common[u][v] and common[v][u] are NOT updated by this logic?
                # common[u][v] is the number of common neighbors of u and v.
                # Adding edge (u,v) does NOT change the set of common neighbors of u and v.
                # It changes the sets N(u) and N(v), but the intersection N(u) cap N(v) 
                # remains the same?
                # Wait. N(u) gains v. N(v) gains u.
                # Does v become a common neighbor of u and v? No, v is not a neighbor of v.
                # Does u become a common neighbor of u and v? No.
                # So common[u][v] stays the same.
                # 
                # However, for any w in N(v) \ {u}, v is now in N(u) and v is in N(v).
                # So v is a common neighbor of u and w?
                # N(u) contains v. N(w) contains v (since w in N(v) and graph is undirected).
                # So yes, v is a common neighbor of u and w.
                # This matches the update `common[u, w] += 1` for w in N(v).

            # Greedy loop
            # We want to ensure min_deg_needed.
            # We will try to add edges to vertices with lowest degree.
            
            # Initial random sparse graph to break symmetry?
            # Or start empty?
            # Starting empty is safer for C4 avoidance.
            
            success = True
            for _ in range(1000): # Max steps
                if time.time() > deadline:
                    success = False
                    break
                    
                # Find vertex with min degree
                min_deg = np.min(deg)
                if min_deg >= min_deg_needed:
                    # Check if all degrees are >= min_deg_needed
                    if np.all(deg >= min_deg_needed):
                        success = True
                        break
                    # If some are high and some low, we might be stuck?
                    # No, we just keep adding to the lowest.
                
                low_vertices = np.where(deg < min_deg_needed)[0]
                if len(low_vertices) == 0:
                    success = True
                    break
                    
                u = random.choice(low_vertices)
                
                # Find a valid v to connect to u
                candidates = []
                for v in range(target_n):
                    if v == u or A[u, v]:
                        continue
                    if can_add(u, v):
                        candidates.append(v)
                        
                if not candidates:
                    # Stuck. 
                    # Try to remove a random edge to unblock? 
                    # For simplicity, just fail this restart.
                    success = False
                    break
                
                v = random.choice(candidates)
                add_edge(u, v)
                
            if success:
                # Final verification
                # Recalculate common to be sure
                C = A.astype(int) @ A.astype(int)
                c4_free = True
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if C[i, j] >= 2:
                            c4_free = False
                            break
                    if not c4_free:
                        break
                
                if c4_free and np.min(deg) >= min_deg_needed:
                    final_edges = []
                    for i in range(target_n):
                        for j in range(i + 1, target_n):
                            if A[i, j]:
                                final_edges.append([i, j])
                    
                    if target_n > best_n:
                        best_n = target_n
                        best_edges = final_edges
                    found = True
                    break # Break restart loop
        
        if best_n == 47:
            break
        if found:
            # If we found a solution for this n, we can stop trying lower n 
            # if we want to maximize n, but we already tried higher n.
            # So we can break the outer loop if we found the max possible?
            # No, we iterate 47, 46...
            # If 47 fails, we try 46. If 46 succeeds, we break.
            break

    if best_n > 0:
        return {'n': best_n, 'edges': best_edges}
    else:
        # Fallback: n=39, empty graph is valid (min deg 0 >= 0)
        return {'n': 39, 'edges': []}
