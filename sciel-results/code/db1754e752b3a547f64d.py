import random
import time
import math

def count_c4(common, n):
    total = 0
    for i in range(n):
        for j in range(i+1, n):
            c = common[i][j]
            if c >= 2:
                total += c * (c - 1) // 2
    return total

def construct():
    n = 31
    m = 91
    start_time = time.time()
    deadline = start_time + 140  # 140s limit
    
    # Initialize random graph
    edges = set()
    while len(edges) < m:
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u != v:
            if u > v: u, v = v, u
            edges.add((u, v))
            
    # Build adjacency and common neighbor matrix
    adj = [[False]*n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        
    common = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            cnt = 0
            for k in range(n):
                if adj[i][k] and adj[j][k]:
                    cnt += 1
            common[i][j] = cnt
            common[j][i] = cnt
            
    current_c4 = count_c4(common, n)
    best_c4 = current_c4
    best_edges = edges.copy()
    
    # Local Search
    T = 1.0
    while time.time() < deadline:
        if T < 0.01:
            break
        # Pick random edge to remove
        e_remove = random.choice(list(edges))
        # Pick random non-edge to add
        while True:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v:
                if u > v: u, v = v, u
                e_add = (u, v)
                if e_add not in edges:
                    break
        
        # Calculate delta
        # Remove e_remove, Add e_add
        # This logic is complex to do in one go without updating state.
        # Simpler: Update state, calc new C4, accept/reject.
        
        # To speed up, we can just do simple swaps if fast enough.
        # O(n) update is fast.
        
        # Apply swap
        u_rem, v_rem = e_remove
        u_add, v_add = e_add
        
        # Store old common values for affected pairs?
        # Or just recalculate? Recalculating is O(n^2).
        # n=31, n^2=961. Very fast.
        # Let's just recalculate common matrix for the new graph?
        # No, that's O(n^3) = 29791.
        # 140s / 30k ops = 4666 ops/s.
        # Python is slow. 30k ops might take 1-5ms.
        # So ~200-1000 iterations per second.
        # That's enough for a decent search.
        
        # Let's try the O(n^2) recalculation of common matrix.
        # It's simpler and less prone to bugs.
        
        # 1. Remove edge
        adj[u_rem][v_rem] = False
        adj[v_rem][u_rem] = False
        edges.remove(e_remove)
        
        # 2. Add edge
        adj[u_add][v_add] = True
        adj[v_add][u_add] = True
        edges.add(e_add)
        
        # 3. Recalculate common
        for i in range(n):
            for j in range(i+1, n):
                cnt = 0
                for k in range(n):
                    if adj[i][k] and adj[j][k]:
                        cnt += 1
                common[i][j] = cnt
                common[j][i] = cnt
                
        new_c4 = count_c4(common, n)
        
        # Accept/Reject
        if new_c4 < current_c4 or random.random() < math.exp((current_c4 - new_c4) / T):
            current_c4 = new_c4
            if new_c4 < best_c4:
                best_c4 = new_c4
                best_edges = edges.copy()
        else:
            # Revert
            adj[u_rem][v_rem] = True
            adj[v_rem][u_rem] = True
            edges.add(e_remove)
            
            adj[u_add][v_add] = False
            adj[v_add][u_add] = False
            edges.remove(e_add)
            
            # Revert common (or recalc again, but we have old common? No, we overwrote it)
            # We need to store old common or recalc.
            # Storing old common is O(n^2) memory.
            # Let's just recalc in revert too?
            # Or store old_common = [row[:] for row in common]
            
        # Cooling
        T *= 0.99
        
    # Convert best_edges to list of lists
    return [list(e) for e in best_edges]
