import time
import sys

sys.setrecursionlimit(5000)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 256
    adj = [0] * N
    
    # Build Graph
    for i in range(N):
        u = i + 1
        for j in range(i + 1, N):
            v = j + 1
            if (u * v) % (u + v) == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # Greedy Initial Solution
    # Min-degree heuristic
    avail = (1 << N) - 1
    best_set = []
    
    # Copy degrees for dynamic update? 
    # For N=256, we can just recompute or update.
    # Simple approach: 
    # While avail:
    #   Find v in avail with min degree in induced subgraph
    #   Add v to set
    #   Remove v and N(v) from avail
    
    current_avail = avail
    while current_avail:
        # Find min degree vertex in current_avail
        min_deg = N
        best_v = -1
        # Iterate through set bits
        m = current_avail
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            # Count neighbors in current_avail
            deg = (adj[v] & current_avail).bit_count()
            if deg < min_deg:
                min_deg = deg
                best_v = v
            m ^= lsb
        
        if best_v == -1:
            break
            
        best_set.append(best_v)
        # Remove best_v and its neighbors
        current_avail &= ~(adj[best_v] | (1 << best_v))
        
    best_size = len(best_set)
    best_solution = best_set[:]
    
    # Branch and Bound
    def pc(x):
        return x.bit_count()
        
    def dfs(avail_mask, current_size, current_set):
        nonlocal best_size, best_solution
        nonlocal deadline, start_time
        
        if time.time() > deadline:
            return
            
        if avail_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_solution = current_set[:]
            return
            
        if current_size + pc(avail_mask) <= best_size:
            return
            
        # Pick a vertex
        # Heuristic: Pick vertex with max degree in avail_mask?
        # Or min degree?
        # In MIS, picking a vertex with high degree to INCLUDE is bad (removes many).
        # Picking a vertex with high degree to EXCLUDE is good (keeps neighbors).
        # But we branch on Include/Exclude.
        # Usually, picking a vertex with high degree leads to stronger pruning in the Include branch.
        # Let's pick the vertex with the highest degree in the subgraph.
        
        max_deg = -1
        best_v = -1
        m = avail_mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            deg = (adj[v] & avail_mask).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v
            m ^= lsb
            
        if best_v == -1:
            return
            
        v_bit = 1 << best_v
        
        # Branch 1: Include best_v
        new_avail_include = avail_mask & ~(adj[best_v] | v_bit)
        current_set.append(best_v)
        dfs(new_avail_include, current_size + 1, current_set)
        current_set.pop()
        
        # Branch 2: Exclude best_v
        new_avail_exclude = avail_mask & ~v_bit
        dfs(new_avail_exclude, current_size, current_set)
        
    dfs(avail, 0, [])
    
    result = [v + 1 for v in best_solution]
    return result
