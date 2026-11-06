import time
import sys

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_edges = []
    best_count = 0
    
    # adj[i] is an integer bitmask of neighbors of i
    adj = [0] * n
    
    # c[u][v] is not explicitly stored, we compute on the fly using bit_count
    # Python 3.8+ has int.bit_count()
    
    def is_valid_add(i, j):
        """
        Check if adding edge (i, j) is valid.
        i and j are not connected yet.
        We need to ensure that for all k, common neighbors of (i, k) and (j, k)
        do not exceed 2.
        """
        # For all k != i, j
        # If k is neighbor of j, then common(i, k) increases by 1.
        # So we need current common(i, k) < 2.
        # If k is neighbor of i, then common(j, k) increases by 1.
        # So we need current common(j, k) < 2.
        
        # Optimize:
        # neighbors_of_j = adj[j]
        # neighbors_of_i = adj[i]
        
        # Check pairs (i, k) where k is in neighbors_of_j
        # For each such k, bit_count(adj[i] & adj[k]) must be < 2
        
        # Check pairs (j, k) where k is in neighbors_of_i
        # For each such k, bit_count(adj[j] & adj[k]) must be < 2
        
        # Note: k can be < i or > i.
        # Since we are building in order, for vertex i, we only connect to < i.
        # But adj[j] might have neighbors > i? No, we build sequentially.
        # If we build i from 0 to 19, when processing i, adj[i] only has bits for < i.
        # adj[j] (where j < i) only has bits for < j (and thus < i).
        # So all relevant k are < i.
        
        mask_j = adj[j]
        mask_i = adj[i]
        
        # Check k in mask_j
        k = mask_j
        while k:
            lsb = k & -k
            k_idx = lsb.bit_length() - 1
            k -= lsb
            # Check common(i, k_idx)
            # common is bit_count(adj[i] & adj[k_idx])
            if adj[i].bit_count() & 0: # placeholder
                pass
            cnt = (adj[i] & adj[k_idx]).bit_count()
            if cnt >= 2:
                return False
        
        # Check k in mask_i
        k = mask_i
        while k:
            lsb = k & -k
            k_idx = lsb.bit_length() - 1
            k -= lsb
            cnt = (adj[j] & adj[k_idx]).bit_count()
            if cnt >= 2:
                return False
                
        return True

    def dfs(i, current_count):
        nonlocal best_edges, best_count
        
        if time.time() > deadline:
            return
            
        if i == n:
            if current_count > best_count:
                best_count = current_count
                # Reconstruct edges
                edges = []
                for u in range(n):
                    mask = adj[u]
                    k = mask
                    while k:
                        lsb = k & -k
                        v = lsb.bit_length() - 1
                        k -= lsb
                        if u < v:
                            edges.append([u, v])
                best_edges = edges
            return

        # Try adding edges to i
        # Heuristic: Try to add as many as possible?
        # We iterate j from i-1 down to 0? Or 0 to i-1?
        # To maximize, we should try to add edges.
        # But we need to backtrack.
        
        # Option 1: Try to add edge (i, j)
        # Option 2: Skip edge (i, j)
        
        # We will iterate through all subsets of neighbors?
        # No, that's 2^i.
        # We do DFS on the choices.
        
        # To speed up, we can try to add edges in a greedy order first.
        # But for backtracking, we must explore both.
        
        # Let's try j from i-1 down to 0.
        # If we can add (i, j), we do it, recurse, then undo.
        # If we can't, we skip.
        # We also need to explore the case where we DON'T add (i, j) even if we can?
        # Yes, because adding it might block future edges.
        # However, for maximizing edges, adding is usually good.
        # But sometimes not.
        
        # We will do:
        # 1. Try adding (i, j)
        # 2. Try not adding (i, j)
        
        # To reduce branching, we can use a heuristic:
        # If adding (i, j) is valid, do it.
        # Then, for the next vertex, we have a constrained state.
        # If we fail to find a good solution, we backtrack.
        
        # Optimization: If i is small, we can be exhaustive.
        # If i is large, we rely on pruning.
        
        # Let's try to add (i, j) first.
        # We iterate j.
        # Actually, we decide the ENTIRE neighbor set of i at once?
        # No, that's 2^i combinations.
        # We decide one edge at a time.
        
        # Let's pick the "most constrained" j?
        # Or just go 0 to i-1.
        
        # To make it deterministic and fast, we will try to add edges in descending order of j.
        # This tends to create high-degree vertices early.
        
        # We need to handle the "skip" case too.
        
        # Efficient DFS:
        # We maintain the state.
        # For j in range(i-1, -1, -1):
        #    if is_valid_add(i, j):
        #       add_edge(i, j)
        #       dfs(i+1, current_count + 1)
        #       remove_edge(i, j)
        #    # Then we MUST also try the case where we do NOT add (i, j)
        #    # But wait, if we iterate j, and we skip (i, j), we still have to decide (i, j-1).
        #    # So the recursion should be:
        #    # dfs(i, current_count, next_j)
        
    # The standard DFS for subset sum style:
    # def dfs(i, current_count, j):
    #    if j < 0:
    #       dfs(i+1, current_count, i) # next vertex
    #    else:
    #       # Option 1: Skip (i, j)
    #       dfs(i, current_count, j-1)
    #       # Option 2: Add (i, j) if valid
    #       if is_valid_add(i, j):
    #          add_edge(i, j)
    #          dfs(i, current_count + 1, j-1)
    #          remove_edge(i, j)
    
    # This is 2^i branches per vertex.
    # For i=10, 1024 branches.
    # For i=19, 500,000 branches.
    # Total nodes in tree is huge.
    # But pruning will help.
    
    # To improve, we can use a heuristic to order j.
    # Or, we can use a "greedy with backtracking" where we only backtrack if we get stuck.
    
    # Let's implement the standard DFS with time limit.
    
    def add_edge(i, j):
        adj[i] |= (1 << j)
        adj[j] |= (1 << i)
        
    def remove_edge(i, j):
        adj[i] &= ~(1 << j)
        adj[j] &= ~(1 << i)
        
    def dfs(i, current_count, j):
        nonlocal best_edges, best_count
        
        if time.time() > deadline:
            return
            
        if i == n:
            if current_count > best_count:
                best_count = current_count
                edges = []
                for u in range(n):
                    mask = adj[u]
                    k = mask
                    while k:
                        lsb = k & -k
                        v = lsb.bit_length() - 1
                        k -= lsb
                        if u < v:
                            edges.append([u, v])
                best_edges = edges
            return

        if j < 0:
            dfs(i + 1, current_count, i)
            return

        # Option 1: Skip (i, j)
        # To prioritize adding, we might want to try Add first?
        # But Add depends on j.
        # If we try Add first, we find high edge counts faster.
        
        # Let's try Add first if valid.
        if is_valid_add(i, j):
            add_edge(i, j)
            dfs(i, current_count + 1, j - 1)
            remove_edge(i, j)
            
        # Option 2: Skip (i, j)
        dfs(i, current_count, j - 1)
        
    # Start DFS
    # To help, we can initialize with a greedy solution to set a good lower bound for best_count.
    # This allows pruning if we can calculate an upper bound.
    # Upper bound: current_count + sum of possible edges for remaining vertices.
    # For vertex k, max edges is k (to 0..k-1).
    # So max remaining = sum_{k=i}^{n-1} k.
    
    def upper_bound(i, current_count):
        # Max edges from i to n-1
        # Vertex i can connect to i vertices.
        # Vertex i+1 to i+1 vertices.
        # ...
        # Sum = i + (i+1) + ... + (n-1)
        # = (n-1)*n/2 - (i-1)*i/2
        max_remaining = (n - 1) * n // 2 - (i - 1) * i // 2
        return current_count + max_remaining

    # Modify dfs to use upper bound
    def dfs_opt(i, current_count, j):
        nonlocal best_edges, best_count
        
        if time.time() > deadline:
            return
            
        if upper_bound(i, current_count) <= best_count:
            return

        if i == n:
            if current_count > best_count:
                best_count = current_count
                edges = []
                for u in range(n):
                    mask = adj[u]
                    k = mask
                    while k:
                        lsb = k & -k
                        v = lsb.bit_length() - 1
                        k -= lsb
                        if u < v:
                            edges.append([u, v])
                best_edges = edges
            return

        if j < 0:
            dfs_opt(i + 1, current_count, i)
            return

        # Try Add first
        if is_valid_add(i, j):
            add_edge(i, j)
            dfs_opt(i, current_count + 1, j - 1)
            remove_edge(i, j)
            
        # Try Skip
        dfs_opt(i, current_count, j - 1)

    # Initial greedy to set best_count
    # Just run a simple greedy
    for i in range(n):
        for j in range(i):
            if is_valid_add(i, j):
                add_edge(i, j)
                
    # Reset adj for DFS
    adj = [0] * n
    best_count = 0
    best_edges = []
    
    # Run DFS
    dfs_opt(0, 0, -1)
    
    # If best_edges is empty, fallback to greedy
    if not best_edges:
        # Re-run greedy
        for i in range(n):
            for j in range(i):
                if is_valid_add(i, j):
                    add_edge(i, j)
        for u in range(n):
            mask = adj[u]
            k = mask
            while k:
                lsb = k & -k
                v = lsb.bit_length() - 1
                k -= lsb
                if u < v:
                    best_edges.append([u, v])
                    
    return best_edges
