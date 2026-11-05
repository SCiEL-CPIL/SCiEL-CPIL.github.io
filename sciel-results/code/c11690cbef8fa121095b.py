import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s buffer

    # We search for n from 40 downwards
    for n in range(40, 25, -1):
        if time.time() > deadline:
            break
            
        # We want to find a symmetric connection set S subset of {1, ..., n//2}
        # such that the circulant graph is triangle-free and alpha < 10.
        
        # Backtracking to generate triangle-free symmetric sets S
        candidates = list(range(1, n // 2 + 1))
        current_S = []
        S_set = set()
        
        # Fast alpha check using bitsets
        def _check_alpha_fast(n, S):
            # Build complement adjacency masks
            # Complement edge i-j if (i-j)%n not in S
            # We want Max Clique in Complement.
            
            adj = [0] * n
            S_set_local = set(S)
            for i in range(n):
                mask = 0
                for j in range(n):
                    if i == j: continue
                    if (i - j) % n not in S_set_local:
                        mask |= (1 << j)
                adj[i] = mask
            
            best = 0
            # Simple B&B with bitsets
            def dfs(cands, size):
                nonlocal best
                if not cands:
                    if size > best:
                        best = size
                    return
                
                # Pruning: if size + bit_count(cands) <= best, return
                if size + cands.bit_count() <= best:
                    return
                
                # Pick a vertex with max degree in cands
                v = -1
                max_deg = -1
                tmp = cands
                while tmp:
                    lsb = tmp & (-tmp)
                    u = lsb.bit_length() - 1
                    deg = (adj[u] & cands).bit_count()
                    if deg > max_deg:
                        max_deg = deg
                        v = u
                    tmp ^= lsb
                
                # Include v
                dfs(cands & adj[v], size + 1)
                # Exclude v
                dfs(cands & ~(1 << v), size)
                
            dfs((1 << n) - 1, 0)
            return best

        def backtrack(idx):
            if time.time() > deadline:
                return False
            
            # Estimate alpha. 
            # If |S| is small, alpha is large.
            # We need alpha < 10.
            # Heuristic: only check if |S| is large enough.
            if len(current_S) >= 2:
                if _check_alpha_fast(n, current_S) < 10:
                    return True

            if idx == len(candidates):
                return False
            
            # Try including candidates[idx] first (greedy for density)
            x = candidates[idx]
            valid = True
            for y in current_S:
                if (x - y) % n in S_set or (x + y) % n in S_set:
                    valid = False
                    break
            
            if valid:
                current_S.append(x)
                S_set.add(x)
                S_set.add((-x) % n)
                if backtrack(idx + 1):
                    return True
                current_S.pop()
                S_set.remove(x)
                S_set.remove((-x) % n)
            
            # Try excluding
            if backtrack(idx + 1):
                return True
            
            return False

        if backtrack(0):
            # Found a valid S
            # Construct edges
            edges = []
            S_set_final = S_set
            for i in range(n):
                for j in range(i + 1, n):
                    if (i - j) % n in S_set_final or (j - i) % n in S_set_final:
                        edges.append([i, j])
            
            # Final verification (slow but safe)
            if _verify(n, edges):
                return {'n': n, 'edges': edges}

    # Fallback: return a known valid graph
    # For n=25, use a specific circulant
    # Or just return a small one to avoid crash
    n0 = 25
    edges0 = []
    S0 = {1, 4, 6, 9, 11, 14, 16, 19, 21, 24} # Just a guess, verify?
    # Actually, let's use a verified small graph.
    # C5 is n=5.
    # Let's just return n=5 to be safe if search fails.
    # But we want high score.
    # The search should succeed.
    
    # If search fails, return a valid small graph
    # n=5, edges (0,1),(1,2),(2,3),(3,4),(4,0)
    return {'n': 5, 'edges': [[0,1],[1,2],[2,3],[3,4],[4,0]]}

def _verify(n, edges):
    # Check triangle-free
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u][v]:
                for w in range(v + 1, n):
                    if adj[u][w] and adj[v][w]:
                        return False
    
    # Check alpha < 10
    # Use simple B&B
    comp_adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and not adj[i][j]:
                comp_adj[i][j] = True
                
    best = 0
    def dfs(cands, size):
        nonlocal best
        if not cands:
            if size > best:
                best = size
            return
        if size + len(cands) <= best:
            return
        
        # Pick vertex with max degree in cands
        v = -1
        max_deg = -1
        for u in cands:
            deg = sum(1 for w in cands if comp_adj[u][w])
            if deg > max_deg:
                max_deg = deg
                v = u
        
        # Include v
        new_cands_in = [u for u in cands if comp_adj[v][u]]
        dfs(new_cands_in, size + 1)
        
        # Exclude v
        new_cands_out = [u for u in cands if u != v]
        dfs(new_cands_out, size)
        
    dfs(list(range(n)), 0)
    return best < 10
