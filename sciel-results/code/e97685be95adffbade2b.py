"""Cochromatic partition via exact chromatic/independence number computation."""
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmask
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    full_mask = (1 << n) - 1
    
    # Helper: check if mask is a clique
    def is_clique(mask):
        if mask == 0:
            return True
        lsb = mask & (-mask)
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        return is_clique(rest) and (rest & ~adj[v]) == 0
    
    # Helper: check if mask is an independent set
    def is_independent(mask):
        if mask == 0:
            return True
        lsb = mask & (-mask)
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        return is_independent(rest) and (rest & adj[v]) == 0
    
    # Find chromatic number via branch and bound
    def find_chromatic_number():
        # Color assignment: color[v] = -1 means uncolored
        color = [-1] * n
        best = n  # upper bound
        
        # Order vertices by degree (descending) for better pruning
        order = sorted(range(n), key=lambda v: -bin(adj[v]).count('1'))
        pos = [0] * n
        for i, v in enumerate(order):
            pos[v] = i
        
        def dfs(idx, used_colors):
            nonlocal best
            if time.time() > deadline:
                return
            if used_colors >= best:
                return
            
            if idx == n:
                best = used_colors
                return
            
            v = order[idx]
            # Try each color
            for c in range(best):
                conflict = False
                for u in range(n):
                    if color[u] == c and (adj[v] & (1 << u)):
                        conflict = True
                        break
                if not conflict:
                    color[v] = c
                    dfs(idx + 1, max(used_colors, c + 1))
                    color[v] = -1
                    if time.time() > deadline:
                        return
            
            # Also try a new color
            if used_colors < best - 1:
                color[v] = used_colors
                dfs(idx + 1, used_colors + 1)
                color[v] = -1
        
        dfs(0, 0)
        return best
    
    # Find maximum independent set size
    def find_max_independent_set():
        best_size = 0
        best_set = 0
        
        def dfs(mask, current_set, current_size):
            nonlocal best_size, best_set
            if time.time() > deadline:
                return
            if current_size >= best_size:
                if current_size > best_size:
                    best_size = current_size
                    best_set = current_set
                return
            
            if mask == 0:
                return
            
            # Pick a vertex from mask
            lsb = mask & (-mask)
            v = lsb.bit_length() - 1
            
            # Option 1: include v
            if (current_set & adj[v]) == 0:
                dfs(mask ^ lsb, current_set | lsb, current_size + 1)
            
            # Option 2: exclude v
            dfs(mask ^ lsb, current_set, current_size)
        
        dfs(full_mask, 0, 0)
        return best_size, best_set
    
    # Find a valid partition with k parts
    def find_partition(k):
        """Find partition into k parts where each part is clique or independent set."""
        parts = []
        remaining = full_mask
        
        def backtrack(remaining, parts, k_left):
            if time.time() > deadline:
                return False
            if k_left == 0:
                return remaining == 0
            
            if remaining == 0:
                return k_left == 0 or all(not p for p in parts)
            
            # Find a valid submask of remaining
            sub = remaining
            while sub > 0:
                if is_clique(sub) or is_independent(sub):
                    parts.append(sub)
                    if backtrack(remaining ^ sub, parts, k_left - 1):
                        return True
                    parts.pop()
                sub = (sub - 1) & remaining
            
            return False
        
        if backtrack(remaining, parts, k):
            return parts
        return None
    
    # Strategy: Try k from 1 upward
    # First get bounds
    chi = find_chromatic_number()
    alpha_size, alpha_set = find_max_independent_set()
    
    # The cochromatic number is min(chi, alpha_size) but we need to verify
    # Actually, cochromatic number z(G) = min{max(χ(G[S]), α(G[S])) : S partition of V}
    # It's between 1 and max(χ(G), α(G))
    
    best_k = max(chi, alpha_size)
    
    # Try to find partition for k = 1, 2, ... up to best_k
    for k in range(1, best_k + 1):
        if time.time() > deadline:
            break
        parts = find_partition(k)
        if parts is not None:
            best_k = k
            break
    
    # If we didn't find a partition, use greedy fallback
    if time.time() > deadline or best_k > 14:
        # Greedy fallback
        es = {tuple(e) for e in edges}
        parts = []
        for v in range(n):
            for part in parts:
                trial = part + [v]
                hits = [(u, w) in es for i, u in enumerate(trial) for w in trial[i+1:]]
                if all(hits) or not any(hits):
                    part.append(v)
                    break
            else:
                parts.append([v])
        return parts
    
    # Convert bitmask parts to lists
    result = []
    for part_mask in parts:
        verts = []
        for v in range(n):
            if part_mask & (1 << v):
                verts.append(v)
        result.append(verts)
    
    # Validate
    def validate(parts):
        seen = [False] * n
        for part in parts:
            if not part:
                return False
            for v in part:
                if not isinstance(v, int) or isinstance(v, bool):
                    return False
                if seen[v]:
                    return False
                seen[v] = True
            is_c = True
            is_i = True
            for i in range(len(part)):
                for j in range(i + 1, len(part)):
                    if adj[part[i]] & (1 << part[j]):
                        is_i = False
                    else:
                        is_c = False
            if not (is_c or is_i):
                return False
        return all(seen)
    
    if not validate(result):
        result = [[v] for v in range(n)]
    
    return result
