"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Precompute adjacency bitmask
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Vertex ordering: Descending degree
    degrees = [bin(adj[v]).count('1') for v in range(n)]
    order = sorted(range(n), key=lambda v: -degrees[v])
    
    # Calculate max clique and independent set sizes for pruning
    max_clique = 0
    max_indep = 0
    for mask in range(1, 1 << n):
        ok_c = True
        ok_i = True
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if (mask & ~adj[v]) != 0:
                ok_c = False
                break
            if (mask & adj[v]) != 0:
                ok_i = False
                break
            m ^= lsb
            if not ok_c and not ok_i:
                break
        if ok_c:
            max_clique = max(max_clique, bin(mask).count('1'))
        if ok_i:
            max_indep = max(max_indep, bin(mask).count('1'))
            
    max_part_size = max(max_clique, max_indep)
    
    # State variables
    parts = [] # list of bitmasks
    types = [] # list of 0 (Clique) or 1 (Independent)
    
    best_count = [n + 1]
    best_solution = [None]
    
    def can_add(v, part_idx):
        mask = parts[part_idx]
        if types[part_idx] == 0: # Clique
            # v must be adjacent to all in mask
            return (mask & ~adj[v]) == 0
        else: # Independent
            # v must NOT be adjacent to any in mask
            return (mask & adj[v]) == 0
            
    def solve(idx):
        if idx == n:
            if len(parts) < best_count[0]:
                best_count[0] = len(parts)
                sol = []
                for i in range(len(parts)):
                    m = parts[i]
                    verts = [j for j in range(n) if m & (1 << j)]
                    sol.append(verts)
                best_solution[0] = sol
            return
        
        # Pruning: Current parts + lower bound on additional parts
        remaining = n - idx
        lb_total = len(parts) + (remaining + max_part_size - 1) // max_part_size
        if lb_total >= best_count[0]:
            return
            
        v = order[idx]
        
        # 1. Try adding to existing parts
        for i in range(len(parts)):
            if can_add(v, i):
                parts[i] |= (1 << v)
                solve(idx + 1)
                parts[i] &= ~(1 << v)
                
        # 2. Try creating new part as Clique
        parts.append(1 << v)
        types.append(0)
        solve(idx + 1)
        parts.pop()
        types.pop()
        
        # 3. Try creating new part as Independent
        parts.append(1 << v)
        types.append(1)
        solve(idx + 1)
        parts.pop()
        types.pop()
        
    solve(0)
    
    if best_solution[0]:
        return best_solution[0]
    else:
        return [[v] for v in range(n)]

# EVOLVE-BLOCK-END
