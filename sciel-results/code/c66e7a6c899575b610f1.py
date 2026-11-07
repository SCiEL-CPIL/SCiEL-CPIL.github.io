"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Greedy fallback
    def greedy():
        parts = []
        types = []
        for v in range(n):
            placed = False
            for i in range(len(parts)):
                mask = parts[i]
                t = types[i]
                # Check validity
                if t == 1: # Clique
                    if (mask & ~adj[v]) == 0:
                        parts[i] = mask | (1 << v)
                        # Type remains 1
                        placed = True
                        break
                elif t == 2: # Indep
                    if (mask & adj[v]) == 0:
                        parts[i] = mask | (1 << v)
                        # Type remains 2
                        placed = True
                        break
                else: # Unknown (size 0 or 1)
                    # If size 0, new bit
                    # If size 1, check edge
                    if mask == 0:
                        parts[i] = (1 << v)
                        # Type remains 0
                        placed = True
                        break
                    else:
                        # Find the single bit
                        # Check if v connects to the one in mask
                        # mask has 1 bit
                        if (mask & adj[v]) != 0:
                            # Connected -> Clique
                            parts[i] = mask | (1 << v)
                            types[i] = 1
                            placed = True
                            break
                        else:
                            # Not connected -> Indep
                            parts[i] = mask | (1 << v)
                            types[i] = 2
                            placed = True
                            break
            if not placed:
                parts.append(1 << v)
                types.append(0)
        
        # Convert masks to lists
        res = []
        for mask in parts:
            lst = []
            for i in range(n):
                if mask & (1 << i):
                    lst.append(i)
            res.append(lst)
        return res

    start_time = time.time()
    deadline = start_time + 50.0
    
    # Solve for min k
    # Lower bound: 1
    # Upper bound: n (each vertex its own part)
    # Greedy gives an upper bound
    g_parts = greedy()
    best_k = len(g_parts)
    best_res = g_parts
    
    # Search for smaller k
    for k in range(1, best_k):
        if time.time() > deadline:
            break
        res = solve_k(k)
        if res:
            best_res = res
            best_k = k
            break # Found optimal for this k, since we go up from 1
            
    return best_res

def solve_k(k):
    n = P['n']
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    parts = []
    types = []
    
    def can_add(v, mask, t):
        if t == 1:
            return (mask & ~adj[v]) == 0
        elif t == 2:
            return (mask & adj[v]) == 0
        else:
            if mask == 0:
                return True
            # mask has 1 bit
            return True # Can always add to size 1, type updates later
    
    def new_type(v, mask, t):
        if t == 1:
            return 1
        elif t == 2:
            return 2
        else:
            if mask == 0:
                return 0
            # mask has 1 bit
            if (mask & adj[v]) != 0:
                return 1
            else:
                return 2

    def search(idx):
        if idx == n:
            return True
        
        v = idx
        
        # Try existing parts
        for i in range(len(parts)):
            if can_add(v, parts[i], types[i]):
                old_mask = parts[i]
                old_type = types[i]
                
                parts[i] = old_mask | (1 << v)
                types[i] = new_type(v, old_mask, old_type)
                
                if search(idx + 1):
                    return True
                
                parts[i] = old_mask
                types[i] = old_type
        
        # Try new part
        if len(parts) < k:
            parts.append(1 << v)
            types.append(0)
            
            if search(idx + 1):
                return True
            
            parts.pop()
            types.pop()
            
        return False
    
    if search(0):
        res = []
        for mask in parts:
            lst = []
            for i in range(n):
                if mask & (1 << i):
                    lst.append(i)
            res.append(lst)
        return res
    return None

# EVOLVE-BLOCK-END
