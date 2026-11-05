import time
import random
import numpy as np

# Deadline guard
_DEADLINE = time.time() + 100.0

def _has_c5(n, nmask):
    """
    Check if graph (given by neighbor bitmasks) contains a C5.
    For each edge (u,v), check if there exist a,b,c with
    u-a-b-c-v forming a 5-cycle.
    Optimized with early exits and bit operations.
    """
    for u in range(n):
        nu = nmask[u]
        # Only consider vertices v > u to avoid double counting
        v_mask = nu & (((1 << n) - 1) ^ ((1 << (u+1)) - 1))
        while v_mask:
            v_bit = v_mask & (-v_mask)
            v = v_bit.bit_length() - 1
            v_mask ^= v_bit
            
            # a in N(u) \ {v}
            a_mask = nu & ~(1 << v)
            while a_mask:
                a_bit = a_mask & (-a_mask)
                a = a_bit.bit_length() - 1
                a_mask ^= a_bit
                # b in N(a) \ {u, v}
                na = nmask[a] & ~(1 << u) & ~(1 << v)
                b_mask = na
                while b_mask:
                    b_bit = b_mask & (-b_mask)
                    b = b_bit.bit_length() - 1
                    b_mask ^= b_bit
                    # c in N(b) \ {a, u, v} and c in N(v)
                    nb = nmask[b] & ~(1 << a) & ~(1 << u) & ~(1 << v)
                    if nb & nmask[v]:
                        return True
    return False

def _max_independent(n, nmask, target=9):
    """
    Check if there's an independent set of size >= target.
    Independent set in G = clique in complement.
    Use optimized branch-and-bound with bitmasks.
    """
    # Complement neighbor masks
    comp_mask = [0] * n
    full_mask = (1 << n) - 1
    for i in range(n):
        comp_mask[i] = full_mask & ~(1 << i) & ~nmask[i]
    
    found = [False]
    
    # Precompute popcount for speed
    def popcount(x):
        return bin(x).count('1')
    
    def search(R_size, P_mask, X_mask):
        if found[0]:
            return
        if R_size >= target:
            found[0] = True
            return
        if P_mask == 0:
            return
        
        # Upper bound: |P| + R_size
        if R_size + popcount(P_mask) < target:
            return
        
        # Pivot: pick vertex in P∪X with max complement-degree in P
        U = P_mask | X_mask
        if U == 0:
            return
        # Pick pivot with max number of neighbors in P (within comp)
        best_pivot = -1
        best_deg = -1
        um = U
        while um:
            ub = um & (-um)
            uv = ub.bit_length() - 1
            um ^= ub
            deg = popcount(comp_mask[uv] & P_mask)
            if deg > best_deg:
                best_deg = deg
                best_pivot = uv
        
        # Candidates: vertices in P NOT adjacent to pivot in comp
        if best_pivot >= 0:
            candidates = P_mask & ~comp_mask[best_pivot]
        else:
            candidates = P_mask
        
        cm = candidates
        while cm:
            cb = cm & (-cm)
            cv = cb.bit_length() - 1
            cm ^= cb
            
            if found[0]:
                return
            
            new_P = P_mask & comp_mask[cv]
            new_X = X_mask & comp_mask[cv]
            
            search(R_size + 1, new_P, new_X)
            
            P_mask = P_mask & ~cb
            X_mask = X_mask | cb
    
    search(0, full_mask, 0)
    return found[0]

def _validate(n, edges):
    """Validate: no C5 and alpha <= 8."""
    nmask = [0] * n
    for u, v in edges:
        nmask[u] |= (1 << v)
        nmask[v] |= (1 << u)
    if _has_c5(n, nmask):
        return False
    if _max_independent(n, nmask, 9):
        return False
    return True

def construct():
    start_time = time.time()
    
    # Start with the known valid 32-vertex graph: 8 disjoint K4s
    best_n = 32
    best_edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i+1, 4):
                best_edges.append([base+i, base+j])
    
    # Try to extend to 33, 34, 35 vertices
    for target_n in [33, 34, 35]:
        if time.time() > _DEADLINE - 25:
            break
        
        n = target_n
        # Build adjacency as bitmasks
        nmask = [0] * n
        # Initialize with 8 K4s
        for b in range(8):
            base = 4 * b
            for i in range(4):
                for j in range(i+1, 4):
                    nmask[base+i] |= (1 << (base+j))
                    nmask[base+j] |= (1 << (base+i))
        
        # Vertices 32..n-1 start isolated
        # We need to add edges to make alpha <= 8
        
        found = False
        max_iters = 300000
        
        # Strategy: connect extra vertices strategically to K4 vertices
        # Each extra vertex needs enough connections to prevent independent sets
        for it in range(max_iters):
            if time.time() > _DEADLINE - 25:
                break
            
            # Focus on extra vertices - they need to be "covered"
            if n > 32:
                extra_verts = list(range(32, n))
                u = random.choice(extra_verts)
                # Connect to a random K4 vertex
                v = random.randint(0, 31)
            else:
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
            
            if u == v or (nmask[u] >> v) & 1:
                continue
            
            # Try adding edge (u,v)
            nmask[u] |= (1 << v)
            nmask[v] |= (1 << u)
            
            # Check C5 first (faster check)
            if _has_c5(n, nmask):
                nmask[u] &= ~(1 << v)
                nmask[v] &= ~(1 << u)
                continue
            
            # Check alpha
            if not _max_independent(n, nmask, 9):
                found = True
                best_n = n
                # Extract edges
                new_edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if (nmask[i] >> j) & 1:
                            new_edges.append([i, j])
                best_edges = new_edges
                break
            
            # Alpha still too large, revert
            nmask[u] &= ~(1 << v)
            nmask[v] &= ~(1 << u)
        
        if found:
            continue
        else:
            break
    
    return {'n': best_n, 'edges': best_edges}
