import random
import time

def construct():
    """
    Attempts to find a valid 2-coloring of K_n with no monochromatic K_{2,2,2}.
    Uses Random Local Search with a fast bitset-based violation checker.
    Targets n=29, the known lower bound for R(K_{2,2,2}).
    """
    start_time = time.time()
    time_limit = 110  # Leave buffer for overhead

    def edge_index(n, u, v):
        if u > v:
            u, v = v, u
        # Number of edges before row u
        # Row i has n-1-i edges
        # Sum_{j=0}^{u-1} (n-1-j)
        # = u*(n-1) - u*(u-1)/2
        # Plus offset in row u: v - u - 1
        return u * (n - 1) - u * (u - 1) // 2 + (v - u - 1)

    def build_adj(n, colors):
        adj0 = [0] * n
        adj1 = [0] * n
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[k]
                if c == 0:
                    adj0[i] |= (1 << j)
                    adj0[j] |= (1 << i)
                else:
                    adj1[i] |= (1 << j)
                    adj1[j] |= (1 << i)
                k += 1
        return adj0, adj1

    def find_violation(n, adj):
        """
        Finds one edge involved in a monochromatic K_{2,2,2} in graph defined by adj.
        Returns (u, v) or None if no violation.
        """
        # Precompute common neighbors for all edges? 
        # n=30 -> 435 edges. 435 * (n^2) is too slow if done naively.
        # But we can just iterate.
        
        # Optimization: Only check edges that exist.
        # For each edge (u,v), common = adj[u] & adj[v].
        # If popcount(common) < 2, skip.
        # Else, for each pair (x,y) in common, check candidates.
        
        # To make it faster, we can precompute list of edges.
        # But building adj is O(n^2).
        
        # Let's iterate u, v.
        for u in range(n):
            au = adj[u]
            # Only consider v > u where edge exists
            # Iterate v in au bits
            # But we need v > u.
            # We can mask out bits <= u.
            mask = au & ~((1 << (u + 1)) - 1)
            while mask:
                lsb = mask & (-mask)
                v = lsb.bit_length() - 1
                mask ^= lsb
                
                common = au & adj[v]
                # Check if at least 2 bits
                if common & (common - 1) == 0:
                    continue
                
                # Extract common bits
                common_bits = common
                verts = []
                while common_bits:
                    lsb_c = common_bits & (-common_bits)
                    idx = lsb_c.bit_length() - 1
                    verts.append(idx)
                    common_bits ^= lsb_c
                
                # Check pairs in verts
                # For speed, if len(verts) is small, just loop.
                # n=30, max common ~ 28.
                for i_x in range(len(verts)):
                    x = verts[i_x]
                    ax = adj[x]
                    for i_y in range(i_x + 1, len(verts)):
                        y = verts[i_y]
                        ay = adj[y]
                        
                        mask_excl = ~((1 << u) | (1 << v) | (1 << x) | (1 << y))
                        
                        # a' in N(v) \cap N(x) \cap N(y)
                        cand_a = adj[v] & ax & ay & mask_excl
                        if cand_a == 0:
                            continue
                        
                        # b' in N(u) \cap N(x) \cap N(y)
                        cand_b = au & ax & ay & mask_excl
                        if cand_b == 0:
                            continue
                        
                        return (u, v)
        return None

    def is_valid(n, colors):
        adj0, adj1 = build_adj(n, colors)
        if find_violation(n, adj0) is not None:
            return False
        if find_violation(n, adj1) is not None:
            return False
        return True

    def search(n, time_budget):
        # Random Local Search with Restart and Perturbation
        # We will try to find a valid coloring.
        
        # Pre-compute edge indices for faster access?
        # colors is a flat list.
        
        for attempt in range(200):
            if time.time() - start_time > start_time + time_budget:
                return None
            
            colors = [random.randint(0, 1) for _ in range(n * (n - 1) // 2)]
            
            # Local Search
            # We use a "tabu" list or just simple random flips to escape local minima.
            # Strategy:
            # 1. Find violation.
            # 2. Flip one edge in the violation.
            # 3. If stuck for X steps, flip a random edge.
            
            stuck_counter = 0
            max_stuck = 20
            
            for step in range(2000):
                if time.time() - start_time > start_time + time_budget:
                    return None
                
                adj0, adj1 = build_adj(n, colors)
                viol0 = find_violation(n, adj0)
                viol1 = find_violation(n, adj1)
                
                if viol0 is None and viol1 is None:
                    return {'n': n, 'colors': colors}
                
                # Pick a violation
                if viol0 is not None:
                    u, v = viol0
                else:
                    u, v = viol1
                
                idx = edge_index(n, u, v)
                
                if stuck_counter >= max_stuck:
                    # Perturbation: flip a random edge
                    idx = random.randint(0, len(colors) - 1)
                    stuck_counter = 0
                    # Reset max_stuck slightly?
                    max_stuck = random.randint(10, 30)
                else:
                    # Check if flipping this edge reduces violations?
                    # Hard to check without full re-eval.
                    # Just flip it.
                    pass
                
                colors[idx] ^= 1
                
                # Simple heuristic: if we keep flipping the same edge, it's bad.
                # We don't track history here, so rely on random restarts.
                
                # Increment stuck counter if we didn't improve?
                # We can't easily measure improvement without counting violations.
                # So we just use fixed perturbation frequency.
                if random.random() < 0.05:
                    stuck_counter += 1
                else:
                    stuck_counter = 0

    # Try n=30 first with a short time limit (5s)
    # If it fails, try n=29 with remaining time.
    
    res = search(30, 5)
    if res:
        return res
        
    res = search(29, 100)
    if res:
        return res
        
    # Fallback: Return a known valid small n
    # The seed provided n=17. We should at least match it.
    # Quadratic residues mod 17
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'colors': [0 if (j - i) % 17 in squares else 1 for i in range(17) for j in range(i + 1, 17)]}
