"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    n = 40
    
    def build_bitsets(edge_set):
        """Build adjacency bitsets for fast C6 checking."""
        adj = [0] * n
        for (a, b) in edge_set:
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
        return adj
    
    def has_c6_bitset(adj, u, v):
        """
        Check if adding edge (u,v) creates a C6.
        A C6 through (u,v): u-w-x-y-z-v with all 6 vertices distinct.
        We need a path of length 4 from u to v in the graph WITHOUT edge (u,v).
        
        Using bitsets:
        - N(u) = neighbors of u (excluding v since edge not yet added)
        - N2(u) = neighbors of neighbors of u (distance 2)
        - N3(u) = distance 3
        - N4(u) = distance 4
        - Check if v is in N4(u) with all intermediate vertices distinct from u, v
        """
        # Remove u and v from consideration for intermediate vertices
        # Path: u - w - x - y - z - v
        # w in N(u)\{v}, x in N(w)\{u,v,w}, y in N(x)\{u,v,w,x}, z in N(y)\{u,v,w,x,y}
        # and z must be neighbor of v
        
        mask_all = (1 << n) - 1
        mask_uv = mask_all ^ (1 << u) ^ (1 << v)
        
        # N(u) excluding v
        nu = adj[u] & mask_uv
        if nu == 0:
            return False
        
        # For each w in N(u), compute N(w) \ {u, v, w}
        # Then for each x in that, compute N(x) \ {u,v,w,x}
        # Then for each y in that, compute N(y) \ {u,v,w,x,y}
        # Then check if any z is in N(v)
        
        # N(v) for final check
        nv = adj[v] & mask_uv
        
        # Iterate over w
        w = nu
        while w:
            lb = w & (-w)
            w ^= lb
            wi = lb.bit_length() - 1
            if wi == v:
                continue
            # N(w) excluding u, v, w
            nw = adj[wi] & mask_uv & ~(1 << wi)
            if nw == 0:
                continue
            # Iterate over x
            x = nw
            while x:
                lbx = x & (-x)
                x ^= lbx
                xi = lbx.bit_length() - 1
                # N(x) excluding u, v, w, x
                nx = adj[xi] & mask_uv & ~(1 << wi) & ~(1 << xi)
                if nx == 0:
                    continue
                # Iterate over y
                y = nx
                while y:
                    lby = y & (-y)
                    y ^= lby
                    yi = lby.bit_length() - 1
                    # N(y) excluding u, v, w, x, y
                    ny = adj[yi] & mask_uv & ~(1 << wi) & ~(1 << xi) & ~(1 << yi)
                    if ny == 0:
                        continue
                    # Check if any z in ny is also in N(v)
                    if ny & nv:
                        return True
        return False
    
    def run_greedy(seed_val):
        """Run greedy edge addition with a given random seed."""
        random.seed(seed_val)
        
        # Start with 8 disjoint K5's (104 edges)
        edge_set = set()
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    edge_set.add((5 * k + a, 5 * k + b))
        
        adj = build_bitsets(edge_set)
        
        # All possible edges not in the initial set
        all_possible = [(a, b) for a in range(n) for b in range(a + 1, n) if (a, b) not in edge_set and (b, a) not in edge_set]
        random.shuffle(all_possible)
        
        for (u, v) in all_possible:
            if time.time() > deadline:
                break
            if (u, v) in edge_set:
                continue
            if not has_c6_bitset(adj, u, v):
                edge_set.add((u, v))
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        return list(edge_set)
    
    best_result = []
    best_count = 0
    
    # Try multiple random orderings
    for trial in range(50):
        if time.time() > deadline:
            break
        result = run_greedy(trial)
        if len(result) > best_count:
            best_count = len(result)
            best_result = result
            # If we've found 105 edges, that's the known best
            if best_count >= 105:
                break
    
    return [[a, b] for (a, b) in sorted(best_result)]
# EVOLVE-BLOCK-END
