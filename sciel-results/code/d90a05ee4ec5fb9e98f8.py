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
        Optimized using bitset intersections to reduce nested iterations.
        """
        mask_all = (1 << n) - 1
        mask_uv = mask_all ^ (1 << u) ^ (1 << v)
        
        # N(u) excluding v
        nu = adj[u] & mask_uv
        if nu == 0:
            return False
        
        # N(v) excluding u
        nv = adj[v] & mask_uv
        if nv == 0:
            return False
        
        # For each w in N(u), we need to check if there's a path w-x-y-z where
        # x in N(w)\{u,v,w}, y in N(x)\{u,v,w,x}, z in N(y)\{u,v,w,x,y} ∩ N(v)
        
        # Precompute for each vertex: adj[vert] & mask_uv (neighbors excluding u,v)
        # We'll iterate over w in N(u)
        w = nu
        while w:
            lb = w & (-w)
            w ^= lb
            wi = lb.bit_length() - 1
            
            # N(w) excluding u, v, w
            nw = adj[wi] & mask_uv & ~(1 << wi)
            if nw == 0:
                continue
            
            # For each x in N(w), compute N(x) \ {u,v,w,x}
            # Then check if any y in that set has a neighbor z in N(v) \ {u,v,w,x,y}
            
            x = nw
            while x:
                lbx = x & (-x)
                x ^= lbx
                xi = lbx.bit_length() - 1
                
                # N(x) excluding u, v, w, x
                nx = adj[xi] & mask_uv & ~(1 << wi) & ~(1 << xi)
                if nx == 0:
                    continue
                
                # For each y in N(x)\{u,v,w,x}, check if N(y) \ {u,v,w,x,y} intersects N(v)
                # Instead of iterating y, we can check:
                # Is there a y in nx such that (adj[y] & mask_uv & ~(1<<wi) & ~(1<<xi) & ~(1<<y)) & nv != 0?
                # This is equivalent to: does nx contain any y where adj[y] has a neighbor in nv
                # excluding u,v,w,x,y?
                
                # Let's try: for each y in nx, check if adj[y] & nv has any vertex other than u,v,w,x,y
                # But that's still O(|nx|) per x.
                
                # Alternative: precompute for each vertex z in nv, which vertices y have z as neighbor?
                # That's adj[z] (symmetric). So y is adjacent to some z in nv iff y in union of adj[z] for z in nv.
                # But we need to exclude u,v,w,x,y from the intersection.
                
                # Let's compute: candidates_y = nx & (union of adj[z] for z in nv)
                # Then for each y in candidates_y, check if (adj[y] & nv) & ~(1<<u) & ~(1<<v) & ~(1<<wi) & ~(1<<xi) & ~(1<<y) != 0
                
                # Actually, let's just iterate y but with early exit
                y = nx
                while y:
                    lby = y & (-y)
                    y ^= lby
                    yi = lby.bit_length() - 1
                    
                    # Check if N(y) \ {u,v,w,x,y} intersects N(v)
                    # N(y) & nv & ~(1<<wi) & ~(1<<xi) & ~(1<<yi)
                    # Note: u and v are already excluded by mask_uv in adj[yi]
                    ny = adj[yi] & nv & ~(1 << wi) & ~(1 << xi) & ~(1 << yi)
                    if ny:
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
        
        added = 0
        for (u, v) in all_possible:
            if time.time() > deadline:
                break
            if (u, v) in edge_set:
                continue
            if not has_c6_bitset(adj, u, v):
                edge_set.add((u, v))
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                added += 1
        
        return list(edge_set)
    
    best_result = []
    best_count = 0
    
    # Try multiple random orderings - with faster C6 check, we can try more
    for trial in range(200):
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
