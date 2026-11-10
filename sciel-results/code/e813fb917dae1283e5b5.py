# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 115  # Safety margin before 150s limit
    n = 40
    
    def build_cover_adj(edge_set):
        """
        Build the bipartite double-cover adjacency matrix.
        Cover has 2n vertices: left = 0..n-1, right = n..2n-1.
        Edge (u,v) in original -> edges (u, v+n) and (v, u+n) in cover.
        C6 in original <-> C4 in cover.
        """
        cover_size = 2 * n
        A = np.zeros((cover_size, cover_size), dtype=np.uint8)
        for (a, b) in edge_set:
            A[a, b + n] = 1
            A[b, a + n] = 1
            A[b + n, a] = 1
            A[a + n, b] = 1
        return A
    
    def check_c6_bulk(A, candidates):
        """
        For each candidate edge (u,v), check if adding it creates a C6.
        In the cover, adding edge (u,v) means adding (u, v+n) and (v, u+n).
        A C4 in cover through these new edges means there exists a path of length 3
        from u to v+n (or v to u+n) in the existing cover graph.
        
        We use Boolean matrix powers: (A^3)[u, v+n] > 0 means a path of length 3 exists.
        But we need to be careful: the path must not use the new edges themselves.
        Since we're checking before adding, A doesn't contain the new edges yet.
        
        Actually, a C6 in original: u-w-x-y-z-v (path of length 5 from u to v).
        In cover: left_u to right_v via path of length 5 in cover.
        Adding edge (u,v) creates left_u -> right_v directly.
        A C4 in cover: left_u -> right_v -> left_y -> right_x -> left_u? No...
        
        Let me reconsider. The double cover of G is a bipartite graph with parts L={l_0..l_{n-1}} and R={r_0..r_{n-1}}.
        Edge (u,v) in G gives edges (l_u, r_v) and (l_v, r_u) in cover.
        
        A C6 in G: v0-v1-v2-v3-v4-v5-v0.
        In cover, this becomes two C6's or a C12... Actually the standard result is:
        G is C6-free iff its bipartite double cover is C4-free? No, that's not right either.
        
        Let me use a different approach: 
        A C6 through edge (u,v) in G is a path u-w-x-y-z-v of length 4 (5 edges) in G - (u,v).
        In the cover, this corresponds to a path from l_u to r_v of length 5 (or l_v to r_u).
        
        So: adding (u,v) creates a C6 iff there's a path of length 5 from l_u to r_v in the current cover.
        That's A^5[l_u, r_v] > 0.
        
        But computing A^5 for each check is expensive. Let's precompute A^2, A^4, etc.
        
        Actually, let's think differently. The number of length-5 paths from l_u to r_v in the cover
        equals the number of C6's through (u,v). We can compute this as:
        sum over w in N(u): (A^4)[w+n, v+n] ... this is getting complex.
        
        Let me use a simpler bulk approach:
        For each candidate (u,v), we need to check if there's a path of length 4 from u to v in G (not using edge (u,v)).
        This is equivalent to: exists w,x,y,z such that u~w, w~x, x~y, y~z, z~v, all distinct.
        
        Bulk approach using matrix multiplication:
        Let M = adjacency matrix of G (n x n, 0/1).
        A path of length 4 from u to v exists iff (M^4)[u,v] > 0 (with appropriate exclusions).
        But M^4 counts walks, not paths. For small graphs, if (M^4)[u,v] > 0, there's likely a path.
        However, we need to exclude walks that revisit vertices or use the edge (u,v) itself.
        
        Given the complexity, let me use a hybrid: bulk precomputation of neighbor sets,
        then vectorized checking.
        """
        # This function will be replaced with a more practical approach below
        return [False] * len(candidates)
    
    def has_c6_fast(adj_bits, u, v):
        """
        Fast C6 check using bitsets.
        C6 through (u,v): u-w-x-y-z-v, all distinct.
        Path of length 4 from u to v avoiding u,v internally.
        """
        mask_all = (1 << n) - 1
        mask_uv = mask_all ^ (1 << u) ^ (1 << v)
        
        nu = adj_bits[u] & mask_uv
        nv = adj_bits[v] & mask_uv
        
        if nu == 0 or nv == 0:
            return False
        
        # For each w in N(u), for each x in N(w), for each y in N(x), check if N(y) intersects N(v)
        # Optimize: precompute N(y) & N(v) for all y
        # Actually, let's do: for each w in N(u), compute reachable = N(w) & mask_uv
        # Then for each x in reachable, compute N(x) & mask_uv, then check intersection with N(v)
        
        # Two-level: for each w in N(u), for each x in N(w)\{u,v}, 
        #   check if any y in N(x)\{u,v,w} has N(y) intersecting N(v)\{u,v,x,y}
        
        # Simpler: iterate w in N(u), then for each x in N(w), check if (N(x) & N(v) & mask_uv) has any y 
        # such that y != u,v,w,x and N(y) & (N(v) & mask_uv & ~(1<<y)) != 0
        # This is still O(deg^4) worst case.
        
        # Let's use the original bitset approach but optimized
        w_iter = nu
        while w_iter:
            lb = w_iter & (-w_iter)
            w_iter ^= lb
            wi = lb.bit_length() - 1
            nw = adj_bits[wi] & mask_uv & ~(1 << wi)
            if nw == 0:
                continue
            x_iter = nw
            while x_iter:
                lbx = x_iter & (-x_iter)
                x_iter ^= lbx
                xi = lbx.bit_length() - 1
                nx = adj_bits[xi] & mask_uv & ~(1 << wi) & ~(1 << xi)
                if nx == 0:
                    continue
                # For each y in nx, check if N(y) intersects nv
                # Combine: does any y in nx have adj[y] & nv != 0?
                # We can compute: union of N(y) for y in nx, intersect with nv
                # But that's not quite right because y must be distinct from u,v,w,x
                # Let's just iterate y
                y_iter = nx
                while y_iter:
                    lby = y_iter & (-y_iter)
                    y_iter ^= lby
                    yi = lby.bit_length() - 1
                    if adj_bits[yi] & nv:
                        return True
        return False
    
    def run_greedy(seed_val):
        """Run greedy edge addition with perturbation when stuck."""
        random.seed(seed_val)
        
        # Start with 8 disjoint K5's (104 edges)
        edge_set = set()
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    edge_set.add((5 * k + a, 5 * k + b))
        
        adj = [0] * n
        for (a, b) in edge_set:
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
        
        # All possible edges not in the initial set
        all_possible = [(a, b) for a in range(n) for b in range(a + 1, n) if (a, b) not in edge_set]
        random.shuffle(all_possible)
        
        i = 0
        stuck_count = 0
        max_stuck = 50  # After this many failed additions, do perturbation
        
        while i < len(all_possible) and time.time() < deadline:
            (u, v) = all_possible[i]
            i += 1
            if (u, v) in edge_set:
                continue
            if not has_c6_fast(adj, u, v):
                edge_set.add((u, v))
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                stuck_count = 0
            else:
                stuck_count += 1
                if stuck_count >= max_stuck:
                    # Perturbation: remove 2-4 random edges to escape local optimum
                    if len(edge_set) > 50:
                        num_remove = random.randint(2, 4)
                        edges_list = list(edge_set)
                        random.shuffle(edges_list)
                        for _ in range(num_remove):
                            if edges_list:
                                rem = edges_list.pop()
                                if rem in edge_set:
                                    edge_set.discard(rem)
                                    a, b = rem
                                    adj[a] &= ~(1 << b)
                                    adj[b] &= ~(1 << a)
                    # Reset stuck counter and re-shuffle remaining candidates
                    stuck_count = 0
                    # Move back a bit to retry some edges
                    i = max(0, i - random.randint(10, 30))
        
        return list(edge_set)
    
    best_result = []
    best_count = 0
    
    # Try multiple random orderings with perturbation
    for trial in range(200):
        if time.time() > deadline - 5:
            break
        result = run_greedy(trial)
        if len(result) > best_count:
            best_count = len(result)
            best_result = result
            if best_count >= 105:
                break
    
    return [[a, b] for (a, b) in sorted(best_result)]
# EVOLVE-BLOCK-END
