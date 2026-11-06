"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def _has_k222(adj, n):
    """Check if graph with adjacency matrix adj on n vertices contains K2,2,2 as a subgraph."""
    # K2,2,2: vertices {a,b,c,d,e,f} with all 12 cross-edges present
    # For each pair of disjoint pairs (a,b), (c,d), (e,f), check all cross-edges
    from itertools import combinations
    verts = range(n)
    # Precompute: for each vertex v, its neighbors as a set for fast lookup
    # Use bitmasks for efficiency
    # For each triple of disjoint pairs, check if all cross-edges exist
    # K2,2,2 has parts of size 2. Let parts be {a,b}, {c,d}, {e,f}
    # Need: a-c, a-d, a-e, a-f, b-c, b-d, b-e, b-f, c-e, c-f, d-e, d-f all edges
    # Iterate over all ways to choose 6 distinct vertices and partition into 3 pairs
    # But that's O(n^6) which is too slow for n=29.
    # Better: for each pair (a,b), for each pair (c,d) disjoint from {a,b},
    # the common neighbors of {a,b} intersected with... hmm.
    # Actually, for K2,2,2 with parts {a,b}, {c,d}, {e,f}:
    # c and d must be common neighbors of a and b
    # e and f must be common neighbors of a and b
    # AND c-e, c-f, d-e, d-f must be edges
    # So: let N_ab = common neighbors of a and b
    # We need two disjoint pairs (c,d) and (e,f) from N_ab such that
    # c-e, c-f, d-e, d-f are all edges.
    # i.e., the subgraph induced by N_ab must contain K_{2,2} (i.e., C4 or K4 or K_{2,2}+extra)
    # Actually we need a K_{2,2} in the induced subgraph on N_ab, where the two sides are {c,d} and {e,f}
    # So check: does the graph induced by N_ab contain a C4? (K_{2,2} is a subgraph of any C4-containing graph)
    # Wait, K_{2,2} IS a C4. So we need a C4 in the induced subgraph on N_ab.
    # But we also need the four vertices to be distinct from a,b and from each other, which is guaranteed.
    
    # For efficiency, use bitmasks
    # adj[i] is a bitmask of neighbors of i
    # common_neigh(a,b) = adj[a] & adj[b]
    # For each pair (a,b), get cn = common_neigh(a,b) & ~(1<<a) & ~(1<<b)
    # Then check if the subgraph induced by cn contains a C4
    # C4 in induced subgraph: exist u,v,w,x in cn such that u-v, v-w, w-x, x-u are edges
    # Equivalently, there exist two vertices u,v in cn that have at least 2 common neighbors in cn
    # (since if u and v share neighbors w and x in cn, then u-w, w-v, v-x, x-u form a C4 if all are in cn and edges exist)
    # Wait: u-w, w-v means w is common neighbor of u,v. Similarly x. Then the cycle is u-w-v-x-u.
    # Edges needed: u-w (yes, w in N(u)), w-v (yes, w in N(v)), v-x (yes, x in N(v)), x-u (yes, x in N(u)).
    # So yes, if two vertices u,v in cn have at least 2 common neighbors in cn, then C4 exists.
    
    for a in range(n):
        for b in range(a+1, n):
            if not (adj[a] >> b) & 1:
                continue
            # a and b are adjacent
            cn_mask = adj[a] & adj[b]
            cn_mask &= ~(1 << a)
            cn_mask &= ~(1 << b)
            # Get list of common neighbors
            cn_list = []
            m = cn_mask
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                cn_list.append(idx)
                m ^= lsb
            # Check if any two vertices in cn_list have >= 2 common neighbors in cn_list
            cn_set = set(cn_list)
            for i in range(len(cn_list)):
                for j in range(i+1, len(cn_list)):
                    u, v = cn_list[i], cn_list[j]
                    # Common neighbors of u and v within cn_set
                    common_uv = adj[u] & adj[v]
                    common_uv &= cn_mask
                    # Count bits
                    cnt = common_uv.bit_count()
                    if cnt >= 2:
                        return True
    return False

def _is_valid(adj, n):
    """Check both color classes are K2,2,2-free."""
    # Color 0 graph: adj
    # Color 1 graph: complement of adj (excluding self-loops)
    if _has_k222(adj, n):
        return False
    # Build complement
    comp = [0] * n
    full = (1 << n) - 1
    for i in range(n):
        comp[i] = full ^ adj[i] ^ (1 << i)  # remove self, complement
    if _has_k222(comp, n):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # Try Paley graph on q=29
    q = 29
    squares = set()
    for x in range(1, q):
        squares.add(x * x % q)
    
    n = q
    adj = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            diff = (j - i) % q
            if diff in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    if _is_valid(adj, n):
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    colors.append(0)
                else:
                    colors.append(1)
        return {'n': n, 'colors': colors}
    
    # Fallback to Paley on 17
    q = 17
    squares = set()
    for x in range(1, q):
        squares.add(x * x % q)
    
    n = q
    adj = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            diff = (j - i) % q
            if diff in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    colors = []
    for i in range(n):
        for j in range(i+1, n):
            if (adj[i] >> j) & 1:
                colors.append(0)
            else:
                colors.append(1)
    return {'n': n, 'colors': colors}
# EVOLVE-BLOCK-END
