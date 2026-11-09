import time
import random
import numpy as np

def _has_mono_k222(adj, n, color):
    """Check if color class contains K_{2,2,2}.
    adj is an n x n boolean matrix where adj[i][j] = True means edge (i,j) has given color.
    K_{2,2,2}: need 6 vertices a,b,c,d,e,f with all 12 cross-edges present.
    Equivalently: need two disjoint pairs (a,b) and (c,d) and (e,f) where
    all edges between the three pairs exist.
    
    More efficiently: K_{2,2,2} exists iff there exist 3 vertices x,y,z such that
    N(x) ∩ N(y) ∩ N(z) has at least... no that's K_{1,1,1}.
    
    Actually K_{2,2,2}: pick 6 vertices partitioned into 3 pairs. All 12 edges between pairs must be present.
    So if we pick two vertices a,b that are NOT connected (in the complement sense, i.e., the edge between them is the OTHER color), then...
    
    Let me think differently. K_{2,2,2} in color c means: there exist vertices a1,a2,b1,b2,c1,c2 such that
    for all i in {a1,a2}, j in {b1,b2}: edge (i,j) is color c
    for all i in {a1,a2}, j in {c1,c2}: edge (i,j) is color c
    for all i in {b1,b2}, j in {c1,c2}: edge (i,j) is color c
    
    So: pick a pair (a1,a2) with edge NOT color c (or just any pair). Then we need N_c(a1) ∩ N_c(a2) to contain a K_{2,2} in color c... no.
    
    Actually: For a fixed pair {u,v} (the two vertices in one part), we need to find 4 more vertices w1,w2,x1,x2 such that
    all of w1,w2,x1,x2 are in N_c(u) ∩ N_c(v), AND the subgraph induced by {w1,w2,x1,x2} in color c contains K_{2,2}.
    K_{2,2} in color c among {w1,w2,x1,x2}: all 4 edges between {w1,w2} and {x1,x2} are color c.
    
    So: for pair {u,v}, let S = N_c(u) ∩ N_c(v). If S contains a K_{2,2} (i.e., there exist disjoint pairs {w1,w2},{x1,x2} ⊂ S with all 4 cross-edges in color c), then we have K_{2,2,2}.
    
    To check if S contains K_{2,2}: for each pair {w1,w2} ⊂ S, check if N_c(w1) ∩ N_c(w2) ∩ S has at least 2 vertices that form... actually just need |N_c(w1) ∩ N_c(w2) ∩ S| >= 2 and those two are distinct from w1,w2. Wait, w1,w2 are in S, and we need x1,x2 in S such that both x1,x2 are adjacent in color c to both w1 and w2. So |N_c(w1) ∩ N_c(w2) ∩ S \ {w1,w2}| >= 2.
    """
    # adj[i][j] is True if edge (i,j) has the specified color
    # For efficiency, use bitmasks
    n_local = n
    # Build bitmask representation: mask[i] is a bitmask of neighbors of i in this color
    masks = [0] * n_local
    for i in range(n_local):
        m = 0
        row = adj[i]
        for j in range(n_local):
            if i != j and row[j]:
                m |= (1 << j)
        masks[i] = m
    
    # For each pair (u,v), compute S = masks[u] & masks[v]
    # Then check if S contains K_{2,2}
    for u in range(n_local):
        for v in range(u + 1, n_local):
            S = masks[u] & masks[v]
            # S should not contain u or v (they're not adjacent to themselves)
            S &= ~(1 << u)
            S &= ~(1 << v)
            # Now check if S has K_{2,2}: need two disjoint pairs with all cross edges
            # Iterate over pairs in S
            s_list = []
            tmp = S
            while tmp:
                lsb = tmp & (-tmp)
                idx = lsb.bit_length() - 1
                s_list.append(idx)
                tmp ^= lsb
            m = len(s_list)
            if m < 4:
                continue
            # Check for K_{2,2} in S
            for a_idx in range(m):
                a = s_list[a_idx]
                for b_idx in range(a_idx + 1, m):
                    b = s_list[b_idx]
                    # Need two more vertices in S adjacent to both a and b in color c
                    common = masks[a] & masks[b] & S
                    common &= ~(1 << a)
                    common &= ~(1 << b)
                    if common.bit_count() >= 2:
                        return True
    return False


def _check_validity(n, colors):
    """Check if the coloring has no monochromatic K_{2,2,2}."""
    num_edges = n * (n - 1) // 2
    if len(colors) != num_edges:
        return False
    # Build adjacency matrices for each color
    adj0 = [[False] * n for _ in range(n)]
    adj1 = [[False] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            if c == 0:
                adj0[i][j] = adj0[j][i] = True
            else:
                adj1[i][j] = adj1[j][i] = True
            idx += 1
    if _has_mono_k222(adj0, n, 0):
        return False
    if _has_mono_k222(adj1, n, 1):
        return False
    return True


def _paley_coloring(n):
    """Generate a Paley-type coloring for prime n ≡ 1 mod 4."""
    if n < 2:
        return []
    # For Paley graph, use quadratic residues
    squares = set()
    for x in range(1, n):
        squares.add((x * x) % n)
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % n
            if d in squares:
                colors.append(0)
            else:
                colors.append(1)
    return colors


def _try_grow(n, colors, deadline):
    """Try to grow the coloring by adding one vertex at a time."""
    # Current state: n vertices, colors list of length n*(n-1)/2
    # Try adding vertex n
    while time.time() < deadline:
        # Try to add a new vertex with various color assignments
        new_n = n + 1
        if new_n > 31:
            break
        found = False
        # Try random colorings for the new edges
        for attempt in range(200):
            if time.time() > deadline:
                break
            new_colors = colors[:]
            for j in range(n):
                # Random color for edge (j, new_n)
                c = random.randint(0, 1)
                new_colors.append(c)
            # Check validity
            if _check_validity(new_n, new_colors):
                n = new_n
                colors = new_colors
                found = True
                break
        if not found:
            # Try more systematic approach: try all 2^n assignments for small n
            if n <= 12:
                for mask in range(1 << n):
                    if time.time() > deadline:
                        break
                    new_colors = colors[:]
                    for j in range(n):
                        c = (mask >> j) & 1
                        new_colors.append(c)
                    if _check_validity(n + 1, new_colors):
                        n = n + 1
                        colors = new_colors
                        found = True
                        break
            if not found:
                break
    return n, colors


def _local_search(n, colors, deadline):
    """Local search: flip edges to try to maintain validity while enabling growth."""
    # This is complex; for now, just do random restarts with growth
    return n, colors


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_n = 0
    best_colors = []
    
    # Start with Paley colorings for primes
    # Paley graphs exist for primes p ≡ 1 mod 4: 5, 13, 17, 29
    # Also try other constructions
    
    # First, try the known Paley coloring for n=17
    for prime in [5, 13, 17, 29]:
        if prime > 31:
            break
        colors = _paley_coloring(prime)
        if _check_validity(prime, colors):
            if prime > best_n:
                best_n = prime
                best_colors = colors
    
    # Try to grow from best_n
    if best_n > 0:
        n, colors = _try_grow(best_n, best_colors, deadline)
        if n > best_n:
            best_n = n
            best_colors = colors
    
    # Try random initial colorings and grow
    random.seed(42)
    for trial in range(50):
        if time.time() > deadline:
            break
        # Start with a random small coloring
        start_n = random.randint(5, 12)
        colors = [random.randint(0, 1) for _ in range(start_n * (start_n - 1) // 2)]
        if not _check_validity(start_n, colors):
            continue
        n, colors = _try_grow(start_n, colors, deadline)
        if n > best_n:
            best_n = n
            best_colors = colors
    
    # Try growing from smaller Paley graphs
    for prime in [5, 13]:
        colors = _paley_coloring(prime)
        if _check_validity(prime, colors):
            n, colors = _try_grow(prime, colors, deadline)
            if n > best_n:
                best_n = n
                best_colors = colors
    
    return {'n': best_n, 'colors': best_colors}
