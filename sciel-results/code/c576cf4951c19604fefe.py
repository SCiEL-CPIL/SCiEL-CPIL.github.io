"""Program for R_5(C4) maximization: greedy randomized construction with dynamic forbidden-pair tracking."""
import time
import random
import numpy as np
from collections import defaultdict

def construct():
    t0 = time.time()
    deadline = t0 + 115.0

    def is_valid(n, colors):
        """Check if no color class contains a C4."""
        for c in range(5):
            adj = [set() for _ in range(n)]
            for idx in range(n * (n - 1) // 2):
                if colors[idx] == c:
                    i, j = idx_to_edge(idx, n)
                    adj[i].add(j)
                    adj[j].add(i)
            for u in range(n):
                for v in range(u + 1, n):
                    if len(adj[u] & adj[v]) >= 2:
                        return False
        return True

    def idx_to_edge(idx, n):
        row_start = 0
        for i in range(n):
            row_len = n - 1 - i
            if idx < row_start + row_len:
                j = i + 1 + (idx - row_start)
                return (i, j)
            row_start += row_len
        return (0, 1)

    def edge_idx(i, j, n):
        return i * n - i * (i + 1) // 2 - 1 + (j - i - 1)

    def greedy_construct(n, seed, time_limit):
        """
        Greedy randomized construction:
        - Process edges in a random order
        - For each edge (u,v), evaluate each color c by counting how many
          "forbidden pairs" would be created (pairs with exactly 1 common neighbor
          in color c that would become 2)
        - Pick the color minimizing new forbidden pairs, with random tie-breaking
        - If no color creates a C4, proceed; otherwise backtrack or restart
        """
        random.seed(seed)
        m = n * (n - 1) // 2
        colors = [-1] * m

        # Precompute edge list
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))

        # Track for each color: common neighbor count for each pair
        # cn[c][u][v] = number of common neighbors of u and v in color c
        # We'll use a dictionary for efficiency
        # Actually, let's track: for each color c, for each vertex pair (u,v),
        # how many common neighbors they have in color c.
        # When we add edge (u,v) in color c, it affects:
        # For each neighbor w of u in color c, and each neighbor x of v in color c,
        # the pair (w,x) gains a common neighbor (which is... wait, no)
        #
        # Actually: adding edge (u,v) in color c means:
        # - u and v now have each other as a neighbor in color c
        # - For any w that is a neighbor of both u and v in color c, 
        #   the pair (w, ?) ... 
        #
        # Let me think differently. A C4 in color c exists if there are vertices
        # a,b,c,d such that edges (a,b), (b,c), (c,d), (d,a) are all in color c.
        # Equivalently, two vertices have >= 2 common neighbors in color c.
        #
        # When we assign color c to edge (u,v):
        # For each w that is adjacent to u in color c, and each x that is adjacent
        # to v in color c, if w != x, then w and x now share u and v? No...
        #
        # Wait. Let me reconsider. If edge (u,v) gets color c, then:
        # - u's color-c neighbors gain v
        # - v's color-c neighbors gain u
        #
        # A C4 is formed if there exist w, x such that:
        # - w is a color-c neighbor of u (before this assignment)
        # - x is a color-c neighbor of v (before this assignment)  
        # - w is a color-c neighbor of x
        # - u is a color-c neighbor of x
        # Hmm, this is getting complicated.
        #
        # Simpler approach: just track common neighbor counts directly.
        # cn[c][(u,v)] = number of vertices w such that (u,w) and (v,w) are both color c.
        #
        # When we assign color c to edge (u,v):
        # For each w in N_c(u) (neighbors of u in color c, excluding v if already there):
        #   cn[c][(w,v)] += 1  (because u is now a common neighbor of w and v in color c)
        #   if cn[c][(w,v)] becomes 2, we have a C4!
        # For each w in N_c(v) (neighbors of v in color c, excluding u):
        #   cn[c][(u,w)] += 1
        #   if cn[c][(u,w)] becomes 2, we have a C4!
        #
        # So the "cost" of assigning color c to edge (u,v) is:
        #   number of w in N_c(u) where cn[c][(w,v)] == 1 (these would become 2)
        # + number of w in N_c(v) where cn[c][(u,w)] == 1 (these would become 2)
        # If this cost > 0, assigning color c creates a C4.
        # If cost == 0, it's safe.
        # Among safe colors, we want to minimize the number of pairs that become "forbidden"
        # (i.e., pairs where cn becomes 1, since they're one step from C4).
        #
        # Let me implement this.

        # N_c[v] = set of neighbors of v in color c
        N = [set() for _ in range(n)]  # N[v] will be per-color, use dict of sets
        # Actually, let's use: adj[c][v] = set of neighbors
        adj = [defaultdict(set) for _ in range(5)]  # adj[c][v] = set of neighbors of v in color c

        # cn[c][(u,v)] for u < v: common neighbor count in color c
        # Use a dict for sparse storage
        cn = [defaultdict(int) for _ in range(5)]

        def get_cn(c, u, v):
            if u > v:
                u, v = v, u
            return cn[c][(u, v)]

        def set_cn(c, u, v, val):
            if u > v:
                u, v = v, u
            if val == 0:
                if (u, v) in cn[c]:
                    del cn[c][(u, v)]
            else:
                cn[c][(u, v)] = val

        # Process edges in random order
        edge_order = list(range(m))
        random.shuffle(edge_order)

        for step in range(m):
            if time.time() > t0 + time_limit:
                return None

            eidx = edge_order[step]
            i, j = idx_to_edge(eidx, n)

            # Evaluate each color
            best_color = -1
            best_cost = float('inf')
            best_forbidden_delta = float('inf')

            for c in range(5):
                # Check if assigning color c to (i,j) creates a C4
                c4_count = 0
                forbidden_delta = 0

                # For each neighbor w of i in color c:
                #   cn[c][(w,j)] increases by 1
                for w in adj[c][i]:
                    if w == j:
                        continue
                    old_cn = get_cn(c, w, j)
                    if old_cn >= 1:
                        c4_count += 1  # would become 2 or more
                    elif old_cn == 0:
                        forbidden_delta += 1  # would become 1

                # For each neighbor w of j in color c:
                #   cn[c][(i,w)] increases by 1
                for w in adj[c][j]:
                    if w == i:
                        continue
                    old_cn = get_cn(c, i, w)
                    if old_cn >= 1:
                        c4_count += 1
                    elif old_cn == 0:
                        forbidden_delta += 1

                if c4_count > 0:
                    continue  # This color creates a C4

                # This color is safe; minimize forbidden_delta
                if forbidden_delta < best_forbidden_delta or (forbidden_delta == best_forbidden_delta and random.random() < 0.5):
                    best_forbidden_delta = forbidden_delta
                    best_color = c

            if best_color == -1:
                # No safe color found; this greedy path failed
                return None

            # Assign best_color to edge (i,j)
            colors[eidx] = best_color
            c = best_color

            # Update data structures
            adj[c][i].add(j)
            adj[c][j].add(i)

            # Update common neighbor counts
            for w in adj[c][i]:
                if w == j:
                    continue
                old_cn = get_cn(c, w, j)
                set_cn(c, w, j, old_cn + 1)

            for w in adj[c][j]:
                if w == i:
                    continue
                old_cn = get_cn(c, i, w)
                set_cn(c, i, w, old_cn + 1)

        # Verify the result
        if is_valid(n, colors):
            return colors
        return None

    # Try greedy construction for decreasing n
    for target_n in [29, 28, 27, 26, 25, 24, 23, 22, 21, 20]:
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        if remaining < 3:
            break

        # Try multiple random seeds
        for attempt in range(50):
            if time.time() > deadline:
                break
            time_per_attempt = min((deadline - time.time()) / max(1, 50 - attempt), 2.0)
            colors = greedy_construct(target_n, seed=42 + attempt * 7, time_limit=time_per_attempt)
            if colors is not None:
                return {'n': target_n, 'colors': colors}

    # Fallback: small known constructions
    for fallback_n in [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]:
        if time.time() > deadline:
            break
        m = fallback_n * (fallback_n - 1) // 2
        colors = [0] * m
        for idx in range(m):
            i, j = idx_to_edge(idx, fallback_n)
            colors[idx] = (j - i) % 5
        if is_valid(fallback_n, colors):
            return {'n': fallback_n, 'colors': colors}

    return {'n': 1, 'colors': []}
