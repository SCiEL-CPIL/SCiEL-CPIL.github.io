import time
import random
import numpy as np

DEADLINE_MARGIN = 10  # seconds before the 150s kill to return

def construct():
    start = time.time()
    deadline = start + DEADLINE_MARGIN

    # Known deterministic 29-vertex Paley coloring
    def paley_29():
        p = 29
        # Quadratic residues mod 29
        sq = set()
        for x in range(1, p):
            sq.add((x * x) % p)
        colors = []
        for i in range(p):
            for j in range(i + 1, p):
                d = (j - i) % p
                colors.append(0 if d in sq else 1)
        return {'n': p, 'colors': colors}

    best = paley_29()
    best_n = best['n']

    # Validation helper using bitmasks
    def is_valid_coloring(n, colors):
        """Check no monochromatic K_{2,2,2}. Returns True if valid."""
        if n < 8:
            return True
        # Build adjacency bitmasks for each color
        adj = [0] * n  # adj[i] has bit j set if color(i,j)==0
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                idx += 1
                if c == 0:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # adj1 = complement (edges with color 1)
        full_mask = (1 << n) - 2  # exclude self
        adj1 = [full_mask ^ adj[i] for i in range(n)]
        
        # A K_{2,2,2} in color c exists iff there exist disjoint sets A={a1,a2}, B={b1,b2}, C={c1,c2}
        # such that all 12 cross-edges between each pair are color c.
        # Equivalently: pick a1 < a2. Their common neighbors in color c must contain b1,b2 (b1!=a1,a2) and c1,c2.
        # And b1,b2 must be connected to c1,c2 in color c.
        # Faster: for each pair (a1,a2), let S = common color-c neighbors of a1,a2 (excluding a1,a2).
        # If |S| >= 4, check if S induces a K_{2,2} in color c.
        
        for color_adj in (adj, adj1):
            for a1 in range(n):
                for a2 in range(a1 + 1, n):
                    common = color_adj[a1] & color_adj[a2]
                    common &= ~(1 << a1)
                    common &= ~(1 << a2)
                    if common.bit_count() < 4:
                        continue
                    # Check if common contains a K_{2,2} in this color
                    # K_{2,2}: need two vertices u,v in common and two vertices w,x in common
                    # such that u-w, u-x, v-w, v-x are all edges in this color.
                    # i.e., u and v have common neighbors in common of size >= 2.
                    # Try all pairs (u,v) in common:
                    # Collect vertices in common
                    verts = []
                    m = common
                    while m:
                        lb = m & (-m)
                        v = lb.bit_length() - 1
                        verts.append(v)
                        m ^= lb
                    k = len(verts)
                    found = False
                    for ui in range(k):
                        u = verts[ui]
                        for vi in range(ui + 1, k):
                            v = verts[vi]
                            cu = color_adj[u] & common & ~(1 << u) & ~(1 << v)
                            if cu.bit_count() >= 2:
                                found = True
                                break
                        if found:
                            break
                    if found:
                        return False
        return True

    # Local search: start from Paley(29), try to extend to 30 or fix if invalid
    def try_local_search(n, colors, time_limit):
        """Attempt to find a valid coloring of K_n via local search. Returns colors or None."""
        random.seed(42)
        cur = colors[:]
        if not is_valid_coloring(n, cur):
            # Try to fix by flipping edges
            for _ in range(2000):
                if time.time() > time_limit:
                    break
                # Find a violation and flip a random edge
                # Simplified: just flip random edges and check
                idx = random.randint(0, len(cur) - 1)
                cur[idx] ^= 1
                if is_valid_coloring(n, cur):
                    return cur
        return cur if is_valid_coloring(n, cur) else None

    # Try to extend to n=30 using local search
    if time.time() < deadline:
        n_target = 30
        # Start from Paley(29), add vertex 29 with random colors
        base_colors = best['colors']
        # Extend to 30
        ext_colors = base_colors[:]
        for j in range(29):
            ext_colors.append(random.randint(0, 1))
        
        result = try_local_search(n_target, ext_colors, deadline - 2)
        if result is not None:
            if is_valid_coloring(n_target, result):
                best = {'n': n_target, 'colors': result}
                best_n = n_target

    # Try n=29 with random restarts if we haven't improved
    if best_n < 29 and time.time() < deadline:
        for attempt in range(50):
            if time.time() > deadline:
                break
            colors = [random.randint(0, 1) for _ in range(29 * 28 // 2)]
            # Local search
            for _ in range(500):
                if time.time() > deadline:
                    break
                if is_valid_coloring(29, colors):
                    best = {'n': 29, 'colors': colors}
                    best_n = 29
                    break
                idx = random.randint(0, len(colors) - 1)
                colors[idx] ^= 1

    return best

# EVOLVE-BLOCK-END
