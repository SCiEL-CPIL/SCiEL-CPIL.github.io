"""
4-coloring of K_n with no monochromatic triangle (R(3,3,3,3)).

Strategy:
  1. Build the known K_32 seed (two Clebsch copies, inter-copy edges color 3).
  2. Use fast bitmask-based monochromatic-triangle counting.
  3. Run Simulated Annealing on the full edge-coloring to minimize triangles
     for target n (50, 48, ...).  If we reach zero triangles, return immediately.
  4. If SA fails for a target, fall back to the best valid coloring found
     during the incremental greedy construction (or the seed K_32).

The bitmask approach (inspiration 2) gives O(1) triangle detection per edge
flip, which is far faster than the O(n^2) per-vertex check in the current
program.  This allows many more SA iterations within the 120 s budget.
"""

import time
import random
import numpy as np


def construct():
    t0 = time.time()
    DEADLINE = t0 + 115.0  # safety margin before 120 s kill

    random.seed(42)
    np.random.seed(42)

    # ---------- GF(16) helpers for the Clebsch seed ----------
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)

    def build_seed_32():
        """Return list of colors for K_32 in lexicographic edge order."""
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                colors.append(
                    log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3
                )
        return colors

    # ---------- Fast triangle counting with bitmasks ----------
    # For a given n and color assignment, maintain for each color c a list of
    # bitmasks: adj[c][i] is a Python int whose bit j is 1 iff edge (i,j) has
    # color c.  The number of monochromatic triangles of color c is
    #   sum_i popcount(adj[c][i] & adj[c][j]) / 3  (each triangle counted 3x)
    # but we only need the total, and we can compute it as
    #   sum over edges (i,j) with color c of popcount(adj[c][i] & adj[c][j]) / 3
    # Actually simpler: total triangles = (1/3) * sum_c sum_i sum_j>i
    #   popcount(adj[c][i] & adj[c][j])  ... no, that's not right either.
    #
    # Correct formula: a monochromatic triangle of color c is a triple (i,j,k)
    # with all three edges color c.  For a fixed color c, the number of such
    # triangles is (1/3) * sum over all edges (i,j) of color c of
    # popcount(adj[c][i] & adj[c][j]).  Each triangle (i,j,k) is counted once
    # for each of its three edges, so dividing by 3 gives the count.
    #
    # For efficiency we maintain the total triangle count incrementally.

    def count_triangles(n, adj):
        """Count total monochromatic triangles given adjacency bitmasks."""
        total = 0
        for c in range(4):
            s = 0
            for i in range(n):
                ai = adj[c][i]
                # Only count edges (i,j) where j > i to avoid double counting
                # within the sum; but the formula counts each triangle 3 times
                # (once per edge), so we sum over ALL ordered pairs i<j where
                # edge (i,j) has color c.
                # Actually, let's just sum over all i, and for each i sum over
                # j > i where adj[c][i] has bit j set.
                # But adj[c][i] is symmetric, so bits are set for both i->j and j->i.
                # We need to be careful. Let's just iterate j from i+1 to n-1.
                mask = ai
                # Clear bits 0..i to only consider j > i
                mask &= ~((1 << (i + 1)) - 1)
                while mask:
                    # Extract lowest set bit
                    lb = mask & (-mask)
                    j = lb.bit_length() - 1
                    s += (adj[c][i] & adj[c][j]).bit_count()
                    mask ^= lb
            total += s
        return total // 3  # each triangle counted 3 times (once per edge)

    # Actually, the above is O(n^2) per call which is fine for n<=50 (1225 edges).
    # But for SA we want O(1) delta per move. Let's use the incremental approach.

    # ---------- SA with incremental triangle counting ----------
    def sa_color(n, init_colors, time_limit):
        """
        Run simulated annealing to minimize monochromatic triangles.
        Returns (colors_list, n_triangles) or (None, None) on timeout.
        """
        # Build adjacency bitmasks
        adj = [[0] * n for _ in range(4)]
        for i in range(n):
            for j in range(i + 1, n):
                idx = i * n - i * (i + 1) // 2 + (j - i - 1)
                c = init_colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)

        # Compute initial triangle count
        def total_triangles():
            total = 0
            for c in range(4):
                s = 0
                for i in range(n):
                    mask = adj[c][i] & ~((1 << (i + 1)) - 1)  # j > i
                    while mask:
                        lb = mask & (-mask)
                        j = lb.bit_length() - 1
                        s += (adj[c][i] & adj[c][j]).bit_count()
                        mask ^= lb
                total += s
            return total // 3

        tri = total_triangles()
        if tri == 0:
            return init_colors, 0

        T = 10.0
        cooling = 0.999
        min_T = 0.01
        best_tri = tri
        best_colors = init_colors[:]

        while time.time() < time_limit:
            if tri == 0:
                break

            # Pick a random edge
            ei = random.randint(0, n * (n - 1) // 2 - 1)
            # Decode edge index to (i, j)
            # Find i such that i*n - i*(i+1)//2 <= ei < (i+1)*n - (i+1)*(i+2)//2
            # Solve: i*n - i*(i+1)//2 <= ei
            # Approximate: i^2/2 ~ ei => i ~ sqrt(2*ei)
            lo, hi = 0, n - 1
            while lo < hi:
                mid = (lo + hi + 1) // 2
                start_mid = mid * n - mid * (mid + 1) // 2
                if start_mid <= ei:
                    lo = mid
                else:
                    hi = mid - 1
            i = lo
            start_i = i * n - i * (i + 1) // 2
            j = i + 1 + (ei - start_i)

            c_old = init_colors[ei]
            c_new = (c_old + 1 + random.randint(0, 2)) % 4

            # Compute delta in triangle count
            # Removing edge (i,j) from color c_old:
            #   This removes triangles where (i,j) is the "third" edge.
            #   Number of such triangles = popcount(adj[c_old][i] & adj[c_old][j])
            #   (these are vertices k such that both (i,k) and (j,k) are color c_old)
            # Adding edge (i,j) to color c_new:
            #   Number of new triangles = popcount(adj[c_new][i] & adj[c_new][j])
            # But wait: when we remove (i,j) from c_old, the adj masks for c_old
            # change, which affects future counts. For the delta calculation,
            # we consider the current state.
            #
            # Current triangles involving edge (i,j) with color c_old:
            #   k_old = popcount(adj[c_old][i] & adj[c_old][j])
            #   This counts vertices k where both (i,k) and (j,k) are color c_old.
            #   Each such k forms a triangle (i,j,k) of color c_old.
            #
            # After removing (i,j) from c_old and adding to c_new:
            #   The edge (i,j) no longer contributes to c_old triangles.
            #   It contributes k_new = popcount(adj[c_new][i] & adj[c_new][j])
            #   to c_new triangles.
            #
            # However, removing (i,j) from c_old also means that for any vertex k,
            # the pair (i,k) or (j,k) might no longer be "connected" through (i,j),
            # but those are already counted in other edges' contributions.
            #
            # The net change in total triangle count is:
            #   delta = k_new - k_old
            #
            # This is because:
            #   - We lose k_old triangles of color c_old (those that include edge (i,j)).
            #   - We gain k_new triangles of color c_new (those that include edge (i,j)).
            #   - No other triangles are affected because the only edge that changed
            #     is (i,j), and a triangle is monochromatic only if all three edges
            #     have the same color. Changing one edge's color can only affect
            #     triangles that include that edge.

            k_old = (adj[c_old][i] & adj[c_old][j]).bit_count()
            k_new = (adj[c_new][i] & adj[c_new][j]).bit_count()
            delta = k_new - k_old

            # Acceptance
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = True  # accept neutral moves to explore
            else:
                if T > min_T:
                    p = np.exp(-delta / T)
                    if random.random() < p:
                        accept = True

            if accept:
                # Update state
                init_colors[ei] = c_new
                adj[c_old][i] &= ~(1 << j)
                adj[c_old][j] &= ~(1 << i)
                adj[c_new][i] |= (1 << j)
                adj[c_new][j] |= (1 << i)
                tri += delta
                if tri < best_tri:
                    best_tri = tri
                    best_colors = init_colors[:]
                    if best_tri == 0:
                        break

            T *= cooling
            if T < min_T:
                T = min_T

        return best_colors, best_tri

    # ---------- Step 1: Build seed K_32 ----------
    seed_colors = build_seed_32()
    best_n = 32
    best_result = {'n': 32, 'colors': seed_colors}

    # ---------- Step 2: Try SA for target n = 50, 48, ... ----------
    # For each target, initialize with random colors (or extend seed) and run SA.
    # If SA finds 0 triangles, return immediately.

    targets = [50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40]
    for target_n in targets:
        if target_n <= best_n:
            continue
        if time.time() > DEADLINE - 10:
            break

        # Try multiple restarts with random initialization
        for restart in range(3):
            if time.time() > DEADLINE - 5:
                break

            # Random initialization
            n_edges = target_n * (target_n - 1) // 2
            init = list(np.random.randint(0, 4, size=n_edges, dtype=np.int8))

            colors, tri = sa_color(target_n, init, time_limit=DEADLINE - 5)
            if colors is not None and tri == 0:
                return {'n': target_n, 'colors': colors}

        # If SA didn't find a perfect coloring, check if it improved on best_n
        # (not directly useful since we need exact 0, but keep the logic simple)

    # ---------- Step 3: Fallback - incremental greedy from seed ----------
    # If SA failed for all targets, try to incrementally extend the seed.
    # This is slower but might find a valid coloring for n > 32.

    def edge_idx(i, j, n):
        return i * n - i * (i + 1) // 2 + (j - i - 1)

    def check_new_vertex_valid(n, colors, new_colors):
        for i in range(n):
            ci = new_colors[i]
            for j in range(i + 1, n):
                cj = new_colors[j]
                if ci == cj:
                    idx = edge_idx(i, j, n)
                    if colors[idx] == ci:
                        return False
        return True

    def try_greedy_add_vertex(n, colors, deadline):
        assignment = [None] * n

        def backtrack(v):
            if time.time() > deadline:
                return None
            if v == n:
                return assignment[:]
            for c in [0, 1, 2, 3]:
                assignment[v] = c
                ok = True
                for i in range(v):
                    if assignment[i] == c:
                        idx = edge_idx(i, v, n)
                        if colors[idx] == c:
                            ok = False
                            break
                if not ok:
                    assignment[v] = None
                    continue
                result = backtrack(v + 1)
                if result is not None:
                    return result
                assignment[v] = None
            return None

        return backtrack(0)

    # Try to extend seed
    n = 32
    colors = seed_colors[:]
    for target in range(32, 51):
        if time.time() > DEADLINE - 5:
            break
        new_colors = try_greedy_add_vertex(n, colors, DEADLINE - 5)
        if new_colors is not None:
            for i in range(n):
                colors.append(new_colors[i])
            n += 1
            if n > best_n:
                best_n = n
                best_result = {'n': n, 'colors': colors[:]}
        else:
            break

    return best_result
