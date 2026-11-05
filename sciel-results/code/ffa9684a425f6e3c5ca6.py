import time
import random
import math

def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit

    # --- GF(16) multiplication for Greenwood-Gleason coloring ---
    def mul(a, b):
        r = 0
        for i in range(4):
            if (b >> i) & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if (r >> i) & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log_map = {}
    x = 1
    for e in range(15):
        log_map[x] = e
        x = mul(x, 2)

    def gg_color(i, j):
        """Greenwood-Gleason coloring of K_16: vertices are elements of GF(16)"""
        return log_map[i ^ j] % 3

    # --- Construct 50-vertex coloring using F_2^5 structure ---
    # The 50-vertex construction: vertices are 0..49
    # Use the known algebraic construction based on quadratic forms over F_2^5
    #
    # Standard construction: partition 50 vertices into 2 groups of 25
    # Actually, the known 50-vertex 4-coloring of K_50 without mono triangles
    # uses the structure: vertices = F_2^5 \ {0} plus some structure
    #
    # Let me use the direct known construction:
    # The 50-vertex coloring can be built from the 16-vertex one by a "doubling"
    # with 4 colors where cross edges use color 3, and within each half we use
    # the GG coloring. But that only gives 32 vertices.
    #
    # The actual 50-vertex construction (Chung) uses a different approach.
    # Let me use a known explicit construction.
    #
    # Actually, let me use the following well-known construction:
    # Take vertices as elements of Z_2 x Z_2 x Z_2 x Z_2 x Z_2 = F_2^5
    # There are 32 elements. We need 50, so this isn't direct.
    #
    # The Chung 50-vertex construction: I'll use a greedy/local search approach
    # starting from a good seed.
    #
    # Let me try a different approach: use the known fact that the 50-vertex
    # coloring exists and use local search to find it quickly.
    #
    # Strategy: Start with a structured initial coloring, then local search.

    N = 50
    num_edges = N * (N - 1) // 2

    # Edge mapping
    edges = []
    edge_to_idx = {}
    for i in range(N):
        for j in range(i + 1, N):
            edge_to_idx[(i, j)] = len(edges)
            edges.append((i, j))

    # Precompute for each edge, the list of triangles it belongs to
    # For edge (u,v), triangles are (u,v,w) for w != u,v
    edge_triangles = [[] for _ in range(num_edges)]
    for idx, (u, v) in enumerate(edges):
        for w in range(N):
            if w == u or w == v:
                continue
            # Find indices of edges (u,w) and (v,w)
            if u < w:
                idx_uw = edge_to_idx[(u, w)]
            else:
                idx_uw = edge_to_idx[(w, u)]
            if v < w:
                idx_vw = edge_to_idx[(v, w)]
            else:
                idx_vw = edge_to_idx[(w, v)]
            edge_triangles[idx].append((w, idx_uw, idx_vw))

    def init_mono_count(colors):
        """Count monochromatic triangles"""
        count = 0
        for i in range(N - 2):
            for j in range(i + 1, N - 1):
                c_ij = colors[edge_to_idx[(i, j)]]
                for k in range(j + 1, N):
                    c_jk = colors[edge_to_idx[(j, k)]]
                    c_ik = colors[edge_to_idx[(i, k)]]
                    if c_ij == c_jk == c_ik:
                        count += 1
        return count

    def init_mono_list(colors):
        """Return list of monochromatic triangles as (i,j,k) tuples"""
        mono_list = []
        for i in range(N - 2):
            for j in range(i + 1, N - 1):
                c_ij = colors[edge_to_idx[(i, j)]]
                for k in range(j + 1, N):
                    c_jk = colors[edge_to_idx[(j, k)]]
                    c_ik = colors[edge_to_idx[(i, k)]]
                    if c_ij == c_jk == c_ik:
                        mono_list.append((i, j, k))
        return mono_list

    def delta_for_change(colors, edge_idx, new_c):
        """Compute change in mono triangle count if we set edge_idx to new_c"""
        old_c = colors[edge_idx]
        if old_c == new_c:
            return 0
        u, v = edges[edge_idx]
        delta = 0
        for w, idx_uw, idx_vw in edge_triangles[edge_idx]:
            c_uw = colors[idx_uw]
            c_vw = colors[idx_vw]
            # Was mono?
            was_mono = (old_c == c_uw == c_vw)
            # Will be mono?
            will_be_mono = (new_c == c_uw == c_vw)
            if was_mono:
                delta -= 1
            if will_be_mono:
                delta += 1
        return delta

    # --- Structured initial coloring ---
    # Use a block structure: divide 50 vertices into blocks
    # Try: 5 blocks of 10 vertices each
    # Within each block: use a 10-vertex 3-coloring (from GG K_16 restricted)
    # Between blocks: use color 3
    #
    # Actually, let me use a simpler structured init:
    # Color based on vertex index modulo patterns

    def structured_init():
        colors = [0] * num_edges
        for idx, (i, j) in enumerate(edges):
            # Use a pattern based on i and j
            # Try: color = (i + j) % 4
            colors[idx] = (i + j) % 4
        return colors

    def gg_based_init():
        """Initialize using Greenwood-Gleason structure where possible"""
        colors = [0] * num_edges
        for idx, (i, j) in enumerate(edges):
            if i < 16 and j < 16:
                colors[idx] = gg_color(i, j)
            elif i >= 32 and j >= 32:
                colors[idx] = gg_color(i - 32, j - 32)
            elif i < 16 and j >= 32:
                # Cross edges between first and third block
                colors[idx] = 3
            elif i >= 16 and i < 32 and j >= 32:
                colors[idx] = 3
            elif i >= 16 and i < 32 and j < 16:
                colors[idx] = 3
            else:
                # Middle block (16-31) and others
                colors[idx] = (i + j) % 4
        return colors

    best_n = 32
    best_colors = None

    # First, get the known 32-vertex coloring as fallback
    n_base = 32
    base_colors = [0] * (n_base * (n_base - 1) // 2)
    idx = 0
    for i in range(n_base):
        for j in range(i + 1, n_base):
            if i < 16 and j < 16:
                c = gg_color(i, j)
            elif i >= 16 and j >= 16:
                c = gg_color(i - 16, j - 16)
            else:
                c = 3
            base_colors[idx] = c
            idx += 1
    best_colors = base_colors[:]

    # --- Local search to find 50-vertex coloring ---
    random.seed(42)  # Reproducibility

    for attempt in range(20):
        if time.time() > deadline - 30:
            break

        # Different initializations
        if attempt == 0:
            colors = gg_based_init()
        elif attempt == 1:
            colors = structured_init()
        else:
            colors = [random.randint(0, 3) for _ in range(num_edges)]

        mono_list = init_mono_list(colors)
        if len(mono_list) == 0:
            return {'n': N, 'colors': colors}

        # Simulated annealing
        temperature = 1.5
        min_temp = 0.01
        cooling = 0.9998
        steps = 0
        max_steps = 50000
        no_improve = 0
        best_mono_count = len(mono_list)
        best_colors_copy = colors[:]

        while len(mono_list) > 0 and steps < max_steps and time.time() < deadline - 30:
            # Pick a random mono triangle
            tri = random.choice(mono_list)
            i, j, k = tri

            # Pick a random edge from the triangle
            edge_choice = random.randint(0, 2)
            if edge_choice == 0:
                u, v = i, j
            elif edge_choice == 1:
                u, v = j, k
            else:
                u, v = i, k

            idx_uv = edge_to_idx[(min(u, v), max(u, v))]
            old_c = colors[idx_uv]

            # Try all 4 colors, pick best
            best_delta = float('inf')
            best_new_c = -1
            for new_c in range(4):
                if new_c == old_c:
                    continue
                d = delta_for_change(colors, idx_uv, new_c)
                if d < best_delta:
                    best_delta = d
                    best_new_c = new_c

            if best_new_c == -1:
                break

            # Acceptance
            if best_delta < 0:
                accept = True
                no_improve = 0
            elif best_delta == 0:
                accept = random.random() < 0.5
                no_improve += 1
            else:
                accept = random.random() < math.exp(-best_delta / temperature)
                no_improve += 1

            if accept:
                colors[idx_uv] = best_new_c
                # Update mono_list
                u, v = edges[idx_uv]
                new_mono_list = []
                for (a, b, c) in mono_list:
                    # Check if this triangle is affected
                    if (a, b) == (u, v) or (b, c) == (u, v) or (a, c) == (u, v) or \
                       (a, b) == (v, u) or (b, c) == (v, u) or (a, c) == (v, u):
                        # Recompute
                        ca_b = colors[edge_to_idx[(min(a, b), max(a, b))]]
                        cb_c = colors[edge_to_idx[(min(b, c), max(b, c))]]
                        ca_c = colors[edge_to_idx[(min(a, c), max(a, c))]]
                        if not (ca_b == cb_c == ca_c):
                            continue
                    new_mono_list.append((a, b, c))

                # Add new mono triangles involving edge (u,v)
                for w, idx_uw, idx_vw in edge_triangles[idx_uv]:
                    c_uw = colors[idx_uw]
                    c_vw = colors[idx_vw]
                    if best_new_c == c_uw == c_vw:
                        tri_key = tuple(sorted([u, v, w]))
                        if tri_key not in new_mono_list:
                            new_mono_list.append(tri_key)

                mono_list = new_mono_list
                steps += 1

                if len(mono_list) < best_mono_count:
                    best_mono_count = len(mono_list)
                    best_colors_copy = colors[:]
                    no_improve = 0

            temperature *= cooling
            if temperature < min_temp:
                temperature = min_temp

            # Reheat if stuck
            if no_improve > 5000:
                temperature = 1.0
                no_improve = 0

        if len(mono_list) == 0:
            return {'n': N, 'colors': colors}

        # Update best if improved
        if best_mono_count < len(init_mono_list(best_colors_copy)):
            pass  # We already track best_colors_copy

    # --- Try to extend to 51 vertices ---
    # If we found a 50-vertex coloring, try to add vertex 50
    # For now, if we haven't found 50, return 32
    # If we have 50, try local search with 51 vertices

    # Check if best_colors_copy has no mono triangles
    if best_colors_copy is not None:
        test_mono = init_mono_list(best_colors_copy)
        if len(test_mono) == 0:
            # Try to extend to 51
            N2 = 51
            num_edges2 = N2 * (N2 - 1) // 2
            edges2 = []
            edge_to_idx2 = {}
            for i in range(N2):
                for j in range(i + 1, N2):
                    edge_to_idx2[(i, j)] = len(edges2)
                    edges2.append((i, j))

            # Initialize: first 50 vertices use best_colors_copy, new vertex random
            colors2 = [0] * num_edges2
            for idx2, (i, j) in enumerate(edges2):
                if i < 50 and j < 50:
                    colors2[idx2] = best_colors_copy[edge_to_idx[(i, j)]]
                else:
                    colors2[idx2] = random.randint(0, 3)

            # Local search on 51 vertices
            # ... (simplified: just try a few steps)
            # For time reasons, skip if not enough time
            if time.time() < deadline - 10:
                mono_list2 = []
                for i in range(N2 - 2):
                    for j in range(i + 1, N2 - 1):
                        c_ij = colors2[edge_to_idx2[(i, j)]]
                        for k in range(j + 1, N2):
                            c_jk = colors2[edge_to_idx2[(j, k)]]
                            c_ik = colors2[edge_to_idx2[(i, k)]]
                            if c_ij == c_jk == c_ik:
                                mono_list2.append((i, j, k))

                # Quick local search
                temp = 1.0
                for step in range(5000):
                    if time.time() > deadline - 5:
                        break
                    if not mono_list2:
                        return {'n': N2, 'colors': colors2}
                    tri = random.choice(mono_list2)
                    i, j, k = tri
                    ec = random.randint(0, 2)
                    if ec == 0:
                        u, v = i, j
                    elif ec == 1:
                        u, v = j, k
                    else:
                        u, v = i, k
                    idx_uv = edge_to_idx2[(min(u, v), max(u, v))]
                    old_c = colors2[idx_uv]
                    best_d = float('inf')
                    best_nc = -1
                    for nc in range(4):
                        if nc == old_c:
                            continue
                        d = 0
                        for w in range(N2):
                            if w == u or w == v:
                                continue
                            if u < w:
                                iuw = edge_to_idx2[(u, w)]
                            else:
                                iuw = edge_to_idx2[(w, u)]
                            if v < w:
                                ivw = edge_to_idx2[(v, w)]
                            else:
                                ivw = edge_to_idx2[(w, v)]
                            cuw = colors2[iuw]
                            cvw = colors2[ivw]
                            if old_c == cuw == cvw:
                                d -= 1
                            if nc == cuw == cvw:
                                d += 1
                        if d < best_d:
                            best_d = d
                            best_nc = nc
                    if best_nc != -1:
                        accept = best_d < 0 or (best_d == 0 and random.random() < 0.5) or \
                                 (best_d > 0 and random.random() < math.exp(-best_d / temp))
                        if accept:
                            colors2[idx_uv] = best_nc
                            # Rebuild mono list (expensive but safe)
                            mono_list2 = []
                            for a in range(N2 - 2):
                                for b in range(a + 1, N2 - 1):
                                    cab = colors2[edge_to_idx2[(a, b)]]
                                    for c in range(b + 1, N2):
                                        cbc = colors2[edge_to_idx2[(b, c)]]
                                        cac = colors2[edge_to_idx2[(a, c)]]
                                        if cab == cbc == cac:
                                            mono_list2.append((a, b, c))
                            temp *= 0.999
                            if temp < 0.01:
                                temp = 0.01

    return {'n': best_n, 'colors': best_colors}
