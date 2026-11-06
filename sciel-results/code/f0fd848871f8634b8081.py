import random
import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Try multiple strategies, keep best result
    best_result = None
    best_n = 0

    # Strategy 1: Try to construct n=26 with focused local search
    r = focused_local_search(26, time_limit=25.0)
    if r is not None:
        best_result = r
        best_n = 26
        # Try to extend to 27
        r27 = try_extend_focused(r, time_limit=40.0)
        if r27 is not None:
            return r27
    elif time.time() < deadline:
        # Strategy 2: Incremental construction
        r = incremental_construct(26, time_limit=20.0)
        if r is not None:
            best_result = r
            best_n = 26
            r27 = try_extend_focused(r, time_limit=40.0)
            if r27 is not None:
                return r27

    # Try more n=26 attempts if time permits
    if best_result is None and time.time() < deadline - 30:
        for attempt in range(3):
            if time.time() > deadline - 30:
                break
            r = focused_local_search(26, time_limit=8.0)
            if r is not None:
                best_result = r
                best_n = 26
                break

    if best_result is not None:
        return best_result

    # Fallback: smaller n
    for n in [25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11]:
        if time.time() > deadline - 5:
            break
        r = focused_local_search(n, time_limit=5.0)
        if r is not None:
            return r

    # Absolute fallback
    return {'n': 11, 'colors': [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]}


def focused_local_search(n, time_limit=25.0):
    """Local search that focuses on edges involved in violations, with incremental updates."""
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx

    # Initialize with random coloring
    colors = [random.randint(0, 4) for _ in range(num_edges)]

    # Build adjacency bitmasks
    adj_mask = [[0] * n for _ in range(5)]
    for idx, (i, j) in enumerate(edges):
        c = colors[idx]
        adj_mask[c][i] |= (1 << j)
        adj_mask[c][j] |= (1 << i)

    # Count violations involving a specific vertex in a specific color
    def count_violations_vertex_color(v, c, adj_mask):
        """Count C4s where vertex v is one of the two 'diagonal' vertices in color c."""
        count = 0
        for u in range(n):
            if u == v:
                continue
            if adj_mask[c][v] & (1 << u):
                # v and u are adjacent in color c
                common = adj_mask[c][v] & adj_mask[c][u]
                k = common.bit_count()
                if k >= 2:
                    count += k * (k - 1) // 2
        return count

    # Total violations = sum over all colors of C4s
    # Each C4 is counted once per pair of diagonal vertices
    def count_total_violations(adj_mask):
        total = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    if adj_mask[c][u] & (1 << v):
                        common = adj_mask[c][u] & adj_mask[c][v]
                        k = common.bit_count()
                        if k >= 2:
                            total += k * (k - 1) // 2
        return total

    # Incremental: compute violation delta when changing edge (i,j) from c_old to c_new
    def compute_delta(i, j, c_old, c_new, adj_mask):
        """
        Compute change in total violations when edge (i,j) changes from c_old to c_new.
        Only C4s involving edge (i,j) or its endpoints in colors c_old/c_new can change.
        """
        delta = 0

        # Remove contribution of edge (i,j) in c_old
        # A C4 uses edge (i,j) if i-j is an edge AND there exist two common neighbors
        # of i and j in c_old (excluding each other)
        # Actually, C4s are defined by pairs of vertices with >=2 common neighbors.
        # Edge (i,j) being in color c_old means i and j are adjacent in c_old.
        # C4s that include edge (i,j): need another pair (k,l) such that 
        # {i,j,k,l} forms C4. This means i and j share >=2 common neighbors in c_old,
        # OR i and k share >=2, etc.
        # 
        # Simpler approach: the total violation count for color c is:
        # sum over all pairs (u,v) adjacent in c of C(common_neighbors(u,v) - 2, 2) + ... 
        # Actually it's: for each pair (u,v) with u<v, if they share >=2 common neighbors in c,
        # add C(k,2) where k = |common|. This counts each C4 exactly once per pair of
        # "opposite" vertices.
        #
        # When we remove edge (i,j) from c_old:
        # - Pairs involving i: (i,v) for v != j. If i-v were adjacent in c_old, removing i-j
        #   doesn't change common neighbors of i and v (unless j was a common neighbor).
        #   j is a common neighbor of i and v iff i-j and v-j are both in c_old.
        # - Similarly for pairs involving j.
        # - Pair (i,j) itself: no longer adjacent, so its contribution becomes 0.
        #
        # Let me think more carefully. The violation count for color c is:
        # V(c) = sum_{u<v, adj_c(u,v)} C(|N_c(u) ∩ N_c(v)|, 2)
        #
        # When edge (i,j) is removed from c_old:
        # 1. Term for pair (i,j) in c_old: was C(|N_c(i) ∩ N_c(j)|, 2), now 0.
        # 2. For pair (i,v) where v≠j: if adj_c(i,v), common neighbors change by ±1 
        #    (j was/is a common neighbor iff adj_c(j,v))
        # 3. For pair (j,v) where v≠i: similarly.
        # 4. For pair (u,v) where u,v≠i,j: no change (i,j not in their neighborhoods 
        #    unless they were adjacent to both, but that doesn't affect u-v common neighbors)
        #    Wait - i and j ARE in the neighborhoods of other vertices. If adj_c(u,i) and 
        #    adj_c(u,j), then i and j are both in N_c(u). But that affects pairs (u, x) 
        #    where x is... no, it affects whether i and j are common neighbors of u and some v.
        #    Actually, N_c(u) ∩ N_c(v) changes when we modify edges from i or j.
        #    If we remove edge (i,j) from c_old, then:
        #    - j is removed from N_c(i)
        #    - i is removed from N_c(j)
        #    - For any other vertex w, N_c(w) is unchanged (since we only changed edge i-j)
        #    So common neighbors of (u,v) change only if u or v is i or j.
        #
        # So the delta is:
        # - Remove contribution of pair (i,j) in c_old
        # - Adjust contributions of pairs (i,v) for v≠j (if adj in c_old)
        # - Adjust contributions of pairs (j,v) for v≠i (if adj in c_old)
        # Then similarly for adding to c_new.

        def pair_contribution(u, v, c, adj):
            """Contribution of pair (u,v) to violation count in color c."""
            if not (adj[c][u] & (1 << v)):
                return 0
            common = adj[c][u] & adj[c][v]
            k = common.bit_count()
            if k >= 2:
                return k * (k - 1) // 2
            return 0

        def adjust_pairs(vertex, other, c, adj, sign):
            """
            Compute delta from adjusting pairs (vertex, v) for v != other.
            sign = -1 for removal, +1 for addition.
            When removing edge (vertex, other) from color c:
            - other is removed from N_c(vertex)
            - For pair (vertex, v): common neighbors lose/gain 'other' iff adj_c(other, v)
            """
            delta = 0
            for v in range(n):
                if v == vertex or v == other:
                    continue
                if adj[c][vertex] & (1 << v):
                    # Pair (vertex, v) exists in color c
                    # Before change: common = N_c(vertex) ∩ N_c(v)
                    # After removing edge (vertex, other): N_c(vertex) loses 'other'
                    # So common loses 'other' iff other ∈ N_c(v), i.e., adj_c(other, v)
                    old_common = adj[c][vertex] & adj[c][v]
                    old_k = old_common.bit_count()
                    
                    # New common: remove 'other' from N_c(vertex), so if other was in common,
                    # it's gone
                    if adj[c][other] & (1 << v):
                        new_k = old_k - 1
                    else:
                        new_k = old_k
                    
                    old_contrib = old_k * (old_k - 1) // 2 if old_k >= 2 else 0
                    new_contrib = new_k * (new_k - 1) // 2 if new_k >= 2 else 0
                    delta += sign * (new_contrib - old_contrib)
            return delta

        # Remove from c_old
        # 1. Pair (i,j) contribution
        old_ij = pair_contribution(i, j, c_old, adj_mask)
        delta -= old_ij

        # 2. Pairs (i, v) for v != j
        delta += adjust_pairs(i, j, c_old, adj_mask, -1)

        # 3. Pairs (j, v) for v != i  
        delta += adjust_pairs(j, i, c_old, adj_mask, -1)

        # Add to c_new
        # 1. Pair (i,j) contribution
        new_ij = pair_contribution(i, j, c_new, adj_mask)
        delta += new_ij

        # 2. Pairs (i, v) for v != j
        delta += adjust_pairs(i, j, c_new, adj_mask, +1)

        # 3. Pairs (j, v) for v != i
        delta += adjust_pairs(j, i, c_new, adj_mask, +1)

        return delta

    current_violations = count_total_violations(adj_mask)
    if current_violations == 0:
        return {'n': n, 'colors': colors}

    best_violations = current_violations
    best_colors = colors[:]

    start_time = time.time()
    temperature = 1.5
    cooling_rate = 0.9997
    no_improve = 0
    max_no_improve = 3000

    # Track which edges are in violations for selection
    def get_violating_edges(adj_mask):
        violating_edges = set()
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    if adj_mask[c][u] & (1 << v):
                        common = adj_mask[c][u] & adj_mask[c][v]
                        k = common.bit_count()
                        if k >= 2:
                            violating_edges.add(edge_idx[(u, v)])
                            temp = common
                            while temp:
                                w = (temp & -temp).bit_length() - 1
                                violating_edges.add(edge_idx[(min(u,w), max(u,w))])
                                violating_edges.add(edge_idx[(min(v,w), max(v,w))])
                                temp &= temp - 1
        return violating_edges

    violating_edges = get_violating_edges(adj_mask)

    while time.time() - start_time < time_limit:
        if no_improve > max_no_improve:
            # Restart
            colors = [random.randint(0, 4) for _ in range(num_edges)]
            adj_mask = [[0] * n for _ in range(5)]
            for idx, (i, j) in enumerate(edges):
                c = colors[idx]
                adj_mask[c][i] |= (1 << j)
                adj_mask[c][j] |= (1 << i)
            current_violations = count_total_violations(adj_mask)
            violating_edges = get_violating_edges(adj_mask)
            temperature = 1.5
            no_improve = 0
            continue

        # Select edge: 70% from violating edges, 30% random
        if violating_edges and random.random() < 0.7:
            e_idx = random.choice(list(violating_edges))
        else:
            e_idx = random.randint(0, num_edges - 1)

        i, j = edges[e_idx]
        old_color = colors[e_idx]

        # Try all 5 colors, pick best using incremental delta
        best_new_color = old_color
        best_new_violations = current_violations

        for new_color in range(5):
            if new_color == old_color:
                continue

            # Apply change temporarily
            adj_mask[old_color][i] &= ~(1 << j)
            adj_mask[old_color][j] &= ~(1 << i)
            adj_mask[new_color][i] |= (1 << j)
            adj_mask[new_color][j] |= (1 << i)

            # Compute delta using the modified adj_mask
            # We need to compute what the violations would be
            # Since we already applied, let's just count the affected parts
            # Actually, let's compute delta before applying
            
            # Revert for now, compute delta properly
            adj_mask[new_color][i] &= ~(1 << j)
            adj_mask[new_color][j] &= ~(1 << i)
            adj_mask[old_color][i] |= (1 << j)
            adj_mask[old_color][j] |= (1 << i)

            # Compute delta
            d = compute_delta(i, j, old_color, new_color, adj_mask)
            new_violations = current_violations + d

            if new_violations < best_new_violations:
                best_new_violations = new_violations
                best_new_color = new_color

        if best_new_color != old_color:
            # Apply best change
            adj_mask[old_color][i] &= ~(1 << j)
            adj_mask[old_color][j] &= ~(1 << i)
            adj_mask[best_new_color][i] |= (1 << j)
            adj_mask[best_new_color][j] |= (1 << i)
            colors[e_idx] = best_new_color

            delta = best_new_violations - current_violations
            if delta < 0 or random.random() < math.exp(-delta / max(temperature, 0.01)):
                current_violations = best_new_violations
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = colors[:]
                    no_improve = 0
                    if best_violations == 0:
                        return {'n': n, 'colors': colors}
                else:
                    no_improve += 1
                # Update violating edges periodically (every 100 steps)
                if no_improve % 100 == 0 or no_improve == 0:
                    violating_edges = get_violating_edges(adj_mask)
            else:
                # Revert
                adj_mask[best_new_color][i] &= ~(1 << j)
                adj_mask[best_new_color][j] &= ~(1 << i)
                adj_mask[old_color][i] |= (1 << j)
                adj_mask[old_color][j] |= (1 << i)
                colors[e_idx] = old_color
                no_improve += 1
        else:
            no_improve += 1

        temperature *= cooling_rate
        if temperature < 0.05:
            temperature = 0.05

    return None


def incremental_construct(n, time_limit=20.0):
    """Try to build a valid coloring incrementally vertex by vertex."""
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = n * (n - 1) // 2

    # Start with a small valid coloring and grow
    colors = [random.randint(0, 4) for _ in range(num_edges)]

    adj_mask = [[0] * n for _ in range(5)]
    for idx, (i, j) in enumerate(edges):
        c = colors[idx]
        adj_mask[c][i] |= (1 << j)
        adj_mask[c][j] |= (1 << i)

    def count_violations():
        total = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    if adj_mask[c][u] & (1 << v):
                        common = adj_mask[c][u] & adj_mask[c][v]
                        k = common.bit_count()
                        if k >= 2:
                            total += k * (k - 1) // 2
        return total

    current_violations = count_violations()
    if current_violations == 0:
        return {'n': n, 'colors': colors}

    start_time = time.time()
    temperature = 1.0

    while time.time() - start_time < time_limit:
        if current_violations == 0:
            return {'n': n, 'colors': colors}

        # Pick a random edge
        e_idx = random.randint(0, num_edges - 1)
        i, j = edges[e_idx]
        old_color = colors[e_idx]
        new_color = random.randint(0, 4)
        if new_color == old_color:
            continue

        # Apply
        adj_mask[old_color][i] &= ~(1 << j)
        adj_mask[old_color][j] &= ~(1 << i)
        adj_mask[new_color][i] |= (1 << j)
        adj_mask[new_color][j] |= (1 << i)
        colors[e_idx] = new_color

        new_violations = count_violations()
        delta = new_violations - current_violations

        if delta < 0 or random.random() < math.exp(-delta / max(temperature, 0.01)):
            current_violations = new_violations
        else:
            # Revert
            adj_mask[new_color][i] &= ~(1 << j)
            adj_mask[new_color][j] &= ~(1 << i)
            adj_mask[old_color][i] |= (1 << j)
            adj_mask[old_color][j] |= (1 << i)
            colors[e_idx] = old_color

        temperature *= 0.999
        if temperature < 0.05:
            temperature = 0.05

    return None


def try_extend_focused(base_result, time_limit=40.0):
    """Extend a valid n-vertex coloring to n+1 vertices using focused search."""
    base_colors = base_result['colors']
    n_base = base_result['n']
    n_new = n_base + 1

    # Build adjacency masks for base
    adj_mask = [[0] * n_base for _ in range(5)]
    idx = 0
    for i in range(n_base):
        for j in range(i + 1, n_base):
            c = base_colors[idx]
            adj_mask[c][i] |= (1 << j)
            adj_mask[c][j] |= (1 << i)
            idx += 1

    def count_violations_new(new_colors):
        """Count C4 violations involving the new vertex."""
        violations = 0
        for c in range(5):
            neighbors = [i for i in range(n_base) if new_colors[i] == c]
            for a in range(len(neighbors)):
                u = neighbors[a]
                for b in range(a + 1, len(neighbors)):
                    v = neighbors[b]
                    common = adj_mask[c][u] & adj_mask[c][v]
                    k = common.bit_count()
                    if k >= 1:
                        violations += k
        return violations

    # Initialize
    new_colors = [random.randint(0, 4) for _ in range(n_base)]
    current_violations = count_violations_new(new_colors)

    if current_violations == 0:
        return build_full_coloring(base_colors, new_colors, n_base)

    best_violations = current_violations
    best_new_colors = new_colors[:]

    start_time = time.time()
    temperature = 1.0
    no_improve = 0

    while time.time() - start_time < time_limit:
        if best_violations == 0:
            break
        if no_improve > 2000:
            new_colors = [random.randint(0, 4) for _ in range(n_base)]
            current_violations = count_violations_new(new_colors)
            temperature = 1.0
            no_improve = 0
            continue

        # Pick a variable: prefer those in violating pairs
        e_idx = random.randint(0, n_base - 1)
        old_c = new_colors[e_idx]
        new_c = random.randint(0, 4)
        if new_c == old_c:
            continue

        new_colors[e_idx] = new_c
        new_violations = count_violations_new(new_colors)

        delta = new_violations - current_violations
        if delta < 0 or random.random() < math.exp(-delta / max(temperature, 0.01)):
            current_violations = new_violations
            if current_violations < best_violations:
                best_violations = current_violations
                best_new_colors = new_colors[:]
                no_improve = 0
                if best_violations == 0:
                    break
            else:
                no_improve += 1
        else:
            new_colors[e_idx] = old_c
            no_improve += 1

        temperature *= 0.999
        if temperature < 0.05:
            temperature = 0.05

    if best_violations == 0:
        return build_full_coloring(base_colors, best_new_colors, n_base)

    return None


def build_full_coloring(base_colors, new_colors, n_base):
    """Build the full (n_base+1)-vertex coloring from base and new vertex colors."""
    n_new = n_base + 1
    final_colors = [0] * (n_new * (n_new - 1) // 2)

    idx = 0
    for i in range(n_new):
        for j in range(i + 1, n_new):
            if j < n_base:
                # Edge in base graph
                base_idx = i * (n_base - 1) - i * (i - 1) // 2 + (j - i - 1)
                final_colors[idx] = base_colors[base_idx]
            else:
                # Edge to new vertex (j == n_base)
                final_colors[idx] = new_colors[i]
            idx += 1

    return {'n': n_new, 'colors': final_colors}
