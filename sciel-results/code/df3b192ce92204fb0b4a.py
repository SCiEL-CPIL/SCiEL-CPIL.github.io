import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    n = 20
    num_pairs = n * (n - 1) // 2  # 190

    # Precompute edge index and vertex pairs
    pair_to_idx = {}
    idx_to_pair = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            pair_to_idx[(i, j)] = idx
            idx_to_pair.append((i, j))
            idx += 1

    # Precompute for each edge (u,v): all other edges that share exactly one endpoint with it
    # i.e., edges (u,w) or (v,w) for w != u,v
    # We need: for each edge e=(u,v), the set of edges adjacent to u or v (excluding e itself)
    # This is used for fast validity checking.

    # Precompute: for each vertex, list of edge indices incident to it
    vertex_edges = [[] for _ in range(n)]
    for e_idx in range(num_pairs):
        u, v = idx_to_pair[e_idx]
        vertex_edges[u].append(e_idx)
        vertex_edges[v].append(e_idx)

    # For each edge e=(u,v), precompute the "conflict" edges:
    # Adding e creates a K2,3 iff there exist w1,w2,w3 such that
    # (u,w1),(u,w2),(u,w3),(v,w1),(v,w2),(v,w3) are all present.
    # Equivalently: among common neighbors of u and v (vertices w with both (u,w) and (v,w) present),
    # if there are >= 3, then K2,3 exists.
    # But also: adding (u,v) could create K2,3 with pairs (u,w) and (v,w') where w != w',
    # no wait - K2,3 is two vertices with 3 common neighbors.
    # The forbidden subgraph is: vertices {a,b} with common neighbors {c,d,e}.
    # So K2,3-free means: for every pair (a,b), |N(a) ∩ N(b)| <= 2.
    #
    # When we add edge (u,v), the only new K2,3 violations that can be created involve
    # the pair (u,v) itself (since (u,v) wasn't an edge before, the pair (u,v) didn't count).
    # Wait no - (u,v) being an edge doesn't affect whether u and v have common neighbors.
    # The common neighbors of u and v are vertices w such that (u,w) and (v,w) are edges.
    # Adding (u,v) doesn't change N(u) ∩ N(v) in terms of which vertices are common neighbors.
    #
    # BUT: adding (u,v) means u is now in N(v) and v is now in N(u).
    # So for any pair (u, w) where w is a neighbor of v: u and w now share v as a common neighbor.
    # Similarly for (v, w) where w is a neighbor of u.
    #
    # So the K2,3 check when adding (u,v):
    # 1. Check pair (u,v): count common neighbors of u and v. If >= 3, violation.
    # 2. For each neighbor w of v (w != u): check if u and w now have >= 3 common neighbors.
    #    Before adding (u,v), they had some count. After, v becomes a common neighbor.
    #    So if they already had >= 2 common neighbors, adding (u,v) makes it 3.
    # 3. For each neighbor w of u (w != v): check if v and w now have >= 3 common neighbors.
    #    Similarly, if they already had >= 2 common neighbors, adding (u,v) makes it 3.

    # Efficient representation: use a 190-bit integer for the edge set.
    # But Python ints are arbitrary precision, so we can use them directly.

    def count_edges(edge_mask):
        return bin(edge_mask).count('1')

    def has_edge(edge_mask, u, v):
        if u > v:
            u, v = v, u
        return (edge_mask >> pair_to_idx[(u, v)]) & 1

    def common_neighbors_count(edge_mask, u, v):
        """Count vertices w such that (u,w) and (v,w) are both in edge_mask."""
        cnt = 0
        for w in range(n):
            if w == u or w == v:
                continue
            eu = pair_to_idx[(min(u,w), max(u,w))]
            ev = pair_to_idx[(min(v,w), max(v,w))]
            if ((edge_mask >> eu) & 1) and ((edge_mask >> ev) & 1):
                cnt += 1
        return cnt

    def is_valid_add(edge_mask, u, v):
        """Check if adding edge (u,v) keeps the graph K2,3-free."""
        if u > v:
            u, v = v, u
        e_idx = pair_to_idx[(u, v)]
        if (edge_mask >> e_idx) & 1:
            return True  # Already present

        # Check 1: common neighbors of u and v
        cn_uv = common_neighbors_count(edge_mask, u, v)
        if cn_uv >= 3:
            return False

        # Check 2: for each neighbor w of v (w != u), check pair (u, w)
        # After adding (u,v), v becomes a common neighbor of u and w.
        # So we need: common_neighbors_count(edge_mask, u, w) < 2 for all neighbors w of v.
        # (If it's already >= 2, adding (u,v) makes it >= 3)
        for w in range(n):
            if w == u or w == v:
                continue
            if has_edge(edge_mask, v, w):
                cn_uw = common_neighbors_count(edge_mask, u, w)
                if cn_uw >= 2:
                    return False

        # Check 3: for each neighbor w of u (w != v), check pair (v, w)
        for w in range(n):
            if w == u or w == v:
                continue
            if has_edge(edge_mask, u, w):
                cn_vw = common_neighbors_count(edge_mask, v, w)
                if cn_vw >= 2:
                    return False

        return True

    def add_edge(edge_mask, u, v):
        if u > v:
            u, v = v, u
        return edge_mask | (1 << pair_to_idx[(u, v)])

    def remove_edge(edge_mask, u, v):
        if u > v:
            u, v = v, u
        return edge_mask & ~(1 << pair_to_idx[(u, v)])

    # Faster validity check using precomputed neighbor lists
    def get_neighbors(edge_mask, vertex):
        """Return list of neighbors of vertex in the graph."""
        nbrs = []
        for w in range(n):
            if w != vertex and has_edge(edge_mask, vertex, w):
                nbrs.append(w)
        return nbrs

    def is_valid_add_fast(edge_mask, u, v):
        """Faster check using neighbor lists."""
        if u > v:
            u, v = v, u
        e_idx = pair_to_idx[(u, v)]
        if (edge_mask >> e_idx) & 1:
            return True

        nbrs_u = get_neighbors(edge_mask, u)
        nbrs_v = get_neighbors(edge_mask, v)
        set_nbrs_u = set(nbrs_u)
        set_nbrs_v = set(nbrs_v)

        # Common neighbors of u and v
        common = set_nbrs_u & set_nbrs_v
        if len(common) >= 3:
            return False

        # For each w in nbrs_v (w != u): check if u,w have >= 2 common neighbors
        for w in nbrs_v:
            if w == u:
                continue
            # Common neighbors of u and w (before adding (u,v))
            # After adding (u,v), v is added to common neighbors of u and w.
            # So we need common_neighbors(u,w) < 2 currently.
            cn = 0
            for x in nbrs_u:
                if x != w and x in set_nbrs_v or (x != w and has_edge(edge_mask, w, x)):
                    pass
            # Actually let's just count directly
            cn_uw = 0
            for x in range(n):
                if x == u or x == w:
                    continue
                if has_edge(edge_mask, u, x) and has_edge(edge_mask, w, x):
                    cn_uw += 1
            if cn_uw >= 2:
                return False

        # For each w in nbrs_u (w != v): check if v,w have >= 2 common neighbors
        for w in nbrs_u:
            if w == v:
                continue
            cn_vw = 0
            for x in range(n):
                if x == v or x == w:
                    continue
                if has_edge(edge_mask, v, x) and has_edge(edge_mask, w, x):
                    cn_vw += 1
            if cn_vw >= 2:
                return False

        return True

    # Even faster: precompute adjacency as bitmasks per vertex
    def adj_mask(edge_mask, vertex):
        """Return bitmask of neighbors of vertex."""
        m = 0
        for w in range(n):
            if w != vertex and has_edge(edge_mask, vertex, w):
                m |= (1 << w)
        return m

    def is_valid_add_v2(edge_mask, u, v):
        """Use vertex adjacency bitmasks for speed."""
        if u > v:
            u, v = v, u
        e_idx = pair_to_idx[(u, v)]
        if (edge_mask >> e_idx) & 1:
            return True

        adj_u = adj_mask(edge_mask, u)
        adj_v = adj_mask(edge_mask, v)

        # Common neighbors of u and v
        common = adj_u & adj_v
        if bin(common).count('1') >= 3:
            return False

        # For each w in adj_v (w != u): need common_neighbors(u,w) < 2
        # common_neighbors(u,w) = popcount(adj_u & adj_w)
        # But we don't have adj_w cached. Let's compute on the fly.
        # Since n=20, this is at most 19 iterations, each O(1) with bit ops.
        tmp = adj_v
        while tmp:
            low = tmp & (-tmp)
            w = low.bit_length() - 1
            tmp ^= low
            if w == u:
                continue
            adj_w = adj_mask(edge_mask, w)
            if bin(adj_u & adj_w).count('1') >= 2:
                return False

        # For each w in adj_u (w != v): need common_neighbors(v,w) < 2
        tmp = adj_u
        while tmp:
            low = tmp & (-tmp)
            w = low.bit_length() - 1
            tmp ^= low
            if w == v:
                continue
            adj_w = adj_mask(edge_mask, w)
            if bin(adj_v & adj_w).count('1') >= 2:
                return False

        return True

    # Greedy construction: build a starting solution
    def greedy_build():
        edge_mask = 0
        # Use minimum common neighbors heuristic
        remaining = list(range(num_pairs))
        random.shuffle(remaining)
        for e_idx in remaining:
            u, v = idx_to_pair[e_idx]
            if is_valid_add_v2(edge_mask, u, v):
                edge_mask |= (1 << e_idx)
        # Second pass
        for e_idx in range(num_pairs):
            u, v = idx_to_pair[e_idx]
            if not ((edge_mask >> e_idx) & 1):
                if is_valid_add_v2(edge_mask, u, v):
                    edge_mask |= (1 << e_idx)
        return edge_mask

    # Simulated Annealing
    def simulated_annealing(edge_mask, max_time=120.0):
        current = edge_mask
        best = edge_mask
        best_count = count_edges(current)
        current_count = best_count

        # Temperature schedule
        T = 1.0
        cooling = 0.9995  # Cool slowly for more exploration

        while time.time() < deadline - 2:
            T *= cooling
            if T < 0.001:
                T = 0.001

            # Generate a random move: flip a random edge
            e_idx = random.randint(0, num_pairs - 1)
            u, v = idx_to_pair[e_idx]
            has = (current >> e_idx) & 1

            if has:
                # Try removing
                new_mask = current & ~(1 << e_idx)
                new_count = current_count - 1
                # Removal is always valid (removing edges can't create K2,3)
                delta = new_count - current_count  # -1
                if delta >= 0 or random.random() < pow(2.718281828, delta / T):
                    current = new_mask
                    current_count = new_count
                    if current_count > best_count:
                        best = current
                        best_count = current_count
            else:
                # Try adding
                if is_valid_add_v2(current, u, v):
                    new_mask = current | (1 << e_idx)
                    new_count = current_count + 1
                    current = new_mask
                    current_count = new_count
                    if current_count > best_count:
                        best = current
                        best_count = current_count
                # If not valid, skip (don't allow invalid states)

        return best, best_count

    # Multi-start simulated annealing
    best_overall = 0
    best_mask_overall = 0

    # Initial greedy solution
    mask = greedy_build()
    best_mask_overall = mask
    best_overall = count_edges(mask)

    # Run SA from greedy
    mask, cnt = simulated_annealing(mask, max_time=60.0)
    if cnt > best_overall:
        best_overall = cnt
        best_mask_overall = mask

    # More restarts if time permits
    restarts = 0
    while time.time() < deadline - 10:
        restarts += 1
        mask = greedy_build()
        cnt = count_edges(mask)
        if cnt > best_overall:
            best_overall = cnt
            best_mask_overall = mask
        mask, cnt = simulated_annealing(mask, max_time=30.0)
        if cnt > best_overall:
            best_overall = cnt
            best_mask_overall = mask

    # Final: polish the best with more SA
    if time.time() < deadline - 3:
        best_mask_overall, best_overall = simulated_annealing(best_mask_overall, max_time=20.0)

    # Convert to edge list
    edges = []
    for e_idx in range(num_pairs):
        if (best_mask_overall >> e_idx) & 1:
            u, v = idx_to_pair[e_idx]
            edges.append([u, v])

    return edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
