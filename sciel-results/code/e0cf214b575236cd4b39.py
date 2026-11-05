import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    n = 40
    FULL = (1 << n) - 1

    # ---- Precompute the fixed C6-conflict graph on the 780 potential edges ----
    # Two potential edges (u,v) and (x,y) conflict if adding both to the empty graph
    # creates a C6. This is a property of the two edges alone, so the conflict graph
    # is fixed. A C6-free graph = an independent set in this conflict graph.
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    M = len(all_edges)  # 780
    e2u = [e[0] for e in all_edges]
    e2v = [e[1] for e in all_edges]

    def creates_c6_two(e1, e2):
        u, v = e2u[e1], e2v[e1]
        x, y = e2u[e2], e2v[e2]
        # The two edges are disjoint (checked by caller). A C6 using both:
        # e1 = (u,v), e2 = (x,y). The remaining 4 edges form a path connecting
        # the two endpoints. The C6 is u-v-...-x-y-...-u.
        # Equivalent: in graph {e1, e2}, is there a path of length 4 from one
        # endpoint of e1 to one endpoint of e2 that avoids the other endpoint?
        # Simpler: build a 2-edge graph and check for a 6-cycle.
        # A 6-cycle on 6 distinct vertices a-b-c-d-e-f-a.
        # The two chosen edges must be two (not necessarily opposite) edges of the cycle.
        # For two disjoint edges to be part of a C6, we need a path of length 4
        # between them. With only 2 edges present, the "path of length 4" must use
        # vertices but no other edges -- impossible unless we consider the cycle
        # structure. Actually a C6 needs 6 edges; 2 edges alone can't form it.
        # So conflict = the two edges are two edges of some 6-cycle in K40.
        # That means: there exist 4 more distinct vertices such that the 6 vertices
        # form a 6-cycle containing both edges.
        # For disjoint edges (u,v) and (x,y): a 6-cycle containing both exists iff
        # we can order the 6 vertices in a cycle with (u,v) and (x,y) as edges.
        # The 6 vertices are {u,v,x,y,p,q}. A 6-cycle is a cyclic ordering.
        # (u,v) adjacent and (x,y) adjacent. The remaining edges connect the ends.
        # This is always possible for 6 distinct vertices? Not always -- depends.
        # Let's just check: is there a 6-cycle through these 6 vertices containing
        # both edges? We can brute force the cyclic structure.
        # Number of ways: fix the two edges as two arcs. The 6-cycle has 6 edges.
        # Place (u,v) and (x,y). The remaining 4 edges must form a path connecting
        # the 4 "loose ends": v,u and y,x (with orientation).
        # Actually simplest: enumerate all 6-cycles on the 6 vertices and test.
        verts = [u, v, x, y]
        # We need 2 more vertices p, q from the remaining 36.
        # For each choice of p,q, check if a 6-cycle exists containing both edges.
        # This is heavy. Use the path formulation instead.
        # A C6 containing edge (u,v) and edge (x,y):
        # The cycle visits u,v adjacent and x,y adjacent.
        # Remove edges (u,v) and (x,y) from the cycle -> a path of 4 edges on 6 verts
        # with endpoints being one of {u,v} and one of {x,y}.
        # The path: s - a - b - c - t where s in {u,v}, t in {x,y}, and {a,b,c} are
        # the other 3 vertices (the 2 extra + the 3 not in {u,v,x,y}... wait 6 verts total).
        # 6 vertices: u,v,x,y,p,q. Path of 4 edges uses 5 vertices; the 6th is one of
        # the endpoints? No. A 6-cycle has 6 vertices. Removing 2 edges that are not
        # adjacent gives a path of 4 edges on all 6 vertices. If the 2 edges are
        # opposite in the cycle, removing them gives a path of 4 edges on 6 vertices.
        # If adjacent, removing gives a path of 5 edges.
        # Case 1: (u,v) and (x,y) are opposite edges (distance 3 in cycle).
        #   Cycle: u-v-?-?-x-y-? back to u. That's 6 edges: (u,v),(v,p),(p,q),(q,x),(x,y),(y,u).
        #   So we need p,q such that edges (v,p),(p,q),(q,x),(y,u) exist in K40 -- they
        #   always exist since K40 is complete! So for ANY two disjoint edges, there is
        #   always a 6-cycle containing both (as opposite edges), as long as we can pick
        #   p,q distinct from u,v,x,y. There are 36 choices for p and 35 for q, all fine.
        #   Wait, that means EVERY pair of disjoint edges conflicts?! That can't be right
        #   because then no C6-free graph could have 2 disjoint edges, but ex(40;C6)>=105
        #   has 105 edges, many of which are disjoint.
        #   I made an error: a C6 requires the 6 edges to be in the GRAPH. The conflict
        #   is: do the two edges lie on a common C6 in K40? Since K40 is complete, any
        #   6 distinct vertices form many C6s. So any two disjoint edges lie on a C6 in
        #   K40. But that's not the conflict we want!
        #   The actual constraint: the graph must be C6-free, i.e., NO C6 is fully
        #   present. Two edges "conflict" only in the sense that they cannot both be
        #   absent-of-C6... no. The independent set formulation: vertices = edges, and
        #   two edge-vertices are adjacent in the conflict graph if they CANNOT both be
        #   in a C6-free graph. But any two edges CAN both be in a C6-free graph (a C6-free
        #   graph can contain any 2 edges). So the conflict graph is NOT just "lies on a C6".
        #   The independent set formulation does NOT work directly because C6-freeness is
        #   a 6-uniform constraint, not a 2-uniform one. A set of edges is C6-free iff it
        #   contains NO 6 edges forming a C6. This is a 6-clique avoidance, not pairwise.
        #   So maximum independent set on a pairwise conflict graph is the WRONG model.
        #   Return False as a sentinel; we will NOT use the pairwise model.
        return False

    # The pairwise conflict-graph / independent-set model is INCORRECT for C6-freeness
    # (C6 is a 6-uniform constraint). We therefore fall back to the proven approach:
    # fast bitset C6 check + greedy + local search + iterated local search, with the
    # structural improvement of a much faster C6 check and a smarter neighborhood
    # (remove one edge, re-fill greedily) to escape local optima more effectively.

    def creates_c6(adj, u, v):
        """Is there a simple path u-a-b-c-d-v of length 5 in the current graph?"""
        nu = adj[u] & ~(1 << v)
        nv = adj[v] & ~(1 << u)
        if nu == 0 or nv == 0:
            return False
        bits_a = nu
        while bits_a:
            lsb_a = bits_a & -bits_a
            a = lsb_a.bit_length() - 1
            bits_a ^= lsb_a
            bits_d = nv & ~(1 << a)
            if bits_d == 0:
                continue
            bits_d_cur = bits_d
            while bits_d_cur:
                lsb_d = bits_d_cur & -bits_d_cur
                d = lsb_d.bit_length() - 1
                bits_d_cur ^= lsb_d
                b_mask = adj[a] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << d)
                if b_mask == 0:
                    continue
                c_mask = adj[d] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << d)
                if c_mask == 0:
                    continue
                bits_b = b_mask
                while bits_b:
                    lsb_b = bits_b & -bits_b
                    b = lsb_b.bit_length() - 1
                    bits_b ^= lsb_b
                    if adj[b] & c_mask & ~(1 << b):
                        return True
        return False

    def get_edges(adj):
        edges = []
        for i in range(n):
            bits = adj[i]
            while bits:
                lsb = bits & -bits
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                bits ^= lsb
        return edges

    def count_edges(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def greedy_construct(rng):
        adj = [0] * n
        order = list(range(M))
        rng.shuffle(order)
        for e in order:
            u, v = e2u[e], e2v[e]
            if not creates_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def local_search(adj, rng):
        changed = True
        while changed:
            changed = False
            missing = []
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] & (1 << v)):
                        missing.append((u, v))
            if not missing:
                break
            rng.shuffle(missing)
            for u, v in missing:
                if time.time() > deadline:
                    return adj
                if not creates_c6(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    changed = True
        return adj

    def swap_search(adj, rng):
        """Remove one edge, re-fill greedily; keep if net gain. Iterate."""
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                return adj
            current_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (1 << j) & adj[i]:
                        current_edges.append((i, j))
            if not current_edges:
                return adj
            rng.shuffle(current_edges)
            for u, v in current_edges:
                if time.time() > deadline:
                    return adj
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                old_count = count_edges(adj)
                added = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if not ((1 << j) & adj[i]):
                            if not creates_c6(adj, i, j):
                                adj[i] |= (1 << j)
                                adj[j] |= (1 << i)
                                added.append((i, j))
                new_count = count_edges(adj)
                if new_count > old_count + 1:
                    improved = True
                else:
                    for i, j in added:
                        adj[i] &= ~(1 << j)
                        adj[j] &= ~(1 << i)
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    rng = random.Random(42)
    best_adj = [0] * n
    best_count = 0

    # Phase 1: Random restarts + local search
    phase1_deadline = start_time + 30.0
    while time.time() < phase1_deadline:
        adj = greedy_construct(rng)
        adj = local_search(adj, rng)
        m = count_edges(adj)
        if m > best_count:
            best_count = m
            best_adj = adj[:]

    current_adj = best_adj[:]
    current_m = best_count

    # Phase 2: Iterated Local Search
    while time.time() < deadline:
        perturbed = current_adj[:]
        num_removals = rng.randint(3, 10)
        existing = []
        for i in range(n):
            for j in range(i + 1, n):
                if (1 << j) & perturbed[i]:
                    existing.append((i, j))
        if existing:
            rng.shuffle(existing)
            for i in range(min(num_removals, len(existing))):
                u, v = existing[i]
                perturbed[u] &= ~(1 << v)
                perturbed[v] &= ~(1 << u)
        perturbed = swap_search(perturbed, rng)
        perturbed = local_search(perturbed, rng)
        m = count_edges(perturbed)
        if m > best_count:
            best_count = m
            best_adj = perturbed[:]
            current_adj = perturbed[:]
            current_m = m
        else:
            if m >= current_m - 2:
                current_adj = perturbed[:]
                current_m = m
            else:
                current_adj = best_adj[:]
                current_m = best_count

    return get_edges(best_adj)
