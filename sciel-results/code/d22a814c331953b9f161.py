import random
import time

_Q3_EDGES = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
_Q3_ADJ = [0] * 8
for _a, _b in _Q3_EDGES:
    _Q3_ADJ[_a] |= (1 << _b)
    _Q3_ADJ[_b] |= (1 << _a)

# For each Q3 edge (qa,qb) and orientation, precompute the 6 unassigned Q3 vertices
# sorted by degree of constraint (number of neighbors in {qa,qb}), descending.
# Also precompute for each unassigned vertex x, the bitmask of its Q3-neighbors
# among the 8 Q3 vertices (for feasibility checks).
_Q3_EDGE_ORIENTS = []
for qa, qb in _Q3_EDGES:
    for (gu_q, gv_q) in [(qa, qb), (qb, qa)]:
        unassigned = [i for i in range(8) if i != gu_q and i != gv_q]
        def _cc(x, a, b):
            c = 0
            if _Q3_ADJ[x] & (1 << a): c += 1
            if _Q3_ADJ[x] & (1 << b): c += 1
            return c
        unassigned.sort(key=lambda x: _cc(x, gu_q, gv_q), reverse=True)
        _Q3_EDGE_ORIENTS.append((gu_q, gv_q, tuple(unassigned)))

def construct():
    start_time = time.time()
    deadline = start_time + 110
    n = 16
    full_mask = (1 << n) - 1

    # Start with K_{4,4,4,4}: partition vertices into 4 groups of 4, connect all across groups.
    # This is Q3-free: Q3 has no independent set of size 4 that spans all 8 vertices
    # in a way that... actually Q3 IS bipartite with parts of size 4. But K_{4,4,4,4}
    # has no edges within parts, so any subgraph is a subgraph of the complete 4-partite.
    # Q3 is bipartite with parts A={0,3,5,6} and B={1,2,4,7}. In K_{4,4,4,4}, if we map
    # Q3 vertices to graph vertices, we need edges only between different parts.
    # Q3 edges always go between its own bipartition A and B. So we need a homomorphism
    # from Q3 to K_{4,4,4,4} that is injective. This means Q3's A-part maps to distinct
    # graph-parts and B-part maps to distinct graph-parts, with no vertex in A sharing
    # a graph-part with a vertex in B that it's adjacent to... Actually it's simpler:
    # K_{4,4,4,4} is the complement of 4 disjoint K4's. Q3 has independence number 4.
    # The 4 vertices of one bipartition of Q3 are independent. If we map them to one
    # part of K_{4,4,4,4}, that works for those 4. The other 4 must map to other parts.
    # But each vertex in B is adjacent to 3 vertices in A. In K_{4,4,4,4}, a vertex in
    # part P2 is adjacent to all vertices in parts P1,P3,P4. So it IS adjacent to all
    # 4 vertices in P1. But in Q3, each B-vertex is adjacent to only 3 of the 4 A-vertices.
    # So the mapping would have extra edges -> not a subgraph isomorphism.
    # More carefully: a subgraph isomorphic to Q3 needs EXACTLY the Q3 edges present.
    # If we map A to part P1 and B to parts P2,P3,P4, then a B-vertex in P2 is adjacent
    # to ALL 4 A-vertices in P1, but in Q3 it's adjacent to only 3. So there's an extra
    # edge -> the induced subgraph on those 8 vertices has more edges than Q3.
    # But the task says "subgraph isomorphic to Q3" which means we need a set of 8 vertices
    # and a bijection such that every Q3 edge maps to a graph edge. It does NOT require
    # that non-edges map to non-edges. So K_{4,4,4,4} might actually contain Q3!
    # Let me check: map Q3 vertex 0->P1, 3->P1, 5->P1, 6->P1 (A-part to P1)
    # 1->P2, 2->P3, 4->P4, 7->P2 (B-part to P2,P3,P4)
    # Q3 edges: (0,1): P1-P2 edge? Yes. (0,2): P1-P3? Yes. (0,4): P1-P4? Yes.
    # (1,3): P2-P1? Yes. (1,5): P2-P1? Yes. (2,3): P3-P1? Yes. (2,6): P3-P1? Yes.
    # (3,7): P1-P2? Yes. (4,5): P4-P1? Yes. (4,6): P4-P1? Yes. (5,7): P1-P2? Yes. (6,7): P1-P2? Yes.
    # All 12 Q3 edges are present! So K_{4,4,4,4} DOES contain Q3. Bad start.

    # Instead, use K_{3,3,3,7} or stick with K_{3,13} but try harder.
    # Actually, let's use a different approach: start with K_{3,13} (39 edges) which is known Q3-free,
    # then greedily add edges with fast Q3 checking.

    # Better: use the complete tripartite K_{4,4,8}? No, that might have Q3 too.
    # Let's just use K_{3,13} as seed but with much faster Q3 detection.

    def make_adj_from_parts(parts):
        """parts: list of lists of vertex indices. Complete multipartite."""
        adj = [0] * n
        for i in range(len(parts)):
            for j in range(i + 1, len(parts)):
                for u in parts[i]:
                    for v in parts[j]:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        return adj

    # K_{3,13}: part0 = {0,1,2}, part1 = {3,...,15}
    adj = make_adj_from_parts([[0,1,2], list(range(3,16))])

    def has_q3_after_add(u, v):
        """Check if adding edge (u,v) to current adj creates a Q3 subgraph.
        Only need to check Q3's that USE the new edge (u,v)."""
        # Temporarily add
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

        # For each Q3 edge orientation where (u,v) maps to that Q3 edge:
        for (qa, qb, unassigned) in _Q3_EDGE_ORIENTS:
            # gu maps to qa, gv maps to qb
            # Try both: u->qa,v->qb and u->qb,v->qa
            for (gu, gv) in [(u, v), (v, u)]:
                # Check: gu must map to qa, gv to qb
                # Feasibility: for each unassigned Q3 vertex x, the graph vertex g
                # must be adjacent (in adj) to all assigned Q3-neighbors of x,
                # and non-adjacent to all assigned Q3-non-neighbors of x (for subgraph iso,
                # we only need edges to exist, not non-edges).
                # Actually for subgraph isomorphism (not induced), we only need:
                # for every Q3 edge (a,b), the mapped pair (g_a, g_b) is an edge in G.
                # We do NOT need non-edges to map to non-edges.
                # So feasibility: for unassigned x, g must be adjacent to all
                # already-assigned Q3-neighbors of x.
                
                # Precompute: for each unassigned position, the set of assigned neighbors
                # and the candidate graph vertices.
                assignment = [None] * 8
                assignment[qa] = gu
                assignment[qb] = gv
                used_mask = (1 << gu) | (1 << gv)

                # For each unassigned Q3 vertex, compute which assigned Q3 vertices it's adjacent to
                # and use that to filter candidates.
                # We'll do backtracking with strong pruning.

                def bt(idx, used_mask):
                    if time.time() > deadline - 5:
                        return False
                    if idx == 6:
                        # All assigned. Verify all Q3 edges are present.
                        for a, b in _Q3_EDGES:
                            ga = assignment[a]
                            gb = assignment[b]
                            if not (adj[ga] & (1 << gb)):
                                return False
                        return True

                    x = unassigned[idx]
                    # Candidate graph vertices: not used, and adjacent to all assigned Q3-neighbors of x
                    candidates = full_mask & ~used_mask
                    # For each assigned Q3 neighbor of x, intersect with adj[g]
                    for y in range(8):
                        if assignment[y] is not None and (_Q3_ADJ[x] & (1 << y)):
                            hy = assignment[y]
                            # g must be adjacent to hy
                            candidates &= adj[hy]  # adj[hy] has bits for all neighbors of hy
                    # Also, g must NOT be gu or gv (already excluded by used_mask)
                    # Iterate candidates
                    while candidates:
                        bit = candidates & (-candidates)
                        g = bit.bit_length() - 1
                        candidates ^= bit
                        assignment[x] = g
                        if bt(idx + 1, used_mask | bit):
                            return True
                        assignment[x] = None
                    return False

                if bt(0, used_mask):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    return True

        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        return False

    best_adj = None
    best_count = -1

    # Multiple restarts with different random orderings
    num_restarts = 8
    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break

        # Reset adj
        if restart == 0:
            adj = make_adj_from_parts([[0,1,2], list(range(3,16))])
        else:
            # Start from best found so far, or fresh K_{3,13}
            if best_adj is not None:
                adj = list(best_adj)
            else:
                adj = make_adj_from_parts([[0,1,2], list(range(3,16))])

        # Collect missing edges
        missing = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    missing.append((i, j))

        random.shuffle(missing)

        for u, v in missing:
            if time.time() > deadline - 10:
                break
            if not has_q3_after_add(u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)

        # Count edges
        count = 0
        for i in range(n):
            count += bin(adj[i]).count('1')
        count //= 2

        if count > best_count:
            best_count = count
            best_adj = list(adj)

    # Build edge list from best
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])

    return edges
