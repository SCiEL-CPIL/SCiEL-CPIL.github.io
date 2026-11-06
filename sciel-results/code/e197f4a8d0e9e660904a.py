"""K4-free graph on 18 vertices minimizing the largest triangle-free induced subgraph."""
import time
import random
from itertools import combinations

N = 18
_DEADLINE = None

def _now():
    return time.time()

def _time_left():
    return _DEADLINE - _now()

def _shrikhande_edges():
    # Shrikhande graph: SRG(16,6,2,2), K4-free, independence number 4.
    # Defined as the complement of the Clebsch graph.
    # Clebsch graph: 16 vertices, each vertex adjacent to 5 others.
    # Vertices are 4-bit strings; two vertices adjacent in Clebsch iff they differ in exactly 1 or 4 bits.
    clebsch_adj = set()
    for a in range(16):
        for b in range(a + 1, 16):
            d = bin(a ^ b).count("1")
            if d == 1 or d == 4:
                clebsch_adj.add((a, b))
    # Shrikhande = complement of Clebsch (on 16 vertices)
    edges = []
    for a in range(16):
        for b in range(a + 1, 16):
            if (a, b) not in clebsch_adj:
                edges.append((a, b))
    return edges

def _is_k4_free(adj):
    # adj is a set of (u,v) with u<v. Check no K4.
    for a in range(N):
        for b in range(a + 1, N):
            if (a, b) not in adj:
                continue
            for c in range(b + 1, N):
                if (a, c) not in adj or (b, c) not in adj:
                    continue
                for d in range(c + 1, N):
                    if (a, d) in adj and (b, d) in adj and (c, d) in adj:
                        return False
    return True

def _max_tri_free(adj):
    """Return the size of the largest triangle-free induced subgraph. Exact via branch and bound."""
    # A subset S is triangle-free iff it contains no triangle.
    # We want max |S| with no triangle in G[S].
    # Equivalently, S is an independent set in the triangle-hypergraph (3-uniform).
    # Branch and bound: pick a vertex, include or exclude.
    # Use the fact that if we've placed a triangle, that branch is invalid.
    # Better: build S incrementally; S is valid iff every 3-subset of S is not a triangle in G.
    # Use a greedy upper bound: |S| + (number of remaining vertices that can still be added)
    # but that's loose. Instead, use: a valid S is triangle-free, so its complement... no.
    # Use the bound: S triangle-free => S is an independent set in the "triangle graph" T where
    # two vertices are adjacent in T if they are part of some triangle? No, that's not right either.
    # Simplest exact: backtracking with the constraint that no triangle is fully included.
    # Upper bound: current_size + (N - next_vertex) is too loose.
    # Use: a triangle-free set is an independent set in the 3-uniform hypergraph of triangles.
    # Upper bound via: pick a triangle, at most 2 of its 3 vertices can be in S.
    # We'll use a simple but effective B&B: order vertices, branch include/exclude,
    # prune if current_size + remaining <= best.
    # To make it fast, use a greedy coloring-like bound on the "conflict" structure.
    # Actually, a triangle-free set S: for any triangle {u,v,w}, |S ∩ {u,v,w}| <= 2.
    # This is a packing constraint. Upper bound: |S| <= N - (number of disjoint triangles) ... not tight.
    # Let's just do B&B with a decent ordering and the simple bound.
    best = [0]
    # Precompute triangles
    tri = []
    for a in range(N):
        for b in range(a + 1, N):
            if (a, b) not in adj:
                continue
            for c in range(b + 1, N):
                if (a, c) in adj and (b, c) in adj:
                    tri.append((a, b, c))
    # For each vertex, the set of triangles it belongs to
    # B&B: we maintain S (set of included vertices) and the next vertex to consider.
    # A partial S is valid iff no triangle is fully contained in S.
    # We add vertices in order; when adding v, we must ensure no triangle {v,x,y} with x,y in S exists.
    # So we need: for each v, the set of pairs (x,y) in S that with v form a triangle.
    # Precompute for each v: list of (x,y) such that {v,x,y} is a triangle.
    tri_pairs = [[] for _ in range(N)]
    for (a, b, c) in tri:
        tri_pairs[a].append((b, c))
        tri_pairs[b].append((a, c))
        tri_pairs[c].append((a, b))
    # For fast check: when S is a set, adding v is OK iff no (x,y) in tri_pairs[v] has both x,y in S.
    # Represent S as a bitmask for speed.
    def can_add(v, S_mask):
        for (x, y) in tri_pairs[v]:
            if (S_mask >> x) & 1 and (S_mask >> y) & 1:
                return False
        return True
    # Upper bound: a triangle-free set is an independent set in the graph where we connect
    # vertices that are in a common triangle? No. Use: the complement of a triangle-free set is
    # a vertex cover of the triangle hypergraph. Hard.
    # Use simple bound: current + (N - next) but tighten with: among remaining vertices,
    # they form a triangle-free set plus current. Upper bound = current_size + (N - next).
    # This is loose but with good ordering it's okay for N=18.
    # Better bound: the remaining vertices R must themselves be triangle-free (when combined with S).
    # So |S ∪ R_selected| <= current + alpha_tri(R), where alpha_tri(R) is max triangle-free in R
    # (ignoring S). We can precompute a greedy bound.
    # For speed, use: current_size + (N - next) and rely on ordering + pruning.
    # Let's use a smarter bound: the set of remaining vertices that can be added to S without
    # forming a triangle with S is some set A; then the answer <= current + max_tri_free_within(A)
    # but that's recursive. Use a greedy: the remaining vertices form a graph H (edges = not in a triangle with S).
    # A triangle-free subset of A is an independent set in the "triangle-in-A" hypergraph.
    # Upper bound: current + (number of vertices in A) is loose.
    # Let's just do B&B with ordering by degree in the triangle-hypergraph (high degree first)
    # and the bound current + (N - next). For N=18 this should be fast enough.
    # Actually, use the bound: current_size + (N - next) is too loose. Use:
    # The remaining vertices must form a set with no triangle among themselves AND no triangle with S.
    # The no-triangle-with-S constraint removes some vertices. Let A = {v >= next : can_add(v, S_mask)}.
    # Then we need a triangle-free subset of A (in the original graph, restricted to A, with no triangle fully in A).
    # Upper bound: current + |A| is still loose but better. And within A, triangles are those fully in A.
    # We can use: current + (|A| - max_packing) ... too complex.
    # Let's use a recursive B&B with memoization on (S_mask, next) is too big.
    # For N=18, a well-ordered B&B should handle it. Use ordering: sort vertices by number of triangles (desc).
    order = sorted(range(N), key=lambda v: -len(tri_pairs[v]))
    # Map: position in order -> vertex
    pos_of = [0] * N
    for i, v in enumerate(order):
        pos_of[v] = i
    # B&B over positions 0..N-1 in `order`.
    # State: pos, S_mask, size.
    # Bound: size + (N - pos) <= best -> prune.
    # Tighter: size + (number of vertices in order[pos:] that can be added) <= best.
    def bb(pos, S_mask, size):
        if size + (N - pos) <= best[0]:
            return
        if pos == N:
            if size > best[0]:
                best[0] = size
            return
        v = order[pos]
        # Try including v
        if can_add(v, S_mask):
            bb(pos + 1, S_mask | (1 << v), size + 1)
        # Try excluding v
        bb(pos + 1, S_mask, size)
    bb(0, 0, 0)
    return best[0]

def _local_search(edges, adj, rng, time_limit):
    """Hill-climb: flip edges to reduce the max triangle-free induced set, keeping K4-free."""
    best_edges = list(edges)
    best_score = _max_tri_free(adj)
    best_adj = set(adj)
    t0 = _now()
    # Precompute all pairs
    all_pairs = [(a, b) for a in range(N) for b in range(a + 1, N)]
    # For each edge flip, check K4-freeness incrementally is complex; just recompute.
    # To speed up, we only consider flips that don't create a K4.
    # Check K4-freeness of a modified adj quickly.
    def k4_free(adj_set):
        return _is_k4_free(adj_set)
    while _time_left() > 2.0:
        improved = False
        # Random order of pairs
        pairs = list(all_pairs)
        rng.shuffle(pairs)
        for (u, v) in pairs:
            if _time_left() < 2.0:
                break
            new_adj = set(best_adj)
            if (u, v) in new_adj:
                new_adj.remove((u, v))
            else:
                new_adj.add((u, v))
            if not k4_free(new_adj):
                continue
            sc = _max_tri_free(new_adj)
            if sc < best_score:
                best_score = sc
                best_adj = new_adj
                best_edges = [e for e in best_edges if e != (u, v) and e != (v, u)]
                if (u, v) not in new_adj:
                    pass
                else:
                    best_edges.append((u, v))
                # Rebuild best_edges properly
                best_edges = list(new_adj)
                improved = True
                if best_score <= 6:
                    return best_edges, best_score
        if not improved:
            # Perturbation: random restart with small random graph
            if _time_left() < 5.0:
                break
            # Start from a random K4-free-ish graph
            new_adj = set()
            for (u, v) in all_pairs:
                if rng.random() < 0.4:
                    new_adj.add((u, v))
            # Remove K4s
            changed = True
            while changed:
                changed = False
                for (a, b, c, d) in combinations(range(N), 4):
                    if all(((x, y) in new_adj) for x, y in combinations([a, b, c, d], 2)):
                        # Remove a random edge of this K4
                        pairs4 = list(combinations([a, b, c, d], 2))
                        rm = rng.choice(pairs4)
                        new_adj.remove(rm)
                        changed = True
            sc = _max_tri_free(new_adj)
            if sc <= best_score:
                best_adj = new_adj
                best_edges = list(new_adj)
                best_score = sc
                if best_score <= 6:
                    return best_edges, best_score
    return best_edges, best_score

def construct():
    global _DEADLINE
    _DEADLINE = _now() + 145.0  # safety margin before 150s
    rng = random.Random(42)
    # Start from Shrikhande on 16 vertices + 2 extra vertices
    base_edges = _shrikhande_edges()
    adj = set(base_edges)
    # Add vertices 16, 17 with some random edges
    for v in (16, 17):
        for u in range(16):
            if rng.random() < 0.4:
                adj.add((u, v) if u < v else (v, u))
    if rng.random() < 0.4:
        adj.add((16, 17))
    # Ensure K4-free
    if not _is_k4_free(adj):
        # Remove edges to break K4s
        changed = True
        while changed:
            changed = False
            for (a, b, c, d) in combinations(range(N), 4):
                if all(((x, y) in adj) for x, y in combinations([a, b, c, d], 2)):
                    pairs4 = list(combinations([a, b, c, d], 2))
                    rm = rng.choice(pairs4)
                    adj.remove(rm)
                    changed = True
    edges = list(adj)
    # Local search
    edges, score = _local_search(edges, adj, rng, _time_left())
    # Final K4 check
    if not _is_k4_free(set(edges)):
        # Fallback to seed
        parts = [range(0, 6), range(6, 12), range(12, 18)]
        return [[a, b] for i in range(3) for j in range(i + 1, 3) for a in parts[i] for b in parts[j]]
    return [[u, v] for (u, v) in edges]
