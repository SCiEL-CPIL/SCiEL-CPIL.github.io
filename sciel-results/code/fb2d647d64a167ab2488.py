import random
import time

N = 18
DEADLINE_MARGIN = 15.0

# Bit helpers
def _bit(i, j):
    return 1 << (i * N + j)

def _sym_set(a, i, j):
    a |= _bit(i, j)
    a |= _bit(j, i)
    return a

def _sym_unset(a, i, j):
    a &= ~_bit(i, j)
    a &= ~_bit(j, i)
    return a

def _has(a, i, j):
    return (a >> (i * N + j)) & 1

def _edges(a):
    out = []
    for i in range(N):
        for j in range(i + 1, N):
            if _has(a, i, j):
                out.append([i, j])
    return out

def _is_k4_free(a):
    # K4-free iff no edge (u,v) has a common neighbour w that is adjacent to both.
    for u in range(N):
        for v in range(u + 1, N):
            if not _has(a, u, v):
                continue
            # common neighbours of u and v
            cu = a >> (u * N)
            cv = a >> (v * N)
            common = cu & cv
            # check if any common neighbour w has edge to both u and v (already true) -> K4 {u,v,w,?}
            # Actually K4 needs a 4th vertex x adjacent to all of u,v,w.
            # Simpler: for each w in common, check if there exists x adjacent to u,v,w.
            # We'll just test all 4-subsets via the 3-clique-with-common-neighbour trick:
            # A K4 exists iff there is a triangle (u,v,w) with a common neighbour x.
            # Equivalent: exists edge (u,v) and two common neighbours w,x that are adjacent.
            # So: for edge (u,v), look at common-neighbour set S; if S contains an edge -> K4.
            # But w,x in S are both adjacent to u and v; if also w-x edge then K4 {u,v,w,x}.
            # We need an edge within S.
            # Check: for each w in S, is there x in S, x>w, with edge w-x?
            w = common
            while w:
                lsb = w & (-w)
                wi = lsb.bit_length() - 1
                w &= w - 1
                rest = w & (a >> (wi * N))
                if rest:
                    return False
    return True

def _max_triangle_free_size(a):
    # Exact branch and bound: largest S subset of [N] inducing no triangle.
    # A set is triangle-free iff for every vertex v in S, S \ ({v} ∪ N(v)) has no edge among itself
    # that is also... simpler: S is triangle-free iff no 3-clique fully inside S.
    # We search for the maximum independent set in the 3-uniform hypergraph of triangles,
    # i.e., maximum set hitting no triangle. Branch on vertices.
    # Represent candidate set as bitmask of available vertices (not yet excluded).
    # We build S; a triangle (i,j,k) forbids all three being in S.
    # Upper bound: |S| + popcount(available).
    best = [0]
    # Precompute for each vertex the list of triangles containing it.
    # triangles list
    tris = []
    tri_of = [[] for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            if not _has(a, i, j):
                continue
            for k in range(j + 1, N):
                if _has(a, i, k) and _has(a, j, k):
                    tris.append((i, j, k))
                    tri_of[i].append((i, j, k))
                    tri_of[j].append((i, j, k))
                    tri_of[k].append((i, j, k))
    # If no triangles, whole set is triangle-free.
    if not tris:
        return N
    # For each vertex, the set of other vertices that form a triangle with it (to prune).
    # state: chosen (bitmask), avail (bitmask of vertices not yet decided and not forbidden)
    # A vertex v can be added only if it doesn't complete a triangle with two already-chosen.
    # Precompute, for each pair (i,j) that is an edge, the set of k completing a triangle.
    # We'll maintain chosen; when considering adding v, check for any chosen pair (i,j) with edge i-j
    # and triangle i,j,v -> i.e., v adjacent to both i and j and i-j edge.
    # Maintain for chosen set: for each chosen vertex i, the set of chosen neighbours of i.
    # Simpler: maintain a bitmask 'conflict' of vertices that cannot be added (would form triangle
    # with two chosen vertices). When we add v, for each pair of chosen vertices (i,j) that are both
    # adjacent to v and to each other, v would form a triangle -> but we check before adding.
    # We'll recompute the forbidden set incrementally.
    # Let's use: forbidden = set of vertices v such that there exist i<j in chosen with v in N(i)∩N(j) and i-j edge.
    # When adding v to chosen, new forbidden vertices = { w : exists i in chosen (before v) with i-j? } 
    # Actually new forbidden from adding v: for each i in chosen_before, if i-v edge, then for each j in
    # chosen_before with i-j edge and v-j edge, w in N(i)∩N(v)∩... hmm.
    # Let me just do a clean recursive with memo-free pruning using a 'bad' bitmask.
    # bad = vertices that would create a triangle if added given current chosen.
    # When we add v: 
    #   new_bad = bad
    #   for each i in chosen_before: if edge(i,v): for each j in chosen_before: if edge(i,j) and edge(v,j): 
    #       then any w in N(i)&N(v) that is also ... no. The triangle would be (i,j,v) but i,j already in chosen ->
    #       that means v is forbidden already. So we only add v if not in bad.
    #   Adding v creates new bad vertices: w such that there exist i in chosen_before with edge(i,v) and edge(i,w) and ... 
    #   need two chosen vertices both adjacent to w and to each other. The pair could be (v, i) for i in chosen_before
    #   with edge(v,i): then w must be adjacent to both v and i and ... that's a triangle (v,i,w). So w in N(v)&N(i).
    #   Also pairs (i,j) already in chosen_before that are both adjacent to w -> those w already in bad.
    #   So new_bad = bad ∪ { w : exists i in chosen_before, edge(v,i) and edge(w,i) and edge(w,v) }
    #   = bad ∪ ( ∪_{i in chosen_before, edge(v,i)} (N(v) & N(i)) )
    #   but only w not already chosen.
    def popcount(x):
        return bin(x).count('1')

    full_mask = (1 << N) - 1
    chosen = 0
    # bad[v] set if v adjacent to two chosen vertices that are adjacent to each other.
    bad = 0
    # adjacency rows
    row = [0] * N
    for i in range(N):
        row[i] = a >> (i * N)

    def rec(chosen, bad, avail):
        # avail: vertices not yet decided (not chosen and not permanently excluded)
        # upper bound
        if popcount(chosen) + popcount(avail) <= best[0]:
            return
        if avail == 0:
            if popcount(chosen) > best[0]:
                best[0] = popcount(chosen)
            return
        # pick a vertex to branch on: choose the one in avail that is in bad? or just lowest.
        # Branch: include or exclude a vertex v in avail.
        # To prune, if v in bad, we cannot include it -> only exclude.
        v = (avail & -avail).bit_length() - 1
        rest = avail & ~_bit(v, v)
        if _has(bad, v, v):
            # cannot include v
            rec(chosen, bad, rest)
        else:
            # include v
            new_bad = bad
            # new forbidden from v with each chosen i adjacent to v
            ci = chosen
            while ci:
                lsb = ci & (-ci)
                ii = lsb.bit_length() - 1
                ci &= ci - 1
                if _has(row[v], ii, ii):  # edge(v, ii)
                    # w in N(v) & N(ii)
                    new_bad |= (row[v] & row[ii])
            new_bad |= chosen  # chosen vertices can't be added again
            rec(chosen | _bit(v, v), new_bad, rest & ~new_bad)
            # exclude v
            rec(chosen, bad, rest)

    rec(0, 0, full_mask)
    return best[0]

def _eval(a):
    # return score (max triangle-free size) if K4-free else None
    if not _is_k4_free(a):
        return None
    return _max_triangle_free_size(a)

def construct():
    t0 = time.time()
    deadline = t0 + 150.0 - DEADLINE_MARGIN
    rng = random.Random(12345)

    # Start from complete tripartite K6,6,6
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    a = 0
    for i in range(3):
        for j in range(i + 1, 3):
            for u in parts[i]:
                for v in parts[j]:
                    a = _sym_set(a, u, v)
    base_score = _eval(a)
    assert base_score is not None, "seed not K4-free"
    best_a = a
    best_score = base_score

    def time_left():
        return deadline - time.time()

    # Simulated annealing over edge flips, with greedy fast-prune then exact verify.
    T = 1.0
    T_min = 0.01
    cooling = 0.999

    # Precompute row for quick triangle check on flip.
    def flip_score(a, u, v):
        # Try adding or removing edge (u,v); return new score if K4-free else None, and new a.
        # We'll just mutate and evaluate; but to prune fast, check K4-freedom incrementally is complex.
        # Instead: build candidate a2, check K4-free (cheap-ish), then exact.
        return None  # placeholder

    iters = 0
    while time_left() > 1.0:
        # propose a move: flip a random edge
        u = rng.randrange(N)
        v = rng.randrange(N)
        if u == v:
            continue
        if u > v:
            u, v = v, u
        a2 = _sym_unset(a, u, v) if _has(a, u, v) else _sym_set(a, u, v)
        # fast K4-free check
        if not _is_k4_free(a2):
            iters += 1
            continue
        # greedy surrogate: quick upper bound on max triangle-free size to prune
        # Compute a fast lower-bound check: if even a trivial triangle-free set is large, skip.
        # Use exact only when promising.
        s2 = _max_triangle_free_size(a2)
        iters += 1
        # accept if better or with probability
        if s2 < best_score:
            a = a2
            best_a = a2
            best_score = s2
            T = max(T_min, T * 1.05)
        else:
            delta = s2 - best_score
            if delta < 0 or rng.random() < pow(2.718281828, -delta / max(T, 1e-6)):
                a = a2
        T = max(T_min, T * cooling)
        # occasionally re-seed with a random K4-free graph to escape local minima
        if iters % 2000 == 0 and time_left() > 5.0:
            # random graph with low density, check K4-free
            ar = 0
            for i in range(N):
                for j in range(i + 1, N):
                    if rng.random() < 0.35:
                        ar = _sym_set(ar, i, j)
            if _is_k4_free(ar):
                sr = _max_triangle_free_size(ar)
                if sr < best_score:
                    a = ar
                    best_a = ar
                    best_score = sr
                else:
                    a = ar  # keep exploring from it

    return _edges(best_a)
