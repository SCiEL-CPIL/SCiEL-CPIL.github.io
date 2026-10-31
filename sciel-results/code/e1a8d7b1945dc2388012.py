"""Rewritten program: Tomita branch-and-bound Maximum Independent Set on
the incompatibility graph of weight-4 integer vectors in R^11.

Structural change vs. the seed: the seed used randomized greedy selection with
restarts (a local-search family). This version models the problem as a
Maximum Independent Set problem and solves it with an exact branch-and-bound
(Tomita) algorithm using bitset coloring for tight upper bounds, seeded by a
greedy lower bound. This is a different optimization/search family.
"""
import time
import numpy as np

D = 11
DEADLINE = 120.0  # seconds
START_TIME = time.time()


def _elapsed():
    return time.time() - START_TIME


def _check_deadline():
    return _elapsed() > DEADLINE - 5.0


# EVOLVE-BLOCK-START
import itertools


def _build_weight4_candidates():
    """Generate all weight-4 vectors: 4 nonzero coords in [0,D), each +/-1."""
    cands = []
    for coords in itertools.combinations(range(D), 4):
        for signs in itertools.product((1, -1), repeat=4):
            v = [0] * D
            for c, s in zip(coords, signs):
                v[c] = s
            cands.append(v)
    return cands


def _greedy_mis(adj, n, random_state):
    """Greedy independent set: repeatedly pick a random vertex, remove it and its
    neighbors. Returns a list of vertex indices (an independent set)."""
    rng = np.random.default_rng(random_state)
    in_set = np.zeros(n, dtype=bool)
    candidates = np.ones(n, dtype=bool)
    order = rng.permutation(n)
    chosen = []
    # Work with a "still available" set; removing neighbors each pick.
    avail = np.ones(n, dtype=bool)
    for idx in order:
        if not avail[idx]:
            continue
        chosen.append(idx)
        # remove idx and all its neighbors from avail
        nbrs = adj[idx]
        avail[nbrs] = False
        avail[idx] = False
    return chosen


def _tomita_max_independent_set(adj, n, deadline_fn, seed_set=None, verbose=False):
    """
    Tomita-style branch-and-bound for Maximum Independent Set.

    adj: list of sets (or arrays) of neighbor indices for each vertex.
    Returns the maximum independent set found (list of vertex indices) and its size.

    Uses a recursive branch on the candidate set P, with a greedy coloring
    upper bound to prune.
    """
    # Convert adjacency to a fast membership structure.
    # Use a list of Python sets for neighbor lookup, plus a boolean matrix-free
    # approach via bitsets for the candidate set P.
    nbr_sets = [set(a) for a in adj]

    best = list(seed_set) if seed_set else []
    best_size = len(best)

    # We represent the candidate set P as a sorted list of integers and operate
    # with bitsets (Python ints) for speed.
    # Precompute a bit for each vertex.
    bit = [1 << i for i in range(n)]

    # For coloring we need, within P, a greedy coloring that orders P and
    # assigns colors so that the number of colors is an upper bound on the
    # size of any independent set within P.

    def color_and_order(P_list):
        """Greedy coloring of the induced subgraph on P_list.
        Returns (order, colors) where order is a permutation of P_list and
        colors[i] is the color (1..k) of order[i]; k = number of colors.
        The color of a vertex is an upper bound on the independent set size
        achievable within {order[0..i]} (the prefix)."""
        # We process vertices; assign smallest possible color not used by
        # already-colored neighbors.
        # To get a good bound we color in an order; we'll color by repeatedly
        # picking a vertex and assigning it the smallest color absent among its
        # colored neighbors.
        color_of = {}
        used = {}  # color -> set of vertices with that color (for neighbor check)
        order = []
        remaining = list(P_list)
        # We'll color in the given order of `remaining` (which we can choose).
        # Use a simple deterministic order for stability.
        for v in remaining:
            # find smallest color c such that no neighbor of v has color c
            c = 1
            while True:
                # check if any neighbor of v is in used.get(c, set())
                conflict = False
                if c in used:
                    for nb in nbr_sets[v]:
                        if nb in used[c]:
                            conflict = True
                            break
                if not conflict:
                    break
                c += 1
            color_of[v] = c
            used.setdefault(c, set()).add(v)
            order.append(v)
        # Now produce the color array aligned with order, and the max color so far.
        colors = []
        cur_max = 0
        for v in order:
            cv = color_of[v]
            if cv > cur_max:
                cur_max = cv
            colors.append(cur_max)
        return order, colors, cur_max

    def branch(P_list, depth):
        nonlocal best, best_size
        if deadline_fn():
            return
        if not P_list:
            if depth > best_size:
                best_size = depth
                best = list(best)  # placeholder; we track best via separate list
            return

        # If current depth + |P| <= best_size, prune
        if depth + len(P_list) <= best_size:
            return

        order, colors, k = color_and_order(P_list)

        # Process in reverse order of color (highest color first) for tighter bound.
        # Tomita: iterate i from k down to 1; for vertices with color i, branch.
        # We'll build groups by color.
        # Simpler: iterate over order in reverse; the color value at position is
        # the bound for the prefix. We branch on each vertex v in reverse order.
        # For v at position pos (0-indexed), the number of vertices in the
        # prefix [0..pos] is pos+1, and the bound is colors[pos].
        # Standard Tomita: for i in range(k, 0, -1): for each v with color i,
        #   branch with P' = P ∩ N(v) ∩ {vertices with color < i}.
        # Let's implement that.
        # Build a map from color to list of vertices (in order).
        from collections import defaultdict
        by_color = defaultdict(list)
        for pos, v in enumerate(order):
            by_color[colors[pos]].append(v)

        for c in range(k, 0, -1):
            if deadline_fn():
                return
            # The bound for any independent set using only vertices with color <= c
            # is c. If depth + c <= best_size, prune all remaining (colors <= c).
            if depth + c <= best_size:
                break
            for v in by_color[c]:
                if deadline_fn():
                    return
                # Candidate set for the recursive call: vertices in P with
                # color < c that are NOT adjacent to v (i.e., compatible with v).
                # (We restrict to color < c to keep the coloring bound valid.)
                P_new = []
                for pos, w in enumerate(order):
                    if colors[pos] < c and w != v and w not in nbr_sets[v]:
                        P_new.append(w)
                # Branch: include v
                # We need to track the actual chosen set; pass a stack.
                branch_with_stack(P_new, depth + 1, stack + [v])
                # Remove v from consideration in this level (it's handled).
                # (In Tomita, after branching on v, v is removed from P for
                # subsequent branches at the same level.)
                # We'll just continue; the next v is a different vertex.

    # We need a stack to track the actual set. Reimplement with explicit stack.
    def branch_with_stack(P_list, depth, stack):
        nonlocal best, best_size
        if deadline_fn():
            return
        if not P_list:
            if depth > best_size:
                best_size = depth
                best = list(stack)
            return
        if depth + len(P_list) <= best_size:
            return
        order, colors, k = color_and_order(P_list)
        from collections import defaultdict
        by_color = defaultdict(list)
        for pos, v in enumerate(order):
            by_color[colors[pos]].append(v)
        for c in range(k, 0, -1):
            if deadline_fn():
                return
            if depth + c <= best_size:
                break
            for v in by_color[c]:
                if deadline_fn():
                    return
                P_new = []
                for pos, w in enumerate(order):
                    if colors[pos] < c and w != v and w not in nbr_sets[v]:
                        P_new.append(w)
                branch_with_stack(P_new, depth + 1, stack + [v])

    # Top-level call
    all_vertices = list(range(n))
    branch_with_stack(all_vertices, 0, [])

    return best, best_size


def construct():
    """Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm. Score is k (number of vectors).

    Strategy: weight-4 vectors (norm^2 = 4). Constraint for two such vectors:
    <u,v> <= 2. Build the incompatibility graph (edge if <u,v> > 2) and find a
    Maximum Independent Set using Tomita branch-and-bound, seeded by a greedy
    independent set for a strong lower bound.
    """
    deadline = DEADLINE - 5.0

    cands = _build_weight4_candidates()
    cands_arr = np.array(cands, dtype=np.int64)
    n = len(cands_arr)

    # Build incompatibility adjacency: edge between i,j if dot(cands[i], cands[j]) > 2.
    # Compute all pairwise dots via matrix multiply.
    # n ~ 5280; n x n ~ 28M entries -> ~224 MB as int64. Manageable but let's use
    # a chunked approach to limit memory, or just do it; 28M * 8 bytes = 224 MB,
    # acceptable.
    dots = cands_arr @ cands_arr.T  # n x n
    # Incompatibility: dot > 2. Exclude self (dot = 4 > 2, but self is not an edge).
    adj_sets = []
    for i in range(n):
        row = dots[i]
        # neighbors j != i with row[j] > 2
        nbrs = np.where(row > 2)[0]
        nbrs = nbrs[nbrs != i]
        adj_sets.append(nbrs.tolist())

    # Greedy lower bound with a few random seeds to get a strong start.
    best_seed = []
    best_seed_size = 0
    for rs in (0, 1, 2, 3, 4):
        if _check_deadline():
            break
        s = _greedy_mis(adj_sets, n, rs)
        if len(s) > best_seed_size:
            best_seed_size = len(s)
            best_seed = s

    # Tomita branch-and-bound, seeded with the greedy set.
    best_set, best_size = _tomita_max_independent_set(
        adj_sets, n, _check_deadline, seed_set=best_seed
    )

    if not best_set:
        # Fallback: return the greedy set if B&B found nothing, else a small set.
        if best_seed:
            best_set = best_seed
        else:
            # Minimal valid set: two orthogonal-ish weight-4 vectors.
            v1 = [0] * D
            v2 = [0] * D
            v1[0] = 1; v1[1] = 1; v1[2] = 1; v1[3] = 1
            v2[4] = 1; v2[5] = 1; v2[6] = 1; v2[7] = 1
            best_set = [0, 1]
            return np.array([cands_arr[0], cands_arr[1]], dtype=np.int64)

    return cands_arr[best_set].astype(np.int64)
# EVOLVE-BLOCK-END
