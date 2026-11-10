"""Linear triple systems on 16 points with small independence number.

Structural rewrite: fast exact independence-number oracle (branch & bound
with dominated-vertex pruning), a balanced greedy initial construction that
maximizes edge count, and a simulated-annealing search over the hyperedge
set (swap/add/remove moves) that keeps the system linear while minimizing
the independence number.
"""
import itertools
import random
import time


# ---------------------------------------------------------------------------
# Fast independence-number oracle for a 3-uniform hypergraph on 16 vertices.
# ---------------------------------------------------------------------------

def _conflict_masks(triples):
    """conflict[v] = bitmask of vertices that share a triple with v."""
    n = 16
    conflict = [0] * n
    for (a, b, c) in triples:
        m = (1 << a) | (1 << b) | (1 << c)
        for v in (a, b, c):
            conflict[v] |= m & ~(1 << v)
    return conflict


def _calc_alpha(triples, time_budget=2.0):
    """Exact independence number via branch & bound with dominated pruning."""
    n = 16
    conflict = _conflict_masks(triples)
    full = (1 << n) - 1
    start = time.time()

    best = [0]

    def popcount(x):
        return bin(x).count('1')

    def search(remaining, chosen, size):
        if size > best[0]:
            best[0] = size
            if best[0] >= 16:
                return
        # Trivial upper bound
        if size + popcount(remaining) <= best[0]:
            return
        if time.time() - start > time_budget:
            return
        if remaining == 0:
            return

        # Dominated-vertex pruning: for each v in remaining, if there is a
        # different u in remaining with (conflict[v] & remaining) a superset of
        # (conflict[u] & remaining), then any independent set that avoids u
        # and contains v can be swapped to contain u instead, so v need not be
        # considered as a "must-choose" candidate; we may drop v from remaining
        # without losing the optimum.
        rem = remaining
        pruned = 0
        verts = [v for v in range(n) if (rem >> v) & 1]
        for i in range(len(verts)):
            vi = verts[i]
            if (rem >> vi) & 1 == 0:
                continue
            cv = conflict[vi] & rem
            for j in range(len(verts)):
                if i == j:
                    continue
                vj = verts[j]
                if (rem >> vj) & 1 == 0:
                    continue
                cu = conflict[vj] & rem
                # If cv is a superset of cu (cv has at least as many blocked
                # vertices as cu), then choosing vj is at least as good as vi.
                if (cv | cu) == cv and cv != cu:
                    rem &= ~(1 << vi)
                    pruned |= (1 << vi)
                    break
        if pruned:
            remaining = rem
            if size + popcount(remaining) <= best[0]:
                return
            if remaining == 0:
                return

        # Branch: pick a vertex to include. Choose the vertex that, if included,
        # removes the most from remaining (most constrained) to shrink the tree.
        verts = [v for v in range(n) if (remaining >> v) & 1]
        # heuristic: pick v maximizing popcount(conflict[v] & remaining)
        v_pick = -1
        v_score = -1
        for v in verts:
            s = popcount(conflict[v] & remaining)
            if s > v_score:
                v_score = s
                v_pick = v

        # Branch 1: include v_pick
        new_rem = remaining & ~(1 << v_pick) & ~conflict[v_pick]
        search(new_rem, chosen | (1 << v_pick), size + 1)

        # Branch 2: exclude v_pick
        new_rem = remaining & ~(1 << v_pick)
        search(new_rem, chosen, size)

    search(full, 0, 0)
    return best[0]


# ---------------------------------------------------------------------------
# Linearity helpers
# ---------------------------------------------------------------------------

def _is_linear(triples):
    n = len(triples)
    for i in range(n):
        si = set(triples[i])
        for j in range(i + 1, n):
            if len(si & set(triples[j])) > 1:
                return False
    return True


def _pair_index(a, b):
    if a > b:
        a, b = b, a
    return a * 16 + b


def _triple_key(t):
    return tuple(sorted(t))


# ---------------------------------------------------------------------------
# Strong initial construction: greedy maximizing edge count with balanced
# degree, which tends to keep the independence number low.
# ---------------------------------------------------------------------------

def _build_initial(seed=12345):
    rng = random.Random(seed)
    n = 16
    used_pairs = set()
    degree = [0] * n
    triples = []

    # Candidate triples ordered by a balance heuristic: prefer triples whose
    # vertices are currently low-degree and whose pairwise distances are spread.
    all_triples = list(itertools.combinations(range(n), 3))

    def candidate_score(t):
        a, b, c = t
        # penalize high-degree vertices
        deg_pen = degree[a] + degree[b] + degree[c]
        # spread: sum of circular distances
        d1 = min(abs(a - b), n - abs(a - b))
        d2 = min(abs(b - c), n - abs(b - c))
        d3 = min(abs(a - c), n - abs(a - c))
        spread = d1 + d2 + d3
        # random jitter to avoid deterministic traps
        jitter = rng.random()
        return (-deg_pen, -spread, jitter)

    # We will repeatedly pick the best feasible candidate. To keep it fast we
    # pre-sort by a static spread score and then refine with degree, but for
    # 16 vertices the full list is small enough to re-score each round.
    for _ in range(len(all_triples)):
        best_t = None
        best_s = None
        for t in all_triples:
            a, b, c = t
            # check pair feasibility
            if (_pair_index(a, b) in used_pairs or
                    _pair_index(a, c) in used_pairs or
                    _pair_index(b, c) in used_pairs):
                continue
            s = candidate_score(t)
            if best_s is None or s < best_s:
                best_s = s
                best_t = t
        if best_t is None:
            break
        a, b, c = best_t
        triples.append([a, b, c])
        used_pairs.add(_pair_index(a, b))
        used_pairs.add(_pair_index(a, c))
        used_pairs.add(_pair_index(b, c))
        degree[a] += 1
        degree[b] += 1
        degree[c] += 1
        # remove best_t from candidate pool to avoid re-evaluating
        all_triples.remove(best_t)

    return triples


# ---------------------------------------------------------------------------
# Simulated annealing over the hyperedge set.
# ---------------------------------------------------------------------------

def _feasible_add(triples, used_pairs, new_triple):
    a, b, c = new_triple
    if a == b or a == c or b == c:
        return False
    return not (_pair_index(a, b) in used_pairs or
                _pair_index(a, c) in used_pairs or
                _pair_index(b, c) in used_pairs)


def _sa_search(triples, time_limit, seed=999):
    rng = random.Random(seed)
    n = 16
    start = time.time()
    deadline = start + time_limit

    triples = [list(t) for t in triples]
    used_pairs = set()
    for t in triples:
        a, b, c = t
        used_pairs.add(_pair_index(a, b))
        used_pairs.add(_pair_index(a, c))
        used_pairs.add(_pair_index(b, c))

    current_alpha = _calc_alpha(triples, time_budget=1.5)
    best_triples = [list(t) for t in triples]
    best_alpha = current_alpha

    T = 1.0
    T_min = 0.05
    cooling = 0.9995

    all_triples = list(itertools.combinations(range(n), 3))

    while time.time() < deadline:
        # Pick a move
        move_type = rng.random()
        if len(triples) > 4 and move_type < 0.55:
            # swap: remove one, add another
            i = rng.randrange(len(triples))
            old = triples[i]
            oa, ob, oc = old
            # temporarily free pairs
            used_pairs.discard(_pair_index(oa, ob))
            used_pairs.discard(_pair_index(oa, oc))
            used_pairs.discard(_pair_index(ob, oc))
            # find a feasible new triple
            cand = None
            for _ in range(60):
                t = all_triples[rng.randrange(len(all_triples))]
                if _feasible_add(triples, used_pairs, t):
                    cand = t
                    break
            if cand is None:
                # restore
                used_pairs.add(_pair_index(oa, ob))
                used_pairs.add(_pair_index(oa, oc))
                used_pairs.add(_pair_index(ob, oc))
                continue
            new_a, new_b, new_c = cand
            used_pairs.add(_pair_index(new_a, new_b))
            used_pairs.add(_pair_index(new_a, new_c))
            used_pairs.add(_pair_index(new_b, new_c))
            triples[i] = [new_a, new_b, new_c]
        elif move_type < 0.8 and len(triples) < 80:
            # add a triple
            cand = None
            for _ in range(60):
                t = all_triples[rng.randrange(len(all_triples))]
                if _feasible_add(triples, used_pairs, t):
                    cand = t
                    break
            if cand is None:
                continue
            a, b, c = cand
            used_pairs.add(_pair_index(a, b))
            used_pairs.add(_pair_index(a, c))
            used_pairs.add(_pair_index(b, c))
            triples.append([a, b, c])
        else:
            # remove a triple
            if not triples:
                continue
            i = rng.randrange(len(triples))
            a, b, c = triples.pop(i)
            used_pairs.discard(_pair_index(a, b))
            used_pairs.discard(_pair_index(a, c))
            used_pairs.discard(_pair_index(b, c))

        # Evaluate
        new_alpha = _calc_alpha(triples, time_budget=1.0)
        delta = new_alpha - current_alpha
        if delta <= 0 or rng.random() < pow(2.718281828459045, -delta / max(T, 1e-6)):
            current_alpha = new_alpha
            if new_alpha < best_alpha:
                best_alpha = new_alpha
                best_triples = [list(t) for t in triples]
        else:
            # revert
            if move_type < 0.55:
                oa, ob, oc = old
                used_pairs.discard(_pair_index(new_a, new_b))
                used_pairs.discard(_pair_index(new_a, new_c))
                used_pairs.discard(_pair_index(new_b, new_c))
                used_pairs.add(_pair_index(oa, ob))
                used_pairs.add(_pair_index(oa, oc))
                used_pairs.add(_pair_index(ob, oc))
                triples[i] = [oa, ob, oc]
            elif move_type < 0.8:
                a, b, c = cand
                used_pairs.discard(_pair_index(a, b))
                used_pairs.discard(_pair_index(a, c))
                used_pairs.discard(_pair_index(b, c))
                triples.pop()
            else:
                used_pairs.add(_pair_index(a, b))
                used_pairs.add(_pair_index(a, c))
                used_pairs.add(_pair_index(b, c))
                triples.append([a, b, c])

        T = max(T_min, T * cooling)

    return best_triples, best_alpha


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before the 150s kill

    # Initial construction
    triples = _build_initial(seed=12345)
    if not _is_linear(triples):
        # Fallback to a simple valid system
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]

    initial_alpha = _calc_alpha(triples, time_budget=3.0)

    # Simulated annealing with the remaining time budget
    remaining = deadline - time.time()
    if remaining > 5.0:
        triples, alpha = _sa_search(triples, remaining - 5.0, seed=999)
    else:
        alpha = initial_alpha

    # Final validity check
    if not _is_linear(triples):
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]

    return triples
