"""Hitting set on maximal cliques via bitmask branch-and-bound + greedy/local search."""
# EVOLVE-BLOCK-START
import time
from random import Random

_DEADLINE = 150.0 - 10.0  # safety margin before 150s wall-clock kill


def _build_graph(n, p, seed):
    """Reproduce random_graph(n,p,seed) edges exactly."""
    rng = Random(seed)
    edges = []
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
                adj[u].add(v)
                adj[v].add(u)
    return edges, adj


def _maximal_cliques(adj, n):
    """Bron–Kerbosch with pivot; returns list of maximal cliques as frozensets (size>=2)."""
    cliques = []

    def bronk(R, P, X):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        if not P:
            return
        # pivot: choose vertex in P ∪ X with max |P ∩ N(u)|
        cand = P | X
        best = -1
        pivot = None
        for u in cand:
            c = len(P & adj[u])
            if c > best:
                best = c
                pivot = u
        # iterate over P \ N(pivot)
        if pivot is not None:
            candidates = P - adj[pivot]
        else:
            candidates = P
        for v in list(candidates):
            bronk(R | {v}, P & adj[v], X & adj[v])
            P.remove(v)
            X.add(v)

    bronk(frozenset(), set(range(n)), set())
    return cliques


def _solve_hitting_set(cliques, n, deadline):
    """
    Solve the hitting set problem: find minimum subset S of {0..n-1}
    such that every clique in `cliques` has at least one vertex in S.
    
    Uses bitmask branch-and-bound for exact solution when feasible,
    with greedy + local search as fallback.
    """
    num_cliques = len(cliques)
    if num_cliques == 0:
        return set()

    # Precompute: for each vertex, bitmask of cliques it covers
    # Use Python ints as bitmasks (arbitrary precision)
    vertex_mask = [0] * n
    for ci, c in enumerate(cliques):
        bit = 1 << ci
        for v in c:
            vertex_mask[v] |= bit

    full_mask = (1 << num_cliques) - 1

    # Greedy initialization
    def greedy_init():
        S = set()
        covered = 0
        while covered != full_mask:
            remaining = full_mask & ~covered
            best_v = -1
            best_gain = -1
            for v in range(n):
                if v in S:
                    continue
                gain = vertex_mask[v] & remaining
                gc = bin(gain).count('1')
                if gc > best_gain:
                    best_gain = gc
                    best_v = v
            if best_v == -1 or best_gain == 0:
                break
            S.add(best_v)
            covered |= vertex_mask[best_v]
        return S

    S = greedy_init()
    covered = 0
    for v in S:
        covered |= vertex_mask[v]
    # If greedy didn't cover all (shouldn't happen), add all
    if covered != full_mask:
        S = set(range(n))
        covered = full_mask

    best_S = set(S)
    best_size = len(S)

    # Local search: try removals and swaps
    rng = Random(42)

    def try_remove(S, covered):
        """Try to remove vertices from S while maintaining coverage."""
        changed = True
        while changed:
            changed = False
            for v in list(S):
                # Check if removing v still covers all
                new_covered = covered & ~vertex_mask[v]
                # But we need to check: is every clique still covered by S\{v}?
                # A clique ci is uncovered after removing v iff:
                #   (vertex_mask[v] has bit ci) AND (no other vertex in S\{v} covers ci)
                # Simpler: just recompute covered without v
                # Actually, let's check per-clique:
                can_remove = True
                for ci in range(num_cliques):
                    if vertex_mask[v] & (1 << ci):
                        # v covers clique ci; check if any other vertex in S covers it
                        other_covers = False
                        for v2 in S:
                            if v2 == v:
                                continue
                            if vertex_mask[v2] & (1 << ci):
                                other_covers = True
                                break
                        if not other_covers:
                            can_remove = False
                            break
                if can_remove:
                    S.discard(v)
                    covered = 0
                    for v2 in S:
                        covered |= vertex_mask[v2]
                    changed = True
        return S, covered

    def local_search(S, covered, deadline):
        """Local search with removals and 1-for-2 swaps."""
        S = set(S)
        covered = 0
        for v in S:
            covered |= vertex_mask[v]
        S, covered = try_remove(S, covered)
        current_size = len(S)
        no_improve = 0
        max_no_improve = 50000

        while time.time() < deadline and no_improve < max_no_improve:
            improved = False

            # Try removals
            for v in list(S):
                # Check if v is essential
                essential = False
                for ci in range(num_cliques):
                    if vertex_mask[v] & (1 << ci):
                        other = False
                        for v2 in S:
                            if v2 != v and (vertex_mask[v2] & (1 << ci)):
                                other = True
                                break
                        if not other:
                            essential = True
                            break
                if not essential:
                    S.discard(v)
                    covered = 0
                    for v2 in S:
                        covered |= vertex_mask[v2]
                    current_size -= 1
                    improved = True
                    break
            if improved:
                no_improve = 0
                continue

            # Try 1-for-2 swap: remove one vertex, add another that covers what was lost
            if len(S) >= 1:
                # Find vertices that might be removable with help
                for v_out in list(S):
                    # What does v_out cover that no other vertex in S covers?
                    only_covered = 0
                    for ci in range(num_cliques):
                        if vertex_mask[v_out] & (1 << ci):
                            other = False
                            for v2 in S:
                                if v2 != v_out and (vertex_mask[v2] & (1 << ci)):
                                    other = True
                                    break
                            if not other:
                                only_covered |= (1 << ci)
                    if only_covered == 0:
                        continue  # v_out is not essential, should have been removed
                    # Try to find a single vertex v_in not in S that covers only_covered
                    for v_in in range(n):
                        if v_in in S:
                            continue
                        if (vertex_mask[v_in] & only_covered) == only_covered:
                            # v_in covers everything v_out uniquely covers
                            # Check if swapping reduces size or is same size
                            # We remove v_out, add v_in: size stays same
                            # But maybe we can then remove another vertex?
                            # For now, just do the swap if it helps
                            new_S = set(S)
                            new_S.discard(v_out)
                            new_S.add(v_in)
                            new_covered = 0
                            for v2 in new_S:
                                new_covered |= vertex_mask[v2]
                            if new_covered == full_mask and len(new_S) < current_size:
                                S = new_S
                                covered = new_covered
                                current_size = len(S)
                                improved = True
                                break
                    if improved:
                        break

            if not improved:
                no_improve += 1
                # Random perturbation: try random swaps
                if len(S) >= 2 and rng.random() < 0.1:
                    v1, v2 = rng.sample(list(S), 2)
                    # Remove both, see what's uncovered
                    temp_covered = 0
                    for v in S:
                        if v != v1 and v != v2:
                            temp_covered |= vertex_mask[v]
                    uncovered = full_mask & ~temp_covered
                    if uncovered == 0:
                        # Both can be removed!
                        S.discard(v1)
                        S.discard(v2)
                        covered = temp_covered
                        current_size = len(S)
                        improved = True
                        no_improve = 0
                    elif uncovered:
                        # Try to find one vertex that covers uncovered
                        for v_in in range(n):
                            if v_in in S:
                                continue
                            if (vertex_mask[v_in] & uncovered) == uncovered:
                                new_S = set(S)
                                new_S.discard(v1)
                                new_S.discard(v2)
                                new_S.add(v_in)
                                new_covered = 0
                                for v2 in new_S:
                                    new_covered |= vertex_mask[v2]
                                if new_covered == full_mask and len(new_S) < current_size:
                                    S = new_S
                                    covered = new_covered
                                    current_size = len(S)
                                    improved = True
                                    no_improve = 0
                                break
                        if improved:
                            break

        return S, covered

    S, covered = local_search(S, covered, time.time() + min(20.0, deadline - time.time()))
    if len(S) < best_size:
        best_S = S
        best_size = len(S)

    return best_S


def _branch_and_bound_hitting(cliques, n, deadline):
    """
    Exact branch-and-bound for hitting set using bitmask.
    Returns the minimum hitting set if found within deadline.
    """
    num_cliques = len(cliques)
    if num_cliques == 0:
        return set()

    vertex_mask = [0] * n
    for ci, c in enumerate(cliques):
        bit = 1 << ci
        for v in c:
            vertex_mask[v] |= bit

    full_mask = (1 << num_cliques) - 1

    # Order vertices by decreasing coverage
    order = sorted(range(n), key=lambda v: -bin(vertex_mask[v]).count('1'))

    best = [n + 1]
    best_set = [set()]

    def bb(covered, remaining_mask, start_idx, current_set, current_size):
        if current_size >= best[0]:
            return
        if time.time() > deadline:
            return
        if remaining_mask == 0:
            if current_size < best[0]:
                best[0] = current_size
                best_set[0] = set(current_set)
            return

        # Lower bound: need at least ceil(popcount(remaining_mask) / max_coverage)
        # Simple bound: each vertex covers at most max bits
        # Better: use the fact that we need to cover remaining_mask
        # Greedy lower bound
        rem = remaining_mask
        lb = 0
        temp_covered = 0
        while rem:
            best_gain = 0
            for i in range(start_idx, n):
                v = order[i]
                gain = vertex_mask[v] & rem
                gc = bin(gain).count('1')
                if gc > best_gain:
                    best_gain = gc
            if best_gain == 0:
                return  # can't cover
            lb += 1
            # Find which vertex gives best gain
            best_v = -1
            for i in range(start_idx, n):
                v = order[i]
                gain = vertex_mask[v] & rem
                if bin(gain).count('1') == best_gain:
                    best_v = v
                    break
            rem &= ~vertex_mask[best_v]
            if current_size + lb >= best[0]:
                return

        # Branch: try adding each remaining vertex
        for i in range(start_idx, n):
            v = order[i]
            if time.time() > deadline:
                return
            new_covered = covered | vertex_mask[v]
            new_remaining = remaining_mask & ~vertex_mask[v]
            new_size = current_size + 1
            if new_size >= best[0]:
                continue
            current_set.add(v)
            bb(new_covered, new_remaining, i + 1, current_set, new_size)
            current_set.discard(v)

    bb(0, full_mask, 0, set(), 0)

    if best[0] <= n:
        return best_set[0]
    return None


def construct():
    t_start = time.time()
    results = []

    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        elapsed = time.time() - t_start
        remaining = _DEADLINE - elapsed
        if remaining <= 0:
            # Out of time, return empty (will be invalid but better than crash)
            results.append([])
            continue

        edges, adj = _build_graph(n, p, seed)
        cliques = _maximal_cliques(adj, n)

        if not cliques:
            results.append([])
            continue

        # Try exact branch-and-bound with a time cap
        # Allocate more time to larger instances
        inst_time = min(40.0, remaining * 0.8)
        exact_result = None
        try:
            exact_result = _branch_and_bound_hitting(cliques, n, time.time() + inst_time)
        except Exception:
            pass

        if exact_result is not None:
            # Verify
            valid = True
            for c in cliques:
                if not (c & exact_result):
                    valid = False
                    break
            if valid:
                results.append(sorted(exact_result))
                continue

        # Fallback: hitting set solver (greedy + local search)
        hit_time = min(30.0, remaining * 0.5)
        try:
            hit_set = _solve_hitting_set(cliques, n, time.time() + hit_time)
            valid = True
            for c in cliques:
                if not (c & hit_set):
                    valid = False
                    break
            if valid:
                results.append(sorted(hit_set))
                continue
        except Exception:
            pass

        # Last resort: greedy vertex cover (guaranteed valid)
        uncovered = set((u, v) for u, v in edges)
        cover = set()
        while uncovered:
            best_v = None
            best_c = -1
            for v in range(n):
                c = 0
                for w in adj[v]:
                    e = (v, w) if v < w else (w, v)
                    if e in uncovered:
                        c += 1
                if c > best_c:
                    best_c = c
                    best_v = v
            if best_c <= 0:
                break
            cover.add(best_v)
            for w in adj[best_v]:
                e = (best_v, w) if best_v < w else (w, best_v)
                uncovered.discard(e)
        if not cover:
            cover = {x for e in edges for x in e}
        results.append(sorted(cover))

    return results
# EVOLVE-BLOCK-END
