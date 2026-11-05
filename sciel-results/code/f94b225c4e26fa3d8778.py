"""Few edges in 5-uniform hypergraphs without Property B.

Strategy:
1. Precompute the known 51-edge construction (even 5-subsets of 10 vertices).
2. Use a fast 2-SAT-style backtracking with unit propagation to verify
   non-2-colorability.
3. Try to find smaller constructions via:
   a) Greedy edge deletion from the 51-edge baseline.
   b) Randomized search for small non-2-colorable hypergraphs on n=10..14.
4. Return the best (fewest edges) valid hypergraph found within time budget.
"""
import time
import random
from itertools import combinations

_DEADLINE = 145.0
_START = time.monotonic()


def _elapsed():
    return time.monotonic() - _START


def _not_2_colorable(n, edges):
    """
    Check if the 5-uniform hypergraph is NOT 2-colorable.
    Uses backtracking with unit propagation on a 2-SAT-like formulation.
    
    For each edge (v1,...,v5), we forbid all-0 and all-1.
    This is equivalent to: for each edge, at least one vertex is 0 AND at least one is 1.
    """
    m = len(edges)
    if m == 0:
        return False
    
    # Build incidence: for each vertex, which edges contain it
    v_edges = [[] for _ in range(n)]
    for ei, e in enumerate(edges):
        for v in e:
            v_edges[v].append(ei)
    
    # Sort vertices by degree (descending) for better branching
    v_order = sorted(range(n), key=lambda v: -len(v_edges[v]))
    
    color = [-1] * n
    # For each edge: count of 0s, count of 1s, count of unassigned
    cnt0 = [0] * m
    cnt1 = [0] * m
    unc = [5] * m
    
    def _propagate():
        """Unit propagation: if an edge has 4 of one color, the 5th must be the other."""
        changed = True
        while changed:
            changed = False
            for ei in range(m):
                c0 = cnt0[ei]
                c1 = cnt1[ei]
                u = unc[ei]
                if c0 >= 5 or c1 >= 5:
                    return False  # contradiction: all same color
                if u == 0:
                    continue
                if c0 == 4 and u == 1:
                    # Must assign 1 to the unassigned vertex in this edge
                    for v in edges[ei]:
                        if color[v] == -1:
                            color[v] = 1
                            cnt1[ei] += 1
                            unc[ei] -= 1
                            for e2 in v_edges[v]:
                                if e2 != ei:
                                    cnt1[e2] += 1
                                    unc[e2] -= 1
                                    if cnt1[e2] >= 5:
                                        return False
                            changed = True
                            break
                elif c1 == 4 and u == 1:
                    # Must assign 0 to the unassigned vertex in this edge
                    for v in edges[ei]:
                        if color[v] == -1:
                            color[v] = 0
                            cnt0[ei] += 1
                            unc[ei] -= 1
                            for e2 in v_edges[v]:
                                if e2 != ei:
                                    cnt0[e2] += 1
                                    unc[e2] -= 1
                                    if cnt0[e2] >= 5:
                                        return False
                            changed = True
                            break
        return True
    
    def _search(idx):
        while idx < n and color[v_order[idx]] != -1:
            idx += 1
        if idx == n:
            # All vertices assigned; check no edge is monochromatic
            for ei in range(m):
                if cnt0[ei] == 5 or cnt1[ei] == 5:
                    return False
            return True  # valid 2-coloring exists
        
        v = v_order[idx]
        # Save state
        color_snap = color[:]
        cnt0_snap = cnt0[:]
        cnt1_snap = cnt1[:]
        unc_snap = unc[:]
        
        for c in (0, 1):
            color[v] = c
            valid = True
            for ei in v_edges[v]:
                if c == 0:
                    cnt0[ei] += 1
                else:
                    cnt1[ei] += 1
                unc[ei] -= 1
                if cnt0[ei] >= 5 or cnt1[ei] >= 5:
                    valid = False
                    break
            if valid and _propagate():
                if _search(idx + 1):
                    return True  # found a valid coloring
            # Restore
            color[:] = color_snap
            cnt0[:] = cnt0_snap
            cnt1[:] = cnt1_snap
            unc[:] = unc_snap
        
        return False  # no valid coloring found
    
    # If _search returns True, a 2-coloring exists -> graph IS 2-colorable
    # We want NOT 2-colorable, so return not _search(0)
    return not _search(0)


def _known_51_edges():
    """The known 51-edge construction: even 5-subsets of 10 vertices."""
    evenset = frozenset((0, 2, 4, 6, 8))
    edges = []
    for c in combinations(range(10), 5):
        if sum(1 for v in c if v in evenset) % 2 == 0:
            edges.append(list(c))
    return edges


def _try_delete_edges(n, edges, rng, time_limit=5.0):
    """
    Try to delete edges while maintaining non-2-colorability.
    Returns the reduced edge list.
    """
    current = list(edges)
    t0 = time.monotonic()
    
    while time.monotonic() - t0 < time_limit:
        if len(current) <= 28:
            break
        # Try to remove a random edge
        idx = rng.randrange(len(current))
        candidate = current[:idx] + current[idx+1:]
        if _not_2_colorable(n, candidate):
            current = candidate
        else:
            # Try a few more random positions
            found = False
            for _ in range(10):
                j = rng.randrange(len(current))
                if j == idx:
                    continue
                candidate = current[:j] + current[j+1:]
                if _not_2_colorable(n, candidate):
                    current = candidate
                    found = True
                    break
            if not found:
                break
    
    return current


def _random_search(n, target_edges, rng, time_limit=5.0):
    """
    Randomly sample hypergraphs with ~target_edges edges and check if non-2-colorable.
    """
    all_edges = list(combinations(range(n), 5))
    t0 = time.monotonic()
    
    while time.monotonic() - t0 < time_limit:
        sample = rng.sample(all_edges, min(target_edges, len(all_edges)))
        if _not_2_colorable(n, sample):
            return [list(e) for e in sample]
    
    return None


def construct():
    if _elapsed() > _DEADLINE - 10:
        return {'n': 10, 'edges': _known_51_edges()}
    
    rng = random.Random(42)
    best_edges = _known_51_edges()
    best_n = 10
    
    # Step 1: Try to reduce the 51-edge construction
    if _elapsed() < _DEADLINE - 30:
        reduced = _try_delete_edges(10, _known_51_edges(), rng, time_limit=10.0)
        if len(reduced) < len(best_edges):
            best_edges = reduced
            best_n = 10
    
    # Step 2: Try random search for smaller constructions
    for n in [10, 11, 12, 13, 14]:
        if _elapsed() > _DEADLINE - 15:
            break
        
        # Try various edge counts below current best
        max_try = min(len(best_edges) - 1, 50)
        for target in range(29, max_try + 1):
            if _elapsed() > _DEADLINE - 15:
                break
            # Limit attempts per target
            result = _random_search(n, target, rng, time_limit=2.0)
            if result is not None and len(result) < len(best_edges):
                best_edges = result
                best_n = n
                break  # Found something better, try to go lower
    
    return {'n': best_n, 'edges': best_edges}
