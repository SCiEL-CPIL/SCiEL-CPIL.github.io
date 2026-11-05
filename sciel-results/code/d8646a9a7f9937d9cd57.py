"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import time
import random
import math
from collections import defaultdict

TOTAL_POINTS = N * N
ALL_MASK = (1 << TOTAL_POINTS) - 1

def point_to_idx(p):
    return p[0] * N + p[1]

def idx_to_point(i):
    return (i // N, i % N)

# Precompute distances and conflict masks
_points = [(i // N, i % N) for i in range(TOTAL_POINTS)]

# Distance matrix
_dist = [[0] * TOTAL_POINTS for _ in range(TOTAL_POINTS)]
for i in range(TOTAL_POINTS):
    xi, yi = _points[i]
    for j in range(i + 1, TOTAL_POINTS):
        xj, yj = _points[j]
        d = (xi - xj) ** 2 + (yi - yj) ** 2
        _dist[i][j] = d
        _dist[j][i] = d

# Conflict mask: conflict[i][j] is a bitmask of points k such that {i,j,k} forms an isosceles triangle
_conflict = [[0] * TOTAL_POINTS for _ in range(TOTAL_POINTS)]
for i in range(TOTAL_POINTS):
    for j in range(i + 1, TOTAL_POINTS):
        mask = 0
        dij = _dist[i][j]
        for k in range(TOTAL_POINTS):
            if k == i or k == j:
                continue
            dik = _dist[i][k]
            djk = _dist[j][k]
            if dij == dik or dij == djk or dik == djk:
                mask |= (1 << k)
        _conflict[i][j] = mask
        _conflict[j][i] = mask

# For each pair (i, j), precompute the set of points that are "covered" by including both i and j
# A point p is covered if p is in the set, or p forms an isosceles triangle with some pair in the set.
# We'll compute coverage incrementally.

def _verify(sol):
    """Verify that sol is a valid maximal no-isosceles set."""
    sol_set = set(sol)
    # Check no isosceles triangles
    for a in range(len(sol)):
        for b in range(a + 1, len(sol)):
            for c in range(b + 1, len(sol)):
                i, j, k = sol[a], sol[b], sol[c]
                if _dist[i][j] == _dist[i][k] or _dist[i][j] == _dist[j][k] or _dist[i][k] == _dist[j][k]:
                    return False
    # Check maximality: every point not in sol must form an isosceles triangle with some pair in sol
    covered = set(sol)
    for a in range(len(sol)):
        for b in range(a + 1, len(sol)):
            i, j = sol[a], sol[b]
            # Add all points that form isosceles with i,j
            mask = _conflict[i][j]
            # Extract set bits
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                covered.add(idx)
                m ^= lsb
    return len(covered) == TOTAL_POINTS

def _compute_covered(sol):
    """Compute the set of covered points given a solution set."""
    covered = set(sol)
    for a in range(len(sol)):
        for b in range(a + 1, len(sol)):
            i, j = sol[a], sol[b]
            mask = _conflict[i][j]
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                covered.add(idx)
                m ^= lsb
    return covered

def _greedy_build(order, stop_early=True):
    """Build a maximal set by greedily adding points in the given order."""
    chosen = []
    valid_mask = ALL_MASK
    for p in order:
        if not (valid_mask & (1 << p)):
            continue
        chosen.append(p)
        valid_mask &= ~(1 << p)
        for q in chosen[:-1]:
            valid_mask &= ~_conflict[p][q]
        if valid_mask == 0 and stop_early:
            break
    return chosen

def _local_search_remove(sol, time_limit):
    """Try to shrink sol by removing points one at a time."""
    current = sol[:]
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        for idx in range(len(current)):
            p = current[idx]
            new_sol = [x for x in current if x != p]
            if len(new_sol) < 2:
                continue
            # Check no isosceles triangles
            valid = True
            for a in range(len(new_sol)):
                for b in range(a + 1, len(new_sol)):
                    for c in range(b + 1, len(new_sol)):
                        i, j, k = new_sol[a], new_sol[b], new_sol[c]
                        if _dist[i][j] == _dist[i][k] or _dist[i][j] == _dist[j][k] or _dist[i][k] == _dist[j][k]:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                continue
            # Check maximality
            covered = _compute_covered(new_sol)
            if len(covered) == TOTAL_POINTS:
                current = new_sol
                improved = True
                break
    return current

def _backtrack_find_maximal(k, time_limit, rng=None):
    """DFS to find a set of exactly k points that is maximal (forbidden covers all)."""
    if rng is None:
        rng = random
    best = [None]
    
    # Precompute for each point the list of points it conflicts with (forms isosceles with some other point)
    # Actually, we need: for a partial set S, what points are still "available" (don't form isosceles with any pair in S)?
    
    def dfs(S, valid_mask, depth):
        if time.time() > time_limit:
            return
        if depth == 0:
            # Check if S is maximal
            covered = _compute_covered(S)
            if len(covered) == TOTAL_POINTS:
                best[0] = S[:]
            return
        
        if valid_mask.bit_count() < depth:
            return
        
        # Get candidates
        m = valid_mask
        candidates = []
        while m:
            lsb = m & -m
            candidates.append(lsb.bit_length() - 1)
            m ^= lsb
        
        rng.shuffle(candidates)
        
        for p in candidates:
            if time.time() > time_limit:
                return
            if best[0] is not None:
                return
            new_valid = valid_mask & ~(1 << p)
            for q in S:
                new_valid &= ~_conflict[p][q]
            if new_valid == 0 and depth > 1:
                continue
            S.append(p)
            dfs(S, new_valid, depth - 1)
            S.pop()
            if best[0] is not None:
                return
    
    dfs([], ALL_MASK, k)
    return best[0]

def _simulated_annealing(initial_sol, time_limit):
    """SA that moves by removing a point and possibly adding points to restore maximality."""
    if not initial_sol:
        return None

    current = initial_sol[:]
    best = current[:]
    best_len = len(current)

    def cost(sol):
        """Lower is better. Penalize non-maximality heavily."""
        if len(sol) < 2:
            return 10000
        # Check independence
        for a in range(len(sol)):
            for b in range(a + 1, len(sol)):
                for c in range(b + 1, len(sol)):
                    i, j, k = sol[a], sol[b], sol[c]
                    if _dist[i][j] == _dist[i][k] or _dist[i][j] == _dist[j][k] or _dist[i][k] == _dist[j][k]:
                        return 10000
        # Compute uncovered
        covered = _compute_covered(sol)
        uncovered = TOTAL_POINTS - len(covered)
        return len(sol) + 50 * uncovered

    T = 2.0
    cooling = 0.995
    cur_cost = cost(current)

    while time.time() < time_limit:
        if len(current) <= 2:
            break
        idx = random.randrange(len(current))
        p = current[idx]
        new_sol = [x for x in current if x != p]
        new_cost = cost(new_sol)

        delta = new_cost - cur_cost
        if delta < 0 or (delta > 0 and random.random() < math.exp(-delta / max(T, 0.01))):
            current = new_sol
            cur_cost = new_cost
            if cur_cost < len(current) + 50:
                covered = _compute_covered(current)
                if len(covered) == TOTAL_POINTS:
                    if len(current) < best_len:
                        best = current[:]
                        best_len = len(current)
        T *= cooling

    if _verify(best):
        return best
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 290.0

    best_sol = None
    best_len = TOTAL_POINTS

    # Phase 1: Greedy constructions
    for attempt in range(50):
        if time.time() > deadline - 60:
            break
        order = list(range(TOTAL_POINTS))
        random.shuffle(order)
        sol = _greedy_build(order, stop_early=True)
        if _verify(sol) and len(sol) < best_len:
            best_sol = sol
            best_len = len(sol)

    # Phase 2: Local search
    if best_sol and time.time() < deadline - 40:
        local_time = min(15.0, deadline - time.time() - 40)
        if local_time > 2:
            improved = _local_search_remove(best_sol, time.time() + local_time)
            if improved and _verify(improved) and len(improved) < best_len:
                best_sol = improved
                best_len = len(best_sol)

    # Phase 3: Backtracking for smaller k
    max_k_to_try = best_len - 1 if best_sol else 20
    for k in range(2, min(max_k_to_try + 1, 25)):
        if time.time() > deadline - 30:
            break
        time_budget = min(5.0, max(1.0, (deadline - time.time() - 30) / max(1, max_k_to_try - k)))
        sol = _backtrack_find_maximal(k, time.time() + time_budget)
        if sol is not None and _verify(sol):
            best_sol = sol
            best_len = k
            if k <= 4:
                break

    # Phase 4: SA refinement
    if best_sol and time.time() < deadline - 10:
        sa_time = min(20.0, deadline - time.time() - 10)
        if sa_time > 3:
            sa_sol = _simulated_annealing(best_sol, time.time() + sa_time)
            if sa_sol and _verify(sa_sol) and len(sa_sol) < best_len:
                best_sol = sa_sol
                best_len = len(best_sol)

    # Final local search
    if best_sol and time.time() < deadline - 5:
        ls_time = min(10.0, deadline - time.time() - 5)
        if ls_time > 1:
            final = _local_search_remove(best_sol, time.time() + ls_time)
            if final and _verify(final) and len(final) < best_len:
                best_sol = final
                best_len = len(final)

    if best_sol is None:
        order = list(range(TOTAL_POINTS))
        random.shuffle(order)
        best_sol = _greedy_build(order, stop_early=True)
        if not _verify(best_sol):
            for _ in range(20):
                order = list(range(TOTAL_POINTS))
                random.shuffle(order)
                best_sol = _greedy_build(order, stop_early=True)
                if _verify(best_sol):
                    break

    return [idx_to_point(i) for i in best_sol]

# EVOLVE-BLOCK-END
