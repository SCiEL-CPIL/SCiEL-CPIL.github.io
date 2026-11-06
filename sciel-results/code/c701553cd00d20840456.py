"""Search for dense linear 3-graphs on 16 points to minimize independence number."""
import time
import random
import itertools
import sys
from collections import defaultdict

# EVOLVE-BLOCK-START

def _compute_independence_number(triples, n=16, timeout=5.0):
    """Compute the independence number (largest subset containing no triple).
    Uses branch-and-bound with memoization on bitmasks."""
    start = time.time()
    ntriples = len(triples)
    if ntriples == 0:
        return n
    # Precompute for each point which triples it belongs to
    point_to_triples = defaultdict(list)
    for ti, tr in enumerate(triples):
        for p in tr:
            point_to_triples[p].append(ti)
    
    # For a candidate independent set S, it's independent iff no triple is fully in S
    # Branch and bound: build set point by point, track which triples are "covered" (have a point in S)
    # A triple is "safe" if it has at least one point NOT in S (i.e., we haven't placed all 3)
    # Actually: a triple is "violated" only if all 3 points are in S.
    # We track for each triple how many of its points are in S. If any reaches 3, invalid.
    
    best = [0]
    # Order points by degree (points in more triples are more constrained)
    point_order = sorted(range(n), key=lambda p: -len(point_to_triples[p]))
    
    # For branch and bound, we go through points in order, decide include or not.
    # State: index in point_order, current set size, for each triple the count of included points.
    # If any triple count reaches 3, prune.
    # Upper bound: current_size + remaining_points
    
    # To keep it fast, use a simpler approach: try all subsets up to size k, check independence.
    # But that's 2^16 = 65536 subsets, each check is O(ntriples * 3). That's feasible.
    
    # Actually, let's do a smarter branch and bound.
    # We'll use recursion with the point order.
    
    sys.setrecursionlimit(10000)
    
    def branch(idx, size, triple_counts):
        if time.time() - start > timeout:
            return
        if size > best[0]:
            best[0] = size
        if idx == n:
            return
        # Upper bound
        if size + (n - idx) <= best[0]:
            return
        p = point_order[idx]
        # Option 1: don't include p
        branch(idx + 1, size, triple_counts)
        # Option 2: include p (if it doesn't complete any triple)
        can_include = True
        new_counts = list(triple_counts)
        for ti in point_to_triples[p]:
            if new_counts[ti] + 1 >= 3:
                can_include = False
                break
            new_counts[ti] += 1
        if can_include:
            branch(idx + 1, size + 1, new_counts)
    
    branch(0, 0, [0] * ntriples)
    return best[0]


def _is_linear(triples):
    """Check that any two triples share at most one point."""
    n = len(triples)
    for i in range(n):
        for j in range(i + 1, n):
            common = set(triples[i]) & set(triples[j])
            if len(common) > 1:
                return False
    return True


def _greedy_construct(random_state, n=16):
    """Greedy construction of a linear 3-graph."""
    # All possible triples
    all_triples = list(itertools.combinations(range(n), 3))
    random_state.shuffle(all_triples)
    
    selected = []
    # For each pair of points, track how many selected triples contain that pair
    pair_count = defaultdict(int)
    
    for tr in all_triples:
        a, b, c = tr
        pairs = [(a, b), (a, c), (b, c)]
        # Check linearity: no pair can be in two triples
        if all(pair_count[p] == 0 for p in pairs):
            selected.append(list(tr))
            for p in pairs:
                pair_count[p] += 1
    
    return selected


def _local_search(triples, random_state, n=16, max_iter=500, timeout=10.0):
    """Local search: try swapping triples to reduce independence number."""
    start = time.time()
    if not _is_linear(triples):
        return triples
    
    current_alpha = _compute_independence_number(triples, n, timeout=2.0)
    best_triples = list(triples)
    best_alpha = current_alpha
    
    all_triples = list(itertools.combinations(range(n), 3))
    
    for iteration in range(max_iter):
        if time.time() - start > timeout:
            break
        
        # Try removing one triple and adding another
        if len(triples) == 0:
            break
        
        improved = False
        # Try each removal
        for i in range(len(triples)):
            if time.time() - start > timeout:
                break
            candidate = triples[:i] + triples[i+1:]
            # Try adding a random triple
            for _ in range(20):
                tr = random_state.choice(all_triples)
                if tr in [tuple(t) for t in candidate]:
                    continue
                candidate_new = candidate + [list(tr)]
                if not _is_linear(candidate_new):
                    continue
                # Check if it reduces independence number
                alpha = _compute_independence_number(candidate_new, n, timeout=1.0)
                if alpha < best_alpha:
                    best_alpha = alpha
                    best_triples = candidate_new
                    triples = candidate_new
                    improved = True
                    break
            if improved:
                break
        
        if not improved and iteration > 100:
            # Try a different strategy: random restart of local search
            break
    
    return best_triples


def _multi_restart_greedy(random_state, n=16, n_restarts=200, timeout=30.0):
    """Multiple restarts of greedy construction."""
    start = time.time()
    best_triples = None
    best_alpha = n + 1
    
    for r in range(n_restarts):
        if time.time() - start > timeout:
            break
        triples = _greedy_construct(random_state, n)
        if not triples:
            continue
        alpha = _compute_independence_number(triples, n, timeout=2.0)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = list(triples)
    
    return best_triples, best_alpha


def _improve_with_backtracking(triples, random_state, n=16, timeout=20.0):
    """Try to add more triples to the system to reduce independence number."""
    start = time.time()
    if not triples:
        return triples
    
    current = list(triples)
    current_alpha = _compute_independence_number(current, n, timeout=2.0)
    
    all_triples = list(itertools.combinations(range(n), 3))
    random_state.shuffle(all_triples)
    
    for tr in all_triples:
        if time.time() - start > timeout:
            break
        if tr in [tuple(t) for t in current]:
            continue
        candidate = current + [list(tr)]
        if not _is_linear(candidate):
            continue
        alpha = _compute_independence_number(candidate, n, timeout=1.5)
        if alpha < current_alpha:
            current = candidate
            current_alpha = alpha
    
    return current


def construct():
    """Construct a linear triple system on 16 points minimizing independence number."""
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill
    
    # Use a fixed seed for reproducibility, but also try random
    random_state = random.Random(42)
    
    best_triples = None
    best_alpha = 17  # Worst case
    
    # Phase 1: Multi-restart greedy
    phase1_timeout = min(30.0, deadline - time.time() - 5.0)
    if phase1_timeout > 5.0:
        triples, alpha = _multi_restart_greedy(random_state, 16, n_restarts=300, timeout=phase1_timeout)
        if triples and alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples
    
    # Phase 2: Local search on best result
    remaining = deadline - time.time() - 5.0
    if remaining > 10.0 and best_triples:
        improved = _local_search(best_triples, random_state, 16, max_iter=1000, timeout=min(remaining, 40.0))
        if improved:
            alpha = _compute_independence_number(improved, 16, timeout=3.0)
            if alpha < best_alpha:
                best_alpha = alpha
                best_triples = improved
    
    # Phase 3: Try to add more triples
    remaining = deadline - time.time() - 5.0
    if remaining > 10.0 and best_triples:
        improved = _improve_with_backtracking(best_triples, random_state, 16, timeout=min(remaining, 30.0))
        if improved:
            alpha = _compute_independence_number(improved, 16, timeout=3.0)
            if alpha < best_alpha:
                best_alpha = alpha
                best_triples = improved
    
    # Phase 4: More random restarts with different seeds if time remains
    remaining = deadline - time.time() - 5.0
    if remaining > 15.0:
        for seed in range(100):
            if time.time() > deadline - 5.0:
                break
            rs = random.Random(seed + 1000)
            triples, alpha = _multi_restart_greedy(rs, 16, n_restarts=50, timeout=min(remaining, 15.0))
            if triples and alpha < best_alpha:
                best_alpha = alpha
                best_triples = triples
            remaining = deadline - time.time() - 5.0
            if remaining < 10.0:
                break
    
    # Fallback: if we have nothing, return the seed
    if best_triples is None:
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    return best_triples

# EVOLVE-BLOCK-END
