"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 256

# EVOLVE-BLOCK-START

def _build_sum_set(selected):
    """Build the set of all pair sums a+b with a<=b."""
    sums = set()
    for i, a in enumerate(selected):
        for j in range(i, len(selected)):
            sums.add(selected[i] + selected[j])
    return sums

def _is_valid_sidon(selected):
    """Check if selected is a valid Sidon set."""
    sums = set()
    for i, a in enumerate(selected):
        for j in range(i, len(selected)):
            b = selected[j]
            s = a + b
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal_fast(selected, N):
    """Check if selected is inclusion-maximal Sidon in [1,N]. Uses precomputed sums."""
    sel_set = set(selected)
    sums = _build_sum_set(selected)
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        # Check if adding x creates a collision
        new_sums = [x + a for a in selected] + [2 * x]
        for s in new_sums:
            if s in sums:
                break
        else:
            # No collision, so x can be added - not maximal
            return False
    return True

def _conflict_graph(N):
    """
    Build conflict graph: two numbers conflict if they cannot coexist in a Sidon set.
    Two numbers a, b conflict if:
      - a == b (trivial, but we only consider distinct)
      - There exist a', b' in {a,b} such that a'+b' collides with some pair involving the other.
    Actually, simpler: a and b conflict if adding both to an empty set creates a sum collision.
    That is, if any of {2a, 2b, a+b} equals another of {2a, 2b, a+b} — but 2a != 2b if a!=b,
    and a+b != 2a iff b!=a, a+b != 2b iff a!=b. So two distinct numbers alone never conflict.
    
    The real conflict is: a and b cannot both be in a Sidon set if there exist other elements.
    But for pairwise conflict in the graph sense for MIS, we need: a and b are adjacent if
    they CANNOT both appear in ANY Sidon set. Two numbers a,b can always appear together in SOME
    Sidon set (just {a,b} is Sidon). So pairwise conflicts don't work directly.
    
    Alternative: model as finding a small maximal independent set. The "independence" here is
    Sidon-ness. Two vertices are "independent" if they can coexist. But the Sidon property is
    not pairwise — it's a global constraint on all pair sums.
    
    Better approach: use a greedy construction with randomization and local search to minimize
    cardinality while maintaining maximality.
    """
    pass

def _greedy_sidon(order, N):
    """Build a Sidon set by processing elements in given order, adding if possible."""
    selected = []
    sums = set()
    for x in order:
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected

def _greedy_sidon_weighted(weights, N):
    """Build a Sidon set using weighted random ordering."""
    order = list(range(1, N + 1))
    # Weighted shuffle: elements with higher weight are more likely to come first
    for i in range(len(order) - 1, 0, -1):
        total_w = sum(weights[j] for j in range(i + 1))
        r = random.random() * total_w
        cum = 0
        for j in range(i + 1):
            cum += weights[j]
            if cum >= r:
                order[i], order[j] = order[j], order[i]
                break
    return _greedy_sidon(order, N)

def _make_maximal(selected, N):
    """
    Given a Sidon set, extend it to be inclusion-maximal by greedily adding elements
    that can be added. This increases cardinality, so we use this only as a repair step.
    """
    sel_set = set(selected)
    sums = _build_sum_set(selected)
    # Try to add missing elements in random order to maximize the chance of a small maximal set
    candidates = [x for x in range(1, N + 1) if x not in sel_set]
    random.shuffle(candidates)
    for x in candidates:
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sel_set.add(x)
            sums.update(added)
    return selected

def _remove_element(selected, x):
    """Remove x from selected, return new list."""
    return [v for v in selected if v != x]

def _try_remove_to_minimize(selected, N, max_iters=200):
    """
    Try to remove elements from a maximal Sidon set while maintaining maximality.
    If removing x still leaves the set maximal (after re-adding if needed), keep it removed.
    """
    best = list(selected)
    for _ in range(max_iters):
        improved = False
        for x in list(best):
            candidate = _remove_element(best, x)
            if not _is_valid_sidon(candidate):
                continue
            # Check if candidate is maximal
            if _is_maximal_fast(candidate, N):
                best = candidate
                improved = True
                break
        if not improved:
            break
    return best

def _swap_local_search(selected, N, max_iters=100):
    """
    Try swap operations: remove x, add y, check if still maximal and smaller.
    """
    best = list(selected)
    sel_set = set(best)
    for _ in range(max_iters):
        improved = False
        for x in list(best):
            for y in range(1, N + 1):
                if y in sel_set:
                    continue
                # Try removing x and adding y
                candidate = [v for v in best if v != x] + [y]
                if _is_valid_sidon(candidate) and _is_maximal_fast(candidate, N):
                    if len(candidate) < len(best):
                        best = candidate
                        sel_set = set(best)
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break
    return best

def construct():
    deadline = time.time() + 55  # Safety margin before 60s limit
    best_set = None
    best_size = N + 1
    
    # Strategy 1: Greedy with random ordering (more trials due to faster ops)
    for trial in range(800):
        if time.time() > deadline:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        selected = _greedy_sidon(order, N)
        # Make it maximal
        selected = _make_maximal(selected, N)
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = list(selected)
    
    # Strategy 2: Start from a larger Sidon set, try to shrink
    # Build a standard greedy Sidon set
    standard = _greedy_sidon(list(range(1, N + 1)), N)
    standard = _make_maximal(standard, N)
    if len(standard) < best_size:
        best_size = len(standard)
        best_set = list(standard)
    
    # Strategy 3: Local search — try removing elements from best_set
    if best_set is not None:
        improved = _try_remove_to_minimize(best_set, N, max_iters=200)
        if len(improved) < best_size:
            best_size = len(improved)
            best_set = improved
    
    # Strategy 4: Random restarts with different initialization
    for trial in range(400):
        if time.time() > deadline:
            break
        # Random subset initialization
        size = random.randint(10, 50)
        init = random.sample(range(1, N + 1), min(size, N))
        # Filter to Sidon
        selected = []
        sums = set()
        for x in init:
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        if not selected:
            continue
        selected = _make_maximal(selected, N)
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = list(selected)
        # Try to shrink
        improved = _try_remove_to_minimize(selected, N, max_iters=100)
        if len(improved) < best_size:
            best_size = len(improved)
            best_set = improved
    
    # Strategy 5: Swap local search on best solution
    if best_set is not None and time.time() < deadline:
        improved = _swap_local_search(best_set, N, max_iters=50)
        if len(improved) < best_size:
            best_size = len(improved)
            best_set = improved
    
    # Final polish: try more removals
    if best_set is not None and time.time() < deadline:
        improved = _try_remove_to_minimize(best_set, N, max_iters=300)
        if len(improved) < best_size:
            best_size = len(improved)
            best_set = improved
    
    if best_set is None:
        # Fallback: standard greedy
        best_set = _greedy_sidon(list(range(1, N + 1)), N)
        best_set = _make_maximal(best_set, N)
    
    return best_set

# EVOLVE-BLOCK-END
