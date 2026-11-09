import time
import random
from fractions import Fraction
from functools import reduce

# EVOLVE-BLOCK-START

def _is_valid_3ap_free(selected, n=256):
    """Check that the set has no nonconstant 3-term arithmetic progression."""
    s = set(selected)
    for a in s:
        for b in s:
            if a == b:
                continue
            c = 2 * b - a
            if c != a and c in s:
                return False
    return True

def _reciprocal_sum_fraction(selected):
    """Exact sum of reciprocals as a Fraction."""
    total = Fraction(0, 1)
    for x in selected:
        total += Fraction(1, x)
    return total

def _reciprocal_sum_float(selected):
    return sum(1.0 / x for x in selected)

def _can_add(candidate, selected_set, selected_list):
    """Check if adding candidate would create a 3-AP."""
    for a in selected_list:
        # candidate as middle: a, candidate, c -> c = 2*candidate - a
        c = 2 * candidate - a
        if c != candidate and c in selected_set:
            return False
        # candidate as end: a, b, candidate -> b = (a + candidate) / 2
        if (a + candidate) % 2 == 0:
            b = (a + candidate) // 2
            if b != candidate and b != a and b in selected_set:
                return False
    return True

def _greedy_construct(start_order, n=256):
    """Greedy: iterate in given order, add if valid."""
    selected = []
    selected_set = set()
    for x in start_order:
        if 1 <= x <= n:
            if _can_add(x, selected_set, selected):
                selected.append(x)
                selected_set.add(x)
    return selected

def _local_search(selected, n=256, time_limit=None):
    """Try swaps and single removals/additions to improve reciprocal sum."""
    selected = sorted(selected)
    selected_set = set(selected)
    current_sum = _reciprocal_sum_float(selected)
    
    if time_limit is None:
        start_time = time.time()
    
    improved = True
    while improved:
        improved = False
        
        # Try removing one element
        for i, elem in enumerate(selected):
            new_sel = selected[:i] + selected[i+1:]
            new_sum = _reciprocal_sum_float(new_sel)
            if new_sum > current_sum + 1e-15:
                selected = new_sel
                selected_set = set(selected)
                current_sum = new_sum
                improved = True
                break
        
        # Try swapping one element for a larger one (or any candidate)
        for i, elem in enumerate(selected):
            # Try replacing elem with candidates from 1 to n
            best_candidate = None
            best_sum = current_sum
            for cand in range(1, n+1):
                if cand == elem or cand in selected_set:
                    continue
                new_sel = selected[:i] + [cand] + selected[i+1:]
                if _is_valid_3ap_free(new_sel, n):
                    s = _reciprocal_sum_float(new_sel)
                    if s > best_sum + 1e-15:
                        best_sum = s
                        best_candidate = cand
            if best_candidate is not None:
                selected[i] = best_candidate
                selected = sorted(selected)
                selected_set = set(selected)
                current_sum = best_sum
                improved = True
        
        # Try adding a new element
        for cand in range(1, n+1):
            if cand not in selected_set:
                if _can_add(cand, selected_set, selected):
                    new_sum = current_sum + 1.0/cand
                    if new_sum > current_sum + 1e-15:
                        selected.append(cand)
                        selected = sorted(selected)
                        selected_set = set(selected)
                        current_sum = new_sum
                        improved = True
        
        if time_limit is not None:
            if time.time() - start_time > time_limit:
                break
    
    return sorted(selected)

def _simulated_annealing(selected, n=256, max_iters=50000, start_temp=1.0, end_temp=0.001, time_limit=None):
    """Simulated annealing to improve reciprocal sum while maintaining 3-AP-free constraint."""
    selected = list(selected)
    selected_set = set(selected)
    current_sum = _reciprocal_sum_float(selected)
    best_selected = list(selected)
    best_sum = current_sum
    
    if time_limit is None:
        start_time = time.time()
    
    for it in range(max_iters):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        temp = start_temp * (end_temp / start_temp) ** (it / max_iters)
        
        # Generate a neighbor: add, remove, or swap
        op = random.choice(['add', 'remove', 'swap'])
        new_selected = list(selected)
        new_selected_set = set(selected)
        
        if op == 'add':
            # Try to add a random element
            candidates = [x for x in range(1, n+1) if x not in new_selected_set]
            if candidates:
                cand = random.choice(candidates)
                if _can_add(cand, new_selected_set, new_selected):
                    new_selected.append(cand)
                    new_selected_set.add(cand)
                else:
                    continue
        elif op == 'remove':
            if new_selected:
                idx = random.randrange(len(new_selected))
                removed = new_selected.pop(idx)
                new_selected_set.discard(removed)
        elif op == 'swap':
            if new_selected:
                idx = random.randrange(len(new_selected))
                removed = new_selected[idx]
                candidates = [x for x in range(1, n+1) if x not in new_selected_set and x != removed]
                if candidates:
                    cand = random.choice(candidates)
                    new_selected[idx] = cand
                    new_selected_set.discard(removed)
                    new_selected_set.add(cand)
                    # Check validity
                    if not _is_valid_3ap_free(new_selected, n):
                        continue
                else:
                    continue
        
        new_sum = _reciprocal_sum_float(new_selected)
        delta = new_sum - current_sum
        
        if delta > 0 or random.random() < __import__('math').exp(delta / temp):
            selected = new_selected
            selected_set = new_selected_set
            current_sum = new_sum
            if current_sum > best_sum:
                best_sum = current_sum
                best_selected = list(selected)
    
    return best_selected

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s
    
    n = 256
    
    # Phase 1: Greedy construction with multiple orderings
    best_selected = []
    best_sum = 0.0
    
    # Ordering 1: natural ascending (original)
    selected = _greedy_construct(range(1, n+1), n)
    s = _reciprocal_sum_float(selected)
    if s > best_sum:
        best_sum = s
        best_selected = list(selected)
    
    # Ordering 2: descending
    selected = _greedy_construct(range(n, 0, -1), n)
    s = _reciprocal_sum_float(selected)
    if s > best_sum:
        best_sum = s
        best_selected = list(selected)
    
    # Ordering 3: random
    random.seed(42)
    for seed_try in range(5):
        if time.time() > deadline:
            break
        order = list(range(1, n+1))
        random.shuffle(order)
        selected = _greedy_construct(order, n)
        s = _reciprocal_sum_float(selected)
        if s > best_sum:
            best_sum = s
            best_selected = list(selected)
    
    # Phase 2: Local search
    if time.time() < deadline - 10:
        improved = _local_search(list(best_selected), n, time_limit=min(10, deadline - time.time()))
        s = _reciprocal_sum_float(improved)
        if s > best_sum:
            best_sum = s
            best_selected = improved
    
    # Phase 3: Simulated annealing
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 5
        sa_result = _simulated_annealing(list(best_selected), n, max_iters=100000, time_limit=remaining)
        s = _reciprocal_sum_float(sa_result)
        if s > best_sum:
            best_sum = s
            best_selected = sa_result
    
    # Final verification
    best_selected = sorted(best_selected)
    if not _is_valid_3ap_free(best_selected, n):
        # Fallback: simple greedy
        best_selected = _greedy_construct(range(1, n+1), n)
    
    return best_selected

# EVOLVE-BLOCK-END
