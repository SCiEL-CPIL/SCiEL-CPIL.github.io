"""Optimized 3-AP-free subset maximizer for reciprocal sum."""
# EVOLVE-BLOCK-START
import time
import random

def _has_3ap(selected, x):
    """Check if adding x to selected creates a 3-term AP. O(|selected|)."""
    # For each a in selected, check if x forms a 3-AP with a and some b in selected
    # Case 1: a, x, b -> b = 2x - a
    # Case 2: x, a, b -> b = 2a - x  
    # Case 3: a, b, x -> b = (a+x)/2
    for a in selected:
        b = 2 * x - a
        if b != a and b != x and b in selected:
            return True
        b = 2 * a - x
        if b != a and b != x and b in selected:
            return True
        if (a + x) % 2 == 0:
            b = (a + x) // 2
            if b != a and b != x and b in selected:
                return True
    return False

def _recip_sum(selected):
    return sum(1.0 / x for x in selected)

def _greedy_ascending(selected=None):
    """Greedy: iterate x from 1 to 256, add if no 3-AP."""
    if selected is None:
        selected = set()
    for x in range(1, 257):
        if x not in selected and not _has_3ap(selected, x):
            selected.add(x)
    return selected

def _greedy_with_order(order, selected=None):
    """Greedy: iterate in given order, add if no 3-AP."""
    if selected is None:
        selected = set()
    for x in order:
        if x not in selected and not _has_3ap(selected, x):
            selected.add(x)
    return selected

def _local_search_improve(selected, deadline):
    """Aggressive local search: additions, swaps, repeated."""
    best_set = set(selected)
    best_sum = _recip_sum(selected)
    current = set(selected)
    
    while time.time() < deadline:
        improved = False
        
        # Phase A: Try adding any missing element
        for x in range(1, 257):
            if time.time() > deadline:
                break
            if x not in current:
                if not _has_3ap(current, x):
                    current.add(x)
                    improved = True
                    s = _recip_sum(current)
                    if s > best_sum:
                        best_sum = s
                        best_set = set(current)
        
        # Phase B: Try 1-for-1 swaps (remove y, add x where x < y)
        for x in range(1, 257):
            if time.time() > deadline:
                break
            if x in current:
                continue
            for y in sorted(current, reverse=True):
                if time.time() > deadline:
                    break
                if y <= x:
                    break
                # Try removing y and adding x
                trial = current - {y}
                if not _has_3ap(trial, x):
                    trial.add(x)
                    s = _recip_sum(trial)
                    if s > best_sum:
                        best_sum = s
                        best_set = set(trial)
                        current = set(trial)
                        improved = True
        
        # Phase C: Try 1-for-2 swaps (remove y1,y2, add x) where x < min(y1,y2)
        # This is expensive, so only do if time permits
        if time.time() < deadline - 5:
            for x in range(1, 257):
                if time.time() > deadline:
                    break
                if x in current:
                    continue
                # Try removing two elements and adding x
                # Only try removing pairs where both > x
                candidates = [y for y in current if y > x]
                for i in range(len(candidates)):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, len(candidates)):
                        if time.time() > deadline:
                            break
                        y1, y2 = candidates[i], candidates[j]
                        trial = current - {y1, y2}
                        if not _has_3ap(trial, x):
                            trial.add(x)
                            s = _recip_sum(trial)
                            if s > best_sum:
                                best_sum = s
                                best_set = set(trial)
                                current = set(trial)
                                improved = True
        
        if not improved:
            break
    
    return best_set

def _perturb_and_refine(base_set, deadline, rng):
    """Perturb the set by removing some random elements, then refine."""
    perturbed = set(base_set)
    # Remove 3-8 random elements
    n_remove = rng.randint(3, 8)
    to_remove = rng.sample(sorted(perturbed), min(n_remove, len(perturbed)))
    for y in to_remove:
        perturbed.discard(y)
    
    # Refill greedily
    for x in range(1, 257):
        if x not in perturbed and not _has_3ap(perturbed, x):
            perturbed.add(x)
    
    # Local search
    return _local_search_improve(perturbed, deadline)

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s limit
    
    rng = random.Random(42)
    
    # Initial greedy ascending
    current = _greedy_ascending()
    best_set = set(current)
    best_sum = _recip_sum(current)
    
    # Multiple local search passes with different perturbations
    n_restarts = 0
    while time.time() < deadline - 10:
        n_restarts += 1
        
        # Try different perturbation strategies
        if n_restarts % 3 == 1:
            # Standard local search on current best
            candidate = _local_search_improve(best_set, deadline - 5)
        elif n_restarts % 3 == 2:
            # Perturb and refine
            candidate = _perturb_and_refine(best_set, deadline - 5, rng)
        else:
            # Greedy with random order, then refine
            order = list(range(1, 257))
            rng.shuffle(order)
            # Bias toward small numbers
            order.sort(key=lambda x: x + rng.random() * 50)
            candidate = _greedy_with_order(order)
            candidate = _local_search_improve(candidate, deadline - 5)
        
        s = _recip_sum(candidate)
        if s > best_sum:
            best_sum = s
            best_set = set(candidate)
    
    # Final refinement with remaining time
    if time.time() < deadline:
        final = _local_search_improve(best_set, deadline)
        if _recip_sum(final) > best_sum:
            best_set = final
            best_sum = _recip_sum(final)
    
    return list(best_set)
# EVOLVE-BLOCK-END
