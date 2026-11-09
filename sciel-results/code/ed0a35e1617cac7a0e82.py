# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _check_addition(selected, x):
    """Check if x can be added to selected without violating the constraint.
    For each element a in selected ∪ {x}, no triple of DISTINCT OTHER elements
    from selected ∪ {x} should have a sum divisible by a.
    """
    s = selected + [x]
    n = len(s)
    if n < 4:
        return True
    # Precompute all triple sums from s
    triple_sums = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                triple_sums.append((s[i] + s[j] + s[k], i, j, k))
    # For each element a at index idx, check if any triple NOT containing idx has sum % a == 0
    for idx, a in enumerate(s):
        for tsum, i, j, k in triple_sums:
            if idx not in (i, j, k):
                if tsum % a == 0:
                    return False
    return True

def _greedy_with_order(order, selected, deadline):
    """Try to add elements in the given order to selected, respecting deadline."""
    for x in order:
        if time.time() > deadline:
            break
        if _check_addition(selected, x):
            selected.append(x)
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin (90s kill, 10s buffer)
    
    best = []
    
    # Initial greedy: try from 192 down
    order = list(range(192, 0, -1))
    selected = _greedy_with_order(order, [], deadline)
    if len(selected) > len(best):
        best = selected[:]
    
    # Try greedy from 1 up
    if time.time() < deadline:
        order = list(range(1, 193))
        selected = _greedy_with_order(order, [], deadline)
        if len(selected) > len(best):
            best = selected[:]
    
    # Randomized restarts with different orderings
    rng = random.Random(42)
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Create a random perturbation of the order
        order = list(range(1, 193))
        # Mix in some bias: larger numbers first with some randomness
        # Weight by inverse of number (smaller numbers more likely to be early)
        weights = [1.0 / v for v in order]
        shuffled = []
        remaining = order[:]
        remaining_weights = weights[:]
        for _ in range(len(remaining)):
            total_w = sum(remaining_weights)
            r = rng.random() * total_w
            cum = 0
            for i, w in enumerate(remaining_weights):
                cum += w
                if cum >= r:
                    shuffled.append(remaining[i])
                    remaining.pop(i)
                    remaining_weights.pop(i)
                    break
        selected = _greedy_with_order(shuffled, [], deadline)
        if len(selected) > len(best):
            best = selected[:]
    
    # One more attempt: greedy but with a local search improvement
    if time.time() < deadline - 5:
        # Try to improve by swapping: remove one, try to add a different one
        current = best[:]
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(len(current)):
                if time.time() > deadline - 2:
                    break
                removed = current[i]
                temp = current[:i] + current[i+1:]
                # Try to add some number not in temp
                for x in range(192, 0, -1):
                    if x in temp:
                        continue
                    if _check_addition(temp, x):
                        new_sel = temp + [x]
                        if len(new_sel) > len(current):
                            current = new_sel
                            improved = True
                            break
                        # Same size but different composition - keep if it allows more later
                        # For simplicity, only accept if strictly better
                if improved:
                    break
    
    return best
# EVOLVE-BLOCK-END
