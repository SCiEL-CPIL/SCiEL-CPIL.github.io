import time
import random
from math import lcm
from functools import reduce

UNIVERSE = list(range(1, 25))
L = reduce(lcm, UNIVERSE)
WEIGHTS = {x: L // x for x in UNIVERSE}

def is_valid(S):
    """Check if S is reciprocal-subsum-free."""
    if len(S) <= 1:
        return True
    S_list = list(S)
    for x in S_list:
        target = WEIGHTS[x]
        others = [WEIGHTS[y] for y in S_list if y != x]
        if sum(others) < target:
            continue
        # Check if target is a subset sum of others
        # Use DP with set, pruning sums > target
        achievable = {0}
        for w in others:
            if w > target:
                continue
            new_sums = set()
            for s in achievable:
                ns = s + w
                if ns <= target:
                    new_sums.add(ns)
            achievable |= new_sums
            if target in achievable:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    best = list(range(13, 25))  # seed fallback, size 12
    best_size = len(best)
    
    # Try greedy constructions with different orders
    random.seed(42)
    
    # Strategy 1: Greedy adding elements in various orders
    orders = [
        list(range(24, 0, -1)),  # descending
        list(range(1, 25)),      # ascending
        list(range(24, 0, -2)) + list(range(23, 0, -2)),  # evens then odds descending
        list(range(12, 0, -1)) + list(range(24, 12, -1)),  # mid-high then high
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        S = set()
        for x in order:
            if time.time() > deadline:
                break
            trial = S | {x}
            if is_valid(trial):
                S = trial
        if len(S) > best_size:
            best = list(S)
            best_size = len(S)
    
    # Strategy 2: Local search - try to add/remove/swap
    S = set(best)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding elements
        for x in UNIVERSE:
            if time.time() > deadline:
                break
            if x in S:
                continue
            trial = S | {x}
            if is_valid(trial):
                S = trial
                best = list(S)
                best_size = len(S)
                improved = True
                break
        # Try swaps
        if not improved:
            for x_in in list(S):
                if time.time() > deadline:
                    break
                for x_out in UNIVERSE:
                    if x_out in S:
                        continue
                    trial = (S - {x_in}) | {x_out}
                    if is_valid(trial) and len(trial) > best_size:
                        S = trial
                        best = list(S)
                        best_size = len(S)
                        improved = True
                        break
                if improved:
                    break
    
    return best
