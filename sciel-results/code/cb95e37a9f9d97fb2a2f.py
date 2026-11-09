import time
import random
from math import gcd
from functools import reduce

UNIVERSE = list(range(1, 25))

# Precompute weights: w_i = L / i where L = lcm(1..24)
def _lcm(a, b):
    return a // gcd(a, b) * b

L = reduce(_lcm, UNIVERSE)
WEIGHTS = [0] + [L // i for i in UNIVERSE]  # 1-indexed

# For fast subset-sum check: given target T and a list of weights (excluding target element),
# check if any subset sums to T.
# Since L is large but we only care about sums <= target, and target = L//x,
# we can use a boolean array or set.

def is_valid(S):
    """Check if set S (set of ints 1..24) is reciprocal-subsum-free."""
    S_list = sorted(S)
    n = len(S_list)
    if n == 0:
        return True
    for i, x in enumerate(S_list):
        target = WEIGHTS[x]
        # Collect weights of all other elements
        others = []
        for j, y in enumerate(S_list):
            if j != i:
                others.append(WEIGHTS[y])
        # Quick check: if sum of others < target, skip
        if sum(others) < target:
            continue
        # Subset sum: can we make target from others?
        # Use a set of achievable sums, capped at target
        achievable = {0}
        for w in others:
            if w > target:
                continue
            new_sums = set()
            for s in achievable:
                ns = s + w
                if ns <= target:
                    new_sums.add(ns)
            achievable.update(new_sums)
            if target in achievable:
                return False
    return True


def find_first_conflict(S):
    """Return (x, witness_list) for first conflict, or None."""
    S_list = sorted(S)
    for i, x in enumerate(S_list):
        target = WEIGHTS[x]
        others_w = []
        others_e = []
        for j, y in enumerate(S_list):
            if j != i:
                others_w.append(WEIGHTS[y])
                others_e.append(y)
        if sum(others_w) < target:
            continue
        # Find a subset that sums to target via backtracking
        # Sort by weight descending for better pruning
        idx_order = sorted(range(len(others_w)), key=lambda k: -others_w[k])
        
        def backtrack(pos, cur_sum, chosen):
            if cur_sum == target:
                return chosen
            if pos == len(idx_order) or cur_sum > target:
                return None
            k = idx_order[pos]
            # Try include
            if cur_sum + others_w[k] <= target:
                res = backtrack(pos + 1, cur_sum + others_w[k], chosen + [others_e[k]])
                if res is not None:
                    return res
            # Try exclude
            res = backtrack(pos + 1, cur_sum, chosen)
            if res is not None:
                return res
            return None
        
        witness = backtrack(0, 0, [])
        if witness is not None:
            return (x, witness)
    return None


def greedy_construct():
    """Greedy: add elements in order, keeping validity."""
    S = set()
    for x in sorted(UNIVERSE):
        new_S = S | {x}
        if is_valid(new_S):
            S = new_S
    return S


def local_search(S, deadline):
    """Try to improve S by adding elements, then swapping."""
    S = set(S)
    remaining = [x for x in UNIVERSE if x not in S]
    
    # Phase A: Try to add any element
    improved = True
    while improved and time.time() < deadline:
        improved = False
        random.shuffle(remaining)
        for x in remaining:
            if time.time() > deadline:
                break
            new_S = S | {x}
            if is_valid(new_S):
                S = new_S
                remaining.remove(x)
                improved = True
    
    # Phase B: Try 1-for-1 swaps to enable further additions
    if time.time() < deadline:
        for _ in range(200):  # limited iterations
            if time.time() > deadline:
                break
            in_S = list(S)
            out_S = [x for x in UNIVERSE if x not in S]
            if not in_S or not out_S:
                break
            random.shuffle(in_S)
            random.shuffle(out_S)
            swap_done = False
            for a in in_S:
                for b in out_S:
                    if time.time() > deadline:
                        break
                    new_S = (S - {a}) | {b}
                    if is_valid(new_S):
                        # Check if this swap enables adding something
                        rem = [x for x in UNIVERSE if x not in new_S]
                        for c in rem:
                            if is_valid(new_S | {c}):
                                S = new_S | {c}
                                swap_done = True
                                break
                        if not swap_done:
                            # Accept swap even without immediate addition if size same
                            # Actually, we only want to swap if it helps. Skip for now.
                            pass
                        if swap_done:
                            break
                if swap_done:
                    break
            if not swap_done:
                break
    
    return S


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Start with a good greedy solution
    best_S = greedy_construct()
    
    # Multiple restarts with random removal + local search
    for restart in range(30):
        if time.time() > deadline - 5:
            break
        
        # Random initial set: start from all, remove random elements to resolve conflicts
        S = set(UNIVERSE)
        
        # Remove elements to resolve conflicts
        max_removals = 12
        for _ in range(max_removals):
            if time.time() > deadline - 3:
                break
            conflict = find_first_conflict(S)
            if conflict is None:
                break
            x, witness = conflict
            # Prefer removing the target x (kills this specific equation)
            # But sometimes removing a witness is better
            # Heuristic: remove x with probability 0.7, else remove a random witness
            if random.random() < 0.7:
                S.remove(x)
            else:
                w = random.choice(witness)
                S.remove(w)
        
        # Local search to improve
        S = local_search(S, deadline)
        
        if len(S) > len(best_S):
            best_S = set(S)
    
    # Final polish: try to add any remaining element
    if time.time() < deadline - 2:
        remaining = [x for x in UNIVERSE if x not in best_S]
        random.shuffle(remaining)
        for x in remaining:
            if time.time() > deadline - 1:
                break
            new_S = best_S | {x}
            if is_valid(new_S):
                best_S = new_S
    
    return list(best_S)
