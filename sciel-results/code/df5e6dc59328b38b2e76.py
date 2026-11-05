from fractions import Fraction
import random
import time

LCM = 5354228880
VALS = {x: LCM // x for x in range(1, 25)}

def can_make_sum(target, others):
    # others is a list of integers
    others.sort(reverse=True)
    n = len(others)
    
    # Memoization? Key: (idx, current_sum). current_sum can be large.
    # But depth is small.
    def dfs(idx, current_sum):
        if current_sum == target:
            return True
        if current_sum > target:
            return False
        if idx == n:
            return False
        
        # Pruning: if current_sum + sum(others[idx:]) < target, return False
        # Precompute suffix sums?
        # For simplicity, just recurse.
        
        if dfs(idx + 1, current_sum + others[idx]):
            return True
        if dfs(idx + 1, current_sum):
            return True
        return False
    
    return dfs(0, 0)

def is_valid(S):
    for a in S:
        target = VALS[a]
        others = [VALS[x] for x in S if x != a]
        if can_make_sum(target, others):
            return False
    return True

def construct():
    deadline = time.time() + 50
    best_size = 0
    best_set = set()
    
    # Start with a good seed
    # Try greedy with random permutations
    for _ in range(500):
        if time.time() > deadline: break
        elements = list(range(1, 25))
        random.shuffle(elements)
        current = set()
        for e in elements:
            if time.time() > deadline: break
            candidate = current | {e}
            if is_valid(candidate):
                current = candidate
        if len(current) > best_size:
            best_size = len(current)
            best_set = current
            
    return list(best_set)
