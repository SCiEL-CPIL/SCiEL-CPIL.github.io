import time
import random
from math import gcd
from functools import reduce

_START = time.time()
_DEADLINE = 85.0

def _deadline_hit():
    return (time.time() - _START) > _DEADLINE

def construct():
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    V = reduce(lcm, range(1, 23))
    weights = [V // i for i in range(1, 23)]
    
    def search_size(target_size):
        def dfs(current_indices, current_sums, available_indices, order_strategy):
            if _deadline_hit():
                return None
            
            if len(current_indices) == target_size:
                return current_indices[:]
            
            if len(current_indices) + len(available_indices) < target_size:
                return None
            
            # Determine order
            if order_strategy == 'desc':
                candidates = sorted(available_indices, reverse=True)
            elif order_strategy == 'asc':
                candidates = sorted(available_indices)
            else:
                candidates = available_indices[:]
                random.shuffle(candidates)
            
            for idx in candidates:
                w = weights[idx]
                
                # Check collision
                # We need to check if any s + w is in current_sums
                # And construct new_sums if no collision
                
                has_collision = False
                new_sums = set()
                for s in current_sums:
                    ns = s + w
                    if ns in current_sums:
                        has_collision = True
                        break
                    new_sums.add(ns)
                
                if not has_collision:
                    new_indices = current_indices + [idx]
                    new_avail = [a for a in available_indices if a != idx]
                    
                    # Recurse
                    res = dfs(new_indices, current_sums.union(new_sums), new_avail, order_strategy)
                    if res is not None:
                        return res
            return None

        # Try a few strategies
        # 'desc' and 'asc' are deterministic. 'random' varies.
        # We'll try 'desc' first, then 'asc', then a few 'random' restarts.
        
        res = dfs([], {0}, list(range(22)), 'desc')
        if res is not None:
            return res
            
        res = dfs([], {0}, list(range(22)), 'asc')
        if res is not None:
            return res
            
        # Try random restarts
        for _ in range(20):
            if _deadline_hit():
                return None
            res = dfs([], {0}, list(range(22)), 'random')
            if res is not None:
                return res
                
        return None

    best = []
    for size in range(22, 0, -1):
        if _deadline_hit():
            break
        res = search_size(size)
        if res is not None:
            best = res
            break
            
    if best:
        return [i + 1 for i in best]
    else:
        return [1, 2, 3]
