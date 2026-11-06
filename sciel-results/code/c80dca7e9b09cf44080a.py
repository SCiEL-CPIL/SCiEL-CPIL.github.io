import time
import math
import random

_START = time.time()
_DEADLINE = 55.0  # Safety margin

N = 24

def _time_left():
    return _DEADLINE - (time.time() - _START)

_L = 1
for k in range(1, 25):
    _L = _L * k // math.gcd(_L, k)
_W = [_L // k for k in range(1, 25)]

def _is_subset_sum(target, items):
    if target < 0:
        return False
    if target == 0:
        return True
    n = len(items)
    if n == 0:
        return False
    
    # Early exit: if sum of all items < target, impossible
    # if min item > target, impossible (unless target is 0)
    if not items:
        return False
    
    # Sort items to potentially prune?
    # Use meet in the middle
    half = n // 2
    left_items = items[:half]
    right_items = items[half:]
    
    left_sums = {0}
    for item in left_items:
        # Optimization: if item > target, we can't add it to any sum <= target
        # But we need sums <= target.
        new_sums = set()
        for s in left_sums:
            ns = s + item
            if ns <= target:
                new_sums.add(ns)
        left_sums |= new_sums
        # Prune sums > target? We already did.
        
    right_sums = {0}
    for item in right_items:
        new_sums = set()
        for s in right_sums:
            ns = s + item
            if ns <= target:
                new_sums.add(ns)
        right_sums |= new_sums
        
    # Check intersection
    # Iterate over smaller set
    if len(left_sums) < len(right_sums):
        for ls in left_sums:
            if (target - ls) in right_sums:
                return True
    else:
        for rs in right_sums:
            if (target - rs) in left_sums:
                return True
    return False

def _is_valid_addition(candidate_set, elem):
    target_weight = _W[elem - 1]
    other_weights = [_W[x - 1] for x in candidate_set]
    
    if _is_subset_sum(target_weight, other_weights):
        return False
    
    for x in candidate_set:
        target_w = _W[x - 1]
        others = [y for y in candidate_set if y != x]
        others.append(elem)
        other_w = [_W[y - 1] for y in others]
        if _is_subset_sum(target_w, other_w):
            return False
            
    return True

def _is_fully_valid(candidate_set):
    n = len(candidate_set)
    for i, x in enumerate(candidate_set):
        target_w = _W[x - 1]
        others = [y for y in candidate_set if y != x]
        other_w = [_W[y - 1] for y in others]
        if _is_subset_sum(target_w, other_w):
            return False
    return True

def _greedy_descending():
    result = []
    for elem in range(24, 0, -1):
        if _is_valid_addition(result, elem):
            result.append(elem)
    return result

def construct():
    random.seed(42)
    initial = _greedy_descending()
    best_set = list(initial)
    best_size = len(best_set)
    
    # Simple local search with strict time limit
    current = set(initial)
    temp = 1.0
    
    while _time_left() > 5.0:
        if not current:
            break
            
        move_type = random.random()
        
        if move_type < 0.5:
            available = [x for x in range(1, 25) if x not in current]
            if not available:
                continue
            elem = random.choice(available)
            if _is_valid_addition(list(current), elem):
                current.add(elem)
                if len(current) > best_size:
                    best_set = list(current)
                    best_size = len(best_set)
        elif move_type < 0.8:
            if len(current) > 1:
                elem = random.choice(list(current))
                current.remove(elem)
        else:
            if len(current) > 0:
                removed = random.choice(list(current))
                available = [x for x in range(1, 25) if x not in current]
                if available:
                    added = random.choice(available)
                    new_set = (current - {removed}) | {added}
                    if _is_fully_valid(list(new_set)):
                        current = new_set
                        if len(current) > best_size:
                            best_set = list(current)
                            best_size = len(best_set)
                            
        temp *= 0.99
        if temp < 0.01:
            temp = 0.01

    return best_set
