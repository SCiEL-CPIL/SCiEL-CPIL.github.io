import time
import random
from itertools import combinations

START_TIME = time.time()
TIME_LIMIT = 55.0

def construct():
    global START_TIME
    START_TIME = time.time()
    best_size = 0
    best_list = []
    numbers = list(range(1, 193))
    
    while time.time() - START_TIME < TIME_LIMIT - 5:
        current_list = []
        triple_sums = set()
        pair_sums = [] # pair_sums[i] contains sums of pairs (j,k) from current_list where j!=i, k!=i
        
        # Choose a strategy: Ascending, Descending, or Random
        strategy = random.choice(['asc', 'desc', 'rand'])
        candidates = list(numbers)
        if strategy == 'asc':
            candidates.sort()
        elif strategy == 'desc':
            candidates.sort(reverse=True)
        else:
            random.shuffle(candidates)
            
        for x in candidates:
            if len(current_list) == 48:
                break
                
            ok = True
            
            # 1. x must not divide any existing triple sum
            for t in triple_sums:
                if t % x == 0:
                    ok = False
                    break
            if not ok:
                continue
                
            # 2. For every existing element a in current_list, 
            #    x + y must not be divisible by a for any y in current_list, y != a
            for idx, a in enumerate(current_list):
                # We need to check pairs (x, y) for all y in current_list, y != a
                # Note: pair_sums[idx] contains sums of (u, v) where u, v are in current_list, u!=idx, v!=idx.
                # We are adding x. New pairs involving x are (x, y).
                # We need (x + y) % a != 0.
                for y in current_list:
                    if y == a:
                        continue
                    if (x + y) % a == 0:
                        ok = False
                        break
                if not ok:
                    break
            
            if ok:
                current_list.append(x)
                n = len(current_list) - 1
                
                # Update triple_sums
                for i in range(n):
                    for j in range(i+1, n):
                        triple_sums.add(x + current_list[i] + current_list[j])
                
                # Update pair_sums
                # For each existing element a (index idx), add x + y to pair_sums[idx] if y != a
                for idx, a in enumerate(current_list[:-1]):
                    for y in current_list[:-1]:
                        if y != a:
                            pair_sums[idx].add(x + y)
                
                # Create pair_sums for x
                new_pairs = set()
                for i in range(n):
                    for j in range(i+1, n):
                        new_pairs.add(current_list[i] + current_list[j])
                pair_sums.append(new_pairs)
        
        if len(current_list) > best_size:
            best_size = len(current_list)
            best_list = list(current_list)
            if best_size == 48:
                break
                
        # Local Search: Try to improve current_list
        # If we have time, try to replace elements.
        # Simple heuristic: If we didn't reach 48, try removing a "bad" element and adding a "good" one.
        # Which element is bad? One that causes many conflicts?
        # Hard to compute quickly.
        # Let's just do random restarts.
        
    return best_list
