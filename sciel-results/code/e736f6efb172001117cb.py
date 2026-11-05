import time
from math import lcm
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n = 128
    numbers = list(range(1, n + 1))
    
    # Precompute lcm table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[i][j] = lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
            
    # Find all bad triples
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))
    
    T = len(bad_triples)
    if T == 0:
        return numbers
        
    # Helper to check if adding x to a set is valid
    # Optimized: only check triples involving x
    # Precompute for each x, the list of (y, z) pairs in bad_triples containing x?
    # Or just check on the fly.
    
    # Greedy initialization
    counts = [0] * (n + 1)
    for (a, b, c) in bad_triples:
        counts[a] += 1
        counts[b] += 1
        counts[c] += 1
        
    active = [True] * T
    active_count = T
    removed = [False] * (n + 1)
    
    while active_count > 0:
        if time.time() > deadline - 10: # Reserve time for LS
            break
        best_x = -1
        best_cnt = -1
        for x in range(1, n + 1):
            if not removed[x] and counts[x] > best_cnt:
                best_cnt = counts[x]
                best_x = x
        if best_x == -1 or best_cnt <= 0:
            break
        removed[best_x] = True
        for i in range(T):
            if not active[i]:
                continue
            a, b, c = bad_triples[i]
            if a == best_x or b == best_x or c == best_x:
                active[i] = False
                active_count -= 1
                for x in (a, b, c):
                    if x != best_x:
                        counts[x] -= 1
                        
    result_set = set(x for x in range(1, n + 1) if not removed[x])
    
    # Local Search
    def can_add(x, s):
        # Check if x can be added to s
        # Need to check all pairs (y, z) in s
        # such that (x, y, z) is a bad triple
        # We can precompute for each x, the set of (y,z) pairs that form a bad triple with x.
        # But s changes.
        # Efficient check:
        # Iterate y in s.
        # Iterate z in s, z > y.
        # Check if (min(x,y,z), mid, max) is in bad_triples.
        # To make this fast, we can create a set of bad triples (sorted).
        pass
        
    # Precompute bad triple set for O(1) lookup
    bad_triple_set = set(bad_triples)
    
    def is_valid_addition(x, s):
        # s is a set of integers
        # Check all pairs in s
        s_list = list(s)
        len_s = len(s_list)
        for i in range(len_s):
            for j in range(i + 1, len_s):
                a = s_list[i]
                b = s_list[j]
                # Form triple (x, a, b)
                t = (min(x, a, b), min(max(x, a, b), max(x, a, b)), max(x, a, b))
                # Simpler:
                vals = sorted([x, a, b])
                if tuple(vals) in bad_triple_set:
                    return False
        return True

    # Improve result_set
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # 1. Try adding back single removed elements
        for x in range(1, n + 1):
            if time.time() > deadline - 5:
                break
            if removed[x]:
                if is_valid_addition(x, result_set):
                    result_set.add(x)
                    removed[x] = False
                    improved = True
        if time.time() > deadline - 5:
            break
            
        # 2. Try 1-for-2 swaps
        # Remove one kept element, add two removed elements
        kept_list = [x for x in range(1, n + 1) if not removed[x]]
        removed_list = [x for x in range(1, n + 1) if removed[x]]
        
        # Shuffle to avoid bias?
        random.shuffle(kept_list)
        random.shuffle(removed_list)
        
        for x in kept_list:
            if time.time() > deadline - 5:
                break
            # Try removing x
            result_set.remove(x)
            added_any = False
            
            # Try adding pairs from removed_list
            # To be efficient, we can just try to add one, then another
            # But they must be compatible with each other and the rest.
            
            # Strategy: Try to add y, then z.
            # y must be valid in result_set (without x).
            # z must be valid in result_set U {y}.
            
            for y in removed_list:
                if time.time() > deadline - 5:
                    break
                if not is_valid_addition(y, result_set):
                    continue
                
                # Tentatively add y
                result_set.add(y)
                
                for z in removed_list:
                    if z == y:
                        continue
                    if not is_valid_addition(z, result_set):
                        continue
                    
                    # If we can add both y and z, we have a 1-for-2 swap
                    result_set.add(z)
                    removed[y] = False
                    removed[z] = False
                    added_any = True
                    improved = True
                    break
                if added_any:
                    break
                else:
                    # Undo y
                    result_set.remove(y)
            
            if not added_any:
                # Put x back
                result_set.add(x)
            # else: x stays removed, y and z are added
        
    result_list = sorted(result_set)
    if not result_list:
        return [1]
        
    # Final validation
    for i in range(len(result_list)):
        for j in range(i + 1, len(result_list)):
            a = result_list[i]
            b = result_list[j]
            for k in range(j + 1, len(result_list)):
                c = result_list[k]
                if tuple(sorted([a, b, c])) in bad_triple_set:
                    # Invalid, should not happen if LS is correct
                    # Fallback to greedy only
                    return sorted(x for x in range(1, n + 1) if not removed[x])
                    
    return result_list
