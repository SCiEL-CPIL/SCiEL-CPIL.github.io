import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55 seconds safety margin
    
    # Precompute: for each x in 1..256, which other values y in 1..256 cannot coexist with x
    # Two values x, y cannot coexist if adding y to a set containing x creates a divisibility conflict
    # among subset sums. We'll compute this dynamically during search.
    
    def is_valid_set(S):
        """Check if set S satisfies all conditions."""
        if len(set(S)) != len(S):
            return False
        if any(x < 1 or x > 256 for x in S):
            return False
        
        n = len(S)
        if n == 0:
            return True
        
        # Generate all nonempty subset sums with counts
        sum_count = {}
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += S[i]
            sum_count[s] = sum_count.get(s, 0) + 1
        
        # Check no two subsets have the same sum
        if any(c > 1 for c in sum_count.values()):
            return False
        
        # Check no two distinct sums divide each other
        sum_list = sorted(sum_count.keys())
        for i in range(len(sum_list)):
            for j in range(i + 1, len(sum_list)):
                a, b = sum_list[i], sum_list[j]
                if b % a == 0:
                    return False
        return True
    
    def get_subset_sums(S):
        """Return the set of all nonempty subset sums of S."""
        n = len(S)
        sums = set()
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += S[i]
            sums.add(s)
        return sums
    
    def can_add_element(S, x):
        """Check if x can be added to set S without violating constraints."""
        if x in S:
            return False
        if x < 1 or x > 256:
            return False
        
        new_S = S + [x]
        n = len(new_S)
        
        # Generate all nonempty subset sums with counts
        sum_count = {}
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += new_S[i]
            sum_count[s] = sum_count.get(s, 0) + 1
        
        # Check no two subsets have the same sum
        if any(c > 1 for c in sum_count.values()):
            return False
        
        # Check no two distinct sums divide each other
        sum_list = sorted(sum_count.keys())
        for i in range(len(sum_list)):
            for j in range(i + 1, len(sum_list)):
                a, b = sum_list[i], sum_list[j]
                if b % a == 0:
                    return False
        return True
    
    # Start with a known valid set
    best_set = [31, 32, 34]
    best_size = 3
    
    # Try multiple greedy strategies with different orderings
    orderings = []
    # Descending
    orderings.append(list(range(256, 0, -1)))
    # Ascending
    orderings.append(list(range(1, 257)))
    # Random shuffles
    for _ in range(20):
        perm = list(range(1, 257))
        random.shuffle(perm)
        orderings.append(perm)
    
    for ordering in orderings:
        if time.time() > deadline:
            break
        
        S = []
        for x in ordering:
            if x in S:
                continue
            if can_add_element(S, x):
                S.append(x)
            if time.time() > deadline:
                break
        
        if len(S) > best_size:
            best_set = S
            best_size = len(S)
    
    # Local search: try to improve by swapping elements
    def local_search(S, max_iters=50):
        S = list(S)
        best = list(S)
        best_size_local = len(S)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            
            # Try adding
            for x in range(256, 0, -1):
                if x not in S:
                    if can_add_element(S, x):
                        S.append(x)
                        best = list(S)
                        best_size_local = len(S)
                        improved = True
                        break
            
            # Try swapping
            if not improved:
                for i in range(len(S)):
                    for x in range(256, 0, -1):
                        if x not in S:
                            new_S = S[:i] + [x] + S[i+1:]
                            if is_valid_set(new_S):
                                S = new_S
                                best = list(S)
                                best_size_local = len(S)
                                improved = True
                                break
                    if improved:
                        break
            
            if not improved:
                break
        
        return best
    
    # Apply local search to best set
    if best_size < 15:
        improved = local_search(best_set, max_iters=30)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(best_set)
    
    # Final: try to add more to best_set
    for x in range(256, 0, -1):
        if x not in best_set:
            if can_add_element(best_set, x):
                best_set.append(x)
    
    return best_set
