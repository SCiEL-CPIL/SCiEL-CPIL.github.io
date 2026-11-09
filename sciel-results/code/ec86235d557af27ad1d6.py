import time
import random
import sys

def construct():
    """
    Finds a set of distinct integers in 1..256 maximizing the size,
    such that all non-empty subset sums are distinct and no two distinct
    subset sums divide one another.
    """
    start_time = time.time()
    time_limit = 55  # Keep 5s margin for evaluator overhead
    
    best_set = [31, 32, 34]
    best_size = len(best_set)
    
    def is_valid_set(S):
        """
        Checks if set S is valid.
        1. All subset sums are distinct.
        2. No two distinct subset sums divide one another.
        """
        if not S:
            return True
        
        sums = {0}
        for x in S:
            new_sums = set()
            for s in sums:
                new_sums.add(s + x)
            sums = sums | new_sums
        
        non_empty_sums = {s for s in sums if s != 0}
        
        # Check distinctness
        if len(non_empty_sums) != (1 << len(S)) - 1:
            return False
            
        # Check divisibility
        sum_list = list(non_empty_sums)
        n = len(sum_list)
        for i in range(n):
            u = sum_list[i]
            for j in range(n):
                if i == j:
                    continue
                v = sum_list[j]
                if u == v:
                    continue
                if v % u == 0 or u % v == 0:
                    return False
        return True

    def try_add_incremental(S, sums_set, x):
        """
        Checks if x can be added to S.
        Returns (new_sums_set, valid)
        S is a list of elements.
        sums_set is the set of all subset sums of S (including 0).
        x is the new element.
        """
        # 1. Generate new sums
        new_sums = set()
        for s in sums_set:
            new_sums.add(s + x)
        
        # 2. Check distinctness (intersection must be empty)
        if sums_set & new_sums:
            return None, False
            
        # 3. Check divisibility
        non_empty_sums = {s for s in sums_set if s != 0}
        non_empty_new = {s for s in new_sums if s != 0}
        
        new_list = list(non_empty_new)
        old_list = list(non_empty_sums)
        
        # Check internal new_sums
        n = len(new_list)
        for i in range(n):
            u = new_list[i]
            for j in range(n):
                if i == j:
                    continue
                v = new_list[j]
                if v % u == 0 or u % v == 0:
                    return None, False
        
        # Check cross new_sums vs old_sums
        for u in new_list:
            for v in old_list:
                if v % u == 0 or u % v == 0:
                    return None, False
            
        # If valid, return the new sums set
        combined_sums = sums_set | new_sums
        return combined_sums, True

    # Search Strategy
    random.seed(42)
    
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Choose strategy
        if restart % 5 == 0:
            candidates = list(range(256, 0, -1))
        elif restart % 5 == 1:
            candidates = list(range(1, 257))
        elif restart % 5 == 2:
            candidates = list(range(1, 257))
            random.shuffle(candidates)
        elif restart % 5 == 3:
            candidates = list(range(1, 257))
            random.shuffle(candidates)
        else:
            candidates = list(range(1, 257))
            random.shuffle(candidates)

        S = []
        sums_set = {0}
        
        for x in candidates:
            if time.time() - start_time > time_limit:
                break
                
            new_sums, valid = try_add_incremental(S, sums_set, x)
            if valid:
                S.append(x)
                sums_set = new_sums
                
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S[:]
                    
                # Stop adding if we reach a size where the next check would be too slow
                if len(S) >= 14:
                    break

    # Final verification of best_set
    if len(best_set) <= 12:
        if not is_valid_set(best_set):
            best_set = [31, 32, 34]
            best_size = 3
    else:
        if time.time() - start_time < time_limit - 5:
             if not is_valid_set(best_set):
                 best_set = [31, 32, 34]
                 best_size = 3

    return list(best_set)

if __name__ == "__main__":
    res = construct()
    print(res)
