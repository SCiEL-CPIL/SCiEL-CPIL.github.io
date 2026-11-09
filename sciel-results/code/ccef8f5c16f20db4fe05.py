import time
import random
from itertools import combinations

def is_valid_set(S):
    """
    Check if the set S is valid.
    S is a list of distinct integers.
    Condition: No element a in S is the mean of any subset T of S\{a} with |T| >= 2.
    """
    if len(S) < 3:
        return True
    
    S_set = set(S)
    n = len(S)
    
    for a in S:
        others = [x for x in S if x != a]
        n_others = len(others)
        
        for k in range(2, n_others + 1):
            target_sum = k * a
            for comb in combinations(others, k):
                if sum(comb) == target_sum:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    # Strategy: Use a 3-AP-free set construction as a strong heuristic.
    # A set with no 3-term arithmetic progression is "average-free" for k=2.
    # For larger k, we need additional checks, but 3-AP-free sets tend to be sparse
    # and often satisfy the full condition.
    
    # Better strategy: Greedy with bitset-based fast checking, but with multiple
    # random orderings to find a larger set.
    
    def construct_with_bitset(order, deadline):
        selected = []
        dp = [0] * 50  # dp[0] to dp[49]
        dp[0] = 1  # Empty set sum 0
        
        for x in order:
            if time.time() > deadline:
                break
            
            # Check 1: Is x the mean of any subset of 'selected'?
            valid = True
            for k in range(2, len(selected) + 1):
                target_sum = k * x
                if target_sum > 24576:
                    continue
                if (dp[k] >> target_sum) & 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check 2: Does adding x make any existing 'a' the mean of a subset containing x?
            for a in selected:
                for k in range(2, len(selected) + 2):
                    target_sum_for_T = k * a - x
                    if target_sum_for_T < 0:
                        continue
                    if target_sum_for_T > 24576:
                        continue
                    if (dp[k-1] >> target_sum_for_T) & 1:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                selected.append(x)
                for k in range(len(selected) - 1, -1, -1):
                    dp[k+1] |= (dp[k] << x)
                
                if len(selected) == 48:
                    break
        
        return selected
    
    # Try multiple orderings to find the largest valid set
    best_set = []
    
    # Ordering 1: Ascending
    order1 = list(range(1, 513))
    result1 = construct_with_bitset(order1, deadline)
    if len(result1) > len(best_set):
        best_set = result1
    if len(best_set) == 48:
        return best_set
    
    # Ordering 2: Descending
    order2 = list(range(512, 0, -1))
    result2 = construct_with_bitset(order2, deadline)
    if len(result2) > len(best_set):
        best_set = result2
    if len(best_set) == 48:
        return best_set
    
    # Ordering 3: Random (multiple tries)
    random.seed(42)
    for trial in range(20):
        if time.time() > deadline:
            break
        order3 = list(range(1, 513))
        random.shuffle(order3)
        result3 = construct_with_bitset(order3, deadline)
        if len(result3) > len(best_set):
            best_set = result3
            if len(best_set) == 48:
                break
    
    # Final verification of the best set
    if len(best_set) > 0 and not is_valid_set(best_set):
        # Fallback: use the first valid subset we can find
        # Try to repair by removing conflicting elements
        best_set = list(best_set)
        changed = True
        while changed and len(best_set) > 0:
            changed = False
            for a in list(best_set):
                others = [x for x in best_set if x != a]
                n_others = len(others)
                for k in range(2, min(n_others, 4) + 1):  # Check small k first
                    target_sum = k * a
                    found = False
                    for comb in combinations(others, k):
                        if sum(comb) == target_sum:
                            found = True
                            break
                    if found:
                        best_set.remove(a)
                        changed = True
                        break
    
    return best_set

if __name__ == "__main__":
    s = construct()
    print(len(s))
