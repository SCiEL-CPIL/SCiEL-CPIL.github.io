import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = 256
    
    def is_valid_addition(x, S_list):
        """Check if adding x to S_list (sorted list) maintains the property.
        S_list is a sorted list of current elements.
        """
        # For all triples (a,b,c) in S_list ∪ {x} with a < b <= c, a ∤ (b+c)
        # New triples must involve x.
        
        # Case A: x is the smallest (x < b <= c, b,c in S_list)
        # Need: for all b <= c in S_list with b > x, (b+c) % x != 0
        # Since S_list is sorted, find elements > x
        for i in range(len(S_list)):
            if S_list[i] > x:
                b = S_list[i]
                for j in range(i, len(S_list)):
                    c = S_list[j]
                    if (b + c) % x == 0:
                        return False
        
        # Case B: x is the middle (a < x <= c, a,c in S_list, a < x <= c)
        # Need: for all a in S_list with a < x, and c in S_list with c >= x, (x+c) % a != 0
        for a in S_list:
            if a < x:
                for c in S_list:
                    if c >= x:
                        if (x + c) % a == 0:
                            return False
        
        # Case C: x is the largest (a < b < x, a,b in S_list)
        # Need: for all a < b in S_list with b < x, (b+x) % a != 0
        for i in range(len(S_list)):
            if S_list[i] < x:
                a = S_list[i]
                for j in range(i + 1, len(S_list)):
                    b = S_list[j]
                    if b < x:
                        if (b + x) % a == 0:
                            return False
        
        return True
    
    def greedy_from_large_to_small():
        S = []
        for x in range(n, 0, -1):
            if is_valid_addition(x, S):
                S.append(x)
        return S
    
    def greedy_random_order():
        S = []
        order = list(range(1, n + 1))
        random.shuffle(order)
        for x in order:
            if is_valid_addition(x, S):
                S.append(x)
        return S
    
    def local_search_improve(S):
        """Try to add excluded elements in descending order."""
        S_set = set(S)
        excluded = sorted(set(range(1, n + 1)) - S_set, reverse=True)
        S_list = sorted(S)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            new_excluded = []
            for x in excluded:
                if is_valid_addition(x, S_list):
                    S_list.append(x)
                    S_list.sort()
                    improved = True
                else:
                    new_excluded.append(x)
            excluded = new_excluded
        return S_list
    
    best_S = []
    
    # Strategy 1: Greedy from large to small
    S1 = greedy_from_large_to_small()
    best_S = S1
    
    # Improve with local search
    if time.time() < deadline - 10:
        S1_imp = local_search_improve(S1)
        if len(S1_imp) > len(best_S):
            best_S = S1_imp
    
    # Strategy 2: Random greedy restarts
    if time.time() < deadline - 10:
        for trial in range(500):
            if time.time() > deadline - 5:
                break
            S2 = greedy_random_order()
            if len(S2) > len(best_S):
                best_S = S2
            # Quick local search on promising solutions
            if len(S2) >= len(best_S) and time.time() < deadline - 10:
                S2_imp = local_search_improve(S2)
                if len(S2_imp) > len(best_S):
                    best_S = S2_imp
    
    # Final local search
    if time.time() < deadline - 5:
        best_S = local_search_improve(best_S)
    
    return sorted(best_S)
