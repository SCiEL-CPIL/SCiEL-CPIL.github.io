"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    """
    Find a large subset of [1, 128] with no three distinct elements whose
    three pairwise gcds are equal.
    """
    start_time = time.time()
    deadline = start_time + 50  # 10s safety margin
    
    N = 128
    
    # Precompute GCD matrix for all pairs in [1, 128]
    # gcd_mat[i][j] = gcd(i, j)
    gcd_mat = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    def has_bad_triple_with_new(selected, new_elem):
        """
        Check if adding new_elem creates a bad triple with any pair in selected.
        A bad triple (a, b, c) has gcd(a,b) == gcd(a,c) == gcd(b,c).
        If we add 'new_elem' to 'selected', we only need to check triples involving 'new_elem'.
        For any pair (a, b) in 'selected', if gcd(a, new) == gcd(b, new) == gcd(a, b), it's a bad triple.
        """
        g_new_a = [0] * (N + 1)
        for a in selected:
            g_new_a[a] = gcd_mat[new_elem][a]
        
        for i in range(len(selected)):
            a = selected[i]
            g_an = g_new_a[a]
            for j in range(i + 1, len(selected)):
                b = selected[j]
                g_bn = g_new_a[b]
                if g_an == g_bn and g_an == gcd_mat[a][b]:
                    return True
        return False

    def greedy_build(order):
        """Build set greedily following given order."""
        selected = []
        selected_set = set()
        for x in order:
            if x not in selected_set:
                if not has_bad_triple_with_new(selected, x):
                    selected.append(x)
                    selected_set.add(x)
        return selected

    def local_search(set_list):
        """Try to improve the set via additions and swaps."""
        set_list = list(set_list)
        set_list_set = set(set_list)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try adding elements
            for x in range(1, N + 1):
                if x not in set_list_set:
                    if not has_bad_triple_with_new(set_list, x):
                        set_list.append(x)
                        set_list_set.add(x)
                        improved = True
                        break
            
            # 2. Try swaps: remove one, add another
            if not improved and time.time() < deadline:
                for i in range(len(set_list)):
                    removed = set_list[i]
                    # Create a temporary list without 'removed'
                    # Optimization: Instead of creating a new list every time, 
                    # we can just check if adding 'x' is valid in the set without 'removed'.
                    # But has_bad_triple_with_new takes a list. 
                    # To keep it fast, we'll do a lighter check or just accept the cost 
                    # since N is small (max 128).
                    
                    # We check if we can swap 'removed' with 'x'
                    # Condition: 
                    # 1. 'x' is not in set_list
                    # 2. The set (set_list - {removed}) + {x} is valid.
                    #    This means:
                    #    a. No bad triple in (set_list - {removed}) (already true)
                    #    b. No bad triple involving 'x' and pairs in (set_list - {removed})
                    
                    # We can reuse the logic: 
                    # Check if 'x' conflicts with any pair in (set_list - {removed})
                    
                    # To avoid O(N^2) inside the swap loop which is O(N*M), 
                    # we limit the number of swap attempts per iteration.
                    
                    candidate_set = set_list[:i] + set_list[i+1:]
                    
                    # Optimization: Only try a few candidates for swap to save time
                    # Randomly pick a few candidates or iterate through all?
                    # Given N=128, iterating all is 128 checks.
                    # Each check is O(N^2) in worst case for has_bad_triple_with_new? 
                    # No, has_bad_triple_with_new is O(K^2) where K is len(candidate_set).
                    # So swap check is O(K^2). Total swap search O(N * M * K^2).
                    # If K ~ 60, K^2 ~ 3600. N*M ~ 128*128 ~ 16000. Total ~ 50M ops.
                    # This might be too slow for Python if done many times.
                    
                    # Let's restrict swap attempts to a subset or use a faster heuristic.
                    # Alternative: Just try to add. If stuck, do a random restart.
                    # The previous code's local search was likely too slow.
                    # Let's simplify: Only do swaps if we haven't improved in a while?
                    # Or just do a limited number of swaps.
                    
                    for x in range(1, N + 1):
                        if x in set_list_set:
                            continue
                        if not has_bad_triple_with_new(candidate_set, x):
                            set_list = candidate_set + [x]
                            set_list_set.remove(removed)
                            set_list_set.add(x)
                            improved = True
                            break
                    if improved:
                        break
                    if time.time() > deadline:
                        break
        
        return set_list

    best_set = []
    
    # Strategy 1: Greedy with natural order
    s1 = greedy_build(range(1, N + 1))
    s1 = local_search(s1)
    if len(s1) > len(best_set):
        best_set = s1
    
    # Strategy 2: Greedy with reverse order
    if time.time() < deadline:
        s2 = greedy_build(range(N, 0, -1))
        s2 = local_search(s2)
        if len(s2) > len(best_set):
            best_set = s2
            
    # Strategy 3: Random restarts
    if time.time() < deadline:
        for _ in range(10):
            if time.time() > deadline:
                break
            random_order = list(range(1, N + 1))
            random.shuffle(random_order)
            s = greedy_build(random_order)
            s = local_search(s)
            if len(s) > len(best_set):
                best_set = s

    # Ensure nonempty
    if not best_set:
        best_set = [1]
    
    return best_set
# EVOLVE-BLOCK-END
