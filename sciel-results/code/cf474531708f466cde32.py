"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    N = 128
    # Precompute GCD matrix for speed
    # gcd_matrix[i][j] = gcd(i+1, j+1)
    gcd_matrix = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_matrix[i][j] = g
            gcd_matrix[j][i] = g

    def is_valid_triple(a, b, c):
        """
        Returns True if the triple (a, b, c) is VALID (i.e., does NOT have equal pairwise gcds).
        The task requires NO three distinct elements whose three pairwise gcds are equal.
        So we want to avoid the case where gcd(a,b) == gcd(a,c) == gcd(b,c).
        """
        g_ab = gcd_matrix[a][b]
        g_ac = gcd_matrix[a][c]
        g_bc = gcd_matrix[b][c]
        # If all three are equal, it's a forbidden triple.
        if g_ab == g_ac == g_bc:
            return False
        return True

    def can_add(element, selected_set):
        """
        Check if 'element' can be added to 'selected_set' without creating a forbidden triple.
        selected_set is a set of integers.
        """
        for a in selected_set:
            for b in selected_set:
                if a == b:
                    continue
                if a > b:
                    a, b = b, a
                # We need to check if (a, b, element) forms a forbidden triple.
                # Since we iterate over all pairs in the existing set, 
                # we cover all combinations of two existing elements with the new one.
                if not is_valid_triple(a, b, element):
                    return False
        return True

    def get_size(selected_set):
        return len(selected_set)

    # Strategy: 
    # 1. Generate several random initial sets using different heuristics.
    # 2. Apply local search to maximize size.
    # 3. Keep the best set found.

    best_set = set()
    
    # Heuristic 1: Greedy by smallest number
    def greedy_smallest():
        s = set()
        for x in range(1, N + 1):
            if time.time() > deadline:
                break
            if can_add(x, s):
                s.add(x)
        return s

    # Heuristic 2: Greedy by largest number
    def greedy_largest():
        s = set()
        for x in range(N, 0, -1):
            if time.time() > deadline:
                break
            if can_add(x, s):
                s.add(x)
        return s

    # Heuristic 3: Random order
    def greedy_random():
        s = set()
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline:
                break
            if can_add(x, s):
                s.add(x)
        return s

    # Heuristic 4: Numbers with few prime factors (coprime-like)
    def greedy_coprime_like():
        s = set()
        # Sort by number of prime factors (ascending)
        def num_prime_factors(n):
            count = 0
            d = 2
            nn = n
            while d * d <= nn:
                while nn % d == 0:
                    count += 1
                    nn //= d
                d += 1
            if nn > 1:
                count += 1
            return count
        
        candidates = sorted(range(1, N + 1), key=num_prime_factors)
        for x in candidates:
            if time.time() > deadline:
                break
            if can_add(x, s):
                s.add(x)
        return s

    # Local Search: Try to add elements that are currently excluded, or swap.
    def local_search(initial_set):
        s = set(initial_set)
        improved = True
        # We will try to add elements from 1..N that are not in s
        # and if that fails, try swapping: remove one element from s, add two? 
        # Simpler: just try to add any element from 1..N not in s.
        # If we can't add, try removing one element and see if we can add two others?
        # Given the constraint is "no equal gcd triples", removing an element might allow adding multiple.
        
        # Basic hill climbing: try to add any element not in s.
        while improved and time.time() < deadline:
            improved = False
            for x in range(1, N + 1):
                if x in s:
                    continue
                if can_add(x, s):
                    s.add(x)
                    improved = True
                    break # Restart loop to check again from beginning? 
                         # Or continue to add more? 
                         # If we add x, it might block y. But adding x increases size.
                         # Let's just add as many as possible in one pass, then repeat.
            
            # If we didn't add anything in a full pass, try swaps?
            # Swaps are complex. Let's stick to addition for now, 
            # but also try random restarts of the addition order.
            
            if not improved:
                # Try a random shuffle of addition attempts?
                # Or try removing an element and see if we can add 2 back.
                # For simplicity and speed, let's just do random restarts of the greedy process 
                # with different seeds/orders in the main loop.
                pass
        
        return s

    # Run multiple greedy strategies
    strategies = [greedy_smallest, greedy_largest, greedy_coprime_like]
    for strat in strategies:
        if time.time() > deadline:
            break
        s = strat()
        s = local_search(s)
        if len(s) > len(best_set):
            best_set = s

    # Run several random greedy attempts
    for _ in range(50):
        if time.time() > deadline:
            break
        s = greedy_random()
        s = local_search(s)
        if len(s) > len(best_set):
            best_set = s

    # Final check: ensure non-empty
    if not best_set:
        # Fallback: just pick 1
        best_set = {1}
        
    return list(best_set)

# EVOLVE-BLOCK-END
