import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 80.0  # 10s margin before 90s limit
    
    N = 256
    # Candidate set: Integers 2..256
    # We suspect the upper half is a good starting point
    # Let's try to optimize the set S
    
    # Precompute divisibility
    # divs[a] = list of b such that a divides b
    divs = [[] for _ in range(N + 1)]
    for a in range(2, N + 1):
        for b in range(a, N + 1, a):
            divs[a].append(b)
            
    # Function to check if a set S is valid
    def is_valid(S):
        S_list = list(S)
        n = len(S_list)
        # If n < 3, valid
        if n < 3:
            return True
        # Check all triples
        # Optimization: for each a, check if a divides b*c for any b,c in S, b!=c, b!=a, c!=a
        for a in S_list:
            # Get divisors of a in S? No, a divides b*c.
            # This is hard to check directly.
            # Just check all pairs b,c
            # O(N^3)
            for i in range(n):
                b = S_list[i]
                if b == a: continue
                for j in range(i + 1, n):
                    c = S_list[j]
                    if c == a: continue
                    if b == c: continue # distinct b,c
                    # Check a | b*c
                    if (b * c) % a == 0:
                        return False
        return True

    # Greedy Deletion
    def greedy_delete(initial_set):
        S = list(initial_set)
        # Count conflicts for each element
        # We recompute from scratch each time for simplicity, N is small
        while True:
            if time.time() > deadline:
                return S
            
            # Find all conflicts
            conflicts = {}
            for a in S:
                conflicts[a] = 0
                
            S_list = S
            n = len(S_list)
            for i in range(n):
                a = S_list[i]
                for j in range(i + 1, n):
                    b = S_list[j]
                    for k in range(j + 1, n):
                        c = S_list[k]
                        # Check if any divides product of other two
                        if (b * c) % a == 0:
                            conflicts[a] += 1
                            conflicts[b] += 1
                            conflicts[c] += 1
                        if (a * c) % b == 0:
                            conflicts[a] += 1
                            conflicts[c] += 1
                            conflicts[b] += 1
                        if (a * b) % c == 0:
                            conflicts[a] += 1
                            conflicts[b] += 1
                            conflicts[c] += 1
            
            # If no conflicts, done
            if all(v == 0 for v in conflicts.values()):
                return S
            
            # Remove element with max conflicts
            max_conf = -1
            elem_to_remove = None
            for elem, cnt in conflicts.items():
                if cnt > max_conf:
                    max_conf = cnt
                    elem_to_remove = elem
            
            if elem_to_remove in S:
                S.remove(elem_to_remove)
            else:
                return S

    # Local Search: Try to add elements back or swap
    def local_search(S):
        S = list(S)
        # Try to add elements from [2..256] not in S
        # If adding x makes it invalid, we can't just add it.
        # We can try to remove one element y from S and add x if it results in a valid set of same size or larger?
        # Since we want max size, we just try to add.
        # If we can't add, we are stuck.
        
        # Better: Start with a large set, delete to make valid.
        # Then try to "recover" by swapping.
        
        # For now, just return S
        return S

    # Strategy:
    # 1. Start with upper half [129..256]
    # 2. Greedy delete
    # 3. Try to add primes from [2..128] if possible
    # 4. Try random restarts with different initial sets or tie-breaking
    
    best_set = []
    
    # Initial Set 1: Upper Half
    initial1 = list(range(129, 257))
    S1 = greedy_delete(initial1)
    best_set = S1
    
    # Initial Set 2: All Primes
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5)+1):
            if n % i == 0: return False
        return True
        
    primes = [x for x in range(2, 257) if is_prime(x)]
    S2 = greedy_delete(primes) # Should be same as primes
    if len(S2) > len(best_set):
        best_set = S2

    # Initial Set 3: Random subset of all numbers?
    # Or numbers with few divisors?
    
    # Let's try a few random restarts of greedy delete on the full set
    # But full set is 255 elements, greedy delete might be slow.
    # Upper half is 128, much faster.
    
    # Try to expand best_set by adding elements
    # Check if we can add any element from [2..256] not in best_set
    current_set = set(best_set)
    for x in range(2, 257):
        if x in current_set:
            continue
        # Try adding x
        new_set = list(current_set) + [x]
        if is_valid(new_set):
            current_set.add(x)
            best_set = list(current_set)
            
    # Try local search: Swap
    # For each x in best_set, try to remove it and add y not in best_set
    # This is O(N^2) * Validation
    # Validation is O(N^3).
    # N=100. 100*100*100^3 = 10^9. Too slow.
    
    # We will just return the best found so far.
    # The upper half set is likely very good.
    
    # One more check: ensure types are correct
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
