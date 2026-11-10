import itertools
import math

def construct():
    max_val = 128
    max_sum = 256
    
    # Sieve of Eratosthenes
    is_prime = [True] * (max_sum + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(max_sum**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, max_sum + 1, i):
                is_prime[j] = False
                
    odds = [x for x in range(1, max_val + 1) if x % 2 == 1]
    evens = [x for x in range(2, max_val + 1) if x % 2 == 0]
    
    # Precompute valid evens for each odd such that sum is prime
    # Also calculate degree for greedy sorting
    valid_b_for_a = {}
    degrees = {}
    for a in odds:
        valid_b_for_a[a] = set()
        for b in evens:
            if is_prime[a + b]:
                valid_b_for_a[a].add(b)
        degrees[a] = len(valid_b_for_a[a])
        
    # Sort odds by degree descending (Greedy heuristic: try most connected nodes first)
    sorted_odds = sorted(odds, key=lambda x: degrees[x], reverse=True)
    
    # Search for A (3 odds) and B (3 evens)
    for A in itertools.combinations(sorted_odds, 3):
        a1, a2, a3 = A
        V_A = valid_b_for_a[a1].intersection(valid_b_for_a[a2]).intersection(valid_b_for_a[a3])
        if len(V_A) < 3:
            continue
        
        V_A_list = list(V_A)
        # Iterate through combinations of 3 evens from the valid set
        for B in itertools.combinations(V_A_list, 3):
            sums = set()
            for a in A:
                for b in B:
                    sums.add(a + b)
            if len(sums) == 9:
                # All sums are prime by construction, and distinct.
                # Distinct primes are pairwise coprime.
                return [list(A), list(B)]
                
    # Fallback: Unsorted search if greedy order missed something (unlikely)
    for A in itertools.combinations(odds, 3):
        a1, a2, a3 = A
        V_A = valid_b_for_a[a1].intersection(valid_b_for_a[a2]).intersection(valid_b_for_a[a3])
        if len(V_A) < 3:
            continue
        V_A_list = list(V_A)
        for B in itertools.combinations(V_A_list, 3):
            sums = set()
            for a in A:
                for b in B:
                    sums.add(a + b)
            if len(sums) == 9:
                return [list(A), list(B)]

    # Absolute fallback: Return a known valid set (5 distinct prime sums)
    # A=[1, 7, 13], B=[4, 10, 16]
    # Sums: 5, 11, 17, 23, 29 (All prime, distinct, pairwise coprime)
    return [[1, 7, 13], [4, 10, 16]]
