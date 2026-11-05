"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random
from math import gcd
from itertools import combinations

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    numbers = list(range(1, N + 1))
    
    # Precompute compatibility: compatible[a][b] = lcm(a,b) <= N
    compatible = [[False] * (N + 1) for _ in range(N + 1)]
    for a in numbers:
        for b in numbers:
            compatible[a][b] = (a * b // gcd(a, b)) <= N
    
    def validate(S):
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                if S_list[i] * S_list[j] // gcd(S_list[i], S_list[j]) > N:
                    return False
        return True
    
    def try_add(S, a):
        return all(compatible[a][b] for b in S)
    
    def greedy(order, S_initial=None):
        S = set(S_initial) if S_initial is not None else set()
        for a in order:
            if a not in S and all(compatible[a][b] for b in S):
                S.add(a)
        return S
    
    def local_search_add(S):
        S = set(S)
        improved = True
        while improved:
            improved = False
            for a in numbers:
                if a not in S and all(compatible[a][b] for b in S):
                    S.add(a)
                    improved = True
                    break
            if time.time() > deadline:
                break
        return S
    
    def local_search_swap(S):
        S = set(S)
        improved = True
        while improved:
            improved = False
            for a in list(S):
                for b in numbers:
                    if b in S:
                        continue
                    # Try removing a and adding b
                    S_new = (S - {a}) | {b}
                    if all(compatible[x][y] for x in S_new for y in S_new):
                        # Check if it's actually an improvement (shouldn't be for same size, but for safety)
                        # Actually we want to find a state where we can then add more.
                        # For now, just accept if valid. We'll rely on the add phase to grow.
                        # But swapping same-size doesn't help cardinality directly.
                        # Let's look for swaps that allow adding a third element?
                        # That's 2-opt. Too slow.
                        # Let's stick to 1-swap for structural improvement, then add.
                        pass
            if time.time() > deadline:
                break
        return S
    
    # Better local search: 1-move (add) and 2-move (swap one for another to enable growth)
    def advanced_local_search(S):
        S = set(S)
        improved = True
        while improved:
            improved = False
            # Phase 1: Add any compatible element
            for a in numbers:
                if a not in S and all(compatible[a][b] for b in S):
                    S.add(a)
                    improved = True
                    break
            if time.time() > deadline:
                break
            
            # Phase 2: Try swapping one element for another to enable adding a third
            if not improved:
                for a in list(S):
                    for b in numbers:
                        if b in S:
                            continue
                        S_temp = (S - {a}) | {b}
                        # Check if S_temp is valid
                        if not all(compatible[x][y] for x in S_temp for y in S_temp):
                            continue
                        # Now try to add a third element c
                        for c in numbers:
                            if c in S_temp:
                                continue
                            if all(compatible[c][x] for x in S_temp):
                                S = S_temp | {c}
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            if time.time() > deadline:
                break
        return S
    
    best_S = set()
    
    # Strategy 1: Greedy with various orderings
    # Order by number of conflicts (degree in conflict graph) - ascending (least conflicts first)
    def conflict_degree(a):
        return sum(1 for b in numbers if not compatible[a][b])
    
    order_deg = sorted(numbers, key=conflict_degree)
    S1 = greedy(order_deg)
    S1 = advanced_local_search(S1)
    if len(S1) > len(best_S):
        best_S = S1
    
    # Order by value ascending
    S2 = greedy(numbers)
    S2 = advanced_local_search(S2)
    if len(S2) > len(best_S):
        best_S = S2
    
    # Order by value descending
    S3 = greedy(numbers[::-1])
    S3 = advanced_local_search(S3)
    if len(S3) > len(best_S):
        best_S = S3
    
    # Strategy 2: Randomized Greedy with Local Search
    random.seed(42)
    for i in range(200):
        if time.time() > deadline:
            break
        order = numbers[:]
        random.shuffle(order)
        S_rand = greedy(order)
        S_rand = advanced_local_search(S_rand)
        if len(S_rand) > len(best_S):
            best_S = S_rand
            # If we found a significantly better one, maybe restart local search more aggressively?
            # For now, just keep going.
    
    # Strategy 3: Structural approach based on prime factors
    # Key insight: If we pick a set of numbers, their pairwise LCMs must be <= 512.
    # Let's try to build a set around a "core" of divisors of a number K <= 512?
    # No, the LCM of the whole set doesn't have to be a single number, just pairwise.
    
    # Let's try a specific structural family:
    # All numbers of the form 2^a * p^b where p is a small prime.
    # Actually, let's just run a max-clique search on a smaller induced subgraph?
    # The graph is dense.
    
    # Let's try to identify the "bottleneck" primes.
    # Large primes > 256 can only appear in numbers that are themselves large primes or small multiples.
    # If p > 256, the only multiple in [1, 512] is p itself.
    # lcm(p, q) for distinct primes p, q > 256 is p*q > 512. So at most one such prime.
    # If p > 256 and x is any other number, lcm(p, x) is a multiple of p.
    # If x is not a multiple of p, lcm(p, x) >= p * (x/gcd(p,x)). Since p is prime and p>x (unless x=p), gcd=1.
    # So lcm(p, x) = p*x. We need p*x <= 512.
    # If p=257, x can be 1, 2 (257*2=514>512). So only x=1.
    # If p=263, x=1.
    # So large primes are very restrictive.
    
    # Let's focus on numbers with small prime factors.
    
    # Strategy 4: Branch and Bound on a reduced set
    # We know the answer is likely around 50-100?
    # Current best from seed is ~1.09 * seed. Seed is likely small.
    # Let's check the size of best_S.
    
    # Let's try to improve best_S with a simple hill climbing on the set itself.
    # Represent S as a list.
    
    # One more strategy: Start with all numbers that divide 512 (powers of 2), then add others.
    # Powers of 2: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512. Size 10.
    # Can we add 3? lcm(3, 512) = 1536 > 512. No.
    # Remove 512, 256. Add 3?
    # Set: {1, 2, 4, 8, 16, 32, 64, 128, 3}.
    # lcm(3, 128) = 384 <= 512. OK.
    # Can we add 6? lcm(6, 128) = 384. lcm(6, 3)=6. OK.
    # Can we add 9? lcm(9, 128) = 1152 > 512. No.
    # Can we add 12? lcm(12, 128) = 384. lcm(12, 9) not in set.
    
    # Let's run a specific greedy that prioritizes numbers with small "span" (max multiple in set).
    
    # Final polish: Try to add any remaining numbers to best_S
    best_S = advanced_local_search(best_S)
    
    # Verify
    if not validate(best_S):
        # Fallback to a known valid set
        return list(range(1, 23))
    
    if not best_S:
        return [1]
        
    return sorted(best_S)

# EVOLVE-BLOCK-END
