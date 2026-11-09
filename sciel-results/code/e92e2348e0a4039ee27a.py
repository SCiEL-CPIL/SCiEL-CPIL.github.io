import time
import math
import sys
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin for 60s limit

    N = 128
    full_mask = (1 << (N + 1)) - 2  # bits 1..N set

    # Precompute LCM table for O(1) lookup
    lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a, N + 1):
            lcm_table[a][b] = a * b // gcd(a, b)
            lcm_table[b][a] = lcm_table[a][b]

    # Precompute bad triples and conflict masks
    # conflict[a][b] is a bitmask of all c such that {a,b,c} forms a bad triple
    conflict = [[0] * (N + 1) for _ in range(N + 1)]
    
    # Also precompute the set of all bad triples for potential other uses, 
    # but primarily we rely on conflict masks.
    # To speed up the "is x compatible with S" check, we need to ensure that 
    # for every pair (s1, s2) in S, x is not in conflict[s1][s2].
    # Or equivalently, for every s in S, x is not in conflict[s][y] for any y in S.
    
    # Let's precompute a slightly different structure for faster local search:
    # For a given set S (represented as bitmask), adding x is valid if:
    # For all s in S, (conflict[s][x] & S_mask) == 0
    # This is O(|S|) per check.
    
    # For simulated annealing, we might want to remove elements.
    # Removing x is always valid (reduces set size).
    # We want to maximize size, so we mostly add.
    
    # Let's precompute degrees for greedy ordering
    conflict_score = [0.0] * (N + 1)
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            # Count how many c make {a,b,c} bad
            # This is popcount of conflict[a][b]
            # But conflict[a][b] contains c > b? No, we filled it symmetrically.
            # Let's just count bits.
            cnt = conflict[a][b].bit_count()
            conflict_score[a] += cnt
            conflict_score[b] += cnt

    # Helper: check if x can be added to set represented by mask S
    # S is a bitmask of elements in the set.
    # We need to know the list of elements in S for the loop.
    # To make this fast, we'll maintain a list or use bit operations cleverly.
    # Since N=128, iterating over bits is fast enough.
    
    def can_add(x, S_mask, S_list):
        """Check if x can be added to the set S (represented by S_mask and S_list)."""
        for s in S_list:
            if conflict[s][x] & S_mask:
                return False
        return True

    def greedy(order):
        S = []
        S_mask = 0
        for x in order:
            if can_add(x, S_mask, S):
                S.append(x)
                S_mask |= (1 << x)
        return S, S_mask

    # Try several greedy orderings to get a good initial solution
    best_set = []
    best_size = 0
    best_mask = 0

    def update_best(S, S_mask):
        nonlocal best_set, best_size, best_mask
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            best_mask = S_mask

    # 1. Natural order
    S, mask = greedy(list(range(1, N + 1)))
    update_best(S, mask)

    # 2. Reverse order
    S, mask = greedy(list(range(N, 0, -1)))
    update_best(S, mask)

    # 3. Sort by conflict score (ascending - prefer low-conflict elements)
    order = sorted(range(1, N + 1), key=lambda x: conflict_score[x])
    S, mask = greedy(order)
    update_best(S, mask)

    # 4. Sort by conflict score (descending)
    order = sorted(range(1, N + 1), key=lambda x: -conflict_score[x])
    S, mask = greedy(order)
    update_best(S, mask)

    # 5. Primes and prime powers first
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    def is_prime_power(n):
        if n < 2:
            return False
        p = n
        while p > 1:
            if p * p > n:
                break
            if n % p == 0:
                n //= p
            else:
                return False
        return True

    primes = [x for x in range(1, N + 1) if is_prime(x)]
    prime_powers = [x for x in range(1, N + 1) if is_prime_power(x) and x not in primes]
    others = [x for x in range(1, N + 1) if x not in primes and x not in prime_powers]
    
    order_pp = primes + prime_powers + others
    S, mask = greedy(order_pp)
    update_best(S, mask)

    # Simulated Annealing
    # State: current set (list and mask)
    # Moves: 
    # 1. Add a random element not in S if valid.
    # 2. Remove a random element from S.
    # 3. Swap: remove one, add another (if valid).
    
    # To speed up "Add a random element", we can precompute candidates?
    # Or just pick random x and check validity.
    
    # Let's refine the SA loop.
    # We will run SA for a fixed number of steps or until time runs out.
    
    def simulated_annealing(initial_set, initial_mask, steps=100000):
        curr_set = initial_set[:]
        curr_mask = initial_mask
        curr_size = len(curr_set)
        
        best_sa_set = curr_set[:]
        best_sa_mask = curr_mask
        best_sa_size = curr_size
        
        T = 1.0
        cooling = 0.9995  # Slow cooling
        
        for i in range(steps):
            if time.time() - start_time > deadline - 2:
                break
            
            # Pick a random move type
            move_type = random.random()
            
            new_set = curr_set
            new_mask = curr_mask
            new_size = curr_size
            
            if move_type < 0.6:
                # Try to add a random element
                # Pick random x from 1..N not in curr_mask
                x = random.randint(1, N)
                if (curr_mask >> x) & 1:
                    continue # Already in set, skip
                if can_add(x, curr_mask, curr_set):
                    new_set = curr_set + [x]
                    new_mask = curr_mask | (1 << x)
                    new_size = curr_size + 1
                else:
                    # If cannot add, maybe do nothing or try another
                    continue
            elif move_type < 0.8:
                # Try to remove a random element
                if curr_size == 0:
                    continue
                idx = random.randint(0, curr_size - 1)
                x = curr_set[idx]
                new_set = curr_set[:idx] + curr_set[idx+1:]
                new_mask = curr_mask & ~(1 << x)
                new_size = curr_size - 1
            else:
                # Swap: remove one, try to add another
                if curr_size == 0:
                    continue
                idx = random.randint(0, curr_size - 1)
                x_remove = curr_set[idx]
                temp_set = curr_set[:idx] + curr_set[idx+1:]
                temp_mask = curr_mask & ~(1 << x_remove)
                
                # Try to add a random element
                x_add = random.randint(1, N)
                if (temp_mask >> x_add) & 1:
                    continue
                if can_add(x_add, temp_mask, temp_set):
                    new_set = temp_set + [x_add]
                    new_mask = temp_mask | (1 << x_add)
                    new_size = len(new_set)
                else:
                    continue

            # Calculate delta
            delta = new_size - curr_size
            
            # Acceptance probability
            if delta > 0:
                # Always accept improvement
                accept = True
            else:
                # Accept with probability exp(delta / T)
                # delta is negative
                if T > 0:
                    prob = math.exp(delta / T)
                    accept = random.random() < prob
                else:
                    accept = False
            
            if accept:
                curr_set = new_set
                curr_mask = new_mask
                curr_size = new_size
                
                if curr_size > best_sa_size:
                    best_sa_size = curr_size
                    best_sa_set = curr_set[:]
                    best_sa_mask = curr_mask
            
            T *= cooling
        
        return best_sa_set, best_sa_mask, best_sa_size

    # Run SA starting from the best greedy solution
    # We can run multiple SA instances with different random seeds or perturbations
    
    random.seed(42)
    
    # Run SA for a while
    sa_set, sa_mask, sa_size = simulated_annealing(best_set, best_mask, steps=50000)
    update_best(sa_set, sa_mask)
    
    # If we have time, run SA again from a slightly perturbed state
    if time.time() - start_time < deadline - 5:
        # Perturb the best set: remove a few random elements
        perturbed_set = best_set[:]
        perturbed_mask = best_mask
        for _ in range(3):
            if not perturbed_set:
                break
            idx = random.randint(0, len(perturbed_set) - 1)
            x = perturbed_set[idx]
            perturbed_set = perturbed_set[:idx] + perturbed_set[idx+1:]
            perturbed_mask &= ~(1 << x)
        
        sa_set2, sa_mask2, sa_size2 = simulated_annealing(perturbed_set, perturbed_mask, steps=50000)
        update_best(sa_set2, sa_mask2)
    
    # Final check: ensure the returned set is valid
    # (SA guarantees validity of moves, but let's double check just in case)
    final_set = best_set
    if not final_set:
        # Fallback: return a single element
        return [1]
        
    return final_set

# Quick test
if __name__ == "__main__":
    result = construct()
    print(f"Found set of size {len(result)}: {result}")
    # Verify
    N = 128
    def lcm(a, b):
        return a * b // gcd(a, b)
    valid = True
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            for k in range(j+1, len(result)):
                a, b, c = result[i], result[j], result[k]
                if lcm(a,b) == lcm(a,c) == lcm(b,c):
                    valid = False
                    print(f"Invalid triple: {a}, {b}, {c}")
    print(f"Valid: {valid}")
