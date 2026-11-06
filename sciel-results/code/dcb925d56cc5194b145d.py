import time
import random
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Precomputation
    N = 512
    max_val = N - 1
    
    # Prime factors for each number 1..511
    # factors[i] is a set of prime factors of i
    factors = [set() for _ in range(max_val + 1)]
    is_prime = [True] * (max_val + 1)
    is_prime[0] = False
    is_prime[1] = False
    primes = []
    for i in range(2, max_val + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i * i, max_val + 1, i):
                is_prime[j] = False
        # Populate factors
        temp = i
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                factors[i].add(p)
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            factors[i].add(temp)
    
    # Weights
    weights = [0.0] * (max_val + 1)
    for i in range(1, max_val + 1):
        weights[i] = 1.0 / (N - i)
        
    def is_coprime_to_all(v, S, prime_counts):
        # prime_counts[p] is count of elements in S divisible by p
        for p in factors[v]:
            if prime_counts[p] > 0:
                return False
        return True

    def greedy_fill(S, prime_counts, start_from=511):
        # S is a set, prime_counts is a dict or array
        # Returns new S and new prime_counts
        new_S = set(S)
        new_pc = prime_counts.copy()
        
        # We want to add numbers. To maximize weight, we prefer larger numbers.
        # Iterate from start_from down to 1
        for v in range(start_from, 0, -1):
            if v in new_S:
                continue
            if is_coprime_to_all(v, new_S, new_pc):
                new_S.add(v)
                for p in factors[v]:
                    new_pc[p] += 1
        return new_S, new_pc

    def calculate_score(S):
        return sum(weights[v] for v in S)

    def local_search(initial_S):
        S = set(initial_S)
        # Calculate initial prime counts
        prime_counts = {}
        for v in S:
            for p in factors[v]:
                prime_counts[p] = prime_counts.get(p, 0) + 1
        
        current_score = calculate_score(S)
        
        improved = True
        while improved:
            improved = False
            # Try removing each u
            # To make it deterministic but varied, we can shuffle the order of u's to check
            # But for speed, we can just check all.
            for u in list(S):
                # Tentatively remove u
                S_temp = S - {u}
                pc_temp = prime_counts.copy()
                for p in factors[u]:
                    pc_temp[p] -= 1
                    if pc_temp[p] == 0:
                        del pc_temp[p] # Keep dict clean, or just let it be 0
                
                # Greedy fill
                S_filled, pc_filled = greedy_fill(S_temp, pc_temp)
                
                score_filled = calculate_score(S_filled)
                
                if score_filled > current_score + 1e-9:
                    S = S_filled
                    prime_counts = pc_filled
                    current_score = score_filled
                    improved = True
                    break # Restart the loop with new S
            
        return S, current_score

    best_set = set()
    best_score = 0.0
    
    # Strategy 1: Greedy descending
    S0 = set()
    pc0 = {}
    S0, pc0 = greedy_fill(S0, pc0)
    score0 = calculate_score(S0)
    if score0 > best_score:
        best_score = score0
        best_set = S0
    
    # Local search from S0
    S_ls, score_ls = local_search(S0)
    if score_ls > best_score:
        best_score = score_ls
        best_set = S_ls
        
    # Randomized restarts
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        # Create a randomized initial set
        # We can shuffle the order of addition in greedy fill
        # Or just pick a random subset of top candidates?
        
        # Let's try a randomized greedy:
        # Shuffle 1..511, but weight the selection?
        # Simpler: Shuffle the list of candidates 511..1 and pick first coprime.
        candidates = list(range(511, 0, -1))
        random.shuffle(candidates)
        
        S_rand = set()
        pc_rand = {}
        for v in candidates:
            if is_coprime_to_all(v, S_rand, pc_rand):
                S_rand.add(v)
                for p in factors[v]:
                    pc_rand[p] = pc_rand.get(p, 0) + 1
        
        # Local search from this random start
        S_rand_ls, score_rand_ls = local_search(S_rand)
        
        if score_rand_ls > best_score:
            best_score = score_rand_ls
            best_set = S_rand_ls
            
        # Occasionally, try a "perturbed" local search from best_set
        if restarts % 5 == 0:
             # Remove a random element from best_set and local search
             if best_set:
                 u = random.choice(list(best_set))
                 S_pert = best_set - {u}
                 pc_pert = {}
                 for v in S_pert:
                     for p in factors[v]:
                         pc_pert[p] = pc_pert.get(p, 0) + 1
                 S_pert_ls, score_pert_ls = local_search(S_pert)
                 if score_pert_ls > best_score:
                     best_score = score_pert_ls
                     best_set = S_pert_ls

    return list(best_set)
# EVOLVE-BLOCK-END
