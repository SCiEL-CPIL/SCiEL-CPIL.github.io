import time
import math
import random
import numpy as np

N = 5000

def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

MAX_PRIME_FOR_PRECOMP = 600
ALL_PRIMES = _primes(MAX_PRIME_FOR_PRECOMP)

prime_indices = {}
for p in ALL_PRIMES:
    indices = [[] for _ in range(p)]
    for x in range(N + 1):
        indices[x % p].append(x)
    prime_indices[p] = indices

def greedy_initialization(prime_order, tie_break='min'):
    residue_choices = {}
    S = np.ones(N + 1, dtype=bool)
    current_size = N + 1
    
    for p in prime_order:
        if p > current_size:
            break
        counts = [0] * p
        for r in range(p):
            cnt = 0
            for x in prime_indices[p][r]:
                if S[x]:
                    cnt += 1
            counts[r] = cnt
        
        if min(counts) > 0:
            if tie_break == 'min':
                min_c = min(counts)
                candidates = [r for r in range(p) if counts[r] == min_c]
                r_to_remove = random.choice(candidates)
            elif tie_break == 'max':
                max_c = max(counts)
                candidates = [r for r in range(p) if counts[r] == max_c]
                r_to_remove = random.choice(candidates)
            else:
                r_to_remove = random.randint(0, p - 1)
            
            residue_choices[p] = r_to_remove
            for x in prime_indices[p][r_to_remove]:
                if S[x]:
                    S[x] = False
                    current_size -= 1
    
    return residue_choices, current_size, S

def local_search(residue_choices, S, primes_list, max_iterations=500, time_limit=None, start_time=None):
    if start_time is None:
        start_time = time.time()
    
    current_size = int(np.sum(S))
    best_size = current_size
    best_choices = dict(residue_choices)
    best_S = S.copy()
    
    relevant_primes = [p for p in primes_list if p <= current_size]
    
    iteration = 0
    while iteration < max_iterations:
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        improved = False
        random.shuffle(relevant_primes)
        
        for p in relevant_primes:
            if p > best_size:
                continue
            
            current_r = residue_choices.get(p, None)
            
            if current_r is not None:
                candidates = [r for r in range(p) if r != current_r]
            else:
                candidates = list(range(p))
            
            random.shuffle(candidates)
            for r in candidates:
                if current_r is None:
                    new_size = current_size - len([x for x in prime_indices[p][r] if S[x]])
                else:
                    add_back = len([x for x in prime_indices[p][current_r] if not S[x]])
                    remove_new = len([x for x in prime_indices[p][r] if S[x]])
                    new_size = current_size + add_back - remove_new
                
                if new_size > best_size:
                    if current_r is None:
                        for x in prime_indices[p][r]:
                            if S[x]:
                                S[x] = False
                    else:
                        for x in prime_indices[p][current_r]:
                            if not S[x]:
                                S[x] = True
                        for x in prime_indices[p][r]:
                            if S[x]:
                                S[x] = False
                    
                    residue_choices[p] = r
                    current_size = new_size
                    best_size = new_size
                    best_choices = dict(residue_choices)
                    best_S = S.copy()
                    improved = True
                    relevant_primes = [pp for pp in primes_list if pp <= best_size]
                    break
        
        iteration += 1
        if not improved:
            if random.random() < 0.2 and best_size > 0:
                perturbed = False
                for _ in range(5):
                    p = random.choice([pp for pp in primes_list if pp <= best_size])
                    old_r = residue_choices.get(p, None)
                    new_r = random.randint(0, p - 1)
                    if new_r == old_r:
                        continue
                    
                    if old_r is None:
                        new_size = current_size - len([x for x in prime_indices[p][new_r] if S[x]])
                    else:
                        add_back = len([x for x in prime_indices[p][old_r] if not S[x]])
                        remove_new = len([x for x in prime_indices[p][new_r] if S[x]])
                        new_size = current_size + add_back - remove_new
                    
                    if new_size > best_size:
                        if old_r is None:
                            for x in prime_indices[p][new_r]:
                                if S[x]:
                                    S[x] = False
                        else:
                            for x in prime_indices[p][old_r]:
                                if not S[x]:
                                    S[x] = True
                            for x in prime_indices[p][new_r]:
                                if S[x]:
                                    S[x] = False
                        residue_choices[p] = new_r
                        current_size = new_size
                        best_size = new_size
                        best_choices = dict(residue_choices)
                        best_S = S.copy()
                        perturbed = True
                        break
                    else:
                        if random.random() < 0.1:
                            if old_r is None:
                                for x in prime_indices[p][new_r]:
                                    if S[x]:
                                        S[x] = False
                            else:
                                for x in prime_indices[p][old_r]:
                                    if not S[x]:
                                        S[x] = True
                                for x in prime_indices[p][new_r]:
                                    if S[x]:
                                        S[x] = False
                            residue_choices[p] = new_r
                            current_size = new_size
                            perturbed = True
                            break
                if perturbed:
                    relevant_primes = [pp for pp in primes_list if pp <= best_size]
                    continue
            else:
                break
    
    return best_choices, best_size, best_S

def construct():
    start_time = time.time()
    deadline = start_time + 290.0
    
    best_size = 0
    best_set = []
    
    active_primes = [p for p in ALL_PRIMES if p <= 500]
    
    strategies = [
        ('min', list(active_primes)),
        ('min', list(reversed(active_primes))),
        ('max', list(active_primes)),
        ('random', list(active_primes)),
    ]
    for _ in range(4):
        shuffled = list(active_primes)
        random.shuffle(shuffled)
        strategies.append(('min', shuffled))
    
    for tie_break, prime_order in strategies:
        if time.time() > deadline - 30:
            break
        
        residue_choices, size, S = greedy_initialization(prime_order, tie_break)
        
        if size > best_size:
            best_size = size
            best_set = [int(x) for x in range(N + 1) if S[x]]
        
        if time.time() > deadline - 60:
            break
        
        residue_choices, size, S = local_search(
            residue_choices, S, active_primes,
            max_iterations=300,
            time_limit=deadline - time.time() - 30,
            start_time=start_time
        )
        
        if size > best_size:
            best_size = size
            best_set = [int(x) for x in range(N + 1) if S[x]]
    
    return sorted(best_set)
