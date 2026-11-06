import time
import random
from fractions import Fraction

def _sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    primes = _sieve(1000)
    MAX_L = 100000
    
    # Candidate prime sets
    # We want sets where sum(1/p) <= 1
    # Set 1: {2, 3, 7} -> 1/2 + 1/3 + 1/7 = 41/42 < 1
    # Set 2: {2, 3, 11} -> 1/2 + 1/3 + 1/11 = 41/66 < 1
    # Set 3: {2, 5, 7} -> 1/2 + 1/5 + 1/7 = 59/70 < 1
    # Set 4: {3, 5, 7, 11, 13, 17, 19, 23} -> Sum approx 0.97
    # Set 5: {2, 3, 11, 13} -> 1/2 + 1/3 + 1/11 + 1/13 = 0.99...
    
    candidate_sets = [
        [2, 3, 7],
        [2, 3, 11],
        [2, 5, 7],
        [3, 5, 7, 11, 13, 17, 19, 23],
        [2, 3, 11, 13],
        [2, 3, 17], # 1/2+1/3+1/17 = 0.91
        [2, 3, 5, 29] # 1/2+1/3+1/5+1/29 = 0.5+0.33+0.2+0.034 = 1.064 (Invalid) -> Remove
    ]
    
    # Filter valid sets
    valid_sets = []
    for p_set in candidate_sets:
        s = sum(Fraction(1, p) for p in p_set)
        if s <= 1:
            valid_sets.append(p_set)
    
    # Add some random valid sets
    for _ in range(5):
        random.shuffle(primes)
        s = Fraction(0)
        p_set = []
        for p in primes:
            c = Fraction(1, p)
            if s + c <= 1:
                s += c
                p_set.append(p)
            if len(p_set) > 10:
                break
        if p_set:
            valid_sets.append(sorted(p_set))

    best_L = 0
    best_classes = []
    
    cover_count = [0] * (MAX_L + 1)
    
    def get_L():
        for i in range(1, MAX_L + 1):
            if cover_count[i] == 0:
                return i - 1
        return MAX_L

    def apply(p, r):
        # r is in 0..p-1
        # covers k such that k % p == r
        # k = r, r+p, ...
        # Note: if r==0, first is p
        start = r if r != 0 else p
        for k in range(start, MAX_L + 1, p):
            cover_count[k] += 1

    def unapply(p, r):
        start = r if r != 0 else p
        for k in range(start, MAX_L + 1, p):
            cover_count[k] -= 1

    def run_local_search(p_set, restarts=50):
        nonlocal best_L, best_classes
        
        for _ in range(restarts):
            if time.time() > deadline:
                return
                
            # Initialize residues
            residues = {p: random.randint(0, p - 1) for p in p_set}
            
            # Reset cover_count
            for i in range(MAX_L + 1):
                cover_count[i] = 0
                
            for p, r in residues.items():
                apply(p, r)
                
            current_L = get_L()
            
            if current_L > best_L:
                best_L = current_L
                best_classes = [[p, residues[p]] for p in p_set]
                
            stuck_count = 0
            max_stuck = 1000
            
            while time.time() < deadline:
                # Pick a random prime to modify
                p = random.choice(p_set)
                old_r = residues[p]
                
                # Pick a new residue
                new_r = random.randint(0, p - 1)
                if new_r == old_r:
                    continue
                
                # Apply change
                unapply(p, old_r)
                apply(p, new_r)
                
                new_L = get_L()
                
                if new_L > current_L:
                    current_L = new_L
                    residues[p] = new_r
                    stuck_count = 0
                    
                    if current_L > best_L:
                        best_L = current_L
                        best_classes = [[p, residues[p]] for p in p_set]
                else:
                    # Revert
                    unapply(p, new_r)
                    apply(p, old_r)
                    stuck_count += 1
                    
                    if stuck_count > max_stuck:
                        # Kick: randomly change a few residues and accept
                        for _ in range(3):
                            pk = random.choice(p_set)
                            rk = random.randint(0, pk - 1)
                            if rk != residues[pk]:
                                unapply(pk, residues[pk])
                                apply(pk, rk)
                                residues[pk] = rk
                        current_L = get_L()
                        if current_L > best_L:
                            best_L = current_L
                            best_classes = [[p, residues[p]] for p in p_set]
                        stuck_count = 0
                        max_stuck = random.randint(500, 2000)

    for p_set in valid_sets:
        if time.time() > deadline:
            break
        run_local_search(p_set, restarts=10)

    if not best_classes:
        # Fallback
        best_classes = [[2, 1], [3, 2], [7, 6]]

    return best_classes
