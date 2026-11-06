import time
import random
from fractions import Fraction
import numpy as np

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
    deadline = start_time + 115
    
    primes = _sieve(1000)
    MAX_L = 100000
    
    def make_greedy_set(skip_primes=None):
        if skip_primes is None:
            skip_primes = set()
        ps = []
        b = Fraction(1)
        for p in primes:
            if p in skip_primes:
                continue
            if b - Fraction(1, p) >= 0:
                ps.append(p)
                b -= Fraction(1, p)
            else:
                break
        return ps
    
    strategies = [
        make_greedy_set(),
        make_greedy_set({2}),
        make_greedy_set({2, 3}),
        [2, 3, 7],
        [3, 5, 7, 11, 13, 17, 19, 23],
        [2, 3, 11],
        [2, 5, 7],
    ]
    
    valid_sets = []
    seen = set()
    for ps in strategies:
        s = sum(Fraction(1, p) for p in ps)
        if s <= 1:
            key = tuple(sorted(ps))
            if key not in seen:
                seen.add(key)
                valid_sets.append(ps)
    
    best_L = 0
    best_classes = []
    
    def compute_L_from_coverage(covered_bool, max_check=MAX_L):
        uncovered = np.where(covered_bool[1:max_check+1] == False)[0]
        if len(uncovered) == 0:
            return max_check
        return int(uncovered[0])
    
    def greedy_and_search(p_set):
        nonlocal best_L, best_classes
        
        covered = np.zeros(MAX_L + 1, dtype=np.int32)
        residues = {}
        
        for p in sorted(p_set):
            best_r = 0
            best_gain = -1
            for r in range(p):
                start = r if r != 0 else p
                idx = np.arange(start, MAX_L + 1, p)
                gain = np.sum(covered[idx] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            residues[p] = best_r
            start = best_r if best_r != 0 else p
            idx = np.arange(start, MAX_L + 1, p)
            covered[idx] += 1
        
        L = compute_L_from_coverage(covered > 0)
        
        if L > best_L:
            best_L = L
            best_classes = [[p, residues[p]] for p in p_set]
        
        stuck = 0
        max_stuck = 200
        while time.time() < deadline:
            p = random.choice(p_set)
            old_r = residues[p]
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue
            
            old_start = old_r if old_r != 0 else p
            new_start = new_r if new_r != 0 else p
            old_idx = np.arange(old_start, MAX_L + 1, p)
            new_idx = np.arange(new_start, MAX_L + 1, p)
            
            covered[old_idx] -= 1
            covered[new_idx] += 1
            
            new_L = compute_L_from_coverage(covered > 0)
            
            if new_L > L:
                L = new_L
                residues[p] = new_r
                stuck = 0
                if L > best_L:
                    best_L = L
                    best_classes = [[p, residues[p]] for p in p_set]
            else:
                covered[new_idx] -= 1
                covered[old_idx] += 1
                stuck += 1
                if stuck > max_stuck:
                    for _ in range(2):
                        pk = random.choice(p_set)
                        rk = random.randint(0, pk - 1)
                        if rk != residues[pk]:
                            so = residues[pk] if residues[pk] != 0 else pk
                            sn = rk if rk != 0 else pk
                            idx_o = np.arange(so, MAX_L + 1, pk)
                            idx_n = np.arange(sn, MAX_L + 1, pk)
                            covered[idx_o] -= 1
                            covered[idx_n] += 1
                            residues[pk] = rk
                    L = compute_L_from_coverage(covered > 0)
                    if L > best_L:
                        best_L = L
                        best_classes = [[p, residues[p]] for p in p_set]
                    stuck = 0
                    max_stuck = random.randint(150, 400)
    
    for p_set in valid_sets:
        if time.time() > deadline - 5:
            break
        greedy_and_search(p_set)
    
    if not best_classes:
        best_classes = [[2, 1], [3, 2], [7, 6]]
    
    return best_classes
