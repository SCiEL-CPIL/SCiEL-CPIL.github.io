import time
import random
import math
import numpy as np

N = 5000
MAX_TIME = 290.0  # Safety margin

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

PRIMES = _primes(N)

def _is_admissible(S, primes):
    len_S = len(S)
    for p in primes:
        if p > len_S:
            break
        residues = set(x % p for x in S)
        if len(residues) == p:
            return p
    return None

def _greedy_numpy(seed=None):
    """
    NumPy-vectorized greedy construction.
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    active = np.ones(N + 1, dtype=bool)
    indices = np.arange(N + 1)
    
    while True:
        count = int(active.sum())
        if count == 0:
            break
        
        viol_p = None
        for p in PRIMES:
            if p > count:
                break
            residues = indices[active] % p
            counts = np.bincount(residues, minlength=p)
            if np.all(counts > 0):
                viol_p = p
                break
        
        if viol_p is None:
            break
        
        p = viol_p
        residues = indices[active] % p
        counts = np.bincount(residues, minlength=p)
        
        min_count = counts.min()
        candidates = np.where(counts == min_count)[0]
        r_to_remove = rng.choice(candidates)
        
        remove_mask = (indices % p == r_to_remove) & active
        active[remove_mask] = False
    
    return indices[active].tolist()

def _greedy_beam(beam_width=5, seed=None):
    """
    Beam search over residue choices at each prime.
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    indices = np.arange(N + 1)
    beams = [np.ones(N + 1, dtype=bool)]
    
    while True:
        new_beams = []
        for beam in beams:
            count = int(beam.sum())
            if count == 0:
                continue
            
            viol_p = None
            for p in PRIMES:
                if p > count:
                    break
                residues = indices[beam] % p
                counts = np.bincount(residues, minlength=p)
                if np.all(counts > 0):
                    viol_p = p
                    break
            
            if viol_p is None:
                new_beams.append(beam)
                continue
            
            p = viol_p
            residues = indices[beam] % p
            counts = np.bincount(residues, minlength=p)
            
            min_count = counts.min()
            candidates = np.where(counts == min_count)[0]
            
            for r in candidates:
                new_beam = beam.copy()
                new_beam[(indices % p == r) & new_beam] = False
                new_beams.append(new_beam)
        
        if not new_beams:
            break
        
        new_beams.sort(key=lambda b: -int(b.sum()))
        beams = new_beams[:beam_width]
        
        all_complete = True
        for beam in beams:
            count = int(beam.sum())
            for p in PRIMES:
                if p > count:
                    break
                residues = indices[beam] % p
                counts = np.bincount(residues, minlength=p)
                if np.all(counts > 0):
                    all_complete = False
                    break
            if not all_complete:
                break
        
        if all_complete:
            break
    
    if not beams:
        return []
    
    best = max(beams, key=lambda b: int(b.sum()))
    return indices[best].tolist()

def construct():
    start_time = time.time()
    best_size = 0
    best_tuple = []
    
    # 1. Deterministic greedy
    res = _greedy_numpy(seed=0)
    if len(res) > best_size:
        best_size = len(res)
        best_tuple = res
    
    # 2. Beam search with small width
    res = _greedy_beam(beam_width=3, seed=0)
    if len(res) > best_size:
        best_size = len(res)
        best_tuple = res
    
    # 3. Many randomized greedy restarts (NumPy makes these fast)
    trial = 1
    while time.time() - start_time < MAX_TIME - 60:
        res = _greedy_numpy(seed=trial)
        if len(res) > best_size:
            best_size = len(res)
            best_tuple = res
        trial += 1
    
    # 4. Larger beam search with remaining time
    if time.time() - start_time < MAX_TIME - 30:
        res = _greedy_beam(beam_width=8, seed=trial)
        if len(res) > best_size:
            best_size = len(res)
            best_tuple = res
    
    return best_tuple
