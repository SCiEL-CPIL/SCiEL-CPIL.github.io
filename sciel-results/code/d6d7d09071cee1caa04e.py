import random
import time
import math

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _calc_delta_energy(seq, chain, occ, i, p_old, p_new):
    if seq[i] != 'H':
        return 0
    delta = 0
    for q in _neighbours(p_old):
        if q in occ:
            j = occ[q]
            if j != i and abs(i - j) > 1 and seq[j] == 'H':
                delta -= 1
    for q in _neighbours(p_new):
        if q in occ:
            j = occ[q]
            if j != i and abs(i - j) > 1 and seq[j] == 'H':
                delta += 1
    return -delta

def _grow_saw(seq, dim, rng):
    n = len(seq)
    if n == 0:
        return []
    chain = [(0,)*dim]
    occ = {(0,)*dim: 0}
    for i in range(1, n):
        last = chain[-1]
        candidates = [nb for nb in _neighbours(last) if nb not in occ]
        if not candidates:
            return None
        nb = rng.choice(candidates)
        chain.append(nb)
        occ[nb] = i
    return chain

def _energy(chain, seq):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if seq[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and seq[j] == 'H':
                    contacts += 1
    return -contacts

def _local_search_sa(seq, chain, occ, time_limit, rng):
    n = len(seq)
    if n < 3:
        return chain, occ
    
    start_time = time.time()
    T = 5.0
    alpha = 0.95
    
    current_energy = _energy(chain, seq)
    best_chain = chain[:]
    best_energy = current_energy
    
    while time.time() - start_time < time_limit:
        i = rng.randint(1, n - 2)
        p_old = chain[i]
        neighbors = list(_neighbours(p_old))
        rng.shuffle(neighbors)
        
        for p_new in neighbors:
            if p_new in occ:
                continue
            
            dE = _calc_delta_energy(seq, chain, occ, i, p_old, p_new)
            
            if dE < 0:
                accept = True
            else:
                if T > 0.01:
                    prob = math.exp(-dE / T)
                    accept = rng.random() < prob
                else:
                    accept = False
            
            if accept:
                del occ[p_old]
                occ[p_new] = i
                chain[i] = p_new
                current_energy += dE
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = chain[:]
                
                break
        
        T *= alpha
        if T < 0.01:
            T = 0.01
    
    return best_chain, {p: i for i, p in enumerate(best_chain)}

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    
    start_time = time.time()
    time_limit = 10.0 # Conservative limit per sequence
    
    best_chain = None
    best_energy = 1000000
    
    seed_val = sum(ord(c) for c in sequence)
    rng = random.Random(seed_val)
    
    # Multi-start SA
    # We will do restarts until time runs out.
    # Each restart gets a slice of time.
    # Let's say 2s per restart.
    restart_time = 2.0
    
    while time.time() - start_time < time_limit:
        remaining = time_limit - (time.time() - start_time)
        if remaining < 0.1:
            break
            
        current_slice = min(restart_time, remaining)
        
        chain = _grow_saw(sequence, dimension, rng)
        if chain is None:
            continue
        
        occ = {p: i for i, p in enumerate(chain)}
        
        chain, occ = _local_search_sa(sequence, chain, occ, current_slice, rng)
        
        e = _energy(chain, sequence)
        
        if e < best_energy:
            best_energy = e
            best_chain = chain
        
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]
