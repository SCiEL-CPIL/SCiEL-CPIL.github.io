import numpy as np
import time
import random

N = 256
MAX_D = N // 3

def _precompute_ap_neighbors():
    """
    neighbors[i] is a list of tuples (j, k, l) such that (i, j, k, l) is a 4-AP 
    (in some order). Specifically, j,k,l are the other three indices.
    """
    neighbors = [[] for _ in range(N)]
    for d in range(1, MAX_D + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            p0 = a - 1
            p1 = p0 + d
            p2 = p0 + 2 * d
            p3 = p0 + 3 * d
            
            neighbors[p0].append((p1, p2, p3))
            neighbors[p1].append((p0, p2, p3))
            neighbors[p2].append((p0, p1, p3))
            neighbors[p3].append((p0, p1, p2))
    return neighbors

_NEIGHBORS = _precompute_ap_neighbors()

def _count_mono4(c):
    """Count monochromatic 4-APs."""
    count = 0
    for d in range(1, MAX_D + 1):
        max_a = N - 3 * d
        if max_a <= 0: continue
        c0 = c[0:max_a]
        c1 = c[d:d + max_a]
        c2 = c[2 * d:2 * d + max_a]
        c3 = c[3 * d:3 * d + max_a]
        mask = (c0 == c1) & (c0 == c2) & (c0 == c3)
        count += int(np.sum(mask))
    return count

def _delta_flip(c, pos, neighbors):
    v_pos = c[pos]
    v_pos_new = 1 - v_pos
    delta = 0
    for n1, n2, n3 in neighbors[pos]:
        v1 = c[n1]
        v2 = c[n2]
        v3 = c[n3]
        if v1 == v2 and v2 == v3:
            if v1 == v_pos:
                delta -= 1
            else:
                delta += 1
    return delta

def _local_search(c, time_limit, rng):
    """Standard hill climbing local search."""
    start = time.time()
    best_c = c.copy()
    best_count = _count_mono4(best_c)
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        order = list(range(N))
        rng.shuffle(order)
        for pos in order:
            if (time.time() - start) >= time_limit:
                break
            delta = _delta_flip(best_c, pos, _NEIGHBORS)
            if delta < 0:
                best_c[pos] = 1 - best_c[pos]
                best_count += delta
                improved = True
    return best_c, best_count

def construct():
    deadline = time.time() + 140 # 10s safety margin
    rng = random.Random(42)
    
    # Initial states: Thue-Morse and Random
    c_tm = np.zeros(N, dtype=np.int8)
    for i in range(1, N):
        c_tm[i] = c_tm[i >> 1] ^ (i & 1)
        
    c_rand = np.array([rng.randint(0,1) for _ in range(N)], dtype=np.int8)
    
    # Start with Thue-Morse, it's usually better
    best_c = c_tm.copy()
    best_count = _count_mono4(best_c)
    
    # Phase 1: Local Search on initial state
    remaining = deadline - time.time()
    if remaining > 10:
        best_c, best_count = _local_search(best_c, min(5.0, remaining - 5), rng)
        
    # Phase 2: Parallel Tempering
    # We will run PT for the remaining time
    # Number of replicas
    R = 12
    # Temperatures: geometric progression
    T_min = 0.5
    T_max = 10.0
    temperatures = np.geomspace(T_min, T_max, R)
    
    # Initialize replicas
    # Replica 0 is our current best
    # Others are random or perturbed
    replicas = [best_c.copy()]
    energies = [best_count]
    
    for k in range(1, R):
        # Perturb best_c
        c_rep = best_c.copy()
        # Flip some random bits
        num_flips = int(N * 0.2 * (k/R)) # More perturbation for higher temp
        for _ in range(num_flips):
            p = rng.randint(0, N-1)
            c_rep[p] = 1 - c_rep[p]
        e_rep = _count_mono4(c_rep)
        replicas.append(c_rep)
        energies.append(e_rep)
        
    # PT Loop
    start_time = time.time()
    steps = 0
    while (time.time() - start_time) < (deadline - time.time() - 10): # Leave 10s for final polish
        # 1. Update each replica
        for k in range(R):
            # Propose flip
            pos = rng.randint(0, N-1)
            delta = _delta_flip(replicas[k], pos, _NEIGHBORS)
            # Metropolis
            if delta < 0 or rng.random() < np.exp(-delta / temperatures[k]):
                replicas[k][pos] = 1 - replicas[k][pos]
                energies[k] += delta
                
        # 2. Swap neighboring replicas
        for k in range(R - 1):
            # Check swap between k and k+1
            # E_k, T_k and E_{k+1}, T_{k+1}
            # Delta E = (E_{k+1} - E_k) / T_k - (E_k - E_{k+1}) / T_{k+1}
            # Wait, the standard criterion is:
            # P(accept) = min(1, exp( - (E_k/T_k + E_{k+1}/T_{k+1}) + (E_{k+1}/T_k + E_k/T_{k+1}) ) )
            # = min(1, exp( -E_k(1/T_k - 1/T_{k+1}) - E_{k+1}(1/T_{k+1} - 1/T_k) ) )
            # Let D = 1/T_k - 1/T_{k+1}
            # Exp arg = -E_k * D + E_{k+1} * D = D * (E_{k+1} - E_k)
            inv_T_k = 1.0 / temperatures[k]
            inv_T_k1 = 1.0 / temperatures[k+1]
            D = inv_T_k - inv_T_k1
            swap_prob = np.exp(D * (energies[k+1] - energies[k]))
            if swap_prob >= 1.0 or rng.random() < swap_prob:
                # Swap
                replicas[k], replicas[k+1] = replicas[k+1], replicas[k]
                energies[k], energies[k+1] = energies[k+1], energies[k]
                
        steps += 1
        
        # Track best
        if energies[0] < best_count: # Replica 0 is the coldest
            best_count = energies[0]
            best_c = replicas[0].copy()
            
    # Phase 3: Final Local Search on best found
    remaining = deadline - time.time()
    if remaining > 2:
        best_c, best_count = _local_search(best_c, min(remaining - 1, 5), rng)
        
    result = [int(x) for x in best_c]
    assert len(result) == 256
    assert all(x in (0, 1) for x in result)
    return result
