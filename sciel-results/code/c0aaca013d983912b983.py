"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute total H-H contact energy for a chain."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _greedy_grow(sequence, dimension, rng, noise_scale=0.0):
    """
    Grow a chain greedily, choosing the next step based on immediate H-H contact gain
    plus a small random noise to allow diversity.
    """
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0): 0}
    
    for i in range(1, len(sequence)):
        last = chain[-1]
        candidates = []
        
        for site in _neighbours(last):
            if site in occupied:
                continue
            
            # Calculate immediate gain
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == "H":
                            gain += 1
            
            # Add noise to break ties and encourage diversity
            score = gain + rng.gauss(0, noise_scale)
            candidates.append((score, site))
        
        if not candidates:
            return None
            
        # Sort by score descending
        candidates.sort(key=lambda x: -x[0])
        
        # Pick the best one (or randomly from top few if noise is high? 
        # For simplicity, just pick best with noise in score)
        best_site = candidates[0][1]
        
        chain.append(best_site)
        occupied[best_site] = i
        
    return chain


def _try_2_exchange(chain, i, j, dimension):
    """
    Attempt a 2-exchange move: swap segments [i..j-1] and [j..k-1] ? 
    Standard 2-exchange in protein folding usually means reversing two segments or swapping them.
    Let's implement a segment swap/reversal hybrid.
    Actually, a simpler and very effective move is 'segment reversal' (3-exchange in some nomenclatures, 
    but let's call it reversal) and 'segment swap'.
    
    Let's implement a move that reverses a sub-chain [i..j].
    This is valid if the new endpoints connect to the rest of the chain without collision.
    """
    if i >= j:
        return None
    
    prefix = chain[:i]
    segment = chain[i:j+1]
    suffix = chain[j+1:]
    
    # Reverse the segment
    new_segment = segment[::-1]
    
    new_chain = prefix + new_segment + suffix
    
    # Check validity
    # 1. Self-avoidance
    if len(set(new_chain)) != len(new_chain):
        return None
        
    # 2. Connectivity (unit steps)
    # Check connection between prefix and new_segment
    if prefix:
        p = prefix[-1]
        s = new_segment[0]
        if abs(p[0]-s[0]) + abs(p[1]-s[1]) + abs(p[2]-s[2]) != 1:
            return None
            
    # Check connection between new_segment and suffix
    if suffix:
        s = new_segment[-1]
        n = suffix[0]
        if abs(s[0]-n[0]) + abs(s[1]-n[1]) + abs(s[2]-n[2]) != 1:
            return None
            
    return new_chain


def _try_3_exchange(chain, i, j, k, dimension):
    """
    Attempt a 3-exchange move: Swap segments [i..j] and [k..l] ? 
    Or more commonly, cut at i, j, k and swap/reverse middle.
    Let's implement a 'segment swap': Swap [i..j] with [k..l].
    """
    if i > j or k > l or j >= k:
        return None
        
    prefix = chain[:i]
    seg1 = chain[i:j+1]
    mid = chain[j+1:k]
    seg2 = chain[k:l+1]
    suffix = chain[l+1:]
    
    # Swap seg1 and seg2
    new_chain = prefix + seg2 + mid + seg1 + suffix
    
    # Check validity
    if len(set(new_chain)) != len(new_chain):
        return None
        
    # Check connectivity
    # prefix -> seg2
    if prefix:
        p = prefix[-1]
        s = seg2[0]
        if abs(p[0]-s[0]) + abs(p[1]-s[1]) + abs(p[2]-s[2]) != 1:
            return None
            
    # seg2 -> mid
    if mid:
        s = seg2[-1]
        m = mid[0]
        if abs(s[0]-m[0]) + abs(s[1]-m[1]) + abs(s[2]-m[2]) != 1:
            return None
            
    # mid -> seg1
    if mid:
        m = mid[-1]
        s = seg1[0]
        if abs(m[0]-s[0]) + abs(m[1]-s[1]) + abs(m[2]-s[2]) != 1:
            return None
            
    # seg1 -> suffix
    if suffix:
        s = seg1[-1]
        n = suffix[0]
        if abs(s[0]-n[0]) + abs(s[1]-n[1]) + abs(s[2]-n[2]) != 1:
            return None
            
    return new_chain


def _local_search(sequence, chain, dimension, time_limit, rng):
    """
    Simulated Annealing with 2-exchanges (reversals) and 3-exchanges (swaps).
    """
    current = list(chain)
    current_energy = _compute_energy(sequence, current)
    best = list(current)
    best_energy = current_energy
    
    n = len(sequence)
    start_time = time.time()
    
    # Cooling schedule
    # Start high, end low
    T = 1.0
    T_end = 0.01
    # Estimate iterations based on time
    # We don't know exact cost per iteration, so we just run until time_limit
    # and adjust T roughly.
    
    while time.time() - start_time < time_limit:
        # Randomly choose a move type
        if rng.random() < 0.5:
            # Reversal (2-exchange style)
            i = rng.randint(0, n - 2)
            j = rng.randint(i + 1, n - 1)
            new_chain = _try_2_exchange(current, i, j, dimension)
        else:
            # Swap (3-exchange style)
            # Pick two disjoint segments
            # To keep it simple, pick random i, j, k, l
            # This might be inefficient if we pick overlapping or invalid ranges often
            # Let's just pick i, j for seg1 and k, l for seg2
            i = rng.randint(0, n - 4)
            j = rng.randint(i + 1, n - 3)
            k = rng.randint(j + 2, n - 2)
            l = rng.randint(k + 1, n - 1)
            new_chain = _try_3_exchange(current, i, j, k, l, dimension)
            
        if new_chain is None:
            continue
            
        new_energy = _compute_energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        # Metropolis acceptance
        if delta < 0:
            accept = True
        else:
            # Accept with probability exp(-delta / T)
            # delta is positive (worse energy, less negative)
            # Energy is negative, so worse means closer to 0.
            # So delta > 0 means we are losing contacts.
            if T > 0:
                accept = rng.random() < (2.718 ** (-delta / T))
            else:
                accept = False
                
        if accept:
            current = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best = list(current)
                best_energy = current_energy
        
        # Cool down
        # Simple exponential cooling
        T *= 0.995
        if T < T_end:
            T = T_end
            
    return best, best_energy


def fold(sequence, dimension):
    """
    Fold using multiple greedy initializations and simulated annealing local search.
    """
    deadline = time.time() + 280  # Safety margin before 300s limit
    
    rng = random.Random(12345 + len(sequence))
    
    # Phase 1: Generate diverse initial chains
    initial_chains = []
    num_initials = 10
    # Allocate 20% of time to generation
    gen_time_limit = 20.0
    gen_deadline = time.time() + gen_time_limit
    
    for i in range(num_initials):
        if time.time() > gen_deadline:
            break
        # Increase noise for later runs to ensure diversity
        noise = 0.1 + 0.1 * i
        chain = _greedy_grow(sequence, dimension, rng, noise_scale=noise)
        if chain is not None:
            initial_chains.append(chain)
            
    if not initial_chains:
        # Fallback: straight line
        return [list(p) for p in [(i, 0, 0) for i in range(len(sequence))]]
        
    # Phase 2: Local search on the best few initial chains
    # Evaluate initial chains
    scored_initials = []
    for chain in initial_chains:
        energy = _compute_energy(sequence, chain)
        scored_initials.append((energy, chain))
        
    scored_initials.sort(key=lambda x: x[0])
    
    # Pick top 3 for refinement
    top_candidates = scored_initials[:3]
    
    best_chain = top_candidates[0][1]
    best_energy = top_candidates[0][0]
    
    # Allocate remaining time to local search
    # Split time among candidates
    remaining_time = deadline - time.time()
    if remaining_time <= 0:
        return [list(p) for p in best_chain]
        
    # Give more time to the best candidate
    # e.g. 50% to best, 25% to 2nd, 25% to 3rd
    time_alloc = [remaining_time * 0.5, remaining_time * 0.25, remaining_time * 0.25]
    
    for i, (init_energy, init_chain) in enumerate(top_candidates):
        if i >= len(time_alloc):
            break
            
        if time.time() > deadline - 5:
            break
            
        t_limit = time_alloc[i]
        if t_limit <= 0:
            continue
            
        refined_chain, refined_energy = _local_search(sequence, init_chain, dimension, t_limit, rng)
        
        if refined_energy < best_energy:
            best_chain = refined_chain
            best_energy = refined_energy
            
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
