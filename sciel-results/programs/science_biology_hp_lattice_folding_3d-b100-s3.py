import time
import random
import math

def _neighbours(point):
    x, y, z = point
    return [(x+1, y, z), (x-1, y, z), (x, y+1, z), (x, y-1, z), (x, y, z+1), (x, y, z-1)]

def _energy(sequence, chain):
    """
    Calculate the energy of a chain.
    Energy = - (number of H-H contacts)
    """
    if not chain:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _greedy_grow(sequence, rng):
    """
    Greedy growth to create an initial self-avoiding walk.
    """
    N = len(sequence)
    if N == 0:
        return []
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, N):
        options = []
        last = chain[-1]
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Heuristic: prefer sites that close loops or are close to other H's
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    if q in occupied:
                        idx = occupied[q]
                        if idx < i - 1 and sequence[idx] == "H":
                            gain += 1
            # Add small random noise to avoid deterministic traps
            score = gain * 10.0 + rng.random()
            options.append((score, site))
        
        if not options:
            return None
            
        best_score, best_site = max(options)
        chain.append(best_site)
        occupied[best_site] = i
        
    return chain

def fold(sequence, dimension):
    N = len(sequence)
    if N == 0:
        return []
    
    start_time = time.time()
    # 240s budget for all sequences. Assume ~9s per sequence.
    # Use 8.0s to be safe.
    deadline = start_time + 8.0
    
    rng = random.Random(42)
    
    # Initialize with best of a few greedy chains
    best_chain = None
    best_energy = 0
    
    for _ in range(10):
        chain = _greedy_grow(sequence, rng)
        if chain:
            e = _energy(sequence, chain)
            if best_chain is None or e < best_energy:
                best_chain = chain
                best_energy = e
                
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0, 0) for i in range(N)]
        best_energy = _energy(sequence, best_chain)
        
    current_chain = best_chain[:]
    current_energy = best_energy
    
    # Tail Regrowth Search
    T = 2.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    while time.time() < deadline:
        # Pick a cut point
        # Bias towards the middle? 
        # Early cuts allow more restructuring but are more expensive to regrow.
        # Late cuts are cheaper but less effective.
        # Uniform is a safe start.
        i = rng.randint(1, N - 2)
        
        head = current_chain[:i]
        # tail = current_chain[i:]
        
        # Prepare data structures for regrowth
        head_map = {p: idx for idx, p in enumerate(head)}
        occupied = set(head_map.keys())
        
        new_tail = []
        tail_map = {} # pos -> global index
        current_pos = head[-1]
        
        valid = True
        # running_energy is not strictly needed if we recompute total, 
        # but we can use it to estimate gain for the next step? 
        # No, we just pick the best next step greedily with noise.
        
        for k in range(i, N):
            options = []
            for site in _neighbours(current_pos):
                if site in occupied:
                    continue
                
                gain = 0
                if sequence[k] == "H":
                    for q in _neighbours(site):
                        # Check if q is in head
                        if q in head_map:
                            idx_h = head_map[q]
                            if sequence[idx_h] == "H":
                                gain += 1
                        # Check if q is in new_tail
                        elif q in tail_map:
                            idx_t = tail_map[q]
                            if sequence[idx_t] == "H":
                                gain += 1
                                
                # Score: maximize gain, add noise
                score = gain * 10.0 + rng.random()
                options.append((score, site))
                
            if not options:
                valid = False
                break
                
            best_score, best_site = max(options)
            new_tail.append(best_site)
            tail_map[best_site] = k
            occupied.add(best_site)
            current_pos = best_site
            
        if not valid:
            continue
            
        new_chain = head + new_tail
        new_energy = _energy(sequence, new_chain)
        
        # Metropolis Acceptance
        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
            current_chain = new_chain
            current_energy = new_energy
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = new_chain[:]
                
            # Cool down
            T *= cooling_rate
            if T < T_min:
                T = T_min
        else:
            # Rejected, no change
            pass
            
        # Occasionally restart if stuck?
        # If T is low and no improvement for a while, we could restart.
        # But the loop is fast, so let it run.
        
    return [list(p) for p in best_chain]
