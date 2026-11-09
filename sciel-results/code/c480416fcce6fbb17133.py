"""
HP lattice protein folding on the cubic lattice.
Strategy: Tail Regrowth with Metropolis Acceptance (Simulated Annealing).
We abandon the GA population approach in favor of a single-trajectory
Monte Carlo method with large-scale moves (tail regrowth). This allows
the chain to restructure significant portions of its backbone,
escaping local optima that trap local search methods like pivots/slides.
The move is guided by a greedy heuristic that favors H-H contacts,
but the acceptance criterion (Metropolis) allows uphill moves early on
to explore the energy landscape.
"""
import random
import time
import math

_START_TIME = time.time()
_DEADLINE = _START_TIME + 325.0  # Safety margin before 330s kill

def _neighbours_list(point):
    """Return list of 6 face-neighbours of point in 3D."""
    x, y, z = point
    return [(x+1,y,z), (x-1,y,z), (x,y+1,z), (x,y-1,z), (x,y,z+1), (x,y,z-1)]

def _energy(sequence, chain):
    """Compute total energy: negative count of non-adjacent H-H contacts."""
    if not chain:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    n = len(chain)
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours_list(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _greedy_grow(sequence, rng, bias=1.0):
    """
    Fast initial self-avoiding walk with heuristic bias towards H contacts.
    Used to generate the initial chain.
    """
    n = len(sequence)
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        last = chain[-1]
        neighbors = _neighbours_list(last)
        
        options = []
        for site in neighbors:
            if site in occupied:
                continue
            
            site_neighbors = _neighbours_list(site)
            h_score = 0
            free_score = 0
            
            for sn in site_neighbors:
                if sn in occupied:
                    idx = occupied[sn]
                    if sequence[idx] == 'H' and abs(idx - i) > 1:
                        h_score += 1
                else:
                    free_score += 1
            
            score = h_score * bias + free_score * 0.1 + rng.random() * 0.5
            options.append((score, site))
        
        if not options:
            return None
        
        options.sort(reverse=True)
        top_k = min(3, len(options))
        site = options[rng.randint(0, top_k - 1)][1]
        
        occupied[site] = i
        chain.append(site)
        
    return chain

def _regrow_tail(sequence, chain, cut_idx, rng):
    """
    Regrow the tail of the chain starting from cut_idx.
    Returns a new chain or None if stuck.
    """
    n = len(sequence)
    if cut_idx >= n:
        return None
        
    head = chain[:cut_idx]
    if not head:
        return None
        
    occupied = {p: i for i, p in enumerate(head)}
    current_pos = head[-1]
    new_tail = []
    
    for k in range(cut_idx, n):
        neighbors = _neighbours_list(current_pos)
        options = []
        
        for site in neighbors:
            if site in occupied:
                continue
            
            site_neighbors = _neighbours_list(site)
            h_score = 0
            free_score = 0
            
            for sn in site_neighbors:
                if sn in occupied:
                    idx = occupied[sn]
                    if sequence[idx] == 'H' and abs(idx - k) > 1:
                        h_score += 1
                else:
                    free_score += 1
            
            # Strong bias towards H contacts to guide the regrowth
            score = h_score * 10.0 + free_score * 0.1 + rng.random() * 0.5
            options.append((score, site))
            
        if not options:
            return None
        
        options.sort(reverse=True)
        top_k = min(3, len(options))
        site = options[rng.randint(0, top_k - 1)][1]
        
        occupied[site] = k
        new_tail.append(site)
        current_pos = site
        
    return head + new_tail

def _sa_fold(sequence, dimension, time_budget, rng):
    """
    Simulated Annealing with Tail Regrowth.
    """
    n = len(sequence)
    deadline = time.time() + time_budget
    
    # Initialize with best of a few greedy chains
    best_chain = None
    best_energy = 0
    
    for _ in range(5):
        chain = _greedy_grow(sequence, rng, bias=1.0 + rng.random())
        if chain:
            e = _energy(sequence, chain)
            if best_chain is None or e < best_energy:
                best_chain = chain
                best_energy = e
                
    if best_chain is None:
        return [(i, 0, 0) for i in range(n)], 0
        
    current_chain = best_chain[:]
    current_energy = best_energy
    
    T = 2.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    while time.time() < deadline:
        # Pick a cut point
        # Bias towards earlier cuts to allow more restructuring
        # But not too early to avoid losing too much valid structure
        # Uniform is a safe start, maybe slightly biased to middle/early
        if n > 2:
            cut_idx = rng.randint(1, n - 2)
        else:
            cut_idx = 1
            
        new_chain = _regrow_tail(sequence, current_chain, cut_idx, rng)
        
        if new_chain is None:
            continue
            
        new_energy = _energy(sequence, new_chain)
        
        # Metropolis Acceptance
        delta_e = new_energy - current_energy
        if delta_e < 0 or (T > T_min and rng.random() < math.exp(-delta_e / T)):
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
            
    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Best conformation using Simulated Annealing with Tail Regrowth.
    """
    n = len(sequence)
    
    elapsed = time.time() - _START_TIME
    remaining = _DEADLINE - elapsed
    if remaining < 1.0:
        return [[i, 0, 0] for i in range(n)]
    
    # Allocate time per sequence
    # We have 26 sequences. Total budget 325s.
    # Average ~12s per sequence.
    # But some are shorter. Let's allocate based on length roughly.
    # Longer chains take longer to evaluate energy and regrow.
    # Let's use a fixed budget of 10s, or remaining - 5s if tight.
    budget = 10.0
    if remaining - budget < 5.0:
        budget = remaining - 5.0
        
    if budget < 1.0:
        budget = 0.5
        
    rng = random.Random(12345 + len(sequence))
    
    # Run SA
    best_chain, best_energy = _sa_fold(sequence, dimension, budget, rng)
    
    # Final check: ensure self-avoidance
    if len(set(best_chain)) != n:
        return [[i, 0, 0] for i in range(n)]
        
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
