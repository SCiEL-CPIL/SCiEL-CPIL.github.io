import random
import time
import math
from collections import defaultdict

def _neighbours(point):
    """Generate the 6 neighbours of a point in 3D cubic lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _grow(sequence, dimension, rng, bias=1.0):
    """
    Randomized greedy chain growth.
    bias > 1: more aggressive contact seeking.
    bias < 1: more random exploration.
    """
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == "H":
                            gain += 1
            # Add randomness
            # Higher bias makes gain more important
            score = gain * bias + rng.random()
            options.append((score, site))
        
        if not options:
            return None
            
        options.sort(reverse=True)
        # Pick from top 3 with probability, or just best
        # Let's pick the best to be fast and deterministic-ish
        site = options[0][1]
        occupied[site] = i
        chain.append(site)
        
    return chain

def _energy(sequence, chain):
    """Calculate total H-H contacts (negative energy)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _tail_refold(chain, sequence, dimension, rng, k):
    """
    Refold the tail starting from index k+1.
    chain[0..k] is fixed.
    Returns new_chain or None if stuck.
    """
    n = len(chain)
    if k >= n - 1:
        return chain[:]
    
    # Fixed part
    fixed = chain[:k+1]
    new_chain = fixed[:]
    
    # Occupied set for fixed part
    occupied = {p: i for i, p in enumerate(fixed)}
    
    # We need to grow from new_chain[-1] to length n
    # Current length is k+1, we need to add n - (k+1) residues.
    # The residues to add are sequence[k+1 .. n-1]
    
    for i in range(k+1, n):
        last = new_chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == "H":
                            gain += 1
            score = gain + rng.random()
            options.append((score, site))
            
        if not options:
            return None
            
        options.sort(reverse=True)
        # Sometimes pick 2nd best to diversify
        if len(options) > 1 and rng.random() < 0.2:
            site = options[1][1]
        else:
            site = options[0][1]
            
        occupied[site] = i
        new_chain.append(site)
        
    return new_chain

def _local_optimize(chain, sequence, dimension, rng, deadline):
    """
    Local optimization using tail refolding moves.
    """
    current_chain = chain[:]
    current_energy = _energy(sequence, current_chain)
    best_chain = current_chain[:]
    best_energy = current_energy
    n = len(current_chain)
    
    # Temperature for simulated annealing of the moves
    T = 1.0
    cooling = 0.995
    
    while time.time() < deadline:
        # Pick a pivot point k
        # Bias towards smaller k (earlier) to allow larger refolds?
        # Or uniform?
        # Let's try uniform 1 to n-2
        k = rng.randint(1, n - 2)
        
        new_chain = _tail_refold(current_chain, sequence, dimension, rng, k)
        
        if new_chain is None:
            T *= cooling
            continue
            
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        # Accept if better or with prob exp(-delta/T)
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = current_chain[:]
                
        T *= cooling
        
        # Early stop if T is too low
        if T < 0.01:
            break
            
    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Main entry point.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    
    rng = random.Random(42)
    
    # Deadline management
    start_time = time.time()
    # Evaluator kills at 330s. Let's aim for 320s to be safe.
    deadline = start_time + 320.0
    
    best_chain = None
    best_energy = 0 # 0 is no contacts (worst for H-rich)
    
    # Phase 1: Generate diverse initial solutions
    # We have ~320s total.
    # Let's allocate 10% for generation, 90% for optimization.
    # Or just interleave.
    
    # Generate a few initial chains
    init_chains = []
    
    # Greedy with different biases
    for bias in [1.0, 1.5, 2.0, 0.5]:
        c = _grow(sequence, dimension, rng, bias=bias)
        if c:
            init_chains.append(c)
            
    # Pure random walks
    for _ in range(5):
        c = _grow(sequence, dimension, rng, bias=0.1)
        if c:
            init_chains.append(c)
            
    if not init_chains:
        # Fallback: straight line
        return [[i, 0, 0] for i in range(n)]
        
    # Phase 2: Optimize
    # We will optimize each chain for a portion of the remaining time.
    # To maximize diversity, we can run them in parallel? No, single process.
    # We can time-slice.
    
    # Let's reserve the last 20% of time for a "best so far" deep dive.
    # For now, just distribute time among initial chains.
    
    num_chains = len(init_chains)
    # Leave some time for the final best chain
    time_for_each = (deadline - time.time()) / (num_chains + 1)
    
    for i, c in enumerate(init_chains):
        if time.time() > deadline - 10.0: # Leave 10s for final
            break
            
        # Time slice for this chain
        chain_deadline = time.time() + time_for_each
        # Cap chain_deadline at global deadline
        if chain_deadline > deadline:
            chain_deadline = deadline
            
        opt_chain, opt_energy = _local_optimize(c, sequence, dimension, rng, chain_deadline)
        
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_chain = opt_chain
            
    # Phase 3: Deep dive on the best chain found
    if best_chain is not None:
        # Use remaining time
        remaining_deadline = deadline
        # If we have significant time left, run another round
        if time.time() < remaining_deadline - 5.0:
            opt_chain, opt_energy = _local_optimize(best_chain, sequence, dimension, rng, remaining_deadline)
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_chain = opt_chain

    if best_chain is None:
        best_chain = init_chains[0]
        
    return [list(p) for p in best_chain]
