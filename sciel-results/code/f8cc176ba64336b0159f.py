"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    """Calculate total energy of a conformation."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _initial_grow(sequence, dimension, rng):
    """
    Generate an initial self-avoiding walk using a greedy randomized approach.
    This ensures we start with a valid conformation for annealing.
    """
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            # Heuristic: prefer sites that create H-H contacts if current is H
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site) 
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            # Slight preference for open space to avoid dead ends
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 10.0 + free * 0.1 + rng.random()
            options.append((score, site))
        
        if not options:
            return None # Dead end, restart needed
        
        # Pick the best option (or random among top few for diversity, but greedy is faster for init)
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
        
    return chain

def _simulate_annealing(sequence, dimension, rng, time_limit, start_chain=None):
    """
    Perform Simulated Annealing starting from a given chain or a new greedy growth.
    Returns the best chain and energy found during this run.
    """
    N = len(sequence)
    if start_chain is None:
        chain = _initial_grow(sequence, dimension, rng)
        if chain is None:
            return None, 0
    else:
        chain = list(start_chain)
    
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    # Cooling schedule parameters
    # Start temperature high enough to accept bad moves, cool down to fine-tune
    # Empirically, T_start ~ 1.0 to 2.0 works well for HP models
    T_start = 2.0
    T_min = 0.01
    # We don't know exact steps in advance, so we use a time-based or step-based cooling.
    # Let's use a step-based cooling with a check on time.
    # Estimated steps: N * 50 iterations per residue is a common heuristic.
    total_steps = N * 100 
    # Cooling rate alpha such that T_end = T_start * alpha^total_steps
    # alpha = (T_min / T_start) ** (1.0 / total_steps)
    if total_steps == 0:
        return best_chain, best_energy
    alpha = (T_min / T_start) ** (1.0 / total_steps)
    
    T = T_start
    start_time = time.time()
    
    for step in range(total_steps):
        if time.time() - start_time > time_limit:
            break
            
        # Select a random residue to move (excluding ends usually to reduce boundary issues, but all is fine)
        # Moving ends can be tricky if they get trapped, but generally ok.
        i = rng.randint(1, N - 2) # Avoid ends for stability in some implementations, but let's try 1..N-2
        
        p = chain[i]
        
        # Try to move to a random neighbor
        # To make it more efficient, we can pick a random direction
        axis = rng.randint(0, dimension - 1)
        step_dir = rng.choice([-1, 1])
        
        new_p = list(p)
        new_p[axis] += step_dir
        new_p = tuple(new_p)
        
        # Check if new position is occupied
        if new_p in occupied:
            continue
            
        # Calculate Delta Energy
        # Remove p from occupied
        del occupied[p]
        
        # Energy change for removing p
        delta = 0
        if sequence[i] == "H":
            # Check neighbors of p for H contacts that are being broken
            for q in _neighbours(p):
                j = occupied.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == "H":
                    # We are breaking a contact
                    delta += 1 # Energy increases by 1 (less negative)
        
        # Place new_p
        occupied[new_p] = i
        
        # Energy change for adding new_p
        if sequence[i] == "H":
            for q in _neighbours(new_p):
                j = occupied.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == "H":
                    # We are making a contact
                    delta -= 1 # Energy decreases by 1 (more negative)
        
        new_energy = current_energy + delta
        
        # Metropolis Criterion
        if new_energy < current_energy or rng.random() < math.exp(-delta / T):
            # Accept move
            chain[i] = new_p
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
        else:
            # Reject move: revert occupied
            del occupied[new_p]
            occupied[p] = i
        
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min

    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Fold the sequence using multiple restarts of Simulated Annealing.
    """
    start_time = time.time()
    time_limit = 320.0 # Safety margin
    
    # Reserve some time for the final best tracking
    # We will run SA loops until time runs out
    
    best_chain_overall = None
    best_energy_overall = 0 # Energy is negative, so 0 is worse than -1
    
    # Initial greedy attempt to have a baseline
    rng = random.Random(12345 + len(sequence))
    
    # Try a few quick greedy grows to find a good starting point for the first SA
    # But SA itself can start from a greedy grow.
    
    while time.time() - start_time < time_limit:
        # Run one SA cycle
        # To prevent a single long SA from consuming all time, we can limit its internal time
        # or just let it run until the global limit is hit. 
        # However, multiple restarts are better. 
        # Let's allocate a portion of time to each restart, or just let the internal loop check global time.
        
        # The internal SA loop checks time. 
        # To encourage restarts, we can artificially cap the SA run time or just rely on the cooling to finish quickly for small N.
        # For large N, cooling might take long. 
        # Let's force a restart if we haven't improved for a while? 
        # Simpler: Just run SA. The internal loop breaks on global time.
        # But if one SA takes 300s, we don't get restarts.
        # Let's limit the SA steps based on time remaining? 
        # Or just accept that the first SA might take most of the time.
        # Better: Limit the SA run to a fixed number of steps or a fraction of total time.
        
        # Let's try: Run SA with a time budget of 50% of remaining time? 
        # No, that's complex. 
        # Let's just run SA. If it finishes early (due to cooling), we restart.
        # If it hits the time limit, we return the best.
        
        chain, energy = _simulate_annealing(sequence, dimension, rng, time_limit - (time.time() - start_time))
        
        if chain is not None and energy < best_energy_overall:
            best_energy_overall = energy
            best_chain_overall = chain
            
        # If the SA finished before the time limit, we should restart to explore a different basin.
        # How do we know if it finished early? 
        # _simulate_annealing returns when it finishes steps or hits time.
        # We can check if the time spent was significantly less than the limit.
        
        elapsed = time.time() - start_time
        if elapsed < time_limit:
             # It finished early (cooling done). Restart.
             # Add a small randomization to the seed for next run
             rng = random.Random(rng.randint(0, 10**9))
        else:
             # It hit the time limit. Break.
             break

    if best_chain_overall is None:
        # Fallback: straight line
        best_chain_overall = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        
    return [list(p) for p in best_chain_overall]
# EVOLVE-BLOCK-END
