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

def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            # Slightly less greedy to allow diversity
            options.append((gain + 0.05 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _delta_single(sequence, chain, occupied, i, new_pos):
    """Calculate energy delta for moving residue i to new_pos."""
    old_pos = chain[i]
    if new_pos == old_pos:
        return 0
    
    # Check if new_pos is valid (free)
    if new_pos in occupied and occupied[new_pos] != i:
        return float('inf')
        
    # Check connectivity: new_pos must be neighbor of i-1 and i+1
    if i > 0:
        prev = chain[i-1]
        if abs(new_pos[0]-prev[0]) + abs(new_pos[1]-prev[1]) + abs(new_pos[2]-prev[2]) != 1:
            return float('inf')
    if i < len(chain)-1:
        next_p = chain[i+1]
        if abs(new_pos[0]-next_p[0]) + abs(new_pos[1]-next_p[1]) + abs(new_pos[2]-next_p[2]) != 1:
            return float('inf')

    delta = 0
    if sequence[i] == 'H':
        # Lost contacts with old_pos
        for q in _neighbours(old_pos):
            j = occupied.get(q)
            if j is not None and j != i:
                if sequence[j] == 'H':
                    delta -= 1
        # Gained contacts with new_pos
        for q in _neighbours(new_pos):
            j = occupied.get(q)
            if j is not None and j != i:
                if sequence[j] == 'H':
                    delta += 1
    return delta

def _sa_search(sequence, chain, rng, deadline):
    """Simulated Annealing local search."""
    n = len(chain)
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    curr_chain = list(chain)
    curr_energy = best_energy
    occupied = {p: i for i, p in enumerate(curr_chain)}
    
    T_start = 1.0
    T_end = 0.01
    # Estimate steps: 10000 steps per second? 
    # Let's just run until deadline.
    
    while time.time() < deadline:
        T = T_start * (T_end / T_start) ** (time.time() - (deadline - 1.0)) / (1.0) # Rough cooling
        # Better cooling: exponential
        # Let's use a fixed number of steps per time chunk? 
        # Simpler: just use a global temperature that cools over the duration of this call.
        # We don't know the exact start time of this call easily without passing it.
        # Let's pass start_time.
        
        # Actually, let's just use a random move and Metropolis acceptance.
        # Temperature: Start high, cool down.
        # We'll use a simple schedule: T = T_start * (0.999 ** steps)
        
        # Pick a move
        i = rng.randint(1, n - 2) # Single move for interior
        # Or pick between single and crank?
        # Let's stick to single for now.
        
        old_pos = curr_chain[i]
        # Generate candidates
        candidates = []
        for site in _neighbours(curr_chain[i-1]):
            if site in _neighbours(curr_chain[i+1]):
                if site not in occupied or occupied[site] == i:
                    if site != old_pos:
                        candidates.append(site)
        
        if not candidates:
            continue
            
        new_pos = rng.choice(candidates)
        
        delta = _delta_single(sequence, curr_chain, occupied, i, new_pos)
        
        if delta < 0 or rng.random() < math.exp(-delta / T):
            # Accept
            occupied[old_pos] = -1
            occupied[new_pos] = i
            curr_chain[i] = new_pos
            curr_energy += delta
            
            if curr_energy < best_energy:
                best_energy = curr_energy
                best_chain = list(curr_chain)
        
        # Cool down
        T *= 0.9999
        
    return best_chain, best_energy

def fold(sequence, dimension):
    start_time = time.time()
    per_seq_deadline = start_time + 10.0
    
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 1
    
    # Phase 1: Growth
    # Run multiple growths
    num_growths = 20
    growth_deadline = per_seq_deadline - 6.0 # Leave 6s for SA
    
    initial_chains = []
    for _ in range(num_growths):
        if time.time() > growth_deadline:
            break
        chain = _grow(sequence, dimension, rng)
        if chain is not None:
            e = _energy(sequence, chain)
            initial_chains.append((e, chain))
            
    if not initial_chains:
        # Fallback
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        best_energy = _energy(sequence, best_chain)
        return [list(p) for p in best_chain]
        
    # Sort by energy
    initial_chains.sort(key=lambda x: x[0])
    
    # Phase 2: SA on best few
    # Take top 3
    top_chains = initial_chains[:3]
    sa_deadline = per_seq_deadline - 0.5
    
    for e, chain in top_chains:
        if time.time() > sa_deadline:
            break
        # Run SA for a portion of the time
        # If 3 chains, give each ~2s
        sa_time_per_chain = 2.0
        chain_deadline = time.time() + sa_time_per_chain
        if chain_deadline > sa_deadline:
            chain_deadline = sa_deadline
            
        new_chain, new_energy = _sa_search(sequence, chain, rng, chain_deadline)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_chain = new_chain
            
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
