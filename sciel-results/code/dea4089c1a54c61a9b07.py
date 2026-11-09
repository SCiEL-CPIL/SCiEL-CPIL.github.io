"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

def _neighbours(point):
    x, y, z = point
    yield (x+1, y, z)
    yield (x-1, y, z)
    yield (x, y+1, z)
    yield (x, y-1, z)
    yield (x, y, z+1)
    yield (x, y, z-1)

def _energy(sequence, chain):
    """Calculate total energy of a conformation."""
    where = {p: i for i, p in enumerate(chain)}
    h_positions = [i for i, c in enumerate(sequence) if c == 'H']
    contacts = 0
    for i in h_positions:
        p = chain[i]
        x, y, z = p
        # Unroll neighbor loop for speed
        for q in ((x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)):
            j = where.get(q)
            if j is not None and j > i + 1:
                if sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _grow_greedy(sequence, dimension, rng, bias=1.0):
    """
    Generate a self-avoiding walk with a greedy bias towards H-H contacts and space.
    """
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    h_set = set(i for i, c in enumerate(sequence) if c == 'H')
    
    for i in range(1, len(sequence)):
        last = chain[-1]
        options = []
        x, y, z = last
        for site in ((x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)):
            if site in occupied:
                continue
            
            score = 0
            # Contact Score
            if i in h_set:
                sx, sy, sz = site
                for q in ((sx+1,sy,sz),(sx-1,sy,sz),(sx,sy+1,sz),(sx,sy-1,sz),(sx,sy,sz+1),(sx,sy,sz-1)):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and j in h_set:
                            score += 2.0
            
            # Space Score
            sx, sy, sz = site
            empty_neighbors = 0
            for q in ((sx+1,sy,sz),(sx-1,sy,sz),(sx,sy+1,sz),(sx,sy-1,sz),(sx,sy,sz+1),(sx,sy,sz-1)):
                if q not in occupied and q != last:
                    empty_neighbors += 1
            score += empty_neighbors * 0.5
            
            # Randomness
            score += rng.random() * 1.0
            options.append((score, site))
        
        if not options:
            return None
            
        options.sort(reverse=True)
        # Use bias to decide how random the choice is
        # bias 1.0 -> always best, bias 0.0 -> random
        if bias > 0.8:
            site = options[0][1]
        else:
            top_k = min(3, len(options))
            # Weighted random from top k
            site = options[rng.randint(0, top_k-1)][1]
            
        chain.append(site)
        occupied[site] = i
        
    return chain

def _tail_regrowth(sequence, chain, occupied, k, rng, temperature=1.0):
    """
    Regrow the tail from index k to end.
    Returns new_chain, new_occupied, or None if stuck.
    """
    if k >= len(chain):
        return chain, occupied
    
    # Remove tail from k to end
    new_occupied = dict(occupied)
    for i in range(k, len(chain)):
        del new_occupied[chain[i]]
    
    new_chain = chain[:k]
    h_set = set(i for i, c in enumerate(sequence) if c == 'H')
    
    for i in range(k, len(sequence)):
        last = new_chain[-1]
        options = []
        x, y, z = last
        for site in ((x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)):
            if site in new_occupied:
                continue
            
            score = 0
            # Contact Score
            if i in h_set:
                sx, sy, sz = site
                for q in ((sx+1,sy,sz),(sx-1,sy,sz),(sx,sy+1,sz),(sx,sy-1,sz),(sx,sy,sz+1),(sx,sy,sz-1)):
                    if q in new_occupied:
                        j = new_occupied[q]
                        if j < i - 1 and j in h_set:
                            score += 2.0
            
            # Space Score
            sx, sy, sz = site
            empty_neighbors = 0
            for q in ((sx+1,sy,sz),(sx-1,sy,sz),(sx,sy+1,sz),(sx,sy-1,sz),(sx,sy,sz+1),(sx,sy,sz-1)):
                if q not in new_occupied and q != last:
                    empty_neighbors += 1
            score += empty_neighbors * 0.5
            
            # Randomness scaled by temperature
            score += rng.random() * temperature
            options.append((score, site))
        
        if not options:
            return None, None
            
        # Boltzmann selection
        if temperature < 0.01:
            options.sort(reverse=True)
            site = options[0][1]
        else:
            scores = [s for s, _ in options]
            max_score = max(scores)
            # Avoid overflow
            exp_scores = [math.exp((s - max_score) / temperature) for s in scores]
            total = sum(exp_scores)
            r = rng.random() * total
            cumulative = 0
            site = options[0][1]
            for idx, (score, s) in enumerate(options):
                cumulative += exp_scores[idx]
                if r <= cumulative:
                    site = s
                    break
        
        new_chain.append(site)
        new_occupied[site] = i
        
    return new_chain, new_occupied

def _sa_fold(sequence, dimension, rng, time_limit):
    """
    Simulated Annealing for a single sequence.
    """
    n = len(sequence)
    if n < 2:
        return [(i, 0, 0) for i in range(n)]
        
    # Initialize with a few random greedy grows, pick the best
    best_init = None
    best_init_energy = float('inf')
    
    for _ in range(10):
        init_chain = _grow_greedy(sequence, dimension, rng, bias=0.5)
        if init_chain is None:
            continue
        e = _energy(sequence, init_chain)
        if e < best_init_energy:
            best_init_energy = e
            best_init = init_chain
            
    if best_init is None:
        best_init = [(i, 0, 0) for i in range(n)]
        best_init_energy = _energy(sequence, best_init)
        
    current_chain = best_init
    current_occupied = {p: i for i, p in enumerate(current_chain)}
    current_energy = best_init_energy
    
    best_chain = list(current_chain)
    best_energy = current_energy
    
    start_time = time.time()
    
    # Annealing schedule
    T = 2.0
    T_min = 0.01
    cooling_rate = 0.995
    
    while T > T_min:
        if time.time() - start_time > time_limit:
            break
            
        # Pick a random cut point k
        # Biased towards the end
        # Tail length L = n - k
        # We want short tails often, but occasionally long ones.
        # Exponential distribution for tail length?
        # Mean tail length ~ 10-20
        tail_len = int(rng.expovariate(1.0 / 15.0))
        tail_len = max(1, min(n - 1, tail_len))
        k = n - tail_len
        
        new_chain, new_occupied = _tail_regrowth(sequence, current_chain, current_occupied, k, rng, T)
        
        if new_chain is None:
            T *= cooling_rate
            continue
            
        new_energy = _energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
            current_chain = new_chain
            current_occupied = new_occupied
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
                
        T *= cooling_rate
        
    return best_chain

def fold(sequence, dimension):
    """
    Best conformation using multiple SA restarts.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
        
    # Total time budget is 330s for 26 sequences.
    # Approx 12s per sequence.
    # We will run multiple SA restarts.
    
    # Allocate 11 seconds per sequence
    seq_time_limit = 11.0
    
    best_chain = None
    best_energy = float('inf')
    
    # Number of restarts
    # Each restart has a time slice
    # Let's do 5 restarts
    num_restarts = 5
    slice_time = seq_time_limit / num_restarts
    
    rng = random.Random(hash(sequence) % (2**32))
    
    for restart in range(num_restarts):
        sa_chain = _sa_fold(sequence, dimension, rng, slice_time)
        
        if sa_chain is not None:
            e = _energy(sequence, sa_chain)
            if e < best_energy:
                best_energy = e
                best_chain = sa_chain
                
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
