"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320  # 10s safety margin before 330s kill


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
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
            options.append((gain + 0.1 * free + rng.random(), site))
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


def _local_search(sequence, chain, dimension, rng, time_budget):
    """
    Refine chain using Monte Carlo single-residue moves.
    Returns best chain found.
    """
    n = len(chain)
    if n < 4:
        return chain

    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy

    occupied = {p: i for i, p in enumerate(current_chain)}

    T_start = 1.5
    T_end = 0.05
    # Estimate max iterations based on time budget
    # Each move is O(1) for collision, O(1) for energy delta approx
    # Let's just run until time budget is exceeded
    end_time = time.time() + time_budget
    T = T_start
    # Cooling schedule: we don't know total iterations, so use a fixed step count per second?
    # Better: run for a fixed number of iterations, but break if time runs out.
    # Let's use a large number of iterations and break on time.
    max_iters = 50000 
    alpha = (T_end / T_start) ** (1.0 / max_iters)
    
    for _ in range(max_iters):
        if time.time() > end_time:
            break
        
        i = rng.randint(1, n - 2)
        old_site = current_chain[i]
        
        # Try to move residue i to a random neighbor
        neighbors = list(_neighbours(old_site))
        rng.shuffle(neighbors)
        
        moved = False
        for new_site in neighbors:
            # Cannot move to previous or next residue
            if new_site == current_chain[i-1] or new_site == current_chain[i+1]:
                continue
            # Check if new_site is occupied by another part of the chain
            if new_site in occupied:
                continue
            
            # Perform move
            # 1. Remove old_site from occupied
            del occupied[old_site]
            # 2. Add new_site to occupied
            occupied[new_site] = i
            # 3. Update chain
            current_chain[i] = new_site
            
            # Calculate energy delta
            # Energy changes only for contacts involving i, i-1, i+1, and their neighbors
            # It's safer and often fast enough to recalculate full energy for small N, 
            # but for large N, delta is better. 
            # Given the time budget, let's try full energy recalc first for simplicity, 
            # but if it's too slow, we optimize. 
            # Actually, _energy is O(N * deg). For N=136, that's ~800 ops. 
            # 50k iterations * 800 ops = 40M ops. Python might handle this in ~2-5s.
            
            new_energy = _energy(sequence, current_chain)
            delta = new_energy - current_energy
            
            if delta < 0 or rng.random() < pow(2.718281828, -delta / T):
                # Accept
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
                moved = True
                break
            else:
                # Reject: revert
                current_chain[i] = old_site
                del occupied[new_site]
                occupied[old_site] = i
        
        T *= alpha
    
    return best_chain


def fold(sequence, dimension):
    """Best conformation of a fixed number of randomized greedy growths with local search."""
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 0
    
    n = len(sequence)
    
    # Determine time allocation
    # We have ~320s total.
    # Let's reserve 10s for safety.
    # We will do a few rounds of (Growth + Local Search).
    # Or, just do Growth many times, and Local Search on the best ones?
    # Strategy: 
    # 1. Generate K initial chains via greedy growth.
    # 2. Select top M chains by energy.
    # 3. Apply local search to each of the M chains with a time slice.
    # 4. Return the best.
    
    # Estimate time per growth: very fast.
    # Estimate time per local search: depends on N.
    
    if n <= 48:
        n_restarts = 100
        ls_time_per_chain = 1.0 # 1s per chain
        top_m = 10
    elif n <= 80:
        n_restarts = 50
        ls_time_per_chain = 2.0
        top_m = 5
    else:
        n_restarts = 20
        ls_time_per_chain = 5.0
        top_m = 3

    # Phase 1: Generate candidates
    candidates = []
    for _ in range(n_restarts):
        if time.time() > _DEADLINE - 10: # Keep 10s buffer
            break
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        value = _energy(sequence, chain)
        candidates.append((value, chain))
        
    if not candidates:
        # Fallback: straight line
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        return [list(p) for p in best]

    # Sort by energy (most negative is best)
    candidates.sort(key=lambda x: x[0])
    
    # Phase 2: Refine top candidates
    # Allocate time based on remaining budget
    time_left = _DEADLINE - 15 - time.time()
    if time_left <= 0:
        best, best_energy = candidates[0]
        return [list(p) for p in best]
    
    # We want to spend time on local search for the top M chains
    # Let's give each chain a slice of the remaining time
    # But we also want to try as many as possible.
    # Let's iterate through top M and give each a portion of time_left / M
    # Or, better, run them sequentially and stop when time is up.
    
    # To maximize chance, let's refine the very best first, then next, etc.
    # If we run out of time, we return the best found so far.
    
    current_best_chain, current_best_energy = candidates[0]
    
    for i in range(min(top_m, len(candidates))):
        if time.time() > _DEADLINE - 5:
            break
        cand_energy, cand_chain = candidates[i]
        # If this candidate is already worse than the best found, and we are late, skip?
        # No, local search might improve it significantly.
        
        # Allocate time slice
        # Let's give each refinement a max of 5 seconds, or remaining time / remaining candidates
        slice_time = min(5.0, ( _DEADLINE - 5 - time.time()) / (top_m - i))
        if slice_time < 0.1:
            break
            
        refined_chain = _local_search(sequence, cand_chain, dimension, rng, slice_time)
        refined_energy = _energy(sequence, refined_chain)
        
        if refined_energy < current_best_energy:
            current_best_chain = refined_chain
            current_best_energy = refined_energy

    return [list(p) for p in current_best_chain]
# EVOLVE-BLOCK-END
