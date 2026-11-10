"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time


def _neighbours2d(point):
    """Yield the 4 unit-step neighbours of a 2D point (tuple)."""
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _compute_energy(sequence, chain):
    """Compute HP energy: -number of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours2d(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _fast_grow(sequence, rng, temperature=0.5):
    """Randomized greedy chain growth biased toward H-H contact formation."""
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours2d(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours2d(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours2d(site) if q not in occupied)
            # Higher weight on gain, some exploration via temperature
            options.append((gain * 3.0 + 0.1 * free + rng.random() * temperature, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _exhaustive_local_search(sequence, chain):
    """
    Exhaustive single-residue move local search.
    Guarantees a local optimum under single-residue moves.
    For small sequences this is fast and reliable.
    """
    n = len(chain)
    if n < 4:
        return chain, _compute_energy(sequence, chain)
    
    best_chain = list(chain)
    best_energy = _compute_energy(sequence, best_chain)
    occupied = {p: i for i, p in enumerate(best_chain)}
    
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            current_pos = best_chain[i]
            for new_pos in _neighbours2d(current_pos):
                if new_pos in occupied:
                    continue
                # Check self-avoidance
                valid = True
                for q in _neighbours2d(new_pos):
                    j = occupied.get(q)
                    if j is not None and abs(j - i) > 1:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Compute delta
                delta = 0
                if sequence[i] == "H":
                    for q in _neighbours2d(current_pos):
                        j = occupied.get(q)
                        if j is not None and abs(j - i) > 1 and sequence[j] == "H":
                            delta += 1
                    for q in _neighbours2d(new_pos):
                        j = occupied.get(q)
                        if j is not None and abs(j - i) > 1 and sequence[j] == "H":
                            delta -= 1
                
                if delta < 0:
                    best_chain[i] = new_pos
                    occupied.pop(current_pos)
                    occupied[new_pos] = i
                    best_energy += delta
                    improved = True
                    break
            if improved:
                break
    
    return best_chain, best_energy


def _local_improve_fast(sequence, chain, max_iter=2000):
    """Fast local search using incremental energy deltas."""
    n = len(chain)
    if n < 4:
        return chain, _compute_energy(sequence, chain)
    
    best_chain = list(chain)
    best_energy = _compute_energy(sequence, best_chain)
    occupied = {p: i for i, p in enumerate(best_chain)}
    
    for _ in range(max_iter):
        improved = False
        for i in range(1, n - 1):
            current_pos = best_chain[i]
            for new_pos in _neighbours2d(current_pos):
                if new_pos in occupied:
                    continue
                valid = True
                for q in _neighbours2d(new_pos):
                    j = occupied.get(q)
                    if j is not None and abs(j - i) > 1:
                        valid = False
                        break
                if not valid:
                    continue
                
                delta = 0
                if sequence[i] == "H":
                    for q in _neighbours2d(current_pos):
                        j = occupied.get(q)
                        if j is not None and abs(j - i) > 1 and sequence[j] == "H":
                            delta += 1
                    for q in _neighbours2d(new_pos):
                        j = occupied.get(q)
                        if j is not None and abs(j - i) > 1 and sequence[j] == "H":
                            delta -= 1
                
                if delta < 0:
                    best_chain[i] = new_pos
                    occupied.pop(current_pos)
                    occupied[new_pos] = i
                    best_energy += delta
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best_chain, best_energy


def _min_conflicts_fold(sequence, max_iter=5000):
    """Min-conflicts style local search."""
    n = len(sequence)
    if n < 4:
        chain = [(i, 0) for i in range(n)]
        return chain, _compute_energy(sequence, chain)
    
    rng = random.Random(42 + n)
    chain = _fast_grow(sequence, rng)
    if chain is None:
        chain = [(i, 0) for i in range(n)]
    
    best_chain = list(chain)
    best_energy = _compute_energy(sequence, best_chain)
    occupied = {p: i for i, p in enumerate(best_chain)}
    
    for _ in range(max_iter):
        best_i, best_delta, best_new_pos = -1, 0, None
        for i in range(1, n - 1):
            current_pos = best_chain[i]
            for new_pos in _neighbours2d(current_pos):
                if new_pos in occupied:
                    continue
                valid = True
                for q in _neighbours2d(new_pos):
                    j = occupied.get(q)
                    if j is not None and abs(j - i) > 1:
                        valid = False
                        break
                if not valid:
                    continue
                
                delta = 0
                if sequence[i] == "H":
                    for q in _neighbours2d(current_pos):
                        j = occupied.get(q)
                        if j is not None and abs(j - i) > 1 and sequence[j] == "H":
                            delta += 1
                    for q in _neighbours2d(new_pos):
                        j = occupied.get(q)
                        if j is not None and abs(j - i) > 1 and sequence[j] == "H":
                            delta -= 1
                
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_new_pos = new_pos
        
        if best_i == -1 or best_delta >= 0:
            break
        
        old_pos = best_chain[best_i]
        best_chain[best_i] = best_new_pos
        occupied.pop(old_pos)
        occupied[best_new_pos] = best_i
        best_energy += best_delta
    
    return best_chain, best_energy


def _regrow_sa_fast(sequence, start_chain, rng, max_iter=1000):
    """Simulated annealing on chain regrowth."""
    n = len(sequence)
    if n < 6:
        return _fast_grow(sequence, rng)
    
    best_chain = list(start_chain)
    best_energy = _compute_energy(sequence, best_chain)
    
    for _ in range(max_iter):
        split = rng.randint(3, n - 3)
        prefix = best_chain[:split]
        occupied = {p: i for i, p in enumerate(prefix)}
        chain = list(prefix)
        
        success = True
        for i in range(split, n):
            options = []
            for site in _neighbours2d(chain[-1]):
                if site in occupied:
                    continue
                gain = 0
                if sequence[i] == "H":
                    gain = sum(1 for q in _neighbours2d(site)
                               if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
                free = sum(1 for q in _neighbours2d(site) if q not in occupied)
                options.append((gain * 3.0 + 0.1 * free + rng.random(), site))
            if not options:
                success = False
                break
            site = max(options)[1]
            occupied[site] = i
            chain.append(site)
        
        if not success:
            continue
        
        energy = _compute_energy(sequence, chain)
        if energy < best_energy:
            best_chain = chain
            best_energy = energy
    
    return best_chain


def fold(sequence, dimension):
    """Best conformation using multiple strategies with exhaustive local search for small sequences."""
    start_time = time.time()
    deadline = start_time + 175.0
    
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 0
    
    n = len(sequence)
    
    # For small sequences, use exhaustive local search to guarantee local optimum
    if n <= 30:
        # More attempts for small sequences
        num_attempts = 50
        for attempt in range(num_attempts):
            if time.time() > deadline:
                break
            
            # Vary temperature for diversity
            temp = 0.3 + rng.random() * 0.7
            chain = _fast_grow(sequence, rng, temperature=temp)
            if chain is None:
                continue
            
            # Exhaustive local search guarantees local optimum
            chain, energy = _exhaustive_local_search(sequence, chain)
            
            if energy < best_energy:
                best, best_energy = chain, energy
    else:
        # For larger sequences, use fast local search
        num_phase1 = max(30, 2000 // n)
        for attempt in range(num_phase1):
            if time.time() > deadline:
                break
            
            temp = 0.3 + rng.random() * 0.7
            chain = _fast_grow(sequence, rng, temperature=temp)
            if chain is None:
                continue
            
            chain, energy = _local_improve_fast(sequence, chain, max_iter=min(2000, max(100, 3000 // n)))
            
            if energy < best_energy:
                best, best_energy = chain, energy
        
        # Phase 2: Min-conflicts
        if best is not None and time.time() < deadline - 15:
            mc_chain, mc_energy = _min_conflicts_fold(sequence, max_iter=min(2000, max(200, 5000 // n)))
            if mc_energy < best_energy:
                best, best_energy = mc_chain, mc_energy
        
        # Phase 3: Regrowth with SA
        if best is not None and time.time() < deadline - 5:
            regrown = _regrow_sa_fast(sequence, best, rng, max_iter=min(300, max(30, 1000 // n)))
            if regrown is not None:
                regrown, regrown_energy = _local_improve_fast(sequence, regrown, max_iter=min(1000, max(50, 2000 // n)))
                if regrown_energy < best_energy:
                    best, best_energy = regrown, regrown_energy
    
    if best is None:
        best = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
