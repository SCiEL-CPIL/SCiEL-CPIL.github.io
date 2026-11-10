"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

_DEADLINE = 320.0
_start_time = None


def _get_remaining_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _DEADLINE - (time.time() - _start_time)


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute energy: -number of H-H contacts on non-adjacent lattice neighbours."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _greedy_grow(sequence, dimension, rng, temp=0.0):
    """One chain growth with temperature-controlled randomness."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        options = []
        last = chain[-1]
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            free = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free += 1
            score = gain * 10.0 + free * 0.5 + rng.random() * (1.0 + temp)
            options.append((score, site))
        
        if not options:
            return None
        
        if temp > 0 and rng.random() < 0.3:
            options.sort(key=lambda x: -x[0])
            k = min(len(options), 1 + int(rng.expovariate(1.0 / max(1, temp))))
            site = options[rng.randint(0, k - 1)][1]
        else:
            site = max(options)[1]
        
        occupied[site] = i
        chain.append(site)
    
    return chain


def _beam_grow(sequence, dimension, rng, beam_width=20):
    """
    Beam search growth: maintain multiple partial chains, expanding each
    and keeping the top beam_width by (contacts, free space) score.
    This explores multiple growth paths simultaneously.
    """
    n = len(sequence)
    start = (0,) * dimension
    
    # Each beam entry: (chain_list, occupied_dict, score)
    beams = [([start], {start: 0}, 0.0)]
    
    for i in range(1, n):
        if _get_remaining_time() < 5.0:
            break
        new_beams = []
        for chain, occupied, _score in beams:
            last = chain[-1]
            for site in _neighbours(last):
                if site in occupied:
                    continue
                gain = 0
                if sequence[i] == "H":
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == "H":
                            gain += 1
                free = 0
                for q in _neighbours(site):
                    if q not in occupied:
                        free += 1
                # Score for this partial chain extension
                new_score = gain * 10.0 + free * 0.3 + rng.random() * 0.5
                new_chain = chain + [site]
                new_occupied = dict(occupied)
                new_occupied[site] = i
                new_beams.append((new_chain, new_occupied, new_score))
        
        if not new_beams:
            return None
        
        # Keep top beam_width by score
        new_beams.sort(key=lambda x: -x[2])
        beams = new_beams[:beam_width]
    
    # Return the best complete chain
    if not beams:
        return None
    return beams[0][0]


def _local_search(sequence, chain, dimension, rng, max_moves=500):
    """Local search with 2-opt reversals."""
    best_chain = list(chain)
    n = len(best_chain)
    best_energy = _energy(sequence, best_chain)
    
    for _ in range(max_moves):
        if _get_remaining_time() < 1.0:
            break
        
        a = rng.randint(0, n - 2)
        b = rng.randint(a + 1, n - 1)
        
        if b - a == 1:
            continue
        
        new_chain = best_chain[:a] + best_chain[a:b+1][::-1] + best_chain[b+1:]
        
        seen = set()
        valid = True
        for p in new_chain:
            if p in seen:
                valid = False
                break
            seen.add(p)
        
        if not valid:
            continue
        
        for k in range(n - 1):
            diff = sum(abs(new_chain[k][d] - new_chain[k+1][d]) for d in range(dimension))
            if diff != 1:
                valid = False
                break
        
        if not valid:
            continue
        
        new_energy = _energy(sequence, new_chain)
        if new_energy < best_energy:
            best_chain = new_chain
            best_energy = new_energy
    
    return best_chain, best_energy


def _polish(sequence, chain, dimension, rng, max_moves=2000):
    """Polish pass: try moving individual residues to adjacent free sites."""
    best_chain = list(chain)
    n = len(best_chain)
    best_energy = _energy(sequence, best_chain)
    
    for _ in range(max_moves):
        if _get_remaining_time() < 1.0:
            break
        
        i = rng.randint(1, n - 2)
        current = best_chain[i]
        
        for site in _neighbours(current):
            if site in best_chain:
                continue
            
            prev = best_chain[i-1]
            next_ = best_chain[i+1]
            
            d1 = sum(abs(site[d] - prev[d]) for d in range(dimension))
            d2 = sum(abs(site[d] - next_[d]) for d in range(dimension))
            
            if d1 != 1 or d2 != 1:
                continue
            
            new_chain = list(best_chain)
            new_chain[i] = site
            new_energy = _energy(sequence, new_chain)
            
            if new_energy < best_energy:
                best_chain = new_chain
                best_energy = new_energy
                break
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Best conformation using beam search growth + local search + polish.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    n = len(sequence)
    rng = random.Random(12345 + n)
    
    best_chain = None
    best_energy = 0
    
    # Phase 1: Beam search growths (structural change from pure greedy)
    # Use wider beams for longer chains to explore more growth paths
    if n <= 48:
        beam_configs = [(10, 50), (15, 40), (20, 30)]
    elif n <= 80:
        beam_configs = [(8, 40), (12, 30), (16, 20)]
    else:
        beam_configs = [(6, 30), (10, 20), (14, 15)]
    
    for beam_width, n_runs in beam_configs:
        for run in range(n_runs):
            if _get_remaining_time() < 5.0:
                break
            chain = _beam_grow(sequence, dimension, rng, beam_width)
            if chain is None:
                continue
            value = _energy(sequence, chain)
            if value < best_energy:
                best_chain = chain
                best_energy = value
    
    # Phase 2: Also do some plain greedy growths for diversity
    n_greedy = 50 if n <= 48 else 30
    for _ in range(n_greedy):
        if _get_remaining_time() < 5.0:
            break
        temp = rng.uniform(0.0, 0.5)
        chain = _greedy_grow(sequence, dimension, rng, temp)
        if chain is None:
            continue
        value = _energy(sequence, chain)
        if value < best_energy:
            best_chain = chain
            best_energy = value
    
    # Phase 3: Local search on best chain
    if best_chain is not None and _get_remaining_time() > 10.0:
        max_moves = 8000 if n <= 48 else 5000
        best_chain, best_energy = _local_search(sequence, best_chain, dimension, rng, max_moves)
    
    # Phase 4: Polish pass
    if best_chain is not None and _get_remaining_time() > 5.0:
        best_chain, best_energy = _polish(sequence, best_chain, dimension, rng)
    
    # Fallback: straight line
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
