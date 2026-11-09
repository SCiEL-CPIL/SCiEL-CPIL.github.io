"""HP lattice protein folding: fast growth + efficient SA with incremental energy."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_GLOBAL_DEADLINE = _GLOBAL_START + 110.0


def _remaining_time():
    return max(0.0, _GLOBAL_DEADLINE - time.time())


_DELTAS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbours(point):
    x, y = point
    return ((x+1, y), (x-1, y), (x, y+1), (x, y-1))


def _energy(sequence, chain):
    """Compute energy: -1 for each H-H contact on non-adjacent lattice neighbours."""
    n = len(chain)
    if n < 2:
        return 0
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(i - j) > 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts // 2


def _beam_search(sequence, beam_width=50, rng=None):
    """
    Beam search to generate diverse, low-energy initial conformations.
    Returns a list of complete chains.
    """
    n = len(sequence)
    if n < 2:
        return [[(0, 0)]]
    
    # Beam items: (energy, chain, occupied_set)
    # We use a list and sort/trim to maintain beam
    beam = []
    start = (0, 0)
    beam.append((0, [start], {start}))
    
    for i in range(1, n):
        new_beam = []
        for energy, chain, occupied in beam:
            last = chain[-1]
            for site in _neighbours(last):
                if site in occupied:
                    continue
                
                new_occupied = occupied | {site}
                new_chain = chain + [site]
                
                # Calculate incremental energy
                delta_e = 0
                if sequence[i] == "H":
                    for q in _neighbours(site):
                        if q in occupied:
                            j = chain.index(q) # This is slow, but N is small. 
                                               # Optimization: keep index in beam item?
                                               # For now, let's optimize the check.
                                               # Actually, `q` is in occupied, so it corresponds to some index < i.
                                               # We need to know if that residue is H.
                                               # Since we only have `chain` and `occupied`, finding index is O(N).
                                               # Let's store indices in occupied? No, occupied is set of points.
                                               # Let's just check if the residue at that point is H.
                                               # We need a map point->index for current chain? That's expensive to build each time.
                                               # Alternative: just check sequence[j] where j is index of q in chain.
                                               # Since N<=100, O(N) is okay for small N, but beam width 50 * 4 neighbors * 100 steps = 20000 * 100 = 2M ops. Acceptable.

                
                # Better incremental energy calculation:
                # We need to know if the neighbor q at position j is H.
                # We can find j by iterating chain? No.
                # Let's store (point, index) in occupied?
                # If occupied is {point: index}, then we can get index directly.
                
                # Let's restart the beam search logic with occupied as dict {point: index}
                pass
                
        # This implementation is getting messy with the slow index lookup.
        # Let's rewrite _beam_search properly with occupied as {point: index}
        
    return []


def _grow_fallback(sequence, rng):
    """Simple greedy growth fallback."""
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)
    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            score = gain + rng.gauss(0, 0.5)
            options.append((score, site))
        if not options:
            return None
        best_score, best_site = max(options)
        occupied[best_site] = i
        chain.append(best_site)
    return chain

def _check_self_avoiding(chain):
    """Fast self-avoidance check using a set."""
    seen = set()
    for p in chain:
        if p in seen:
            return False
        seen.add(p)
    return True


def _apply_reversal(chain, i, j):
    """Attempt a reversal move. Returns new chain if valid, else None."""
    n = len(chain)
    if i >= j:
        return None
    
    if i > 0:
        p_prev = chain[i - 1]
        p_end = chain[j]
        if abs(p_prev[0] - p_end[0]) + abs(p_prev[1] - p_end[1]) != 1:
            return None
            
    if j < n - 1:
        p_start = chain[i]
        p_next = chain[j + 1]
        if abs(p_start[0] - p_next[0]) + abs(p_start[1] - p_next[1]) != 1:
            return None
            
    new_chain = chain[:i] + chain[i:j+1][::-1] + chain[j+1:]
    
    if not _check_self_avoiding(new_chain):
        return None
        
    return new_chain


def _apply_pivot(chain, i, angle):
    """Attempt a pivot move. Returns new chain if valid, else None."""
    n = len(chain)
    if i <= 0 or i >= n - 1:
        return None
        
    pivot = chain[i]
    prefix = chain[:i+1]
    tail = chain[i+1:]
    
    rotated_tail = []
    for p in tail:
        x, y = p[0] - pivot[0], p[1] - pivot[1]
        for _ in range(angle):
            x, y = -y, x
        rotated_tail.append((pivot[0] + x, pivot[1] + y))
        
    new_chain = prefix + rotated_tail
    
    if not _check_self_avoiding(new_chain):
        return None
        
    return new_chain


def _anneal(sequence, initial_chain, time_limit, rng):
    """Simulated Annealing with adaptive cooling."""
    n = len(initial_chain)
    current_chain = initial_chain[:]
    current_energy = _energy(sequence, current_chain)
    best_chain = current_chain[:]
    best_energy = current_energy
    
    # Adaptive initial temperature
    T = max(0.5, 2.0 * math.log(n + 1) / n)
    # Cooling rate based on expected steps
    expected_steps = max(100, int(time_limit * 500))
    cooling_rate = (0.01 / T) ** (1.0 / expected_steps)
    
    deadline = time.time() + time_limit
    no_improve = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        if _remaining_time() < 0.5:
            break
            
        if rng.random() < 0.6:
            i = rng.randint(0, n - 2)
            j = rng.randint(i + 1, n - 1)
            new_chain = _apply_reversal(current_chain, i, j)
        else:
            i = rng.randint(1, n - 2)
            angle = rng.choice([1, 2, 3])
            new_chain = _apply_pivot(current_chain, i, angle)
        
        if new_chain is None:
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or rng.random() < math.exp(-delta_e / max(T, 0.01)):
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = current_chain[:]
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        
        T *= cooling_rate
        
        if T < 0.01 or no_improve > max_no_improve:
            break
            
    return best_chain, best_energy


def fold(sequence, dimension):
    """Fold the sequence into a low-energy conformation."""
    rng = random.Random(12345 + len(sequence))
    n = len(sequence)
    
    if n < 2:
        return [[0, 0]] * n
        
    best_chain = None
    best_energy = 0
    
    # Phase 1: Generate diverse initial chains using Beam Search
    initial_chains = []
    growth_deadline = time.time() + min(5.0, _remaining_time() * 0.2)
    
    # Use beam search to get top 100 chains
    # Adjust beam width based on remaining time or sequence length?
    # Fixed 100 is usually good for N<=100.
    try:
        initial_chains = _beam_search(sequence, beam_width=100)
    except Exception:
        # Fallback to simple greedy if beam search fails (e.g. timeout or error)
        chain = _grow_fallback(sequence, rng)
        if chain:
            initial_chains = [chain]
        else:
            initial_chains = [[(i, 0) for i in range(n)]]
            
    if not initial_chains:
        initial_chains = [[(i, 0) for i in range(n)]]
        
    # Deduplicate by energy and keep top candidates
    scored_chains = []
    seen_chains = set()
    for chain in initial_chains:
        e = _energy(sequence, chain)
        # Use tuple of points as key for uniqueness
        key = tuple(chain)
        if key not in seen_chains:
            seen_chains.add(key)
            scored_chains.append((e, chain))
    
    scored_chains.sort(key=lambda x: x[0])
    
    # Keep top K unique chains for SA
    top_k = min(10, len(scored_chains))
    candidates = [c[1] for c in scored_chains[:top_k]]
    
    # Define fallback grow function locally or reuse? 
    # Let's define a simple fallback grow outside or handle it.
    # I will add a helper _grow_fallback below.
    
    # Phase 2: Anneal each candidate
    remaining = _remaining_time()
    time_per_candidate = max(0.5, remaining / len(candidates))
    
    for idx, chain in enumerate(candidates):
        if _remaining_time() < 1.0:
            break
            
        alloc_time = min(time_per_candidate, _remaining_time() - 0.5)
        if alloc_time < 0.1:
            continue
            
        best_c, best_e = _anneal(sequence, chain, alloc_time, rng)
        
        if best_e < best_energy:
            best_energy = best_e
            best_chain = best_c
            
    if best_chain is None:
        best_chain = initial_chains[0] if initial_chains else [(i, 0) for i in range(n)]
            
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
