"""HP lattice protein folding: Beam search + efficient local search."""
# EVOLVE-BLOCK-START
import time
import math
import random

_START_TIME = time.time()
_DEADLINE = 195.0  # Safety margin before 210s kill


def _remaining_time():
    return max(0.0, _DEADLINE - (time.time() - _START_TIME))


def _neighbours(point):
    """Yield the 4 neighbours of a 2D lattice point."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _energy(sequence, chain):
    """Compute HP energy: -number of H-H contacts (non-adjacent in chain)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _partial_score(sequence, chain, occupied):
    """
    Score a partial chain for beam search.
    Prioritizes forming H-H contacts and maintaining free space for future growth.
    """
    i = len(chain) - 1
    score = 0.0
    if sequence[i] == "H":
        for q in _neighbours(chain[i]):
            j = occupied.get(q)
            if j is not None and j < i - 1 and sequence[j] == "H":
                score += 1.0
    # Small bonus for free space to avoid trapping the chain
    free = sum(1 for q in _neighbours(chain[i]) if q not in occupied)
    score += 0.1 * free
    return score


def _beam_search(sequence, dimension, beam_width=100, max_time=5.0):
    """
    Beam search over self-avoiding walks with contact-aware scoring.
    Returns a list of (energy, chain) for the top complete chains found.
    """
    n = len(sequence)
    start = (0, 0)
    # Each beam entry: (negative_score, chain, occupied_dict)
    # We store negative score to use min-heap or sort ascending
    beam = [(-_partial_score(sequence, [start], {start: 0}), [start], {start: 0})]
    
    start_time = time.time()
    
    for depth in range(1, n):
        if time.time() - start_time > max_time:
            break
        new_beam = []
        for neg_score, chain, occupied in beam:
            last_site = chain[-1]
            for site in _neighbours(last_site):
                if site in occupied:
                    continue
                new_chain = chain + [site]
                new_occupied = dict(occupied)
                new_occupied[site] = depth
                score = _partial_score(sequence, new_chain, new_occupied)
                new_beam.append((-score, new_chain, new_occupied))
        
        if not new_beam:
            return []
        
        # Keep top beam_width entries
        # Sort by score (ascending, so most negative = best score first)
        new_beam.sort(key=lambda x: x[0])
        beam = new_beam[:beam_width]
        
        if not beam:
            break
            
    # Calculate actual energy for the remaining candidates
    results = []
    for neg_score, chain, _ in beam:
        if len(chain) == n:
            e = _energy(sequence, chain)
            results.append((e, chain))
            
    # If we ran out of time before completing the chain, we can't use partials directly
    # But usually beam search completes for N=100 within 5s if beam_width is moderate.
    # If it doesn't, we return empty and fallback to other methods.
    
    return results


def _apply_reversal(chain, i, j):
    """
    Apply a reversal move: reverse chain[i:j+1].
    Returns the new chain if valid, else None.
    """
    n = len(chain)
    if n < 4:
        return None
    
    # Check boundary connectivity
    # If i > 0, chain[i-1] must connect to chain[j] (which becomes the new start of the segment)
    if i > 0:
        if _manhattan(chain[i - 1], chain[j]) != 1:
            return None
    # If j < n-1, chain[i] (which becomes the new end of the segment) must connect to chain[j+1]
    if j < n - 1:
        if _manhattan(chain[i], chain[j + 1]) != 1:
            return None
            
    # Perform reversal
    new_chain = chain.copy()
    new_chain[i:j+1] = chain[i:j+1][::-1]
    
    # Check self-avoidance
    # Only the segment [i, j] changes position relative to the rest, 
    # but since we reversed it, we just need to ensure no collisions.
    # A full set check is O(N), which is fast enough for N=100.
    if len(set(new_chain)) != n:
        return None
        
    return new_chain


def _apply_pivot(chain, pivot_idx):
    """
    Apply a pivot move: rotate the tail chain[pivot_idx+1:] around chain[pivot_idx].
    Returns a list of valid new chains (one for each valid rotation 90, 180, 270).
    """
    n = len(chain)
    if n < 4 or pivot_idx >= n - 2:
        return []
    
    pivot_pt = chain[pivot_idx]
    tail = chain[pivot_idx+1:]
    
    # Relative coordinates of the tail
    rel_tail = [(pt[0] - pivot_pt[0], pt[1] - pivot_pt[1]) for pt in tail]
    
    valid_chains = []
    
    # Rotation 90 deg: (x, y) -> (-y, x)
    rot1 = [(pivot_pt[0] - dy, pivot_pt[1] + dx) for dx, dy in rel_tail]
    # Rotation 180 deg: (x, y) -> (-x, -y)
    rot2 = [(pivot_pt[0] - dx, pivot_pt[1] - dy) for dx, dy in rel_tail]
    # Rotation 270 deg: (x, y) -> (y, -x)
    rot3 = [(pivot_pt[0] + dy, pivot_pt[1] - dx) for dx, dy in rel_tail]
    
    head = chain[:pivot_idx+1]
    
    for rot in [rot1, rot2, rot3]:
        new_chain = head + rot
        # Check self-avoidance
        if len(set(new_chain)) == n:
            valid_chains.append(new_chain)
            
    return valid_chains


def _anneal(sequence, chain, max_time, seed):
    """
    Simulated annealing with reversal and pivot moves.
    """
    n = len(chain)
    if n <= 2:
        return list(chain), _energy(sequence, chain)

    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy

    T = 3.0
    T_min = 0.05
    # Estimate steps: assume ~1000 steps/sec for safety
    max_steps = int(max_time * 500) 
    if max_steps < 100:
        max_steps = 100
        
    cooling = (T_min / T) ** (1.0 / max_steps)
    rng = random.Random(seed)

    start = time.time()
    steps = 0
    while T > T_min and steps < max_steps and (time.time() - start) < max_time:
        r = rng.random()
        new_chain = None
        
        if r < 0.5:
            # Reversal
            i = rng.randint(0, n - 3)
            j = rng.randint(i + 2, n - 1)
            new_chain = _apply_reversal(chain, i, j)
        else:
            # Pivot
            pivot_idx = rng.randint(1, n - 3)
            candidates = _apply_pivot(chain, pivot_idx)
            if candidates:
                new_chain = rng.choice(candidates)
                
        if new_chain is None:
            T *= cooling
            steps += 1
            continue
            
        new_energy = _energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        if delta_e <= 0 or rng.random() < math.exp(-delta_e / T):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
                
        T *= cooling
        steps += 1

    return best_chain, best_energy


def _polish(sequence, chain, max_time, seed):
    """
    High-precision polish: only accept moves that improve energy.
    """
    n = len(chain)
    if n <= 2:
        return list(chain), _energy(sequence, chain)

    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy

    rng = random.Random(seed)
    start = time.time()

    while (time.time() - start) < max_time:
        improved = False
        r = rng.random()
        
        if r < 0.5:
            # Try reversals
            i = rng.randint(0, n - 3)
            j = rng.randint(i + 2, n - 1)
            new_chain = _apply_reversal(chain, i, j)
            if new_chain:
                new_energy = _energy(sequence, new_chain)
                if new_energy < current_energy:
                    chain = new_chain
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
                    improved = True
        else:
            # Try pivots
            pivot_idx = rng.randint(1, n - 3)
            candidates = _apply_pivot(chain, pivot_idx)
            for new_chain in candidates:
                new_energy = _energy(sequence, new_chain)
                if new_energy < current_energy:
                    chain = new_chain
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
                    improved = True
                    break
                    
        if not improved and rng.random() < 0.05:
            # Occasionally break out if stuck for a while? 
            # Simple polish just runs until time is up.
            pass

    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Fold a sequence using Beam Search initialization and Simulated Annealing.
    """
    n = len(sequence)
    rng = random.Random(12345 + n)
    best_chain = None
    best_energy = 0

    # Phase 1: Beam Search to generate diverse initial conformations
    # Allocate time based on sequence length
    beam_time = min(5.0, _remaining_time() / 2.0)
    beam_width = 150 if n <= 50 else 100
    
    initial_chains = _beam_search(sequence, dimension, beam_width=beam_width, max_time=beam_time)
    
    if not initial_chains:
        # Fallback: Simple linear chain if beam search fails or times out
        best_chain = [[i, 0] for i in range(n)]
        best_energy = _energy(sequence, best_chain)
        # Return immediately if no time left
        if _remaining_time() < 1.0:
            return [list(p) if isinstance(p, tuple) else p for p in best_chain]
    else:
        # Sort by energy (most negative first)
        initial_chains.sort(key=lambda x: x[0])
        # Take the top 5 unique chains
        unique_chains = []
        seen = set()
        for e, c in initial_chains:
            # Simple uniqueness check (tuple of tuples)
            key = tuple(tuple(p) for p in c)
            if key not in seen:
                seen.add(key)
                unique_chains.append((e, c))
            if len(unique_chains) >= 5:
                break
        
        if unique_chains:
            best_energy, best_chain = unique_chains[0]

    # Phase 2: Anneal the top initial conformations
    if _remaining_time() > 2.0:
        num_to_anneal = min(len(unique_chains) if 'unique_chains' in locals() and unique_chains else 1, 3)
        time_per_anneal = max(1.0, (_remaining_time() - 2.0) / max(1, num_to_anneal))
        
        for idx in range(num_to_anneal):
            if _remaining_time() < 2.0:
                break
            
            if idx < len(unique_chains):
                start_chain = unique_chains[idx][1]
            else:
                start_chain = best_chain
            
            t = min(time_per_anneal, _remaining_time() - 1.0)
            if t <= 0:
                break
                
            annealed_chain, annealed_energy = _anneal(sequence, start_chain, max_time=t, seed=12345 + idx)
            
            if annealed_energy < best_energy:
                best_chain = annealed_chain
                best_energy = annealed_energy

    # Phase 3: Polish the best result
    if best_chain is not None and _remaining_time() > 1.0:
        polish_time = min(3.0, _remaining_time() - 0.5)
        polished_chain, polished_energy = _polish(sequence, best_chain, max_time=polish_time, seed=99999)
        if polished_energy < best_energy:
            best_chain = polished_chain
            best_energy = polished_energy

    if best_chain is None:
        best_chain = [[i, 0] for i in range(n)]

    return [list(p) if isinstance(p, tuple) else p for p in best_chain]
# EVOLVE-BLOCK-END
