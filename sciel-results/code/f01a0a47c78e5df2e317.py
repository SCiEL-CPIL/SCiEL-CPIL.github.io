"""HP lattice protein folding: beam search with incremental contacts + multi-start SA."""
import random
import time
import math
from heapq import heappush, heappop

_GLOBAL_START = time.time()
_GLOBAL_DEADLINE = _GLOBAL_START + 115.0


def _remaining_time():
    return max(0.0, _GLOBAL_DEADLINE - time.time())


_DELTAS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbours(point):
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _energy(sequence, chain):
    """Compute energy: -1 for each H-H contact on non-adjacent lattice neighbours."""
    n = len(chain)
    if n < 2:
        return 0
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    h_positions = set()
    for i, c in enumerate(sequence):
        if c == "H":
            h_positions.add(i)
    contacts = 0
    for i, p in enumerate(chain):
        if i in h_positions:
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(i - j) > 1 and j in h_positions:
                    contacts += 1
    return -contacts // 2


def _beam_search(sequence, beam_width, rng, time_limit):
    """
    Beam search over self-avoiding walks with incremental contact scoring.
    Returns list of (energy, chain) for the beam at the end.
    """
    n = len(sequence)
    if n == 0:
        return [(0, [])]
    if n == 1:
        return [(0, [(0, 0)])]
    
    deadline = time.time() + time_limit
    
    # Each state: (score, chain, occupied_set)
    # We use a list of states, sorted by score (higher is better)
    beam = [(-1000.0, [(0, 0)], {(0, 0)})]
    
    for i in range(1, n):
        if time.time() > deadline:
            break
        
        new_beam = []
        for score, chain, occupied in beam:
            current = chain[-1]
            options = []
            for site in _neighbours(current):
                if site in occupied:
                    continue
                
                # Compute incremental score
                inc_score = 0.0
                if sequence[i] == 'H':
                    for q in _neighbours(site):
                        if q in occupied:
                            # Find which residue is at q
                            for j, p in enumerate(chain):
                                if p == q and j < i - 1 and sequence[j] == 'H':
                                    inc_score += 1.0
                                    break
                
                # Compactness bonus: prefer sites with fewer free neighbours
                free_count = sum(1 for q in _neighbours(site) if q not in occupied)
                inc_score -= 0.1 * free_count
                
                # Small random noise
                inc_score += rng.random() * 0.3
                
                new_occupied = occupied | {site}
                new_chain = chain + [site]
                new_score = score + inc_score
                options.append((new_score, new_chain, new_occupied))
            
            # Keep top options from this parent
            options.sort(key=lambda x: x[0], reverse=True)
            new_beam.extend(options[:max(2, beam_width // 2)])
        
        # Keep top beam_width states
        new_beam.sort(key=lambda x: x[0], reverse=True)
        beam = new_beam[:beam_width]
        
        if not beam:
            break
    
    # Return final states with actual energy
    results = []
    for score, chain, occupied in beam:
        if len(chain) == n:
            e = _energy(sequence, chain)
            results.append((e, chain))
    
    return results


def _grow_compact(sequence, rng, compactness):
    """
    Greedy chain growth with a compactness parameter.
    Returns a valid self-avoiding chain or None.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    for i in range(1, n):
        current = chain[-1]
        options = []
        for site in _neighbours(current):
            if site in occupied:
                continue
            
            score = 0.0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == "H":
                            score += 2.0
                        elif j < i - 1:
                            score += 0.5
            free_count = sum(1 for q in _neighbours(site) if q not in occupied)
            score -= compactness * free_count * 0.3
            score += rng.random() * 0.5
            options.append((score, site))
        
        if not options:
            return None
        
        options.sort(key=lambda x: x[0], reverse=True)
        best_site = options[0][1]
        occupied[best_site] = i
        chain.append(best_site)
    
    return chain


def _is_valid_chain(chain):
    """Check self-avoidance and unit steps."""
    if not chain:
        return True
    seen = set()
    for p in chain:
        if p in seen:
            return False
        seen.add(p)
    for i in range(1, len(chain)):
        dx = abs(chain[i][0] - chain[i-1][0])
        dy = abs(chain[i][1] - chain[i-1][1])
        if dx + dy != 1:
            return False
    return True


def _apply_reversal(chain, i, j):
    """
    Attempt a reversal move. Returns new chain if valid, else None.
    0 <= i < j < len(chain).
    """
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
    
    # Quick self-avoidance check on the reversed segment
    seen = set(new_chain[:i])
    for p in new_chain[i:j+1]:
        if p in seen:
            return None
        seen.add(p)
    for p in new_chain[j+1:]:
        if p in seen:
            return None
        seen.add(p)
    
    return new_chain


def _apply_pivot(chain, i, angle):
    """
    Attempt a pivot move. Rotate tail chain[i+1:] around chain[i].
    angle is 1, 2, or 3 (multiples of 90 degrees).
    Returns new chain if valid, else None.
    """
    n = len(chain)
    if i <= 0 or i >= n - 1:
        return None
        
    pivot = chain[i]
    tail = chain[i+1:]
    prefix = chain[:i+1]
    
    rotated_tail = []
    for p in tail:
        x, y = p[0] - pivot[0], p[1] - pivot[1]
        for _ in range(angle):
            x, y = -y, x
        rotated_tail.append((pivot[0] + x, pivot[1] + y))
        
    new_chain = prefix + rotated_tail
    
    # Quick self-avoidance check
    seen = set(prefix)
    for p in rotated_tail:
        if p in seen:
            return None
        seen.add(p)
    
    return new_chain


def _anneal(sequence, initial_chain, time_limit, rng):
    """
    Simulated Annealing starting from initial_chain.
    Returns best_chain, best_energy.
    """
    n = len(initial_chain)
    if n < 3:
        return initial_chain[:], _energy(sequence, initial_chain)
    
    current_chain = initial_chain[:]
    current_energy = _energy(sequence, current_chain)
    best_chain = current_chain[:]
    best_energy = current_energy
    
    T = 2.0
    cooling_rate = 0.999
    
    deadline = time.time() + time_limit
    
    stall_counter = 0
    max_stall = 200
    
    while time.time() < deadline:
        if _remaining_time() < 1.0:
            break
        
        if rng.random() < 0.5:
            i = rng.randint(0, n - 2)
            j = rng.randint(i + 1, n - 1)
            new_chain = _apply_reversal(current_chain, i, j)
            if new_chain is None:
                continue
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
            stall_counter = 0
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = current_chain[:]
        else:
            stall_counter += 1
        
        T *= cooling_rate
        if T < 0.05:
            if rng.random() < 0.3:
                T = 1.5
            else:
                break
        
        if stall_counter > max_stall:
            if rng.random() < 0.5:
                i = rng.randint(0, n - 2)
                j = rng.randint(i + 1, n - 1)
                new_chain = _apply_reversal(current_chain, i, j)
                if new_chain is not None:
                    current_chain = new_chain
                    current_energy = _energy(sequence, current_chain)
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = current_chain[:]
            stall_counter = 0
        
        if _remaining_time() < 0.5:
            break
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Fold the sequence into a low-energy conformation using beam search + multi-start SA.
    """
    rng = random.Random(42 + len(sequence) * 7)
    n = len(sequence)
    
    if n < 2:
        return [[0, 0]] * n
        
    best_chain = None
    best_energy = 0
    
    # Phase 1: Beam search for good initial conformations
    beam_results = []
    beam_widths = [20, 50, 100]
    beam_time = min(15.0, _remaining_time() * 0.25)
    
    for bw in beam_widths:
        if _remaining_time() < 5.0:
            break
        t_alloc = beam_time / len(beam_widths)
        results = _beam_search(sequence, bw, rng, t_alloc)
        beam_results.extend(results)
    
    # Phase 2: Greedy growth for additional diverse starts
    compactness_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    growth_results = []
    growth_time = min(5.0, _remaining_time() * 0.1)
    
    for compactness in compactness_values:
        if _remaining_time() < 3.0:
            break
        for _ in range(3):
            if _remaining_time() < 3.0:
                break
            chain = _grow_compact(sequence, rng, compactness)
            if chain is not None and _is_valid_chain(chain):
                e = _energy(sequence, chain)
                growth_results.append((e, chain))
    
    # Combine all candidates
    all_candidates = beam_results + growth_results
    
    if not all_candidates:
        # Fallback: straight line
        all_candidates = [(0, [(i, 0) for i in range(n)])]
    
    # Sort by energy and keep top K
    all_candidates.sort(key=lambda x: x[0])
    top_k = min(8, len(all_candidates))
    candidates = [c[1] for c in all_candidates[:top_k]]
    
    # Phase 3: Anneal each candidate
    total_time_remaining = _remaining_time()
    time_per_candidate = total_time_remaining / len(candidates)
    
    for idx, chain in enumerate(candidates):
        if _remaining_time() < 1.5:
            break
            
        alloc_time = min(time_per_candidate, _remaining_time() - 1.0)
        if alloc_time < 0.2:
            continue
            
        best_c, best_e = _anneal(sequence, chain, alloc_time, rng)
        
        if best_e < best_energy:
            best_energy = best_e
            best_chain = best_c
    
    # Fallback
    if best_chain is None:
        if all_candidates:
            best_chain = all_candidates[0][1]
        else:
            best_chain = [(i, 0) for i in range(n)]

    # Phase 4: Polish pass
    if _remaining_time() > 3.0 and best_chain is not None:
        polish_time = min(_remaining_time() - 1.5, 8.0)
        if polish_time > 1.0:
            polish_rng = random.Random(99 + len(sequence))
            polish_chain = best_chain[:]
            polish_energy = best_energy
            
            n_p = len(polish_chain)
            if n_p >= 3:
                T = 0.8
                cooling_rate = 0.9995
                deadline = time.time() + polish_time
                stall_counter = 0
                max_stall = 100
                
                while time.time() < deadline:
                    if _remaining_time() < 1.0:
                        break
                    
                    if rng.random() < 0.6:
                        i = rng.randint(0, n_p - 2)
                        j = rng.randint(i + 1, n_p - 1)
                        new_chain = _apply_reversal(polish_chain, i, j)
                        if new_chain is None:
                            continue
                    else:
                        i = rng.randint(1, n_p - 2)
                        angle = rng.choice([1, 2, 3])
                        new_chain = _apply_pivot(polish_chain, i, angle)
                        if new_chain is None:
                            continue
                    
                    new_energy = _energy(sequence, new_chain)
                    delta_e = new_energy - polish_energy
                    
                    if delta_e < 0 or rng.random() < math.exp(-delta_e / max(T, 0.01)):
                        polish_chain = new_chain
                        polish_energy = new_energy
                        stall_counter = 0
                        if polish_energy < best_energy:
                            best_energy = polish_energy
                            best_chain = polish_chain[:]
                    else:
                        stall_counter += 1
                    
                    T *= cooling_rate
                    if T < 0.03:
                        if rng.random() < 0.2:
                            T = 0.5
                        else:
                            break
                    
                    if stall_counter > max_stall:
                        if rng.random() < 0.5:
                            i = rng.randint(0, n_p - 2)
                            j = rng.randint(i + 1, n_p - 1)
                            new_chain = _apply_reversal(polish_chain, i, j)
                            if new_chain is not None:
                                polish_chain = new_chain
                                polish_energy = _energy(sequence, polish_chain)
                                if polish_energy < best_energy:
                                    best_energy = polish_energy
                                    best_chain = polish_chain[:]
                        stall_counter = 0
    
    return [list(p) for p in best_chain]
