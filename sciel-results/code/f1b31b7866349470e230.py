"""HP lattice protein folding: greedy growth + SA polishing."""
# EVOLVE-BLOCK-START
import random
import time
import math

_START_TIME = time.time()
DEADLINE = 195.0  # Safety margin before 210s kill

def _remaining_time():
    return DEADLINE - time.time()

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    """Compute energy: -number of H-H contacts (non-adjacent in chain)."""
    n = len(chain)
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    contacts = 0
    for i in range(n):
        if sequence[i] == 'H':
            p = chain[i]
            x, y = p[0], p[1]
            # Check 4 neighbors
            for nx, ny in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
                j = where.get((nx, ny))
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _grow(sequence, dimension, rng):
    """Randomized greedy chain growth with contact-aware bias."""
    n = len(sequence)
    if n <= 2:
        return [(0,0), (1,0)][:n]
    
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                sx, sy = site
                for nx, ny in ((sx+1,sy),(sx-1,sy),(sx,sy+1),(sx,sy-1)):
                    j = occupied.get((nx, ny))
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            free = 0
            sx, sy = site
            for nx, ny in ((sx+1,sy),(sx-1,sy),(sx,sy+1),(sx,sy-1)):
                if (nx, ny) not in occupied:
                    free += 1
            score = gain * 2.0 + free * 0.1 + rng.random() * 0.8
            options.append((score, site))
        if not options:
            return None
        best_site = max(options, key=lambda x: x[0])[1]
        chain.append(best_site)
        occupied[best_site] = i
    
    return chain

def _local_polish(sequence, chain, rng, time_budget):
    """Local search polishing: try single-residue moves and short reversals."""
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
    
    current = list(chain)
    current_energy = _energy(sequence, current)
    best = list(current)
    best_energy = current_energy
    
    start_time = time.time()
    T = 0.3
    T_min = 0.01
    
    while T > T_min and (time.time() - start_time) < time_budget and _remaining_time() > 10:
        move_type = rng.random()
        improved = False
        
        if move_type < 0.5:
            # Single residue move
            i = rng.randint(1, n - 2)
            old_pos = current[i]
            for new_pos in _neighbours(old_pos):
                if new_pos == current[i-1] or new_pos == current[i+1]:
                    continue
                if new_pos in current:
                    continue
                # Check if valid
                current[i] = new_pos
                new_energy = _energy(sequence, current)
                delta = new_energy - current_energy
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    current_energy = new_energy
                    improved = True
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best = list(current)
                    break
                else:
                    current[i] = old_pos
        elif move_type < 0.8:
            # Short reversal (length 2-4)
            length = rng.randint(2, min(4, n - 2))
            start_idx = rng.randint(1, n - length - 1)
            end_idx = start_idx + length - 1
            # Save original
            orig = current[start_idx:end_idx+1][:]
            # Reverse
            current[start_idx:end_idx+1] = orig[::-1]
            # Check validity
            valid = True
            for i in range(start_idx, end_idx+1):
                if i > 0 and current[i] == current[i-1]:
                    valid = False
                    break
                if i < n-1 and current[i] == current[i+1]:
                    valid = False
                    break
            if valid:
                # Check self-avoidance in modified region
                seen = set()
                for i in range(start_idx, end_idx+1):
                    if current[i] in seen:
                        valid = False
                        break
                    seen.add(current[i])
                # Check against rest
                if valid:
                    rest = set(current[:start_idx]) | set(current[end_idx+1:])
                    if seen & rest:
                        valid = False
            
            if valid:
                new_energy = _energy(sequence, current)
                delta = new_energy - current_energy
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    current_energy = new_energy
                    improved = True
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best = list(current)
                else:
                    current[start_idx:end_idx+1] = orig
            else:
                current[start_idx:end_idx+1] = orig
        else:
            # Try a small rotation of a segment
            if n >= 5:
                pivot_idx = rng.randint(1, n - 3)
                seg_len = rng.randint(2, min(3, n - pivot_idx - 1))
                if seg_len >= 2:
                    pivot = current[pivot_idx]
                    segment = current[pivot_idx+1:pivot_idx+1+seg_len]
                    orig_segment = segment[:]
                    # Rotate 90 degrees
                    rotated = []
                    valid = True
                    for p in segment:
                        dx = p[0] - pivot[0]
                        dy = p[1] - pivot[1]
                        new_p = (pivot[0] - dy, pivot[1] + dx)
                        if new_p in current:
                            valid = False
                            break
                        rotated.append(new_p)
                    if valid:
                        # Check adjacency
                        if rotated[0] != current[pivot_idx]:
                            # First rotated point must be adjacent to pivot
                            if abs(rotated[0][0]-pivot[0]) + abs(rotated[0][1]-pivot[1]) != 1:
                                valid = False
                        if valid and seg_len > 1:
                            last_rot = rotated[-1]
                            next_after = current[pivot_idx+1+seg_len]
                            if abs(last_rot[0]-next_after[0]) + abs(last_rot[1]-next_after[1]) != 1:
                                valid = False
                        if valid:
                            current[pivot_idx+1:pivot_idx+1+seg_len] = rotated
                            new_energy = _energy(sequence, current)
                            delta = new_energy - current_energy
                            if delta < 0 or rng.random() < math.exp(-delta / T):
                                current_energy = new_energy
                                improved = True
                                if new_energy < best_energy:
                                    best_energy = new_energy
                                    best = list(current)
                            else:
                                current[pivot_idx+1:pivot_idx+1+seg_len] = orig_segment
        
        if not improved:
            T *= 0.99
        else:
            T = max(T * 0.999, T_min)
    
    return best, best_energy

def fold(sequence, dimension):
    """Fold sequence using greedy growth + local polishing."""
    n = len(sequence)
    rng = random.Random(12345 + n)
    
    if n <= 2:
        return [[i, 0] for i in range(n)]
    
    best_chain = None
    best_energy = 0
    
    # Phase 1: Greedy growth - run as many as time allows
    candidates = []
    max_growth_time = _remaining_time() * 0.6
    
    while _remaining_time() > 15.0:
        elapsed = time.time() - _START_TIME
        if elapsed > max_growth_time:
            break
        chain = _grow(sequence, dimension, rng)
        if chain is not None:
            energy = _energy(sequence, chain)
            candidates.append((energy, chain))
            if best_chain is None or energy < best_energy:
                best_energy = energy
                best_chain = chain
    
    if not candidates:
        # Fallback: straight line
        return [[i, 0] for i in range(n)]
    
    # Sort candidates by energy
    candidates.sort(key=lambda x: x[0])
    
    # Phase 2: Polish top candidates with local search
    polish_time_total = _remaining_time() * 0.35
    num_to_polish = min(len(candidates), 8)
    
    if num_to_polish > 0 and polish_time_total > 1.0:
        time_per_polish = polish_time_total / num_to_polish
        for i in range(num_to_polish):
            if _remaining_time() < 10.0:
                break
            energy, chain = candidates[i]
            # Better candidates get slightly more time
            time_for_this = time_per_polish * (1.5 - i * 0.1)
            time_for_this = max(0.3, time_for_this)
            
            polished_chain, polished_energy = _local_polish(sequence, chain, rng, time_for_this)
            
            if polished_energy < best_energy:
                best_energy = polished_energy
                best_chain = polished_chain
    
    if best_chain is None:
        best_chain = candidates[0][1]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
