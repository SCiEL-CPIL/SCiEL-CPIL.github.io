"""Rewritten HP lattice protein folding with improved growth and local search."""
import random
import time
import math


def _neighbours(point):
    """Generate 4-neighbours of a 2D lattice point."""
    x, y = point
    return [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]


def _compute_energy(sequence, chain):
    """Compute the energy (negative count of non-adjacent H-H contacts)."""
    n = len(chain)
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, (x, y) in enumerate(chain):
        if sequence[i] == 'H':
            for nx, ny in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]:
                j = pos_to_idx.get((nx, ny))
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _greedy_grow(sequence, rng):
    """Grow chain greedily preferring H-H contacts."""
    n = len(sequence)
    start = (0, 0)
    chain = [start]
    occupied = {start}
    
    for i in range(1, n):
        last = chain[-1]
        candidates = []
        
        for pos in _neighbours(last):
            if pos in occupied:
                continue
            
            score = 0.0
            
            if sequence[i] == 'H':
                for nx, ny in _neighbours(pos):
                    if (nx, ny) in occupied:
                        for j, p in enumerate(chain):
                            if p == (nx, ny) and j < i - 1 and sequence[j] == 'H':
                                score += 1.0
                                break
            
            free = sum(1 for nx, ny in _neighbours(pos) if (nx, ny) not in occupied)
            score += 0.05 * free
            score += rng.random() * 0.01
            
            candidates.append((score, pos))
        
        if not candidates:
            return None
        
        candidates.sort(key=lambda x: -x[0])
        best_pos = candidates[0][1]
        chain.append(best_pos)
        occupied.add(best_pos)
    
    return chain


def _local_search(sequence, chain, rng, max_steps=400, time_limit=0.4):
    """Improve chain using simulated annealing."""
    n = len(chain)
    if n <= 2:
        return chain, _compute_energy(sequence, chain)
    
    is_h = [c == 'H' for c in sequence]
    current_chain = [tuple(p) for p in chain]
    current_energy = _compute_energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    start_time = time.time()
    temperature = 1.0
    cooling = 0.997
    
    pos_to_idx = {p: i for i, p in enumerate(current_chain)}
    
    steps = 0
    while steps < max_steps and (time.time() - start_time) < time_limit:
        steps += 1
        i = rng.randint(1, n - 2)
        old_pos = current_chain[i]
        
        neighbours = _neighbours(old_pos)
        rng.shuffle(neighbours)
        
        for new_pos in neighbours:
            if new_pos in pos_to_idx:
                continue
            
            # Check new_pos doesn't conflict with adjacent chain positions
            if i > 0 and new_pos == current_chain[i-1]:
                continue
            if i < n-1 and new_pos == current_chain[i+1]:
                continue
            
            # Compute delta energy
            delta = 0
            
            # Lost contacts from old position
            for nx, ny in _neighbours(old_pos):
                j = pos_to_idx.get((nx, ny))
                if j is not None and j != i - 1 and j != i + 1 and is_h[i] and is_h[j]:
                    delta += 1
            
            # Gained contacts at new position
            for nx, ny in _neighbours(new_pos):
                j = pos_to_idx.get((nx, ny))
                if j is not None and j != i - 1 and j != i + 1 and is_h[i] and is_h[j]:
                    delta -= 1
            
            # Apply
            del pos_to_idx[old_pos]
            pos_to_idx[new_pos] = i
            current_chain[i] = new_pos
            
            new_energy = current_energy + delta
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(current_chain)
            
            if new_energy <= current_energy or rng.random() < math.exp(-(new_energy - current_energy) / max(temperature, 1e-10)):
                current_energy = new_energy
            else:
                del pos_to_idx[new_pos]
                pos_to_idx[old_pos] = i
                current_chain[i] = old_pos
            
            break
        
        temperature *= cooling
    
    return [list(p) for p in best_chain], best_energy


def fold(sequence, dimension):
    """Fold HP sequence into low-energy self-avoiding conformation."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    if n == 2:
        return [[0, 0], [1, 0]]
    
    deadline = time.time() + 115
    
    rng = random.Random(12345 + n)
    
    best_chain = None
    best_energy = 0
    
    # Phase 1: Greedy growths
    for _ in range(100):
        if time.time() > deadline:
            break
        chain = _greedy_grow(sequence, rng)
        if chain is None:
            continue
        energy = _compute_energy(sequence, chain)
        if energy < best_energy:
            best_energy = energy
            best_chain = [list(p) for p in chain]
    
    if best_chain is None:
        best_chain = [[i, 0] for i in range(n)]
        best_energy = _compute_energy(sequence, best_chain)
    
    # Phase 2: Local search on best solution
    if time.time() < deadline - 1:
        improved, improved_energy = _local_search(sequence, best_chain, rng, max_steps=600, time_limit=min(0.6, deadline - time.time() - 0.5))
        if improved_energy < best_energy:
            best_energy = improved_energy
            best_chain = improved
    
    # Verify self-avoidance
    seen = set()
    for p in best_chain:
        if tuple(p) in seen:
            best_chain = [[i, 0] for i in range(n)]
            break
        seen.add(tuple(p))
    
    return best_chain
