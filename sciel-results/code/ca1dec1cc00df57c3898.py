"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import time
import random
import sys

_TIME_LIMIT = 195.0  # safety margin before 210s kill
_start_time = time.time()
_DEADLINE = _start_time + _TIME_LIMIT


def _time_up():
    return time.time() >= _DEADLINE


def _neighbours2d(point):
    """Yield the 4 neighbors of a 2D lattice point."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _energy(sequence, chain):
    """Compute the HP contact energy: -number of H-H non-bonded contacts."""
    n = len(sequence)
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    
    contacts = 0
    for i in range(n):
        if sequence[i] == 'H':
            x, y = chain[i]
            for nx, ny in ((x+1,y), (x-1,y), (x,y+1), (x,y-1)):
                j = where.get((nx, ny))
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _greedy_grow(sequence, dimension, rng, start=(0, 0)):
    """
    Greedy chain growth with stochastic selection.
    Returns a valid self-avoiding walk or None if stuck.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        last = chain[-1]
        x, y = last
        candidates = []
        
        for nx, ny in ((x+1,y), (x-1,y), (x,y+1), (x,y-1)):
            if (nx, ny) in occupied:
                continue
            
            # Compute gain: number of new H-H contacts if we place here
            gain = 0
            if sequence[i] == 'H':
                for qnx, qny in ((nx+1,ny), (nx-1,ny), (nx,ny+1), (nx,ny-1)):
                    j = occupied.get((qnx, qny))
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            
            # Count free neighbors for future flexibility
            free = 0
            for qnx, qny in ((nx+1,ny), (nx-1,ny), (nx,ny+1), (nx,ny-1)):
                if (qnx, qny) not in occupied:
                    free += 1
            
            # Score: prioritize contacts, then free space, add randomness
            score = gain * 10.0 + free * 0.5 + rng.random()
            candidates.append((score, (nx, ny)))
        
        if not candidates:
            return None
        
        # Select best candidate
        best_score, best_site = max(candidates)
        chain.append(best_site)
        occupied[best_site] = i
    
    return chain


def _local_search(sequence, chain, max_iterations=5000):
    """
    Local search using kink flips and short regrows.
    Returns improved chain or original if no improvement found.
    """
    n = len(sequence)
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    rng = random.Random(42)
    
    for iteration in range(max_iterations):
        if _time_up():
            break
        
        # Try different move types
        move_type = rng.random()
        
        if move_type < 0.6:
            # Kink flip: reverse a short segment
            if n < 4:
                continue
            # Pick a random position and segment length
            seg_len = rng.randint(3, min(8, n - 1))
            start_idx = rng.randint(0, n - 1 - seg_len)
            
            # Check if reversing this segment maintains self-avoidance
            new_chain = list(chain)
            new_chain[start_idx:start_idx + seg_len] = reversed(chain[start_idx:start_idx + seg_len])
            
            # Check self-avoidance
            positions = set()
            valid = True
            for p in new_chain:
                if p in positions:
                    valid = False
                    break
                positions.add(p)
            
            if not valid:
                continue
            
            # Check unit steps
            for i in range(1, n):
                dx = abs(new_chain[i][0] - new_chain[i-1][0])
                dy = abs(new_chain[i][1] - new_chain[i-1][1])
                if dx + dy != 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            new_energy = _energy(sequence, new_chain)
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = new_chain
                chain = new_chain
                current_energy = new_energy
        
        elif move_type < 0.85:
            # Short regrow: remove last k residues and regrow
            k = rng.randint(3, min(10, n - 1))
            if k >= n:
                continue
            
            prefix = chain[:n-k]
            occupied = {p: i for i, p in enumerate(prefix)}
            
            # Regrow the last k residues
            new_suffix = []
            success = True
            for i in range(n-k, n):
                last = prefix[-1] if prefix else (0, 0)
                x, y = last
                candidates = []
                
                for nx, ny in ((x+1,y), (x-1,y), (x,y+1), (x,y-1)):
                    if (nx, ny) in occupied:
                        continue
                    
                    gain = 0
                    if sequence[i] == 'H':
                        for qnx, qny in ((nx+1,ny), (nx-1,ny), (nx,ny+1), (nx,ny-1)):
                            j = occupied.get((qnx, qny))
                            if j is not None and j < i - 1 and sequence[j] == 'H':
                                gain += 1
                    
                    free = 0
                    for qnx, qny in ((nx+1,ny), (nx-1,ny), (nx,ny+1), (nx,ny-1)):
                        if (qnx, qny) not in occupied:
                            free += 1
                    
                    score = gain * 10.0 + free * 0.5 + rng.random()
                    candidates.append((score, (nx, ny)))
                
                if not candidates:
                    success = False
                    break
                
                best_score, best_site = max(candidates)
                new_suffix.append(best_site)
                occupied[best_site] = i
                prefix.append(best_site)
            
            if not success:
                continue
            
            new_chain = list(prefix)
            new_energy = _energy(sequence, new_chain)
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = new_chain
                chain = new_chain
                current_energy = new_energy
        else:
            # Single residue move: try moving one residue
            if n < 3:
                continue
            idx = rng.randint(1, n - 2)  # avoid ends for simplicity
            x, y = chain[idx]
            
            for nx, ny in ((x+1,y), (x-1,y), (x,y+1), (x,y-1)):
                if (nx, ny) == chain[idx-1] or (nx, ny) == chain[idx+1]:
                    continue
                if (nx, ny) in set(chain):
                    continue
                
                new_chain = list(chain)
                new_chain[idx] = (nx, ny)
                
                # Check self-avoidance
                positions = set(new_chain)
                if len(positions) != n:
                    continue
                
                new_energy = _energy(sequence, new_chain)
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = new_chain
                    chain = new_chain
                    current_energy = new_energy
    
    return best_chain


def fold(sequence, dimension):
    """
    Fold an HP sequence into a low-energy self-avoiding conformation.
    Returns a list of len(sequence) lattice points as lists of length dimension.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    rng = random.Random(12345 + n)
    
    best_chain = None
    best_energy = 0  # start with unfolded (energy 0)
    
    # Phase 1: Multiple greedy growths with different random seeds
    num_growth_attempts = min(500, max(50, 20000 // n))
    
    for attempt in range(num_growth_attempts):
        if _time_up():
            break
        
        # Try different starting positions and orientations
        start_x = rng.randint(-5, 5)
        start_y = rng.randint(-5, 5)
        start = (start_x, start_y)
        
        chain = _greedy_grow(sequence, dimension, rng, start)
        if chain is None:
            continue
        
        energy = _energy(sequence, chain)
        if energy < best_energy:
            best_energy = energy
            best_chain = list(chain)
    
    # If no valid chain found, use a simple straight line
    if best_chain is None:
        best_chain = [[i, 0] for i in range(n)]
        best_energy = 0
    else:
        best_chain = [list(p) for p in best_chain]
    
    # Phase 2: Local search on the best chain
    if not _time_up():
        best_chain = _local_search(sequence, best_chain, max_iterations=20000)
    
    # Convert to lists of length dimension
    result = []
    for p in best_chain:
        if len(p) == dimension:
            result.append(list(p))
        else:
            # Pad to dimension if needed
            padded = list(p) + [0] * (dimension - len(p))
            result.append(padded)
    
    return result
# EVOLVE-BLOCK-END
