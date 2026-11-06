import random
import time
import sys

# Global deadline
_DEADLINE = None
_START_TIME = None

def _init_deadline():
    global _DEADLINE, _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    if _DEADLINE is None:
        _DEADLINE = _START_TIME + 290  # 290s safety margin before 300s kill

def _time_left():
    _init_deadline()
    return _DEADLINE - time.time()


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute energy of a chain configuration."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _energy_delta_place(sequence, chain, occupied, pos, index):
    """Compute energy change when placing residue at index at position pos."""
    if sequence[index] != 'H':
        return 0
    delta = 0
    for q in _neighbours(pos):
        if q in occupied:
            j = occupied[q]
            if j < index - 1 and sequence[j] == 'H':
                delta -= 1  # energy decreases by 1 (more negative)
    return delta


def _energy_delta_remove(sequence, chain, occupied, pos, index):
    """Compute energy change when removing residue at index from position pos."""
    if sequence[index] != 'H':
        return 0
    delta = 0
    for q in _neighbours(pos):
        if q in occupied:
            j = occupied[q]
            if j < index - 1 and sequence[j] == 'H':
                delta += 1  # energy increases by 1 (less negative)
    return delta


def _greedy_seed(sequence, dimension, rng):
    """Generate a good initial seed using greedy growth."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        best_site = None
        best_score = -999999
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            # Score: prefer H-H contacts, then free space
            score = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == 'H':
                            score += 10
            # Free space bonus
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score += 0.1 * free
            score += rng.random() * 0.5
            
            if score > best_score:
                best_score = score
                best_site = site
        
        if best_site is None:
            return None
        
        occupied[best_site] = i
        chain.append(best_site)
    
    return chain


def _sa_fold(sequence, dimension, rng, max_iters):
    """Simulated annealing for a single restart."""
    # Generate initial seed
    chain = _greedy_seed(sequence, dimension, rng)
    if chain is None:
        return None, 0
    
    n = len(sequence)
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _compute_energy(sequence, chain)
    
    best_chain = list(chain)
    best_energy = current_energy
    
    # SA parameters
    T_start = 2.0
    T_end = 0.05
    cooling = (T_end / T_start) ** (1.0 / max(1, max_iters))
    T = T_start
    
    for it in range(max_iters):
        if _time_left() < 2.0:
            break
        
        # Choose a move type
        move_type = rng.random()
        
        if move_type < 0.7:
            # Pivot move: rotate a subchain around a pivot point
            if n < 4:
                continue
            pivot = rng.randint(1, n - 2)
            length = min(rng.randint(1, n - pivot - 1), 6)
            
            # Try to rotate the segment chain[pivot:pivot+length+1] around chain[pivot]
            # We'll try all 24 orientations (rotations) of the subchain
            pivot_pos = chain[pivot]
            segment = chain[pivot+1:pivot+1+length]
            
            # Generate all possible rotations
            def rotate_points(points, axis, angle):
                # For simplicity, we'll try all 6 axis permutations and sign combinations
                # Actually, let's just try small perturbations
                pass
            
            # Simpler: try moving one residue in the segment to a random nearby free site
            idx = pivot + rng.randint(1, length)
            old_pos = chain[idx]
            
            # Try all neighbors of old_pos
            candidates = []
            for new_pos in _neighbours(old_pos):
                if new_pos == chain[idx-1] or (idx + 1 < n and new_pos == chain[idx+1]):
                    continue
                if new_pos in occupied:
                    # Check if it's the residue we're moving
                    if occupied[new_pos] != idx:
                        continue
                # Check self-avoidance with other residues
                valid = True
                for j in range(n):
                    if j == idx or j == idx-1 or (j == idx+1 and j < n):
                        continue
                    if chain[j] == new_pos:
                        valid = False
                        break
                if valid:
                    candidates.append(new_pos)
            
            if not candidates:
                continue
            
            new_pos = rng.choice(candidates)
            
            # Check if new_pos is already occupied by another residue
            if new_pos in occupied and occupied[new_pos] != idx:
                continue
            
            # Compute energy change
            delta = 0
            if sequence[idx] == 'H':
                # Remove contacts from old position
                for q in _neighbours(old_pos):
                    if q in occupied:
                        j = occupied[q]
                        if j < idx - 1 and sequence[j] == 'H':
                            delta += 1  # removing a contact increases energy
                # Add contacts at new position
                for q in _neighbours(new_pos):
                    if q in occupied:
                        j = occupied[q]
                        if j != idx and j < idx - 1 and sequence[j] == 'H':
                            delta -= 1  # adding a contact decreases energy
            
            new_energy = current_energy + delta
            
            # Metropolis criterion
            if delta <= 0 or rng.random() < pow(2.718281828, -delta / T):
                # Accept move
                del occupied[old_pos]
                chain[idx] = new_pos
                occupied[new_pos] = idx
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
        
        elif move_type < 0.85:
            # Crankshaft move: rotate two adjacent bonds
            if n < 5:
                continue
            idx = rng.randint(1, n - 3)
            p0 = chain[idx]
            p1 = chain[idx+1]
            p2 = chain[idx+2]
            
            # Try to move p1 to a new position that keeps p0-p1 and p1-p2 as unit steps
            # New position must be a neighbor of both p0 and p2 (or we move p0 and p2)
            # Simpler: try moving p1 to a random free neighbor
            candidates = []
            for new_p1 in _neighbours(p1):
                if new_p1 == p0 or new_p1 == p2:
                    continue
                # Check if new_p1 is free
                if new_p1 in occupied and occupied[new_p1] != idx+1:
                    continue
                # Check self-avoidance
                valid = True
                for j in range(n):
                    if j == idx+1:
                        continue
                    if chain[j] == new_p1:
                        valid = False
                        break
                if valid:
                    candidates.append(new_p1)
            
            if not candidates:
                continue
            
            new_p1 = rng.choice(candidates)
            
            # Compute energy change
            delta = 0
            if sequence[idx+1] == 'H':
                for q in _neighbours(p1):
                    if q in occupied:
                        j = occupied[q]
                        if j < idx and sequence[j] == 'H':
                            delta += 1
                for q in _neighbours(new_p1):
                    if q in occupied:
                        j = occupied[q]
                        if j != idx+1 and j < idx and sequence[j] == 'H':
                            delta -= 1
            
            new_energy = current_energy + delta
            
            if delta <= 0 or rng.random() < pow(2.718281828, -delta / T):
                del occupied[p1]
                chain[idx+1] = new_p1
                occupied[new_p1] = idx+1
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
        
        else:
            # Simple local move: try to move the last few residues
            idx = n - 1
            old_pos = chain[idx]
            candidates = []
            for new_pos in _neighbours(old_pos):
                if new_pos == chain[idx-1]:
                    continue
                if new_pos in occupied and occupied[new_pos] != idx:
                    continue
                valid = True
                for j in range(n):
                    if j == idx:
                        continue
                    if chain[j] == new_pos:
                        valid = False
                        break
                if valid:
                    candidates.append(new_pos)
            
            if not candidates:
                continue
            
            new_pos = rng.choice(candidates)
            
            delta = 0
            if sequence[idx] == 'H':
                for q in _neighbours(old_pos):
                    if q in occupied:
                        j = occupied[q]
                        if j < idx - 1 and sequence[j] == 'H':
                            delta += 1
                for q in _neighbours(new_pos):
                    if q in occupied:
                        j = occupied[q]
                        if j != idx and j < idx - 1 and sequence[j] == 'H':
                            delta -= 1
            
            new_energy = current_energy + delta
            
            if delta <= 0 or rng.random() < pow(2.718281828, -delta / T):
                del occupied[old_pos]
                chain[idx] = new_pos
                occupied[new_pos] = idx
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
        
        T *= cooling
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Fold a protein sequence using simulated annealing with multiple restarts."""
    _init_deadline()
    
    n = len(sequence)
    rng = random.Random(42 + len(sequence))
    
    best_chain = None
    best_energy = 1
    
    # Determine number of restarts based on time budget
    # Each restart with SA takes roughly O(n * iters) time
    # Let's do multiple restarts with varying SA iteration counts
    
    # First, get a good greedy seed
    greedy = _greedy_seed(sequence, dimension, rng)
    if greedy is not None:
        e = _compute_energy(sequence, greedy)
        best_chain = list(greedy)
        best_energy = e
    
    # Now do SA restarts
    while _time_left() > 5.0:
        # Determine iterations based on remaining time and sequence length
        # Estimate: each iteration is O(n) for energy delta
        # Target: use ~40% of remaining time per restart
        time_per_restart = _time_left() * 0.4
        # Rough estimate: 1 microsecond per iteration per residue
        iters = int(time_per_restart * 1e6 / max(1, n))
        iters = max(100, min(iters, 50000))
        
        chain, energy = _sa_fold(sequence, dimension, rng, iters)
        
        if chain is not None and energy < best_energy:
            best_chain = list(chain)
            best_energy = energy
        
        # Check if we've found a very good solution
        if best_energy <= -n // 2:
            break
    
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
