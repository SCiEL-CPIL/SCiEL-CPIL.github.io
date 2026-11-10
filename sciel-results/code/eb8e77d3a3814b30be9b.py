import random
import time
import math

# Global deadline guard: must finish well before 330s wall-clock.
_TOTAL_START = time.time()
_SAFETY_MARGIN = 15.0  # seconds to leave before hard kill

def _remaining_time():
    """Return remaining seconds before our safety deadline."""
    return _TOTAL_START + 330.0 - _SAFETY_MARGIN - time.time()

def _neighbours(point):
    """Generate 6 neighbors of a point in 3D lattice."""
    for axis in range(3):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    """Calculate energy: -number of H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _fold_subchain(sequence, start, end, dimension, rng, time_limit):
    """
    Fold a sub-chain (sequence[start:end]) independently.
    Uses randomized growth with multiple restarts.
    Returns (chain, energy) for the sub-chain, or None if failed.
    """
    sub_seq = sequence[start:end]
    n = len(sub_seq)
    if n == 0:
        return [], 0
    
    best_chain = None
    best_energy = float('inf')
    
    # Number of restarts based on sub-chain length
    if n <= 10:
        num_restarts = 20
    elif n <= 20:
        num_restarts = 10
    else:
        num_restarts = 5
    
    for _ in range(num_restarts):
        if time.time() > time_limit:
            break
        
        # Randomized growth
        start_pos = (0, 0, 0)
        chain = [start_pos]
        occupied = {start_pos: 0}
        
        success = True
        for i in range(1, n):
            options = []
            for site in _neighbours(chain[-1]):
                if site in occupied:
                    continue
                
                score = 0.0
                # Prefer sites that create H-H contacts
                if sub_seq[i] == "H":
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and sub_seq[j] == "H":
                            score += 10.0
                
                # Prefer free space
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                score += 0.3 * free
                score += rng.random() * 0.5
                
                options.append((score, site))
            
            if not options:
                success = False
                break
            
            # Pick with some randomness
            if rng.random() < 0.7:
                site = max(options, key=lambda x: x[0])[1]
            else:
                site = options[rng.randrange(len(options))][1]
            
            occupied[site] = i
            chain.append(site)
        
        if success:
            e = _energy(sub_seq, chain)
            if e < best_energy:
                best_energy = e
                best_chain = list(chain)
    
    if best_chain is None:
        return None
    return best_chain, best_energy

def _merge_chains(sequence, chain1, chain2, start_idx, dimension, rng, time_limit):
    """
    Attempt to merge two independently-folded sub-chains into a valid full chain.
    chain1 covers sequence[0:start_idx], chain2 covers sequence[start_idx:].
    We need to find a valid conformation where the end of chain1 is adjacent to the start of chain2.
    
    Strategy: Use local search to adjust the positions of residues near the boundary.
    """
    n1 = len(chain1)
    n2 = len(chain2)
    n = n1 + n2
    
    # Initial attempt: place chain2 relative to chain1
    # The last residue of chain1 must be adjacent to the first residue of chain2
    end1 = chain1[-1]
    
    # Try to find a valid placement of chain2
    # First, check if chain2 can be placed as-is (translated) so that chain2[0] is adjacent to end1
    # We need to find a translation vector t such that chain2[0] + t is adjacent to end1,
    # and no position in the translated chain2 conflicts with chain1
    
    best_full_chain = None
    best_energy = float('inf')
    
    # Generate candidate translations
    candidates = []
    for delta in _neighbours(end1):
        # delta is where chain2[0] would go
        t = (delta[0] - chain2[0][0], delta[1] - chain2[0][1], delta[2] - chain2[0][2])
        
        # Check for conflicts
        translated_chain2 = [(x + t[0], y + t[1], z + t[2]) for x, y, z in chain2]
        
        # Check if any position in translated_chain2 conflicts with chain1
        # (except that translated_chain2[0] should be adjacent to chain1[-1], not equal)
        conflict = False
        occupied1 = set(chain1)
        for i, pos in enumerate(translated_chain2):
            if i == 0:
                # chain2[0] should be adjacent to chain1[-1], not equal
                if pos in occupied1:
                    conflict = True
                    break
            else:
                if pos in occupied1:
                    conflict = True
                    break
        
        if not conflict:
            # Also check self-conflicts within translated_chain2 (shouldn't happen if chain2 is valid)
            candidates.append((t, translated_chain2))
    
    if not candidates:
        # No valid direct placement; try local search to fix
        # Pick a random neighbor of end1 as target for chain2[0]
        target = list(end1)
        axis = rng.randrange(3)
        target[axis] += rng.choice([-1, 1])
        target = tuple(target)
        t = (target[0] - chain2[0][0], target[1] - chain2[0][1], target[2] - chain2[0][2])
        translated_chain2 = [(x + t[0], y + t[1], z + t[2]) for x, y, z in chain2]
        candidates.append((t, translated_chain2))
    
    # For each candidate, try to refine with local search
    for t, trans_c2 in candidates:
        full_chain = chain1 + trans_c2
        e = _energy(sequence, full_chain)
        
        if e < best_energy:
            best_energy = e
            best_full_chain = list(full_chain)
        
        # Local search: try to improve by moving residues near the boundary
        if time.time() < time_limit:
            current_chain = list(full_chain)
            current_e = e
            occupied = {p: i for i, p in enumerate(current_chain)}
            
            # Focus on residues within 5 of the boundary
            boundary_start = max(0, n1 - 5)
            boundary_end = min(n, n1 + 5)
            
            for _ in range(200):
                if time.time() > time_limit:
                    break
                
                # Pick a residue near the boundary to move
                i = rng.randrange(boundary_start, boundary_end)
                if i == 0 or i == n - 1:
                    continue
                
                p_prev = current_chain[i-1]
                p_next = current_chain[i+1]
                p_old = current_chain[i]
                
                # Check if p_prev and p_next are distance 2 apart
                dist = abs(p_prev[0]-p_next[0]) + abs(p_prev[1]-p_next[1]) + abs(p_prev[2]-p_next[2])
                if dist != 2:
                    continue
                
                # Find the other common neighbor
                p_new = None
                for q in _neighbours(p_prev):
                    if q == p_old:
                        continue
                    if abs(q[0]-p_next[0]) + abs(q[1]-p_next[1]) + abs(q[2]-p_next[2]) == 1:
                        p_new = q
                        break
                
                if p_new is None or p_new in occupied:
                    continue
                
                # Calculate energy delta
                delta_e = 0
                if sequence[i] == "H":
                    for q in _neighbours(p_old):
                        j = occupied.get(q)
                        if j is not None and j != i and sequence[j] == "H" and abs(i-j) > 1:
                            delta_e += 1
                    for q in _neighbours(p_new):
                        j = occupied.get(q)
                        if j is not None and j != i and sequence[j] == "H" and abs(i-j) > 1:
                            delta_e -= 1
                
                if delta_e < 0:
                    occupied.pop(p_old)
                    occupied[p_new] = i
                    current_chain[i] = p_new
                    current_e += delta_e
                    if current_e < best_energy:
                        best_energy = current_e
                        best_full_chain = list(current_chain)
    
    if best_full_chain is None:
        return None
    return best_full_chain, best_energy

def fold(sequence, dimension):
    """
    Fold using divide-and-conquer: split sequence into sub-chains,
    fold each independently, then merge.
    """
    rng = random.Random(12345 + len(sequence))
    n = len(sequence)
    
    if n <= 10:
        # Small chain: just use direct growth with multiple restarts
        best_chain = None
        best_energy = float('inf')
        
        for _ in range(30):
            if _remaining_time() < 5.0:
                break
            
            start_pos = (0, 0, 0)
            chain = [start_pos]
            occupied = {start_pos: 0}
            
            success = True
            for i in range(1, n):
                options = []
                for site in _neighbours(chain[-1]):
                    if site in occupied:
                        continue
                    score = 0.0
                    if sequence[i] == "H":
                        for q in _neighbours(site):
                            j = occupied.get(q)
                            if j is not None and j < i - 1 and sequence[j] == "H":
                                score += 10.0
                    free = sum(1 for q in _neighbours(site) if q not in occupied)
                    score += 0.3 * free
                    score += rng.random() * 0.5
                    options.append((score, site))
                
                if not options:
                    success = False
                    break
                
                if rng.random() < 0.7:
                    site = max(options, key=lambda x: x[0])[1]
                else:
                    site = options[rng.randrange(len(options))][1]
                
                occupied[site] = i
                chain.append(site)
            
            if success:
                e = _energy(sequence, chain)
                if e < best_energy:
                    best_energy = e
                    best_chain = list(chain)
        
        if best_chain is None:
            best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        return [list(p) for p in best_chain]
    
    # Divide into two sub-chains
    # Split point: try multiple split points
    split_points = []
    if n <= 30:
        split_points = [n // 2]
    elif n <= 60:
        split_points = [n // 3, 2 * n // 3, n // 2]
    else:
        split_points = [n // 4, n // 2, 3 * n // 4]
    
    best_chain = None
    best_energy = float('inf')
    
    for split in split_points:
        if _remaining_time() < 20.0:
            break
        
        time_per_sub = _remaining_time() / 3.0 - 5.0
        
        # Fold sub-chain 1: sequence[0:split]
        result1 = _fold_subchain(sequence, 0, split, dimension, rng, time.time() + time_per_sub)
        if result1 is None:
            continue
        chain1, e1 = result1
        
        # Fold sub-chain 2: sequence[split:]
        result2 = _fold_subchain(sequence, split, n, dimension, rng, time.time() + time_per_sub)
        if result2 is None:
            continue
        chain2, e2 = result2
        
        # Merge
        merge_result = _merge_chains(sequence, chain1, chain2, split, dimension, rng, time.time() + _remaining_time() / 2.0)
        if merge_result is not None:
            full_chain, full_e = merge_result
            if full_e < best_energy:
                best_energy = full_e
                best_chain = full_chain
    
    # Fallback: if divide-and-conquer failed, use direct growth
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
