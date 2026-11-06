"""HP lattice protein folding: multi-start greedy growth + local search + simulated annealing."""
# EVOLVE-BLOCK-START
import time
import random
import math

_T0 = time.time()
_DEADLINE = 195.0  # 210s wall-clock limit, 15s safety margin


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _neighbours2(point):
    x, y = point
    return ((x+1, y), (x-1, y), (x, y+1), (x, y-1))


def _energy(seq, chain):
    """Compute energy: -number of H-H non-bonded contacts."""
    n = len(chain)
    if n < 3:
        return 0
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    h_indices = [i for i, c in enumerate(seq) if c == 'H']
    contacts = 0
    for i in h_indices:
        x, y = chain[i]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = pos_to_idx.get((x+dx, y+dy))
            if j is not None and j > i + 1 and seq[j] == 'H':
                contacts += 1
    return -contacts


def _grow(seq, dim, rng):
    """Randomized greedy chain growth preferring H-H contacts."""
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(seq)):
        last = chain[-1]
        x, y = last
        options = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            site = (x+dx, y+dy)
            if site in occupied:
                continue
            gain = 0
            if seq[i] == 'H':
                for ddx, ddy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    q = (site[0]+ddx, site[1]+ddy)
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and seq[j] == 'H':
                        gain += 1
            free = 0
            for ddx, ddy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if (site[0]+ddx, site[1]+ddy) not in occupied:
                    free += 1
            score = gain * 3.0 + free * 0.1 + rng.random() * 0.5
            options.append((score, site))
        
        if not options:
            return None
        
        options.sort(key=lambda t: -t[0])
        # Pick from top options with some randomness
        if rng.random() < 0.7 and len(options) > 1:
            k = min(rng.randint(0, 2), len(options)-1)
            site = options[k][1]
        else:
            site = options[0][1]
        
        occupied[site] = i
        chain.append(site)
    
    return chain


def _local_search(seq, chain, rng, max_iter=500):
    """Local search: try moving each residue to adjacent empty positions."""
    n = len(chain)
    if n < 3:
        return chain
    chain = list(chain)
    best_energy = _energy(seq, chain)
    occupied = {p: i for i, p in enumerate(chain)}
    
    improved = True
    iters = 0
    while improved and iters < max_iter and _remaining() > 5.0:
        improved = False
        iters += 1
        for i in range(n):
            x, y = chain[i]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                new_pos = (x+dx, y+dy)
                if new_pos in occupied:
                    continue
                # Check self-avoidance: new_pos must not be occupied by any other residue
                # Also check that moving doesn't break connectivity
                # We need to verify the path from i to i+1 and i-1 to i still works
                # Actually, we just move residue i to new_pos, but we must ensure
                # the chain remains a valid self-avoiding walk.
                # This is complex; let's use a simpler approach: only try moves
                # where the new position is adjacent to at least one of the neighbors
                # and the old position becomes free.
                
                # Check that new_pos is not occupied
                if new_pos in occupied:
                    continue
                
                # Try the move
                old_pos = chain[i]
                chain[i] = new_pos
                del occupied[old_pos]
                occupied[new_pos] = i
                
                # Check validity: no two residues share the same position
                # (already ensured by occupied dict)
                # Check connectivity: consecutive residues must be adjacent
                valid = True
                for k in range(n - 1):
                    ax, ay = chain[k]
                    bx, by = chain[k+1]
                    if abs(ax-bx) + abs(ay-by) != 1:
                        valid = False
                        break
                
                if not valid:
                    # Revert
                    chain[i] = old_pos
                    del occupied[new_pos]
                    occupied[old_pos] = i
                    continue
                
                new_energy = _energy(seq, chain)
                if new_energy < best_energy:
                    best_energy = new_energy
                    improved = True
                else:
                    # Revert
                    chain[i] = old_pos
                    del occupied[new_pos]
                    occupied[old_pos] = i
    
    return chain


def _simulated_annealing(seq, chain, rng, max_iter=2000, T0_temp=1.0, T1_temp=0.01):
    """Simulated annealing for local refinement."""
    n = len(chain)
    if n < 3:
        return chain
    chain = list(chain)
    current_energy = _energy(seq, chain)
    best_chain = list(chain)
    best_energy = current_energy
    occupied = {p: i for i, p in enumerate(chain)}
    
    temp = T0_temp
    cooling = (T1_temp / T0_temp) ** (1.0 / max(max_iter, 1))
    
    for it in range(max_iter):
        if _remaining() < 5.0:
            break
        temp *= cooling
        
        # Pick a random residue to move
        i = rng.randint(0, n-1)
        x, y = chain[i]
        dx, dy = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        new_pos = (x+dx, y+dy)
        
        if new_pos in occupied:
            continue
        
        old_pos = chain[i]
        chain[i] = new_pos
        del occupied[old_pos]
        occupied[new_pos] = i
        
        # Check connectivity
        valid = True
        for k in range(n - 1):
            ax, ay = chain[k]
            bx, by = chain[k+1]
            if abs(ax-bx) + abs(ay-by) != 1:
                valid = False
                break
        
        if not valid:
            chain[i] = old_pos
            del occupied[new_pos]
            occupied[old_pos] = i
            continue
        
        new_energy = _energy(seq, chain)
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-10)):
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(chain)
        else:
            chain[i] = old_pos
            del occupied[new_pos]
            occupied[old_pos] = i
    
    return best_chain


def _segment_relocation(seq, chain, rng, max_tries=100):
    """Try relocating segments of the chain to improve energy."""
    n = len(chain)
    if n < 5:
        return chain
    best_chain = list(chain)
    best_energy = _energy(seq, chain)
    
    for _ in range(max_tries):
        if _remaining() < 5.0:
            break
        # Pick a random segment
        length = rng.randint(2, min(5, n-2))
        start = rng.randint(1, n - length - 1)
        end = start + length - 1
        
        # Extract segment
        segment = chain[start:end+1]
        
        # Try to relocate the segment by shifting it
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            new_segment = [(x+dx, y+dy) for x, y in segment]
            
            # Check if new positions are free (not occupied by non-segment residues)
            occupied = {p: i for i, p in enumerate(chain)}
            for k in range(start, end+1):
                del occupied[chain[k]]
            
            valid = True
            for pos in new_segment:
                if pos in occupied:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check connectivity at boundaries
            if start > 0:
                ax, ay = chain[start-1]
                bx, by = new_segment[0]
                if abs(ax-bx) + abs(ay-by) != 1:
                    continue
            if end < n-1:
                ax, ay = new_segment[-1]
                bx, by = chain[end+1]
                if abs(ax-bx) + abs(ay-by) != 1:
                    continue
            
            # Check internal connectivity of new segment
            for k in range(len(new_segment)-1):
                ax, ay = new_segment[k]
                bx, by = new_segment[k+1]
                if abs(ax-bx) + abs(ay-by) != 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Apply move
            new_chain = chain[:start] + new_segment + chain[end+1:]
            new_energy = _energy(seq, new_chain)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = new_chain
    
    return best_chain


def fold(sequence, dimension):
    """Fold the HP sequence using multi-start greedy + local search + SA."""
    if dimension != 2:
        # Fallback for other dimensions: straight line
        return [[i] + [0]*(dimension-1) for i in range(len(sequence))]
    
    n = len(sequence)
    rng = random.Random(12345 + n)
    
    best_chain = None
    best_energy = 0  # Energy is negative, so 0 is worst
    
    # Multi-start greedy growth
    num_starts = 50
    for s in range(num_starts):
        if _remaining() < 10.0:
            break
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        e = _energy(sequence, chain)
        if best_chain is None or e < best_energy:
            best_chain = chain
            best_energy = e
    
    if best_chain is None:
        # Fallback: straight line
        return [[i, 0] for i in range(n)]
    
    # Local search on best chain
    best_chain = _local_search(sequence, best_chain, rng, max_iter=300)
    best_energy = _energy(sequence, best_chain)
    
    # Try multiple local searches from different starts
    for s in range(10):
        if _remaining() < 10.0:
            break
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        chain = _local_search(sequence, chain, rng, max_iter=200)
        e = _energy(sequence, chain)
        if e < best_energy:
            best_chain = chain
            best_energy = e
    
    # Simulated annealing on best
    if _remaining() > 20.0:
        sa_chain = _simulated_annealing(sequence, best_chain, rng, max_iter=1000, T0_temp=0.5, T1_temp=0.01)
        sa_energy = _energy(sequence, sa_chain)
        if sa_energy < best_energy:
            best_chain = sa_chain
            best_energy = sa_energy
    
    # Segment relocation
    if _remaining() > 15.0:
        rel_chain = _segment_relocation(sequence, best_chain, rng, max_tries=50)
        rel_energy = _energy(sequence, rel_chain)
        if rel_energy < best_energy:
            best_chain = rel_chain
            best_energy = rel_energy
    
    # Final local search
    if _remaining() > 10.0:
        best_chain = _local_search(sequence, best_chain, rng, max_iter=500)
        best_energy = _energy(sequence, best_chain)
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
