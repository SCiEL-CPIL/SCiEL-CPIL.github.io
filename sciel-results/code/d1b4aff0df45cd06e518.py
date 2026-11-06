"""HP lattice protein folding: beam search with contact-maximizing heuristic."""
# EVOLVE-BLOCK-START
import time
import math

_T0 = time.time()
_DEADLINE = 195.0  # wall-clock limit with safety margin


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _compute_energy(sequence, chain):
    """Compute energy: -number of H-H non-bonded contacts."""
    n = len(chain)
    if n < 3:
        return 0
    pos_to_idx = {}
    for i, p in enumerate(chain):
        pos_to_idx[p] = i
    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        x, y = chain[i]
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nb = (x + dx, y + dy)
            j = pos_to_idx.get(nb)
            if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts


def _beam_search(sequence, max_width, time_limit):
    """
    Beam search for HP lattice folding.
    State: (chain, energy, occupied_set, pos_to_idx)
    We grow the chain one residue at a time, keeping the top `max_width` states.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    start_time = time.time()
    
    # Initial state: first residue at origin
    # Try all 4 starting directions for the second residue
    beam = []
    for dx, dy in [(1, 0), (-1, 1), (0, 1), (0, -1)]:
        chain = [(0, 0), (dx, dy)]
        occupied = {(0, 0), (dx, dy)}
        pos_to_idx = {(0, 0): 0, (dx, dy): 1}
        energy = 0  # No contacts possible with 2 residues
        beam.append((chain, energy, occupied, pos_to_idx))
    
    if n == 2:
        return beam[0][0]
    
    # Grow from residue 2 to n-1
    for i in range(2, n):
        new_beam = []
        is_h = sequence[i] == 'H'
        
        for chain, energy, occupied, pos_to_idx in beam:
            last = chain[-1]
            # Try all 4 directions
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                new_pos = (last[0] + dx, last[1] + dy)
                if new_pos in occupied:
                    continue
                
                # Compute new contacts if this residue is H
                new_energy = energy
                if is_h:
                    for ndx, ndy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        nb = (new_pos[0] + ndx, new_pos[1] + ndy)
                        j = pos_to_idx.get(nb)
                        if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                            new_energy -= 1
                
                new_chain = chain + [new_pos]
                new_occupied = occupied | {new_pos}
                new_pos_to_idx = dict(pos_to_idx)
                new_pos_to_idx[new_pos] = i
                
                new_beam.append((new_chain, new_energy, new_occupied, new_pos_to_idx))
        
        if not new_beam:
            # Dead end - should not happen for reasonable widths
            return None
        
        # Sort by energy (more negative is better) and keep top max_width
        new_beam.sort(key=lambda x: x[1])
        beam = new_beam[:max_width]
        
        # Time check
        if time.time() - start_time > time_limit:
            break
    
    if not beam:
        return None
    
    # Return the best chain
    best = beam[0][0]
    return best


def _local_refine(sequence, chain, time_limit):
    """
    Local refinement using kink moves and small segment relocations.
    """
    n = len(chain)
    if n < 4:
        return chain, _compute_energy(sequence, chain)
    
    start_time = time.time()
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    T = 2.0
    cooling = 0.999
    min_T = 0.1
    
    occupied = set(chain)
    
    while True:
        if time.time() - start_time > time_limit:
            break
        if _remaining() < 5.0:
            break
        
        # Try kink move
        i = None
        for _ in range(50):
            i = 1 + int((time.time() * 1000000) % (n - 2)) if n > 3 else None
            if i is not None:
                break
        if i is None:
            i = 1 + int(hash(str(time.time())) % max(1, n - 2))
        
        mid = chain[i]
        left = chain[i - 1]
        right = chain[i + 1]
        
        candidates = []
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            new_mid = (mid[0] + dx, mid[1] + dy)
            if new_mid in occupied:
                continue
            if abs(new_mid[0] - left[0]) + abs(new_mid[1] - left[1]) != 1:
                continue
            if abs(new_mid[0] - right[0]) + abs(new_mid[1] - right[1]) != 1:
                continue
            candidates.append(new_mid)
        
        if candidates:
            new_mid = candidates[int(time.time() * 1000000) % len(candidates)]
            new_chain = list(chain)
            new_chain[i] = new_mid
            
            # Verify
            if len(set(new_chain)) == n:
                new_energy = _compute_energy(sequence, new_chain)
                delta = new_energy - current_energy
                if delta <= 0 or (T > min_T and math.exp(-delta / T) > ((time.time() * 1000000) % 1000) / 1000.0):
                    chain = new_chain
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(new_chain)
        
        T *= cooling
        if T < min_T:
            T = min_T
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Fold sequence into low-energy conformation."""
    if dimension != 2:
        return [[i] + [0] * (dimension - 1) for i in range(len(sequence))]
    
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    if n == 2:
        return [[0, 0], [1, 0]]
    
    best_chain = None
    best_energy = 0
    
    start_time = time.time()
    time_budget = min(_remaining() - 5.0, 180.0)
    
    # Adaptive beam width based on sequence length
    if n <= 20:
        widths = [400, 200, 100]
        num_passes = 3
    elif n <= 35:
        widths = [300, 150, 80]
        num_passes = 3
    elif n <= 50:
        widths = [200, 100, 60]
        num_passes = 3
    elif n <= 70:
        widths = [150, 80, 50]
        num_passes = 2
    else:
        widths = [100, 60, 40]
        num_passes = 2
    
    for pass_idx in range(num_passes):
        if _remaining() < 10.0:
            break
        if time.time() - start_time > time_budget:
            break
        
        width = widths[pass_idx % len(widths)]
        time_left = min(time_budget - (time.time() - start_time), _remaining() - 8.0)
        if time_left < 2.0:
            break
        
        # Run beam search
        chain = _beam_search(sequence, width, time_left * 0.6)
        if chain is None or len(chain) != n:
            continue
        
        # Verify
        if len(set(chain)) != n:
            continue
        valid = True
        for i in range(n - 1):
            if abs(chain[i][0] - chain[i + 1][0]) + abs(chain[i][1] - chain[i + 1][1]) != 1:
                valid = False
                break
        if not valid:
            continue
        
        # Local refinement
        time_left_refine = min(time_budget - (time.time() - start_time), _remaining() - 5.0)
        if time_left_refine < 1.0:
            break
        
        chain, energy = _local_refine(sequence, chain, time_left_refine * 0.4)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = list(chain)
    
    if best_chain is None:
        # Fallback: simple linear chain
        best_chain = [(i, 0) for i in range(n)]
    
    return [[x, y] for x, y in best_chain]
# EVOLVE-BLOCK-END
