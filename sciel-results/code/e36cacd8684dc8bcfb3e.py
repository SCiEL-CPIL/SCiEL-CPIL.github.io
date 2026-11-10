"""HP lattice protein folding: Parallel Tempering Monte Carlo on a fixed SAW scaffold."""
import random
import math
import time

# EVOLVE-BLOCK-START

def _neighbours(point):
    """Yield the 6 neighbours of a point on the cubic lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _compute_energy(sequence, chain):
    """Compute the HP energy: -number of non-consecutive H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _delta_energy(sequence, chain, old_positions, new_positions, indices):
    """
    Calculate the change in energy when residues at `indices` move from `old_positions` to `new_positions`.
    `chain` is the updated chain (with new positions).
    We need to check contacts involving the moved residues and their neighbors.
    """
    # This function is tricky because we need to know the state BEFORE and AFTER.
    # It's safer to just recompute energy for the local neighborhood or use a full recompute if moves are small.
    # Given the complexity of tracking all affected contacts (a moved residue affects its 6 neighbors,
    # and those neighbors' contacts with other residues), and the fact that N is small (<140),
    # a full recompute is O(N) if we use a pre-computed neighbor list or similar.
    # However, the current _compute_energy is O(N^2) or O(N*6).
    # Let's stick to full recompute for correctness and simplicity, but optimize _compute_energy.
    
    # Actually, let's optimize _compute_energy to be faster.
    # The current _compute_energy is:
    # 1. Build dict: O(N)
    # 2. Loop N: O(N)
    # 3. Inner loop 6 neighbors: O(N*6)
    # Total O(N). This is acceptable.
    
    # So, we will just call _compute_energy(sequence, chain) after the move.
    # To make this faster, we can maintain a set of H positions.
    pass

def _try_kink(chain, occupied, i, sequence, dimension, rng):
    """
    Try to flip a single residue at position i to one of its 6 neighbors.
    The new position must not be occupied by any other part of the chain.
    The new position must not be the same as chain[i-1] or chain[i+1] (to avoid 180 degree turn, 
    although 180 degree turn is technically allowed in SAW if it doesn't self-intersect, 
    but it's a "backtrack". Usually we forbid i-1 and i+1 to keep it a proper SAW step? 
    No, SAW just forbids self-intersection. Backtracking is allowed if it doesn't hit itself.
    However, a kink usually implies moving to a neighbor that isn't the previous or next.
    Let's allow any unoccupied neighbor.
    """
    n = len(chain)
    if i <= 0 or i >= n - 1:
        return False, 0, None
    
    p = chain[i]
    new_p = None
    for q in _neighbours(p):
        if q in occupied:
            continue
        # Check if q is the previous or next residue
        if i > 0 and q == chain[i-1]:
            continue
        if i < n - 1 and q == chain[i+1]:
            continue
        new_p = q
        break
    
    if new_p is None:
        return False, 0, None
    
    # Tentatively apply
    old_p = p
    chain[i] = new_p
    occupied.remove(old_p)
    occupied.add(new_p)
    
    new_e = _compute_energy(sequence, chain)
    
    return True, new_e, old_p

def _try_crankshaft(chain, occupied, i, sequence, dimension, rng):
    """
    Try to rotate a segment of length 2 (residues i and i+1) around the bond (i-1, i).
    Or rotate segment i, i+1 around bond i-1, i? 
    Standard crankshaft: pick a segment of length L (usually 2 or 3).
    Here we pick residues i and i+1. We pivot them around chain[i-1].
    The new position of chain[i] must be a neighbor of chain[i-1] (excluding chain[i-2] and chain[i+1]).
    Then chain[i+1] must be placed relative to new chain[i].
    """
    n = len(chain)
    if i <= 0 or i >= n - 1:
        return False, 0, None, None
    
    # Segment is chain[i], chain[i+1]
    # Pivot around chain[i-1]
    p0 = chain[i-1]
    p1_old = chain[i]
    p2_old = chain[i+1]
    
    # Remove segment from occupied
    occupied.remove(p1_old)
    occupied.remove(p2_old)
    
    # Find valid new position for p1
    valid_p1s = []
    for q in _neighbours(p0):
        if q == p1_old:
            continue
        if i >= 2 and q == chain[i-2]:
            continue
        if q in occupied:
            continue
        valid_p1s.append(q)
        
    if not valid_p1s:
        occupied.add(p1_old)
        occupied.add(p2_old)
        return False, 0, None, None
        
    new_p1 = rng.choice(valid_p1s)
    
    # Find valid position for p2 relative to new_p1
    # p2 must be neighbor of new_p1, not occupied, not p0, not new_p1's previous (which is p0? no, p1 is new_p1)
    # p2 must not be p0 (chain[i-1])
    # p2 must not be in occupied (which now contains everything except p1, p2)
    valid_p2s = []
    for q in _neighbours(new_p1):
        if q == p0:
            continue
        if q in occupied:
            continue
        valid_p2s.append(q)
        
    if not valid_p2s:
        occupied.add(p1_old)
        occupied.add(p2_old)
        return False, 0, None, None
        
    new_p2 = rng.choice(valid_p2s)
    
    # Apply
    chain[i] = new_p1
    chain[i+1] = new_p2
    occupied.add(new_p1)
    occupied.add(new_p2)
    
    new_e = _compute_energy(sequence, chain)
    
    return True, new_e, (p1_old, p2_old), (new_p1, new_p2)

def _try_pivot(chain, occupied, i, sequence, dimension, rng):
    """
    Pivot move: Rotate tail chain[i:] around bond (i-1, i).
    """
    n = len(chain)
    if i <= 0 or i >= n:
        return False, 0, None
    
    tail_old = chain[i:]
    p_pivot = chain[i-1]
    
    # Remove tail
    for p in tail_old:
        occupied.remove(p)
        
    # Find valid new head for tail
    valid_heads = []
    for q in _neighbours(p_pivot):
        if i >= 2 and q == chain[i-2]:
            continue
        if q in occupied:
            continue
        valid_heads.append(q)
        
    if not valid_heads:
        for p in tail_old:
            occupied.add(p)
        return False, 0, None
        
    new_head = rng.choice(valid_heads)
    new_tail = [new_head]
    occupied.add(new_head)
    
    success = True
    for k in range(1, len(tail_old)):
        last = new_tail[-1]
        valid_nexts = []
        for q in _neighbours(last):
            if q in occupied:
                continue
            if len(new_tail) > 1 and q == new_tail[-2]:
                continue
            valid_nexts.append(q)
            
        if not valid_nexts:
            success = False
            break
            
        next_p = rng.choice(valid_nexts)
        new_tail.append(next_p)
        occupied.add(next_p)
        
    if not success:
        for p in new_tail:
            occupied.remove(p)
        for p in tail_old:
            occupied.add(p)
        return False, 0, None
        
    chain[i:] = new_tail
    new_e = _compute_energy(sequence, chain)
    
    return True, new_e, tail_old

def fold(sequence, dimension):
    """
    Parallel Tempering Monte Carlo for HP protein folding.
    """
    n = len(sequence)
    rng = random.Random(42)
    
    # Initial conformation: a simple snake-like path
    chain = []
    occupied = set()
    current = (0, 0, 0)
    chain.append(current)
    occupied.add(current)
    
    directions = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
    
    for i in range(1, n):
        moved = False
        dirs = directions[:]
        rng.shuffle(dirs)
        for d in dirs:
            new_pos = (current[0] + d[0], current[1] + d[1], current[2] + d[2])
            if new_pos not in occupied:
                chain.append(new_pos)
                occupied.add(new_pos)
                current = new_pos
                moved = True
                break
        if not moved:
            # Fallback
            chain = [tuple([i] + [0]*(dimension-1)) for i in range(n)]
            occupied = set(chain)
            break
    
    if len(chain) != n:
        chain = [tuple([i] + [0]*(dimension-1)) for i in range(n)]
        occupied = set(chain)
    
    current_energy = _compute_energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    
    # Parallel Tempering setup
    num_temps = 5
    # Temperatures: high to low
    # T=1.0 is high (random), T=0.01 is low (greedy)
    temps = [1.0, 0.5, 0.2, 0.1, 0.05]
    
    # Initialize replicas
    replicas = []
    for t in range(num_temps):
        c = chain[:]
        occ = set(chain)
        # Perturb initial state for higher temps to diversify
        if t > 0:
             # Do some random moves to perturb
             for _ in range(10):
                 i = rng.randint(1, n-2)
                 # Try a random pivot or kink
                 # For simplicity, just randomize a bit
                 pass
        e = _compute_energy(sequence, c)
        replicas.append({'chain': c, 'occupied': occ, 'energy': e})
        
    start_time = time.time()
    time_budget = 220  # Leave margin
    
    # Number of moves per replica per sweep
    moves_per_sweep = 50
    
    while time.time() - start_time < time_budget:
        # Perform moves for each replica
        for t in range(num_temps):
            rep = replicas[t]
            c = rep['chain']
            occ = rep['occupied']
            e = rep['energy']
            
            for _ in range(moves_per_sweep):
                # Choose move type
                # Kink: fast, local
                # Crankshaft: medium, local
                # Pivot: slow, global
                move_type = rng.randint(0, 2)
                
                if move_type == 0:
                    # Kink
                    i = rng.randint(1, n-2)
                    success, new_e, old_p = _try_kink(c, occ, i, sequence, dimension, rng)
                    if success:
                        delta_e = new_e - e
                        if delta_e < 0 or rng.random() < math.exp(-delta_e / temps[t]):
                            e = new_e
                            rep['energy'] = e
                            if e < best_energy:
                                best_energy = e
                                best_chain = c[:]
                        else:
                            # Revert
                            c[i] = old_p
                            occ.remove(c[i]) # Wait, c[i] is new_p. We need to remove new_p and add old_p
                            # _try_kink modified c and occ.
                            # If rejected, we must revert.
                            # _try_kink returned old_p.
                            # Current c[i] is new_p.
                            occ.remove(c[i])
                            c[i] = old_p
                            occ.add(old_p)
                elif move_type == 1:
                    # Crankshaft
                    i = rng.randint(1, n-2)
                    success, new_e, old_pos, new_pos = _try_crankshaft(c, occ, i, sequence, dimension, rng)
                    if success:
                        delta_e = new_e - e
                        if delta_e < 0 or rng.random() < math.exp(-delta_e / temps[t]):
                            e = new_e
                            rep['energy'] = e
                            if e < best_energy:
                                best_energy = e
                                best_chain = c[:]
                        else:
                            # Revert
                            c[i] = old_pos[0]
                            c[i+1] = old_pos[1]
                            occ.remove(new_pos[0])
                            occ.remove(new_pos[1])
                            occ.add(old_pos[0])
                            occ.add(old_pos[1])
                else:
                    # Pivot
                    i = rng.randint(1, n-1)
                    success, new_e, old_tail = _try_pivot(c, occ, i, sequence, dimension, rng)
                    if success:
                        delta_e = new_e - e
                        if delta_e < 0 or rng.random() < math.exp(-delta_e / temps[t]):
                            e = new_e
                            rep['energy'] = e
                            if e < best_energy:
                                best_energy = e
                                best_chain = c[:]
                        else:
                            # Revert
                            c[i:] = old_tail
                            # Rebuild occupied for tail? 
                            # _try_pivot removed old_tail and added new_tail.
                            # If rejected, remove new_tail (current c[i:]) and add old_tail.
                            for p in c[i:]:
                                occ.remove(p)
                            for p in old_tail:
                                occ.add(p)
                            c[i:] = old_tail
        
        # Swap attempts between neighboring replicas
        for t in range(num_temps - 1):
            if rng.random() < 0.5:
                # Metropolis swap
                e1 = replicas[t]['energy']
                e2 = replicas[t+1]['energy']
                T1 = temps[t]
                T2 = temps[t+1]
                
                # Probability of swap
                # exp(-(e2/T2 - e1/T1 - e1/T1 + e2/T2)?) 
                # Delta E_swap = e2/T2 + e1/T1 - (e1/T2 + e2/T1)
                # Accept if delta < 0 or rand < exp(-delta)
                delta = (e2/T2 + e1/T1) - (e1/T2 + e2/T1)
                
                if delta < 0 or rng.random() < math.exp(-delta):
                    # Swap
                    replicas[t], replicas[t+1] = replicas[t+1], replicas[t]
    
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
