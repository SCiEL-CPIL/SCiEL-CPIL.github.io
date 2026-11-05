"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

_DEADLINE = 320.0  # safety margin before 330s wall-clock kill
_START_TIME = None

def _get_deadline():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _START_TIME + _DEADLINE

def _time_left():
    return _get_deadline() - time.time()

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _compute_h_contacts(sequence, chain):
    """Return set of (i,j) pairs where both are H and adjacent on lattice but not chain-adjacent."""
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    contacts = set()
    for i, p in enumerate(chain):
        if sequence[i] != "H":
            continue
        for q in _neighbours(p):
            j = where.get(q)
            if j is not None and j > i + 1 and sequence[j] == "H":
                contacts.add((i, j))
    return contacts

def _local_search(sequence, chain, dimension, rng, max_iters=500):
    """Monte Carlo local search with incremental energy updates."""
    if chain is None:
        return None
    n = len(chain)
    if n < 4:
        return chain
    
    # Precompute H positions
    h_positions = [i for i, c in enumerate(sequence) if c == 'H']
    
    occupied = {p: i for i, p in enumerate(chain)}
    current_contacts = _compute_h_contacts(sequence, chain)
    current_energy = -len(current_contacts)
    best_chain = list(chain)
    best_energy = current_energy
    best_contacts = set(current_contacts)
    
    # Helper: check if two residues are adjacent on lattice
    def _lattice_adjacent(p1, p2):
        return sum(abs(a-b) for a, b in zip(p1, p2)) == 1
    
    # Precompute contact count per residue pair for faster updates
    # We'll track contacts as a set of frozenset pairs for O(1) lookup
    
    for it in range(max_iters):
        if _time_left() < 1.0:
            break
        
        # Choose move type: pivot (50%) or crank (50%)
        move_type = rng.random()
        
        if move_type < 0.5:
            # Pivot move: rotate subchain around a bond
            if n < 4:
                continue
            pivot_idx = rng.randint(1, n - 3)
            orig_dir = tuple(chain[pivot_idx + 1][k] - chain[pivot_idx][k] for k in range(3))
            
            # Generate 4 possible new directions (excluding original and opposite)
            possible_dirs = []
            for ax in range(3):
                for s in (-1, 1):
                    d = [0, 0, 0]
                    d[ax] = s
                    d = tuple(d)
                    if d != orig_dir and d != tuple(-x for x in orig_dir):
                        possible_dirs.append(d)
            
            if not possible_dirs:
                continue
            
            new_dir = possible_dirs[rng.randint(0, len(possible_dirs) - 1)]
            
            subchain_len = n - pivot_idx - 1
            subchain_old = list(chain[pivot_idx + 1:])
            
            # Try each possible direction
            found_valid = False
            for try_dir in possible_dirs:
                # Build new subchain by translating/rotating
                new_subchain = [None] * subchain_len
                new_subchain[0] = tuple(chain[pivot_idx][k] + try_dir[k] for k in range(3))
                
                # Check if first position is valid
                if new_subchain[0] in occupied:
                    continue
                
                # Build rest of subchain maintaining unit steps
                valid = True
                for k in range(1, subchain_len):
                    old_dx = subchain_old[k][0] - subchain_old[k-1][0]
                    old_dy = subchain_old[k][1] - subchain_old[k-1][1]
                    old_dz = subchain_old[k][2] - subchain_old[k-2][2]
                    
                    new_pos = tuple(new_subchain[k-1][m] + subchain_old[k][m] - subchain_old[k-1][m] for m in range(3))
                    
                    if new_pos in occupied:
                        valid = False
                        break
                    new_subchain[k] = new_pos
                
                if not valid:
                    continue
                
                # Check self-collision within new subchain
                seen = set()
                for pos in new_subchain:
                    if pos in seen:
                        valid = False
                        break
                    seen.add(pos)
                
                if not valid:
                    continue
                
                # Check bond connections
                if not _lattice_adjacent(chain[pivot_idx], new_subchain[0]):
                    continue
                
                if pivot_idx + subchain_len < n:
                    if not _lattice_adjacent(new_subchain[-1], chain[pivot_idx + subchain_len]):
                        continue
                
                # This move is valid - compute energy change
                # Remove old subchain positions from occupied
                for pos in subchain_old:
                    del occupied[pos]
                
                # Add new subchain positions
                for k, pos in enumerate(new_subchain):
                    occupied[pos] = pivot_idx + 1 + k
                
                # Compute new contacts
                new_chain = chain[:pivot_idx+1] + new_subchain
                new_contacts = _compute_h_contacts(sequence, new_chain)
                new_energy = -len(new_contacts)
                
                if new_energy < current_energy:
                    # Accept
                    chain = new_chain
                    current_energy = new_energy
                    current_contacts = new_contacts
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
                        best_contacts = set(current_contacts)
                    found_valid = True
                    break
                else:
                    # Reject - restore occupied
                    for pos in subchain_old:
                        occupied[pos] = chain.index(pos) if pos in occupied else pivot_idx + 1 + subchain_old.index(pos)
                    # Rebuild occupied from chain
                    occupied = {p: i for i, p in enumerate(chain)}
            
            if found_valid:
                occupied = {p: i for i, p in enumerate(chain)}
        
        else:
            # Crank move: translate a subchain
            if n < 5:
                continue
            start_idx = rng.randint(1, n - 3)
            end_idx = rng.randint(start_idx + 1, n - 1)
            subchain = list(chain[start_idx:end_idx + 1])
            
            # Try all 6 translation directions
            accepted = False
            for ax in range(3):
                if accepted:
                    break
                for s in (-1, 1):
                    new_subchain = []
                    for p in subchain:
                        np = list(p)
                        np[ax] += s
                        new_subchain.append(tuple(np))
                    
                    # Check self-collision within new subchain
                    seen = set()
                    can_place = True
                    for np in new_subchain:
                        if np in seen:
                            can_place = False
                            break
                        seen.add(np)
                    
                    if not can_place:
                        continue
                    
                    # Remove old positions
                    for p in subchain:
                        del occupied[p]
                    
                    # Check new positions against occupied
                    for np in new_subchain:
                        if np in occupied:
                            # Restore and break
                            for i, p in enumerate(subchain):
                                occupied[p] = start_idx + i
                            can_place = False
                            break
                    
                    if not can_place:
                        continue
                    
                    # Check bond connections
                    if start_idx > 0:
                        prev = chain[start_idx - 1]
                        if not _lattice_adjacent(prev, new_subchain[0]):
                            for i, p in enumerate(subchain):
                                occupied[p] = start_idx + i
                            continue
                    
                    if end_idx < n - 1:
                        next_p = chain[end_idx + 1]
                        if not _lattice_adjacent(new_subchain[-1], next_p):
                            for i, p in enumerate(subchain):
                                occupied[p] = start_idx + i
                            continue
                    
                    # Apply move
                    for i, np in enumerate(new_subchain):
                        occupied[np] = start_idx + i
                    
                    # Compute new energy
                    new_chain = chain[:start_idx] + new_subchain + chain[end_idx + 1:]
                    new_contacts = _compute_h_contacts(sequence, new_chain)
                    new_energy = -len(new_contacts)
                    
                    if new_energy < current_energy:
                        # Accept
                        chain = new_chain
                        current_energy = new_energy
                        current_contacts = new_contacts
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = list(chain)
                            best_contacts = set(current_contacts)
                        accepted = True
                        break
                    else:
                        # Reject, restore
                        for i, p in enumerate(subchain):
                            occupied[p] = start_idx + i
    
    return best_chain

def _simulated_annealing(sequence, chain, dimension, rng, max_iters=1000):
    """Simulated annealing local search."""
    if chain is None:
        return None
    n = len(chain)
    if n < 4:
        return chain
    
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    # Simulated annealing parameters
    T_start = 5.0
    T_end = 0.1
    cooling_rate = (T_end / T_start) ** (1.0 / max_iters)
    T = T_start
    
    for it in range(max_iters):
        if _time_left() < 1.0:
            break
        
        T *= cooling_rate
        
        # Choose move: pivot or crank
        if rng.random() < 0.5:
            # Pivot move
            if n < 4:
                continue
            pivot_idx = rng.randint(1, n - 3)
            orig_dir = tuple(chain[pivot_idx + 1][k] - chain[pivot_idx][k] for k in range(3))
            
            possible_dirs = []
            for ax in range(3):
                for s in (-1, 1):
                    d = [0, 0, 0]
                    d[ax] = s
                    d = tuple(d)
                    if d != orig_dir and d != tuple(-x for x in orig_dir):
                        possible_dirs.append(d)
            
            if not possible_dirs:
                continue
            
            try_dir = possible_dirs[rng.randint(0, len(possible_dirs) - 1)]
            subchain_len = n - pivot_idx - 1
            subchain_old = list(chain[pivot_idx + 1:])
            
            new_subchain = [None] * subchain_len
            new_subchain[0] = tuple(chain[pivot_idx][k] + try_dir[k] for k in range(3))
            
            if new_subchain[0] in occupied:
                continue
            
            valid = True
            for k in range(1, subchain_len):
                new_pos = tuple(new_subchain[k-1][m] + subchain_old[k][m] - subchain_old[k-1][m] for m in range(3))
                if new_pos in occupied:
                    valid = False
                    break
                new_subchain[k] = new_pos
            
            if not valid:
                continue
            
            seen = set()
            for pos in new_subchain:
                if pos in seen:
                    valid = False
                    break
                seen.add(pos)
            
            if not valid:
                continue
            
            if not _lattice_adjacent(chain[pivot_idx], new_subchain[0]):
                continue
            
            if pivot_idx + subchain_len < n and not _lattice_adjacent(new_subchain[-1], chain[pivot_idx + subchain_len]):
                continue
            
            # Apply and compute
            for pos in subchain_old:
                del occupied[pos]
            for k, pos in enumerate(new_subchain):
                occupied[pos] = pivot_idx + 1 + k
            
            new_chain = chain[:pivot_idx+1] + new_subchain
            new_energy = _energy(sequence, new_chain)
            
            delta = new_energy - current_energy
            if delta < 0 or (T > 0.01 and rng.random() < pow(2.71828, -delta / T)):
                chain = new_chain
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
            else:
                for pos in subchain_old:
                    occupied[pos] = chain.index(pos) if pos in occupied else pivot_idx + 1 + subchain_old.index(pos)
                occupied = {p: i for i, p in enumerate(chain)}
        else:
            # Crank move
            if n < 5:
                continue
            start_idx = rng.randint(1, n - 3)
            end_idx = rng.randint(start_idx + 1, n - 1)
            subchain = list(chain[start_idx:end_idx + 1])
            
            for ax in range(3):
                for s in (-1, 1):
                    new_subchain = []
                    for p in subchain:
                        np = list(p)
                        np[ax] += s
                        new_subchain.append(tuple(np))
                    
                    seen = set()
                    can_place = True
                    for np in new_subchain:
                        if np in seen:
                            can_place = False
                            break
                        seen.add(np)
                    
                    if not can_place:
                        continue
                    
                    for p in subchain:
                        del occupied[p]
                    
                    for np in new_subchain:
                        if np in occupied:
                            for i, p in enumerate(subchain):
                                occupied[p] = start_idx + i
                            can_place = False
                            break
                    
                    if not can_place:
                        continue
                    
                    if start_idx > 0 and not _lattice_adjacent(chain[start_idx - 1], new_subchain[0]):
                        for i, p in enumerate(subchain):
                            occupied[p] = start_idx + i
                        continue
                    
                    if end_idx < n - 1 and not _lattice_adjacent(new_subchain[-1], chain[end_idx + 1]):
                        for i, p in enumerate(subchain):
                            occupied[p] = start_idx + i
                        continue
                    
                    for i, np in enumerate(new_subchain):
                        occupied[np] = start_idx + i
                    
                    new_chain = chain[:start_idx] + new_subchain + chain[end_idx + 1:]
                    new_energy = _energy(sequence, new_chain)
                    
                    delta = new_energy - current_energy
                    if delta < 0 or (T > 0.01 and rng.random() < pow(2.71828, -delta / T)):
                        chain = new_chain
                        current_energy = new_energy
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = list(chain)
                        break
                    else:
                        for i, p in enumerate(subchain):
                            occupied[p] = start_idx + i
    
    return best_chain

def _lattice_adjacent(p1, p2):
    return sum(abs(a-b) for a, b in zip(p1, p2)) == 1

def fold(sequence, dimension):
    """Best conformation using multiple strategies: greedy growth + local search + simulated annealing."""
    n = len(sequence)
    rng = random.Random(12345 + n)
    best, best_energy = None, 1
    
    # Number of restarts based on sequence length and time budget
    time_per_restart = 0.3 if n <= 50 else (0.5 if n <= 80 else 1.0)
    max_restarts = int(_time_left() / time_per_restart)
    max_restarts = max(20, min(max_restarts, 1000))
    
    for r in range(max_restarts):
        if _time_left() < 3.0:
            break
        
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        # Local search with simulated annealing
        if _time_left() > 2.0:
            # Alternate between local search and simulated annealing
            if r % 2 == 0:
                max_ls = 200 if n <= 50 else (150 if n <= 80 else 80)
                chain = _local_search(sequence, chain, dimension, rng, max_iters=max_ls)
            else:
                max_sa = 300 if n <= 50 else (200 if n <= 80 else 100)
                chain = _simulated_annealing(sequence, chain, dimension, rng, max_iters=max_sa)
        
        if chain is None:
            continue
        
        value = _energy(sequence, chain)
        if value < best_energy:
            best, best_energy = chain, value
    
    if best is None:
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
