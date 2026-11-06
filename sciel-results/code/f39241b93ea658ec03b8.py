"""HP lattice protein folding on the cubic lattice."""
import random
import math
import time

def _neighbours(point):
    """Generator of 6-neighbours on the cubic lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _neighbours_list(point):
    """Return list of 6-neighbor tuples for faster iteration."""
    result = []
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            result.append(tuple(q))
    return result

def _beam_search_growth(sequence, dimension, rng, beam_width=200):
    """
    Beam search for chain growth.
    Each beam entry: (chain, occupied_dict, score)
    Score = negative contact count (we want to minimize = maximize contacts)
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[(0,) * dimension]]
    
    is_h = [1 if s == 'H' else 0 for s in sequence]
    
    # Initialize: first residue at origin
    start = (0,) * dimension
    initial_occupied = {start: 0}
    beams = [([start], initial_occupied, 0)]  # chain, occupied, energy_so_far
    
    for i in range(1, n):
        if not beams:
            break
        new_beams = []
        for chain, occupied, energy in beams:
            last = chain[-1]
            for site in _neighbours(last):
                if site in occupied:
                    continue
                # Compute new energy contribution
                new_energy = energy
                if is_h[i]:
                    for nb in _neighbours(site):
                        j = occupied.get(nb, None)
                        if j is not None and j < i - 1 and is_h[j]:
                            new_energy -= 1
                new_occupied = dict(occupied)
                new_occupied[site] = i
                new_chain = chain + [site]
                new_beams.append((new_chain, new_occupied, new_energy))
        
        if not new_beams:
            break
        
        # Sort by energy (ascending = most negative = best) and keep top beam_width
        new_beams.sort(key=lambda x: x[2])
        beams = new_beams[:beam_width]
    
    if not beams:
        return None
    
    # Return the best complete chain
    best = min(beams, key=lambda x: x[2])
    return best[0]

def _compute_energy(sequence, chain):
    """Compute the energy of a complete chain."""
    n = len(chain)
    if n < 2:
        return 0
    is_h = [1 if s == 'H' else 0 for s in sequence]
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i in range(n):
        if not is_h[i]:
            continue
        for nb in _neighbours(chain[i]):
            j = pos_to_idx.get(nb, None)
            if j is not None and j > i + 1 and is_h[j]:
                contacts += 1
    return -contacts

def _apply_pivot_move(chain, k, axis, direction, dimension):
    """
    Apply a pivot move: rotate segment k..n-1 around chain[k-1].
    Returns new chain or None if invalid (self-intersection).
    k is 1..n-1, pivot is at index k-1.
    """
    n = len(chain)
    if k < 1 or k >= n:
        return None
    
    pivot = chain[k-1]
    px, py, pz = pivot
    
    # Build set of head positions for collision check
    head_set = set(chain[:k])
    
    # Inline rotation for speed
    new_tail = []
    if axis == 0 and direction == 1:
        # (x, y, z) -> (x, -z, y)
        for i in range(k, n):
            p = chain[i]
            nx = px + (p[0] - px)
            ny = pz - (p[2] - pz)
            nz = py + (p[1] - py)
            new_p = (nx, ny, nz)
            if new_p in head_set:
                return None
            new_tail.append(new_p)
    elif axis == 0 and direction == -1:
        # (x, y, z) -> (x, z, -y)
        for i in range(k, n):
            p = chain[i]
            nx = px + (p[0] - px)
            ny = pz + (p[2] - pz)
            nz = py - (p[1] - py)
            new_p = (nx, ny, nz)
            if new_p in head_set:
                return None
            new_tail.append(new_p)
    elif axis == 1 and direction == 1:
        # (x, y, z) -> (z, y, -x)
        for i in range(k, n):
            p = chain[i]
            nx = pz + (p[2] - pz)
            ny = py + (p[1] - py)
            nz = px - (p[0] - px)
            new_p = (nx, ny, nz)
            if new_p in head_set:
                return None
            new_tail.append(new_p)
    elif axis == 1 and direction == -1:
        # (x, y, z) -> (-z, y, x)
        for i in range(k, n):
            p = chain[i]
            nx = pz - (p[2] - pz)
            ny = py + (p[1] - py)
            nz = px + (p[0] - px)
            new_p = (nx, ny, nz)
            if new_p in head_set:
                return None
            new_tail.append(new_p)
    elif axis == 2 and direction == 1:
        # (x, y, z) -> (-y, x, z)
        for i in range(k, n):
            p = chain[i]
            nx = py - (p[1] - py)
            ny = px + (p[0] - px)
            nz = pz + (p[2] - pz)
            new_p = (nx, ny, nz)
            if new_p in head_set:
                return None
            new_tail.append(new_p)
    else:
        # (x, y, z) -> (y, -x, z)
        for i in range(k, n):
            p = chain[i]
            nx = py + (p[1] - py)
            ny = px - (p[0] - px)
            nz = pz + (p[2] - pz)
            new_p = (nx, ny, nz)
            if new_p in head_set:
                return None
            new_tail.append(new_p)
    
    return chain[:k] + new_tail

def _compute_delta_energy(sequence, chain, k, new_chain, is_h):
    """
    Compute energy difference when replacing tail (k..n-1) with new_tail.
    Only cross-contacts between head (0..k-1) and tail change.
    is_h is precomputed list of 0/1 for H residues.
    """
    n = len(chain)
    if k < 1 or k >= n:
        return 0
    
    # Build position maps for old tail and new tail
    old_tail = chain[k:]
    new_tail = new_chain[k:]
    
    old_tail_pos_set = set(old_tail)
    new_tail_pos_set = set(new_tail)
    
    # Map positions to indices in tails
    old_tail_idx = {p: i for i, p in enumerate(old_tail)}
    new_tail_idx = {p: i for i, p in enumerate(new_tail)}
    
    # Count old cross-contacts: H in head adjacent to H in old_tail
    old_cross = 0
    head = chain[:k]
    for i in range(k):
        if not is_h[i]:
            continue
        p = head[i]
        px, py, pz = p
        # Check 6 neighbors inline
        for nb in ((px+1,py,pz),(px-1,py,pz),(px,py+1,pz),(px,py-1,pz),(px,py,pz+1),(px,py,pz-1)):
            j = old_tail_idx.get(nb, -1)
            if j >= 0 and is_h[k + j]:
                old_cross += 1
    
    # Count new cross-contacts
    new_cross = 0
    for i in range(k):
        if not is_h[i]:
            continue
        p = head[i]
        px, py, pz = p
        for nb in ((px+1,py,pz),(px-1,py,pz),(px,py+1,pz),(px,py-1,pz),(px,py,pz+1),(px,py,pz-1)):
            j = new_tail_idx.get(nb, -1)
            if j >= 0 and is_h[k + j]:
                new_cross += 1
    
    # Energy = -contacts, so delta = -(new_cross - old_cross) = old_cross - new_cross
    return old_cross - new_cross

def _local_search(sequence, chain, dimension, rng, max_time, is_h):
    """
    Pivot-move local search with simulated annealing.
    Returns best chain found.
    """
    n = len(chain)
    if n < 2:
        return chain
    
    current_chain = chain[:]
    current_energy = _compute_energy(sequence, current_chain)
    best_chain = current_chain[:]
    best_energy = current_energy
    
    T_max = 8.0
    T_min = 0.05
    
    start_time = time.time()
    
    while time.time() - start_time < max_time:
        # Choose pivot point k (1 to n-1)
        k = rng.randint(1, n - 1)
        axis = rng.randint(0, dimension - 1)
        direction = rng.choice([-1, 1])
        
        new_chain = _apply_pivot_move(current_chain, k, axis, direction, dimension)
        if new_chain is None:
            continue
        
        delta = _compute_delta_energy(sequence, current_chain, k, new_chain, is_h)
        
        # Metropolis acceptance
        if delta < 0:
            accept = True
        else:
            elapsed = time.time() - start_time
            progress = min(1.0, elapsed / max_time)
            T = T_max - (T_max - T_min) * progress
            if T < T_min:
                T = T_min
            prob = math.exp(-delta / T)
            accept = rng.random() < prob
        
        if accept:
            current_chain = new_chain
            current_energy += delta
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = current_chain[:]
    
    return best_chain

def _multi_start_beam_growth(sequence, dimension, rng, beam_width, num_starts, time_budget):
    """
    Run beam search multiple times with different random seeds and return the best chain.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[(0,) * dimension]]
    
    is_h = [1 if s == 'H' else 0 for s in sequence]
    best_chain = None
    best_energy = float('inf')
    
    start_time = time.time()
    per_start_budget = time_budget / max(1, num_starts)
    
    for s in range(num_starts):
        if time.time() - start_time >= time_budget:
            break
        
        # Vary the beam width slightly for diversity
        bw = max(50, beam_width + rng.randint(-50, 50))
        
        chain = _beam_search_growth(sequence, dimension, rng, bw)
        if chain is not None:
            energy = _compute_energy(sequence, chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = chain
    
    if best_chain is None:
        return None
    
    return best_chain

def _refined_local_search(sequence, chain, dimension, rng, max_time, is_h):
    """
    Refined local search with adaptive pivot selection and multiple cooling schedules.
    """
    n = len(chain)
    if n < 2:
        return chain
    
    current_chain = chain[:]
    current_energy = _compute_energy(sequence, current_chain)
    best_chain = current_chain[:]
    best_energy = current_energy
    
    start_time = time.time()
    
    # Try multiple cooling schedules
    schedules = [
        (10.0, 0.05, 0.0),   # exponential decay
        (8.0, 0.03, 0.5),    # slower cooling
        (6.0, 0.01, 0.8),    # fast cooling
    ]
    
    for T_max, T_min, plateau in schedules:
        remaining = max_time - (time.time() - start_time)
        if remaining <= 0.1:
            break
        
        # Run SA for a fraction of remaining time
        sa_time = remaining * 0.5
        if sa_time < 0.1:
            break
        
        sa_start = time.time()
        
        while time.time() - sa_start < sa_time:
            k = rng.randint(1, n - 1)
            axis = rng.randint(0, dimension - 1)
            direction = rng.choice([-1, 1])
            
            new_chain = _apply_pivot_move(current_chain, k, axis, direction, dimension)
            if new_chain is None:
                continue
            
            delta = _compute_delta_energy(sequence, current_chain, k, new_chain, is_h)
            
            if delta < 0:
                accept = True
            else:
                elapsed = time.time() - sa_start
                progress = min(1.0, elapsed / sa_time)
                if progress < plateau:
                    T = T_max
                else:
                    T = T_max - (T_max - T_min) * ((progress - plateau) / (1 - plateau))
                if T < T_min:
                    T = T_min
                prob = math.exp(-delta / T)
                accept = rng.random() < prob
            
            if accept:
                current_chain = new_chain
                current_energy += delta
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = current_chain[:]
    
    # Final polish: deterministic local search with small moves
    remaining = max_time - (time.time() - start_time)
    if remaining > 0.5:
        polish_start = time.time()
        improved = True
        while improved and time.time() - polish_start < remaining * 0.5:
            improved = False
            for k in range(1, n):
                for axis in range(dimension):
                    for direction in (-1, 1):
                        new_chain = _apply_pivot_move(current_chain, k, axis, direction, dimension)
                        if new_chain is None:
                            continue
                        delta = _compute_delta_energy(sequence, current_chain, k, new_chain, is_h)
                        if delta < 0:
                            current_chain = new_chain
                            current_energy += delta
                            improved = True
                            if current_energy < best_energy:
                                best_energy = current_energy
                                best_chain = current_chain[:]
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break
    
    return best_chain

def fold(sequence, dimension):
    """
    Fold the protein chain using multi-start beam search initialization and refined local search.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0] * dimension]
    
    # Global deadline guard
    if not hasattr(fold, '_start_time'):
        fold._start_time = time.time()
    
    elapsed_global = time.time() - fold._start_time
    total_budget = 290.0
    remaining = total_budget - elapsed_global
    if remaining <= 0:
        return [[i] + [0] * (dimension - 1) for i in range(n)]
    
    # Allocate time per call - be more generous for shorter chains
    if n <= 50:
        per_call_budget = min(remaining * 0.45, 22.0)
    elif n <= 100:
        per_call_budget = min(remaining * 0.4, 20.0)
    else:
        per_call_budget = min(remaining * 0.35, 18.0)
    if per_call_budget < 0.5:
        per_call_budget = 0.5
    
    rng = random.Random(12345 + len(sequence) * 7 + n)
    
    is_h = [1 if s == 'H' else 0 for s in sequence]
    
    # Phase 1: Multi-start beam search initialization
    init_budget = per_call_budget * 0.45
    num_starts = max(1, min(6, int(init_budget / 1.5)))
    
    # Adjust beam width based on chain length - wider for shorter chains
    if n <= 50:
        beam_width = 800
    elif n <= 100:
        beam_width = 500
    else:
        beam_width = 300
    
    best_init_chain = None
    best_init_energy = float('inf')
    
    # Try multiple starts with varying beam widths
    for s in range(num_starts):
        if time.time() - (fold._start_time) >= total_budget - 10:
            break
        
        # Vary beam width for diversity
        bw = max(150, beam_width + rng.randint(-150, 150))
        chain = _beam_search_growth(sequence, dimension, rng, bw)
        if chain is not None:
            energy = _compute_energy(sequence, chain)
            if energy < best_init_energy:
                best_init_energy = energy
                best_init_chain = chain
    
    # Fallback: greedy growth with look-ahead
    if best_init_chain is None:
        start = (0,) * dimension
        chain, occupied = [start], {start: 0}
        success = True
        for i in range(1, n):
            options = []
            for site in _neighbours(chain[-1]):
                if site in occupied:
                    continue
                gain = 0
                if is_h[i]:
                    for nb in _neighbours(site):
                        j = occupied.get(nb, None)
                        if j is not None and j < i - 1 and is_h[j]:
                            gain += 1
                free = sum(1 for nb in _neighbours(site) if nb not in occupied)
                # Penalize moves that create dead-ends
                dead_end_penalty = 0
                if free <= 1 and i < n - 2:
                    dead_end_penalty = 0.5
                options.append((gain + 0.1 * free - dead_end_penalty + rng.random(), site))
            if not options:
                success = False
                break
            site = max(options)[1]
            occupied[site] = i
            chain.append(site)
        if success:
            energy = _compute_energy(sequence, chain)
            if energy < best_init_energy:
                best_init_energy = energy
                best_init_chain = chain
    
    if best_init_chain is None:
        best_init_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_init_energy = 0
    
    # Phase 2: Refined local search with remaining budget
    elapsed_so_far = time.time() - fold._start_time
    remaining_after_init = total_budget - elapsed_so_far
    ls_budget = min(remaining_after_init * 0.9, per_call_budget * 0.6)
    
    if ls_budget > 0.5:
        best_chain = _refined_local_search(sequence, best_init_chain, dimension, rng, ls_budget, is_h)
    else:
        best_chain = best_init_chain
    
    return [list(p) for p in best_chain]
