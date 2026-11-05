"""
HP Lattice Protein Folding on Cubic Lattice
Redesigned: contact-based energy with explicit H-H contact counting during growth,
look-ahead scoring, multi-move local search with simulated annealing.
"""
import math
import random
import time

# EVOLVE-BLOCK-START

# Precompute neighbor offsets for speed
_NEIGHBOR_OFFSETS = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]

def _count_hh_contacts(sequence, chain):
    """
    Count H-H contacts efficiently using a set of H positions.
    Returns number of non-bonded H-H contacts.
    """
    n = len(chain)
    if n == 0:
        return 0
    
    # Build position -> index map and set of H positions
    pos_to_idx = {}
    h_positions = set()
    for i, p in enumerate(chain):
        pos_to_idx[p] = i
        if sequence[i] == 'H':
            h_positions.add(p)
    
    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        x, y, z = chain[i]
        for dx, dy, dz in _NEIGHBOR_OFFSETS:
            j = pos_to_idx.get((x+dx, y+dy, z+dz), -1)
            if j != -1 and abs(i - j) > 1 and sequence[j] == 'H':
                contacts += 1
    return contacts // 2

def _energy(sequence, chain):
    """Energy = -number of H-H contacts."""
    return -_count_hh_contacts(sequence, chain)

def _grow_with_lookahead(sequence, rng, temp=1.0, bias=4.0, lookahead=3):
    """
    Grow a self-avoiding walk with contact-based bias and look-ahead scoring.
    The lookahead considers how many future H residues could potentially
    form contacts with the current placement.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0): 0}
    
    # Precompute H positions in sequence for faster lookups
    h_indices = [i for i in range(n) if sequence[i] == 'H']
    
    for i in range(1, n):
        last = chain[-1]
        lx, ly, lz = last
        neighbors = [
            (lx+1, ly, lz), (lx-1, ly, lz),
            (lx, ly+1, lz), (lx, ly-1, lz),
            (lx, ly, lz+1), (lx, ly, lz-1)
        ]
        
        available = []
        for p in neighbors:
            if p not in occupied:
                available.append(p)
        
        if not available:
            return None
        
        scored = []
        for p in available:
            score = 0.0
            px, py, pz = p
            
            if sequence[i] == 'H':
                # Count immediate H-H contacts
                contact_gain = 0
                for dx, dy, dz in _NEIGHBOR_OFFSETS:
                    j = occupied.get((px+dx, py+dy, pz+dz), -1)
                    if j != -1 and abs(i - j) > 1 and sequence[j] == 'H':
                        contact_gain += 1
                score += contact_gain * bias
                
                # Look-ahead: count how many future H residues are within
                # distance that could potentially form contacts
                # (within Chebyshev distance 2 from this position)
                if lookahead > 0:
                    future_h_count = 0
                    for fi in range(i + 2, min(i + 2 + lookahead, n)):
                        if sequence[fi] == 'H':
                            # Check if position fi could be near p
                            # We don't know where fi will go, so use a heuristic:
                            # count how many free neighbor positions are available
                            # that could serve as intermediate positions
                            pass  # Simplified: just add small bonus for being near other H
                    # Instead, use a simpler heuristic: prefer positions that have
                    # more free neighbors (more flexibility for future growth)
                    free_neighbors = 0
                    for dx, dy, dz in _NEIGHBOR_OFFSETS:
                        if (px+dx, py+dy, pz+dz) not in occupied:
                            free_neighbors += 1
                    score += free_neighbors * 0.3
            else:
                # P residue: prefer positions with more free neighbors
                free_neighbors = 0
                px, py, pz = p
                for dx, dy, dz in _NEIGHBOR_OFFSETS:
                    if (px+dx, py+dy, pz+dz) not in occupied:
                        free_neighbors += 1
                score += free_neighbors * 0.2
            
            scored.append((score, p))
        
        # Boltzmann selection
        max_score = max(s for s, _ in scored)
        weights = []
        total_w = 0.0
        for s, p in scored:
            w = math.exp((s - max_score) / temp)
            weights.append(w)
            total_w += w
        
        if total_w == 0:
            chosen = rng.choice(available)
        else:
            r = rng.random() * total_w
            cumulative = 0.0
            chosen = scored[0][1]
            for w, (_, p) in zip(weights, scored):
                cumulative += w
                if r <= cumulative:
                    chosen = p
                    break
        
        chain.append(chosen)
        occupied[chosen] = i
    
    return chain

def _regrow_tail(sequence, chain, start_idx, rng, temp=0.5):
    """
    Re-grow the chain from start_idx to the end, keeping chain[:start_idx] fixed.
    """
    n = len(sequence)
    if start_idx >= n:
        return chain[:]
    
    head = chain[:start_idx]
    occupied = {p: i for i, p in enumerate(head)}
    
    new_chain = head[:]
    
    for i in range(start_idx, n):
        last = new_chain[-1]
        lx, ly, lz = last
        neighbors = [
            (lx+1, ly, lz), (lx-1, ly, lz),
            (lx, ly+1, lz), (lx, ly-1, lz),
            (lx, ly, lz+1), (lx, ly, lz-1)
        ]
        
        prev_pos = new_chain[-2] if len(new_chain) > 1 else None
        available = []
        for p in neighbors:
            if p in occupied:
                continue
            if p == prev_pos:
                continue
            available.append(p)
            
        if not available:
            return None
            
        scored = []
        for p in available:
            score = 0
            if sequence[i] == 'H':
                px, py, pz = p
                for dx, dy, dz in _NEIGHBOR_OFFSETS:
                    j = occupied.get((px+dx, py+dy, pz+dz), -1)
                    if j != -1 and abs(i - j) > 1 and sequence[j] == 'H':
                        score += 1
            scored.append((score, p))
            
        max_s = max(s for s, _ in scored)
        weights = [math.exp((s - max_s) / temp) for s, _ in scored]
        total_w = sum(weights)
        if total_w == 0:
            chosen = rng.choice(available)
        else:
            r = rng.random() * total_w
            cumulative = 0
            for w, (_, p) in zip(weights, scored):
                cumulative += w
                if r <= cumulative:
                    chosen = p
                    break
                    
        new_chain.append(chosen)
        occupied[chosen] = i
        
    return new_chain

def _edge_move(sequence, chain, idx, rng):
    """
    Try to move a single residue to an adjacent position.
    Returns new chain or None if invalid.
    """
    n = len(chain)
    if idx <= 0 or idx >= n - 1:
        return None
    
    p = chain[idx]
    px, py, pz = p
    
    # Try all 6 neighbor positions
    candidates = []
    for dx, dy, dz in _NEIGHBOR_OFFSETS:
        new_p = (px+dx, py+dy, pz+dz)
        # Check self-avoidance: new position must not be occupied by any other residue
        # except the ones we're moving from
        valid = True
        for j, q in enumerate(chain):
            if j == idx:
                continue
            if q == new_p:
                valid = False
                break
        # Also check it's not adjacent to the chain neighbors (would create bond violation)
        # Actually, we just need it to be a unit step from chain[idx-1] and chain[idx+1]
        # But moving idx means it must still be connected to idx-1 and idx+1
        # So new_p must be adjacent to chain[idx-1] AND chain[idx+1]
        if valid:
            # Check adjacency to neighbors in chain
            prev_p = chain[idx-1]
            next_p = chain[idx+1]
            if (abs(new_p[0]-prev_p[0]) + abs(new_p[1]-prev_p[1]) + abs(new_p[2]-prev_p[2]) != 1):
                valid = False
            if valid and (abs(new_p[0]-next_p[0]) + abs(new_p[1]-next_p[1]) + abs(new_p[2]-next_p[2]) != 1):
                valid = False
        if valid:
            candidates.append(new_p)
    
    if not candidates:
        return None
    
    new_p = rng.choice(candidates)
    new_chain = chain[:]
    new_chain[idx] = new_p
    return new_chain

def _local_search(sequence, chain, rng, time_budget=2.0):
    """
    Local search using multiple move types with Simulated Annealing.
    Moves: pivot-regrow, edge moves.
    """
    n = len(chain)
    if n < 4:
        return chain
    
    c = chain[:]
    current_energy = _energy(sequence, c)
    best_chain = c[:]
    best_energy = current_energy
    
    start_time = time.time()
    T = 1.5
    T_min = 0.01
    cooling_rate = 0.995
    
    while time.time() - start_time < time_budget:
        move_type = rng.random()
        
        if move_type < 0.7:
            # Pivot-regrow move
            pivot = rng.randint(1, n - 2)
            new_chain = _regrow_tail(sequence, c, pivot, rng, temp=0.5)
        else:
            # Edge move
            idx = rng.randint(1, n - 2)
            new_chain = _edge_move(sequence, c, idx, rng)
        
        if new_chain is None:
            T = max(T_min, T * cooling_rate)
            continue
            
        new_energy = _energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        if delta_e <= 0 or rng.random() < math.exp(-delta_e / T):
            c = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = c[:]
                
        T = max(T_min, T * cooling_rate)
        
    return best_chain

def _mccg_fold(sequence, dimension, time_limit, seed):
    """
    Main folding function: MCCG with contact bias + local search.
    """
    rng = random.Random(seed)
    n = len(sequence)
    
    best_chain = None
    best_energy = 0
    
    start_time = time.time()
    
    # Phase 1: Multiple MCCG runs with varying parameters
    attempts = 0
    max_attempts = 8000
    
    top_chains = []  # (energy, chain)
    
    while attempts < max_attempts:
        elapsed = time.time() - start_time
        if elapsed > time_limit * 0.55:
            break
        
        # Vary temperature and bias for diversity
        temp = 0.3 + rng.random() * 1.2
        bias = 3.0 + rng.random() * 5.0
        
        chain = _grow_with_lookahead(sequence, rng, temp=temp, bias=bias)
        if chain is not None:
            e = _energy(sequence, chain)
            if e < best_energy:
                best_energy = e
                best_chain = chain[:]
            
            top_chains.append((e, chain))
            top_chains.sort(key=lambda x: x[0])
            if len(top_chains) > 10:
                top_chains.pop()
        
        attempts += 1
    
    # Phase 2: Local search on the best chains
    remaining = time_limit - (time.time() - start_time)
    if remaining > 0.5 and best_chain is not None:
        # Allocate time among top chains
        num_ls = min(4, len(top_chains))
        ls_time_each = min(2.0, remaining * 0.45 / max(1, num_ls))
        
        for e, chain in top_chains[:num_ls]:
            if time.time() - start_time > time_limit - 0.2:
                break
            
            refined = _local_search(sequence, chain, rng, time_budget=ls_time_each)
            refined_e = _energy(sequence, refined)
            
            if refined_e < best_energy:
                best_energy = refined_e
                best_chain = refined[:]
    
    # Phase 3: If time remains, do more MCCG + quick local search
    remaining = time_limit - (time.time() - start_time)
    if remaining > 1.0 and best_chain is not None:
        # More growth attempts
        while time.time() - start_time < time_limit - 0.5:
            temp = 0.3 + rng.random() * 1.0
            bias = 4.0 + rng.random() * 4.0
            chain = _grow_with_lookahead(sequence, rng, temp=temp, bias=bias)
            if chain is not None:
                e = _energy(sequence, chain)
                if e < best_energy:
                    best_energy = e
                    best_chain = chain[:]
        
        # Final local search on best
        if time.time() - start_time < time_limit - 0.3:
            final_time = min(3.0, time_limit - (time.time() - start_time) - 0.2)
            if final_time > 0.5:
                refined = _local_search(sequence, best_chain, rng, time_budget=final_time)
                refined_e = _energy(sequence, refined)
                if refined_e < best_energy:
                    best_energy = refined_e
                    best_chain = refined[:]
    
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
    
    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Fold the sequence using MCCG with contact-based bias and local search.
    """
    n = len(sequence)
    # Scale time with length: longer chains need more time
    time_limit = 1.5 + 0.05 * n
    if time_limit > 10.0:
        time_limit = 10.0
    if time_limit < 1.5:
        time_limit = 1.5

    seed = hash(sequence) % (2**32)
    
    chain, energy = _mccg_fold(sequence, dimension, time_limit, seed)
    
    return [list(p) for p in chain]

# EVOLVE-BLOCK-END
