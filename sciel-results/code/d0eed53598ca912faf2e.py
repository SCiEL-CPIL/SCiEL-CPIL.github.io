"""Rewritten program: multi-restart SA with tail-regrowth + pivot moves,
incremental energy, and a global deadline guard."""

import random
import time
import math
import sys

# Global deadline: 330 s wall-clock budget for all sequences.
_GLOBAL_DEADLINE = 325.0
_start_wall = time.time()


def _remaining_time():
    return max(0.0, _GLOBAL_DEADLINE - (time.time() - _start_wall))


def _energy(sequence, chain):
    """Calculate total energy of a conformation."""
    where = {p: i for i, p in enumerate(chain)}
    h_positions = [i for i, c in enumerate(sequence) if c == 'H']
    contacts = 0
    for i in h_positions:
        p = chain[i]
        x, y, z = p
        for q in ((x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)):
            j = where.get(q)
            if j is not None and j > i + 1:
                if sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _pivot_move(sequence, chain, occupied, k, rng):
    """
    Attempt a pivot move: rotate the sub-chain from index k to end
    around the bond between k-1 and k.
    Returns (new_chain, new_occupied, new_energy) or (None, None, None) if invalid.
    """
    n = len(chain)
    if k < 1 or k >= n:
        return None, None, None
    
    # The pivot point is chain[k-1], and we rotate chain[k:] around the bond (k-1, k)
    # We try all 6 possible new directions for the bond (k-1, k), excluding the current one
    px, py, pz = chain[k-1]
    cx, cy, cz = chain[k]
    
    # Current direction from k-1 to k
    cur_dx = cx - px
    cur_dy = cy - py
    cur_dz = cz - pz
    
    # All 6 unit directions
    all_dirs = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    valid_dirs = [d for d in all_dirs if d != (cur_dx, cur_dy, cur_dz)]
    
    if not valid_dirs:
        return None, None, None
    
    # Pick a random direction
    new_dir = valid_dirs[rng.randint(0, len(valid_dirs)-1)]
    ndx, ndy, ndz = new_dir
    
    # New position for chain[k]
    new_k_pos = (px + ndx, py + ndy, pz + ndz)
    
    # Check if new_k_pos is occupied (excluding positions k..n-1 which we're moving)
    # We need to check against occupied[0..k-1]
    for i in range(k):
        if chain[i] == new_k_pos:
            return None, None, None
    
    # Now we need to check if the rest of the tail can be placed
    # This is complex; for simplicity, we'll try to regrow the tail greedily
    # from the new position with some randomness
    
    # Actually, let's do a simpler approach: just check if new_k_pos is free
    # and then regrow the tail
    new_occupied = {chain[i]: i for i in range(k)}
    
    new_chain = list(chain[:k])
    new_chain.append(new_k_pos)
    new_occupied[new_k_pos] = k
    
    h_set = set(i for i, c in enumerate(sequence) if c == 'H')
    
    # Regrow tail from k+1 to end
    for i in range(k+1, n):
        last = new_chain[-1]
        lx, ly, lz = last
        options = []
        for site in ((lx+1,ly,lz),(lx-1,ly,lz),(lx,ly+1,lz),(lx,ly-1,lz),(lx,ly,lz+1),(lx,ly,lz-1)):
            if site in new_occupied:
                continue
            
            score = 0.0
            if i in h_set:
                sx, sy, sz = site
                for q in ((sx+1,sy,sz),(sx-1,sy,sz),(sx,sy+1,sz),(sx,sy-1,sz),(sx,sy,sz+1),(sx,sy,sz-1)):
                    if q in new_occupied:
                        j = new_occupied[q]
                        if j < i - 1 and j in h_set:
                            score += 3.0
            
            sx, sy, sz = site
            empty_neighbors = 0
            for q in ((sx+1,sy,sz),(sx-1,sy,sz),(sx,sy+1,sz),(sx,sy-1,sz),(sx,sy,sz+1),(sx,sy,sz-1)):
                if q not in new_occupied and q != last:
                    empty_neighbors += 1
            score += empty_neighbors * 0.5
            
            score += rng.random() * 0.5
            options.append((score, site))
        
        if not options:
            return None, None, None
        
        options.sort(reverse=True)
        # Pick from top options with some randomness
        top_k = min(3, len(options))
        site = options[rng.randint(0, top_k-1)][1]
        new_chain.append(site)
        new_occupied[site] = i
    
    new_energy = _energy(sequence, new_chain)
    return new_chain, new_occupied, new_energy


def _grow_greedy(sequence, rng, bias=1.0):
    """
    Generate a self-avoiding walk with a greedy bias towards H-H contacts and space.
    """
    n = len(sequence)
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    h_set = set(i for i, c in enumerate(sequence) if c == 'H')

    for i in range(1, n):
        last = chain[-1]
        options = []
        x, y, z = last
        for site in ((x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)):
            if site in occupied:
                continue

            score = 0.0
            if i in h_set:
                sx, sy, sz = site
                for q in ((sx+1,sy,sz),(sx-1,sy,sz),(sx,sy+1,sz),(sx,sy-1,sz),(sx,sy,sz+1),(sx,sy,sz-1)):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and j in h_set:
                            score += 3.0

            sx, sy, sz = site
            empty_neighbors = 0
            for q in ((sx+1,sy,sz),(sx-1,sy,sz),(sx,sy+1,sz),(sx,sy-1,sz),(sx,sy,sz+1),(sx,sy,sz-1)):
                if q not in occupied and q != last:
                    empty_neighbors += 1
            score += empty_neighbors * 0.8

            score += rng.random() * (1.0 - bias) * 2.0
            options.append((score, site))

        if not options:
            return None

        options.sort(reverse=True)
        if bias > 0.7:
            site = options[0][1]
        else:
            top_k = min(4, len(options))
            site = options[rng.randint(0, top_k-1)][1]

        chain.append(site)
        occupied[site] = i

    return chain


def _tail_regrowth(sequence, chain, occupied, k, rng, temperature=1.0):
    """
    Regrow the tail from index k to end.
    Returns new_chain, new_occupied, or (None, None) if stuck.
    """
    n = len(chain)
    if k >= n:
        return chain, occupied

    new_occupied = dict(occupied)
    for i in range(k, n):
        del new_occupied[chain[i]]

    new_chain = chain[:k]
    h_set = set(i for i, c in enumerate(sequence) if c == 'H')

    for i in range(k, n):
        last = new_chain[-1]
        options = []
        x, y, z = last
        for site in ((x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)):
            if site in new_occupied:
                continue

            score = 0.0
            if i in h_set:
                sx, sy, sz = site
                for q in ((sx+1,sy,sz),(sx-1,sy,sz),(sx,sy+1,sz),(sx,sy-1,sz),(sx,sy,sz+1),(sx,sy,sz-1)):
                    if q in new_occupied:
                        j = new_occupied[q]
                        if j < i - 1 and j in h_set:
                            score += 3.0

            sx, sy, sz = site
            empty_neighbors = 0
            for q in ((sx+1,sy,sz),(sx-1,sy,sz),(sx,sy+1,sz),(sx,sy-1,sz),(sx,sy,sz+1),(sx,sy,sz-1)):
                if q not in new_occupied and q != last:
                    empty_neighbors += 1
            score += empty_neighbors * 0.8

            score += rng.random() * temperature
            options.append((score, site))

        if not options:
            return None, None

        if temperature < 0.05:
            options.sort(reverse=True)
            site = options[0][1]
        else:
            scores = [s for s, _ in options]
            max_score = max(scores)
            exp_scores = [math.exp((s - max_score) / temperature) for s in scores]
            total = sum(exp_scores)
            r = rng.random() * total
            cumulative = 0.0
            site = options[0][1]
            for idx, (_, s) in enumerate(options):
                cumulative += exp_scores[idx]
                if r <= cumulative:
                    site = s
                    break

        new_chain.append(site)
        new_occupied[site] = i

    return new_chain, new_occupied


def _sa_fold(sequence, rng, time_limit):
    """
    Simulated Annealing with tail regrowth and pivot moves.
    """
    n = len(sequence)
    if n < 2:
        return [(i, 0, 0) for i in range(n)]

    # Multiple greedy initializations with different biases
    best_init = None
    best_init_energy = float('inf')
    biases = [0.3, 0.6, 0.9, 1.0]
    for b in biases:
        init_chain = _grow_greedy(sequence, rng, bias=b)
        if init_chain is None:
            continue
        e = _energy(sequence, init_chain)
        if e < best_init_energy:
            best_init_energy = e
            best_init = init_chain

    if best_init is None:
        best_init = [(i, 0, 0) for i in range(n)]
        best_init_energy = _energy(sequence, best_init)

    current_chain = best_init
    current_occupied = {p: i for i, p in enumerate(current_chain)}
    current_energy = best_init_energy

    best_chain = list(current_chain)
    best_energy = current_energy

    start_time = time.time()
    T = 2.0
    T_min = 0.05
    cooling_rate = 0.998

    stagnation_counter = 0
    max_stagnation = 500
    accept_count = 0
    total_count = 0

    while T > T_min:
        if time.time() - start_time > time_limit:
            break

        total_count += 1
        move_type = rng.random()
        
        if move_type < 0.6:
            # Tail regrowth
            if move_type < 0.3:
                tail_len = rng.randint(1, 8)
            elif move_type < 0.45:
                tail_len = rng.randint(8, min(25, n-1))
            else:
                tail_len = rng.randint(25, max(25, n//2))
            
            tail_len = max(1, min(n - 1, tail_len))
            k = n - tail_len

            new_chain, new_occupied = _tail_regrowth(sequence, current_chain, current_occupied, k, rng, T)

            if new_chain is None:
                T *= cooling_rate
                continue

            new_energy = _energy(sequence, new_chain)
        else:
            # Pivot move
            k = rng.randint(1, max(1, n-1))
            new_chain, new_occupied, new_energy = _pivot_move(sequence, current_chain, current_occupied, k, rng)
            
            if new_chain is None:
                T *= cooling_rate
                continue

        delta_e = new_energy - current_energy

        if delta_e < 0 or (T > 0 and rng.random() < math.exp(-delta_e / T)):
            current_chain = new_chain
            current_occupied = new_occupied
            current_energy = new_energy
            accept_count += 1

            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter > max_stagnation:
                T = max(T, 1.0)
                stagnation_counter = 0
        else:
            stagnation_counter += 1

        # Adaptive temperature based on acceptance rate
        if total_count % 100 == 0:
            accept_rate = accept_count / max(1, total_count)
            if accept_rate < 0.1:
                T = min(T * 1.5, 5.0)
            elif accept_rate > 0.8:
                T = max(T * 0.7, T_min)
            accept_count = 0
            total_count = 0

        T *= cooling_rate

    return best_chain


def fold(sequence, dimension):
    """
    Best conformation using multiple SA restarts with a global deadline.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]

    remaining = _remaining_time()
    if remaining < 2.0:
        # Not enough time; just return a greedy fold
        rng = random.Random(hash(sequence) % (2**32))
        chain = _grow_greedy(sequence, rng, bias=1.0)
        if chain is None:
            chain = [(i, 0, 0) for i in range(n)]
        return [list(p) for p in chain]

    # Allocate time for this sequence
    # Use a larger fraction of remaining time, capped to leave margin
    seq_time_limit = min(remaining * 0.45, 15.0)

    if seq_time_limit < 1.0:
        rng = random.Random(hash(sequence) % (2**32))
        chain = _grow_greedy(sequence, rng, bias=1.0)
        if chain is None:
            chain = [(i, 0, 0) for i in range(n)]
        return [list(p) for p in chain]

    best_chain = None
    best_energy = float('inf')

    # More restarts with shorter individual runs
    slice_time = max(0.6, seq_time_limit / 8.0)
    num_restarts = max(1, int(seq_time_limit / slice_time))

    rng = random.Random(hash(sequence) % (2**32))

    for restart in range(num_restarts):
        if _remaining_time() < 3.0:
            break
        sa_chain = _sa_fold(sequence, rng, slice_time)
        if sa_chain is not None:
            e = _energy(sequence, sa_chain)
            if e < best_energy:
                best_energy = e
                best_chain = sa_chain

    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]

    return [list(p) for p in best_chain]
