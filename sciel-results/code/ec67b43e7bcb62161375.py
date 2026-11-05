import random
import time
import sys
import math
from collections import deque

# EVOLVE-BLOCK-START

def _neighbours(point):
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

def _energy(sequence, chain):
    n = len(chain)
    if n == 0:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _energy_delta_contact(sequence, pos_dict, i, old_pos, new_pos):
    if sequence[i] != 'H':
        return 0
    delta = 0
    for q in _neighbours(old_pos):
        j = pos_dict.get(q)
        if j is not None and j != i and sequence[j] == 'H' and abs(j - i) > 1:
            delta += 1
    for q in _neighbours(new_pos):
        j = pos_dict.get(q)
        if j is not None and j != i and sequence[j] == 'H' and abs(j - i) > 1:
            delta -= 1
    return delta

def _greedy_grow(sequence, rng=None, temperature=0.5):
    n = len(sequence)
    if n == 0:
        return [], 0
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    for i in range(1, n):
        last = chain[-1]
        candidates = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == 'H':
                            gain += 1
            open_count = 0
            for q in _neighbours(site):
                if q not in occupied:
                    open_count += 1
            score = gain * 10 + open_count
            candidates.append((score, site))
        
        if not candidates:
            return None, 0
        
        # Softmax selection with temperature
        if len(candidates) == 1:
            best_site = candidates[0][1]
        else:
            scores = [c[0] for c in candidates]
            max_score = max(scores)
            # Exponentiate with temperature
            exp_scores = [math.exp((s - max_score) / temperature) for s in scores]
            total = sum(exp_scores)
            r = rng.random() * total
            cumsum = 0
            for idx, es in enumerate(exp_scores):
                cumsum += es
                if r <= cumsum:
                    best_site = candidates[idx][1]
                    break
            else:
                best_site = candidates[-1][1]
        
        chain.append(best_site)
        occupied[best_site] = i
    
    return chain, _energy(sequence, chain)

def _random_walk(sequence, rng):
    n = len(sequence)
    if n == 0:
        return []
    chain = [(0, 0)]
    occupied = {(0, 0)}
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for i in range(1, n):
        last = chain[-1]
        candidates = []
        for dx, dy in dirs:
            new_pos = (last[0] + dx, last[1] + dy)
            if new_pos not in occupied:
                candidates.append(new_pos)
        if not candidates:
            return None
        new_pos = rng.choice(candidates)
        chain.append(new_pos)
        occupied.add(new_pos)
    return chain

def _pivot_move(sequence, chain, pos_dict, pivot_idx, rng):
    """
    Attempt a pivot move: rotate the tail of the chain (from pivot_idx+1 to end)
    around the pivot point. Returns (new_chain, new_energy) or (None, None).
    """
    n = len(chain)
    if pivot_idx < 1 or pivot_idx >= n - 1:
        return None, None
    
    head = chain[:pivot_idx + 1]
    tail = chain[pivot_idx + 1:]
    pivot = chain[pivot_idx]
    
    # Build occupied set for head
    occupied_head = set(head)
    
    # Try rotating tail around pivot
    # The tail starts at tail[0] which is adjacent to pivot
    # We try all 3 possible rotations (excluding the original direction)
    
    # Get the vector from pivot to tail[0]
    dx0 = tail[0][0] - pivot[0]
    dy0 = tail[0][1] - pivot[1]
    
    # Possible new directions (rotate 90, 180, 270 degrees)
    rotations = [
        (-dy0, dx0),   # 90 deg CCW
        (-dx0, -dy0),  # 180 deg
        (dy0, -dx0),   # 270 deg CCW
    ]
    
    for ndx, ndy in rotations:
        new_tail = [pivot]
        new_occupied = set(occupied_head)
        
        # Try to place the tail starting from the new direction
        # We need to retrace the tail shape but rotated
        # Actually, let's try a simpler approach: just try random self-avoiding
        # walks from the new starting position
        
        new_start = (pivot[0] + ndx, pivot[1] + ndy)
        if new_start in occupied_head:
            continue
        
        new_tail = [new_start]
        new_occupied = set(occupied_head) | {new_start}
        
        success = True
        for k in range(1, len(tail)):
            last_pos = new_tail[-1]
            candidates = []
            for nb in _neighbours(last_pos):
                if nb not in new_occupied:
                    candidates.append(nb)
            if not candidates:
                success = False
                break
            # Choose randomly or greedily
            if rng.random() < 0.7:
                # Prefer positions that create H-H contacts
                best_score = -999
                best_pos = None
                for c in candidates:
                    score = 0
                    if sequence[pivot_idx + k] == 'H':
                        for q in _neighbours(c):
                            if q in new_occupied:
                                j = None
                                # Need to check if the position at q belongs to head or tail
                                # This is complex, so just use random for simplicity
                                score += 1
                    if score > best_score:
                        best_score = score
                        best_pos = c
                if best_pos is not None:
                    new_tail.append(best_pos)
                    new_occupied.add(best_pos)
                else:
                    new_tail.append(rng.choice(candidates))
                    new_occupied.add(new_tail[-1])
            else:
                new_tail.append(rng.choice(candidates))
                new_occupied.add(new_tail[-1])
        
        if not success:
            continue
        
        # Check connection: new_tail[-1] should be the end, no connection needed to anything else
        # Actually, the tail is the end of the chain, so no connection constraint
        
        new_chain = head + new_tail[1:]  # Skip pivot since it's already in head
        
        # Verify self-avoidance
        if len(set(new_chain)) != n:
            continue
        
        new_energy = _energy(sequence, new_chain)
        return new_chain, new_energy
    
    return None, None

def _local_search(sequence, chain, deadline, rng):
    n = len(sequence)
    if n < 3:
        return list(chain), _energy(sequence, chain)

    current_chain = list(chain)
    pos_dict = {p: i for i, p in enumerate(current_chain)}
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    T = 2.0
    cooling = 0.9995
    min_T = 0.05
    
    steps = 0
    no_improve_count = 0
    
    while time.time() < deadline:
        T = max(T * cooling, min_T)
        steps += 1
        
        # Choose move type
        move_type = rng.random()
        
        if move_type < 0.5:
            # Single residue move
            i = rng.randint(1, n - 2)
            old_pos = current_chain[i]
            prev_pos = current_chain[i - 1]
            next_pos = current_chain[i + 1]
            
            candidates = []
            for nb in _neighbours(prev_pos):
                if nb == old_pos:
                    continue
                if nb in _neighbours(next_pos):
                    if nb in pos_dict and pos_dict[nb] != i:
                        continue
                    candidates.append(nb)
            
            if not candidates:
                continue
            
            new_pos = rng.choice(candidates)
            delta = _energy_delta_contact(sequence, pos_dict, i, old_pos, new_pos)
            
            if delta < 0 or rng.random() < math.exp(-delta / T):
                del pos_dict[old_pos]
                current_chain[i] = new_pos
                pos_dict[new_pos] = i
                current_energy += delta
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
        elif move_type < 0.8:
            # Segment move
            max_seg = min(n // 3, 10)
            if max_seg < 2:
                max_seg = 2
            seg_len = rng.randint(2, max_seg)
            i = rng.randint(0, n - seg_len)
            j = i + seg_len - 1
            
            head = current_chain[:i]
            tail = current_chain[j+1:]
            
            occupied_ht = set(head) | set(tail)
            
            anchor = head[-1] if head else (0, 0)
            start_candidates = []
            for nb in _neighbours(anchor):
                if nb not in occupied_ht:
                    start_candidates.append(nb)
            
            if not start_candidates:
                continue
            
            start_pos = rng.choice(start_candidates)
            new_segment = [start_pos]
            new_occupied = set(occupied_ht) | {start_pos}
            
            success = True
            for k in range(1, seg_len):
                last_pos = new_segment[-1]
                next_candidates = []
                for nb in _neighbours(last_pos):
                    if nb not in new_occupied:
                        next_candidates.append(nb)
                
                if not next_candidates:
                    success = False
                    break
                
                next_pos = rng.choice(next_candidates)
                new_segment.append(next_pos)
                new_occupied.add(next_pos)
            
            if not success:
                continue
            
            if tail:
                tail_start = tail[0]
                d = abs(new_segment[-1][0] - tail_start[0]) + abs(new_segment[-1][1] - tail_start[1])
                if d != 1:
                    continue
            
            new_chain = head + new_segment + tail
            
            if len(set(new_chain)) != n:
                continue
            
            new_energy = _energy(sequence, new_chain)
            delta = new_energy - current_energy
            
            if delta < 0 or rng.random() < math.exp(-delta / T):
                current_chain = new_chain
                current_energy = new_energy
                pos_dict = {p: idx for idx, p in enumerate(current_chain)}
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
        else:
            # Pivot move
            pivot_idx = rng.randint(1, n - 2)
            new_chain, new_energy = _pivot_move(sequence, current_chain, pos_dict, pivot_idx, rng)
            
            if new_chain is not None:
                delta = new_energy - current_energy
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    current_chain = new_chain
                    current_energy = new_energy
                    pos_dict = {p: idx for idx, p in enumerate(current_chain)}
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
        
        if no_improve_count > 200:
            T = max(T * 2.0, 1.5)
            no_improve_count = 0
    
    return best_chain, best_energy

def _polish_pass(sequence, chain, deadline, rng):
    n = len(sequence)
    if n < 3:
        return list(chain), _energy(sequence, chain)
    
    current_chain = list(chain)
    pos_dict = {p: i for i, p in enumerate(current_chain)}
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(1, n - 1):
            if time.time() >= deadline:
                break
            old_pos = current_chain[i]
            prev_pos = current_chain[i - 1]
            next_pos = current_chain[i + 1]
            
            candidates = []
            for nb in _neighbours(prev_pos):
                if nb == old_pos:
                    continue
                if nb in _neighbours(next_pos):
                    if nb in pos_dict and pos_dict[nb] != i:
                        continue
                    candidates.append(nb)
            
            for new_pos in candidates:
                delta = _energy_delta_contact(sequence, pos_dict, i, old_pos, new_pos)
                if delta < 0:
                    del pos_dict[old_pos]
                    current_chain[i] = new_pos
                    pos_dict[new_pos] = i
                    current_energy += delta
                    improved = True
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
                    break
    
    return best_chain, best_energy

def _restart_local_search(sequence, n_restarts, deadline, rng):
    n = len(sequence)
    best_chain = None
    best_energy = 0
    
    start_time = time.time()
    per_restart = (deadline - start_time) / max(1, n_restarts)
    
    for r in range(n_restarts):
        restart_deadline = start_time + (r + 1) * per_restart
        if time.time() >= deadline:
            break
        
        # Generate initial conformation with varied temperatures
        temp = 0.3 + 0.4 * (r / max(1, n_restarts - 1))
        if rng.random() < 0.7:
            init_chain, init_energy = _greedy_grow(sequence, rng, temperature=temp)
            if init_chain is None:
                init_chain = _random_walk(sequence, rng)
                if init_chain is None:
                    init_chain = [(i, 0) for i in range(n)]
                    init_energy = _energy(sequence, init_chain)
        else:
            init_chain = _random_walk(sequence, rng)
            if init_chain is None:
                init_chain, init_energy = _greedy_grow(sequence, rng, temperature=temp)
                if init_chain is None:
                    init_chain = [(i, 0) for i in range(n)]
                    init_energy = _energy(sequence, init_chain)
            else:
                init_energy = _energy(sequence, init_chain)
        
        ls_deadline = restart_deadline - 0.05
        ls_chain, ls_energy = _local_search(sequence, init_chain, ls_deadline, rng)
        
        if time.time() < deadline - 0.1:
            polish_deadline = min(restart_deadline - 0.02, deadline - 0.1)
            if polish_deadline > time.time() + 0.05:
                final_chain, final_energy = _polish_pass(sequence, ls_chain, polish_deadline, rng)
                if final_energy < ls_energy:
                    ls_chain, ls_energy = final_chain, final_energy
        
        if ls_energy < best_energy:
            best_energy = ls_energy
            best_chain = ls_chain
    
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    return best_chain, best_energy

def fold(sequence, dimension):
    start_time = time.time()
    n = len(sequence)
    
    # Safety margin: use 115s out of 120s budget
    deadline = start_time + 115.0
    
    if n == 0:
        return []
    
    if n <= 30:
        n_restarts = 40
    elif n <= 50:
        n_restarts = 30
    elif n <= 70:
        n_restarts = 22
    else:
        n_restarts = 16
    
    rng = random.Random(12345 + n)
    
    best_chain, best_energy = _restart_local_search(sequence, n_restarts, deadline, rng)
    
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
