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

def _greedy_grow(sequence, rng=None):
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
        candidates.sort(key=lambda x: -x[0])
        best_site = candidates[0][1]
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

def _pivot_move(chain, i, rng):
    """Rotate a sub-chain around position i. Returns new chain or None."""
    n = len(chain)
    if i < 1 or i >= n - 1:
        return None
    
    # The segment to rotate is chain[i+1:]
    # Pivot at chain[i]
    # We try all 3 possible rotations (90, 180, 270 degrees)
    # But we need to check self-avoidance
    
    prefix = chain[:i+1]
    segment = chain[i+1:]
    
    if not segment:
        return None
    
    # The segment starts at chain[i+1] and goes through the rest
    # We rotate the entire tail around chain[i]
    # Actually, let's rotate a random-length segment starting at some point
    
    # Let's pick a random start for the segment to rotate
    # Actually, the standard pivot move rotates the tail from position i onwards
    # around the bond (i-1, i) or (i, i+1)
    
    # Let's use a different approach: pick two indices a < b
    # and rotate the segment chain[a+1:b+1] around chain[a]
    
    a = i
    # Random length of segment to rotate
    max_len = min(n - a - 1, 15)
    if max_len < 2:
        return None
    seg_len = rng.randint(2, max_len)
    b = a + seg_len
    
    if b >= n:
        return None
    
    # Segment is chain[a+1 : b+1]
    seg = chain[a+1:b+1]
    
    # The pivot is chain[a]
    # The first point of seg is chain[a+1]
    # We rotate seg around chain[a]
    
    # Compute the vector from chain[a] to chain[a+1]
    dx = seg[0][0] - chain[a][0]
    dy = seg[0][1] - chain[a][1]
    
    # Rotation options: 90, 180, 270 degrees
    # 90 deg CW: (dx, dy) -> (dy, -dx)
    # 180 deg: (dx, dy) -> (-dx, -dy)
    # 270 deg CW: (dx, dy) -> (-dy, dx)
    
    rotations = [
        (dy, -dx),
        (-dx, -dy),
        (-dy, dx)
    ]
    
    rng.shuffle(rotations)
    
    for rot_dx, rot_dy in rotations:
        # Check if the rotated first point is valid
        new_first = (chain[a][0] + rot_dx, chain[a][1] + rot_dy)
        
        # Check if new_first is not in prefix or in the part of seg after rotation
        # We'll build the full new chain and check self-avoidance
        
        # Rotate the entire segment
        new_seg = [new_first]
        
        # For the rest of the segment, we need to rotate each point
        # Actually, we rotate the entire segment as a rigid body around chain[a]
        # So each point (x, y) in seg becomes:
        # relative to chain[a]: (x - chain[a][0], y - chain[a][1])
        # rotated: depends on rotation
        
        # Let's compute the rotation matrix
        if rot_dx == dy and rot_dy == -dx:
            # 90 deg CW
            def rotate(pt):
                rx = pt[0] - chain[a][0]
                ry = pt[1] - chain[a][1]
                return (chain[a][0] + ry, chain[a][1] - rx)
        elif rot_dx == -dx and rot_dy == -dy:
            # 180 deg
            def rotate(pt):
                rx = pt[0] - chain[a][0]
                ry = pt[1] - chain[a][1]
                return (chain[a][0] - rx, chain[a][1] - ry)
        else:
            # 270 deg CW
            def rotate(pt):
                rx = pt[0] - chain[a][0]
                ry = pt[1] - chain[a][1]
                return (chain[a][0] - ry, chain[a][1] + rx)
        
        new_seg = [rotate(pt) for pt in seg]
        
        # Check that new_seg[0] is adjacent to chain[a]
        if abs(new_seg[0][0] - chain[a][0]) + abs(new_seg[0][1] - chain[a][1]) != 1:
            continue
        
        # Build new chain
        new_chain = prefix + new_seg + chain[b+1:]
        
        # Check self-avoidance
        if len(set(new_chain)) != n:
            continue
        
        # Check that the connection at the end is valid
        # new_seg[-1] should be adjacent to chain[b+1] if b+1 < n
        if b + 1 < n:
            if abs(new_seg[-1][0] - chain[b+1][0]) + abs(new_seg[-1][1] - chain[b+1][1]) != 1:
                continue
        
        return new_chain
    
    return None

def _reversal_move(chain, rng):
    """Reverse a segment of the chain. Returns new chain or None."""
    n = len(chain)
    if n < 4:
        return None
    
    # Pick two indices a < b
    a = rng.randint(0, n - 3)
    b = rng.randint(a + 2, n - 1)
    
    # Segment is chain[a:b+1]
    seg = chain[a:b+1]
    new_seg = list(reversed(seg))
    
    # Check that the connections are valid
    # chain[a-1] should be adjacent to new_seg[0] = chain[b]
    if a > 0:
        if abs(chain[a-1][0] - new_seg[0][0]) + abs(chain[a-1][1] - new_seg[0][1]) != 1:
            return None
    
    # new_seg[-1] = chain[a] should be adjacent to chain[b+1]
    if b + 1 < n:
        if abs(new_seg[-1][0] - chain[b+1][0]) + abs(new_seg[-1][1] - chain[b+1][1]) != 1:
            return None
    
    new_chain = chain[:a] + new_seg + chain[b+1:]
    
    if len(set(new_chain)) != n:
        return None
    
    return new_chain

def _polish_pass(sequence, chain, deadline, rng):
    n = len(chain)
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

def _local_search(sequence, chain, deadline, rng):
    n = len(chain)
    if n < 3:
        return list(chain), _energy(sequence, chain)

    current_chain = list(chain)
    pos_dict = {p: i for i, p in enumerate(current_chain)}
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    T = 3.0
    cooling = 0.999
    min_T = 0.01
    
    start_time = time.time()
    steps = 0
    no_improve_count = 0
    
    while time.time() < deadline:
        T = max(T * cooling, min_T)
        steps += 1
        
        # Choose move type
        r = rng.random()
        if r < 0.3:
            # Single residue move
            i = rng.randint(1, n - 2)
            
            if i == 0 or i == n - 1:
                neighbor_idx = 1 if i == 0 else n - 2
                neighbor_pos = current_chain[neighbor_idx]
                old_pos = current_chain[i]
                
                candidates = []
                for nb in _neighbours(neighbor_pos):
                    if nb == old_pos:
                        continue
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
            else:
                prev_pos = current_chain[i - 1]
                next_pos = current_chain[i + 1]
                old_pos = current_chain[i]
                
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
        elif r < 0.6:
            # Pivot move
            i = rng.randint(1, n - 2)
            new_chain = _pivot_move(current_chain, i, rng)
            if new_chain is not None:
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
        elif r < 0.8:
            # Reversal move
            new_chain = _reversal_move(current_chain, rng)
            if new_chain is not None:
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
            # Segment move
            max_seg = min(n // 2, 12)
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
        
        if no_improve_count > 300:
            T = max(T * 2.0, 2.0)
            no_improve_count = 0
    
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
        
        if rng.random() < 0.6:
            init_chain, init_energy = _greedy_grow(sequence, rng)
            if init_chain is None:
                init_chain = _random_walk(sequence, rng)
                if init_chain is None:
                    init_chain = [(i, 0) for i in range(n)]
                    init_energy = _energy(sequence, init_chain)
        else:
            init_chain = _random_walk(sequence, rng)
            if init_chain is None:
                init_chain, init_energy = _greedy_grow(sequence, rng)
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
    
    deadline = start_time + 11.0
    
    if n == 0:
        return []
    
    if n <= 30:
        n_restarts = 30
    elif n <= 50:
        n_restarts = 25
    elif n <= 70:
        n_restarts = 20
    else:
        n_restarts = 15
    
    rng = random.Random(12345 + n)
    
    best_chain, best_energy = _restart_local_search(sequence, n_restarts, deadline, rng)
    
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
