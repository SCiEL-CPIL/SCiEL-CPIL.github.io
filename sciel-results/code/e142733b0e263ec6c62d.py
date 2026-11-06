"""Improved HP lattice protein folding with ILS and Reversal moves."""
# EVOLVE-BLOCK-START
import random
import time

_START_TIME = None

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_remaining():
    _init_time()
    # Budget is 120s CPU. Safety margin to 110s.
    return 110.0 - (time.time() - _START_TIME)

def _neighbours(point):
    """Yield 4-neighbours on square lattice."""
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

def _count_contacts(sequence, chain):
    """Calculate total H-H contacts for a chain."""
    n = len(chain)
    if n == 0:
        return 0
    
    occupied = {p: i for i, p in enumerate(chain)}
    contacts = 0
    is_h = [c == 'H' for c in sequence]
    
    for i in range(n):
        if not is_h[i]:
            continue
        x, y = chain[i]
        # Check 4 neighbors
        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        for nx, ny in neighbors:
            j = occupied.get((nx, ny))
            if j is not None and j > i + 1 and is_h[j]:
                contacts += 1
    return contacts

def _backtrack_grow(sequence, dimension, rng, max_depth=50):
    """
    Generate a valid self-avoiding walk using randomized backtracking.
    This ensures we start with a valid conformation, unlike greedy which might fail or be suboptimal.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    is_h = [c == 'H' for c in sequence]
    
    # Recursive backtracking with random ordering
    def dfs(idx, chain, occupied):
        if idx == n:
            return True
        
        current = chain[-1]
        neighbors = _neighbours(current)
        rng.shuffle(neighbors)
        
        for site in neighbors:
            if site in occupied:
                continue
            
            # Heuristic: Prefer sites that create H-H contacts if possible
            # But keep it mostly random for diversity in ILS
            # We can bias slightly towards 'H' placements if they create contacts
            # For now, pure random with backtracking is robust.
            
            occupied[site] = idx
            chain.append(site)
            
            if dfs(idx + 1, chain, occupied):
                return True
            
            # Backtrack
            del occupied[site]
            chain.pop()
        
        return False
    
    # To avoid deep recursion issues or getting stuck, we limit the number of backtracks
    # or just rely on the fact that N is small (<=100).
    # However, naive backtracking can be exponential.
    # We use a simple randomized greedy with backtracking only when stuck.
    
    # Alternative: Randomized Greedy with Backtracking
    # Let's implement a simpler version: Random Walk with Backtracking
    # If we get stuck, we backtrack to the last decision point.
    
    # For efficiency, let's use a simple randomized greedy first, and if it fails, backtrack.
    # Actually, for HP folding, a pure random walk often fails to complete for N>30.
    # A "Look-ahead" greedy is better.
    
    # Let's stick to a robust Randomized Greedy that backtracks if it hits a dead end.
    
    def grow_with_backtrack():
        chain = [(0, 0)]
        occupied = {(0, 0): 0}
        
        for i in range(1, n):
            current = chain[-1]
            neighbors = _neighbours(current)
            valid_neighbors = [nb for nb in neighbors if nb not in occupied]
            
            if not valid_neighbors:
                return None # Dead end
            
            # Score neighbors: prefer those with more free space (flexibility)
            # and those that create H-H contacts.
            scored = []
            for nb in valid_neighbors:
                free_space = sum(1 for q in _neighbours(nb) if q not in occupied)
                contact_gain = 0
                if is_h[i]:
                    for q in _neighbours(nb):
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and is_h[j]:
                            contact_gain += 1
                score = contact_gain * 10.0 + free_space * 0.1 + rng.random()
                scored.append((score, nb))
            
            scored.sort(key=lambda x: x[0], reverse=True)
            
            # With some probability, pick a random valid neighbor to avoid local bias
            if rng.random() < 0.2:
                site = rng.choice(valid_neighbors)
            else:
                site = scored[0][1]
            
            occupied[site] = i
            chain.append(site)
            
        return chain
    
    # Try simple greedy first
    chain = grow_with_backtrack()
    if chain:
        return chain
    
    # If greedy fails (unlikely for N<=100 with free space bias), fallback to pure random walk with backtracking
    # This is expensive, so we only do it if time permits.
    if _time_remaining() < 5.0:
        return None
        
    # Simple Backtracking Search
    def bt_search(idx, chain, occupied):
        if idx == n:
            return True
        current = chain[-1]
        neighbors = _neighbours(current)
        rng.shuffle(neighbors)
        for nb in neighbors:
            if nb in occupied:
                continue
            occupied[nb] = idx
            chain.append(nb)
            if bt_search(idx + 1, chain, occupied):
                return True
            chain.pop()
            del occupied[nb]
        return False
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    if bt_search(1, chain, occupied):
        return chain
        
    return None

def _simulated_annealing(sequence, chain, rng, max_steps=500):
    """
    Refine the chain using Simulated Annealing with Kink and Reversal moves.
    """
    n = len(chain)
    if n < 3:
        return chain, -_count_contacts(sequence, chain)
    
    occupied = {p: i for i, p in enumerate(chain)}
    current_contacts = _count_contacts(sequence, chain)
    current_energy = -current_contacts
    is_h = [c == 'H' for c in sequence]
    
    start_temp = 5.0
    end_temp = 0.1
    
    for step in range(max_steps):
        progress = step / max(1, max_steps)
        temp = start_temp * (end_temp / start_temp) ** progress
        
        move_type = rng.random()
        
        if move_type < 0.5:
            # Kink move
            if n < 3:
                continue
            i = rng.randint(1, n - 2)
            prev = chain[i-1]
            curr = chain[i]
            next_ = chain[i+1]
            
            candidates = set()
            for p in _neighbours(prev):
                if p == curr:
                    continue
                dist = abs(p[0] - next_[0]) + abs(p[1] - next_[1])
                if dist == 1:
                    candidates.add(p)
            
            if not candidates:
                continue
                
            new_pos = rng.choice(list(candidates))
            
            if new_pos in occupied:
                continue
            
            del occupied[curr]
            occupied[new_pos] = i
            chain[i] = new_pos
            
            new_contacts = _count_contacts(sequence, chain)
            new_energy = -new_contacts
            delta = new_energy - current_energy
            
            if delta < 0 or (temp > 0.01 and rng.random() < pow(2.718281828, -delta / temp)):
                current_energy = new_energy
                current_contacts = new_contacts
            else:
                chain[i] = curr
                del occupied[new_pos]
                occupied[curr] = i
                
        elif move_type < 0.8:
            # Reversal move: Reverse a segment of the chain
            # This is a powerful move for escaping local minima.
            # Pick two indices i, j (1 <= i < j <= n-1)
            # Reverse the segment chain[i..j]
            # The new chain must be self-avoiding.
            # The segment is removed and reinserted in reverse.
            # Connectivity:
            # chain[i-1] is connected to chain[j] (new first of reversed seg)
            # chain[j+1] is connected to chain[i] (new last of reversed seg)
            
            if n < 4:
                continue
                
            i = rng.randint(1, n - 2)
            j = rng.randint(i + 1, n - 1)
            
            # Check if the reversal is valid
            # The new segment is chain[j], chain[j-1], ..., chain[i]
            # We need to check if any of these positions are occupied by non-segment residues.
            # The segment positions are currently occupied by the segment itself.
            # So we just need to check if the reversed sequence hits any *other* occupied spots.
            # Actually, the set of positions is the same, just the order is reversed.
            # So self-avoidance is automatically preserved if the original was valid!
            # Wait, no. The *sequence* of positions is reversed.
            # The set of occupied cells is identical.
            # So the only thing that changes is the *order* of residues in those cells.
            # This changes which residues are adjacent in the chain, and thus which contacts are formed.
            # It does NOT change the self-avoidance property of the set of points.
            # Therefore, a reversal move is ALWAYS valid in terms of self-avoidance,
            # provided the endpoints connect correctly.
            
            # Let's verify connectivity:
            # Old: ... chain[i-1] - chain[i] - ... - chain[j] - chain[j+1] ...
            # New: ... chain[i-1] - chain[j] - chain[j-1] - ... - chain[i] - chain[j+1] ...
            
            # We need chain[i-1] to be adjacent to chain[j].
            # And chain[j+1] to be adjacent to chain[i].
            
            p_prev = chain[i-1]
            p_new_start = chain[j]
            if abs(p_prev[0] - p_new_start[0]) + abs(p_prev[1] - p_new_start[1]) != 1:
                continue
                
            if j < n - 1:
                p_next = chain[j+1]
                p_new_end = chain[i]
                if abs(p_next[0] - p_new_end[0]) + abs(p_next[1] - p_new_end[1]) != 1:
                    continue
            
            # If valid, apply reversal
            # Update chain list
            chain[i:j+1] = chain[i:j+1][::-1]
            
            # Update occupied map
            # The positions are the same, but the indices mapping to them changes.
            # We need to rebuild the occupied map for the segment.
            for k in range(i, j+1):
                del occupied[chain[k]]
            for k in range(i, j+1):
                occupied[chain[k]] = k
                
            new_contacts = _count_contacts(sequence, chain)
            new_energy = -new_contacts
            delta = new_energy - current_energy
            
            if delta < 0 or (temp > 0.01 and rng.random() < pow(2.718281828, -delta / temp)):
                current_energy = new_energy
                current_contacts = new_contacts
            else:
                # Revert
                chain[i:j+1] = chain[i:j+1][::-1]
                for k in range(i, j+1):
                    del occupied[chain[k]]
                for k in range(i, j+1):
                    occupied[chain[k]] = k
        else:
            # Segment move (length 2 or 3)
            seg_len = rng.choice([2, 3])
            if n < seg_len + 2:
                continue
            
            start_idx = rng.randint(1, n - seg_len - 1)
            end_idx = start_idx + seg_len - 1
            
            seg_indices = list(range(start_idx, end_idx + 1))
            old_seg = [chain[i] for i in seg_indices]
            
            anchor_start = chain[start_idx - 1]
            anchor_end = chain[end_idx + 1]
            
            found_valid = False
            for _ in range(20): # Reduced attempts for speed
                pos = anchor_start
                path = [pos]
                for _ in range(seg_len):
                    neighbors = _neighbours(pos)
                    pos = rng.choice(neighbors)
                    path.append(pos)
                
                if path[-1] != anchor_end:
                    continue
                
                new_seg = path[1:]
                
                collision = False
                for p in new_seg:
                    if p in occupied:
                        is_old_seg_pos = any(p == old_p for old_p in old_seg)
                        if not is_old_seg_pos:
                            collision = True
                            break
                
                if collision:
                    continue
                
                if len(set(new_seg)) != len(new_seg):
                    continue
                
                # Apply
                for i in seg_indices:
                    del occupied[chain[i]]
                
                for k, idx in enumerate(seg_indices):
                    new_pos = new_seg[k]
                    occupied[new_pos] = idx
                    chain[idx] = new_pos
                
                new_contacts = _count_contacts(sequence, chain)
                new_energy = -new_contacts
                delta = new_energy - current_energy
                
                if delta < 0 or (temp > 0.01 and rng.random() < pow(2.718281828, -delta / temp)):
                    current_energy = new_energy
                    current_contacts = new_contacts
                    found_valid = True
                else:
                    # Revert
                    for k, idx in enumerate(seg_indices):
                        del occupied[chain[idx]]
                        old_pos = old_seg[k]
                        occupied[old_pos] = idx
                        chain[idx] = old_pos
                
                break
            
    return chain, current_energy

def fold(sequence, dimension):
    """
    Fold an HP sequence on the square lattice.
    Uses Iterated Local Search with Backtracking Growth and SA with Reversal moves.
    """
    _init_time()
    
    n = len(sequence)
    if n == 0:
        return []
    
    if _time_remaining() < 1.0:
        return [[i, 0] for i in range(n)]
    
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0
    
    max_iterations = 1000
    iterations = 0
    
    while _time_remaining() > 5.0 and iterations < max_iterations:
        iterations += 1
        
        # 1. Generate initial conformation
        chain = _backtrack_grow(sequence, dimension, rng)
        if chain is None:
            # Fallback: simple straight line if growth fails (rare)
            chain = [(i, 0) for i in range(n)]
        
        # 2. Simulated Annealing
        # Adjust steps based on time
        sa_steps = 200
        if _time_remaining() < 20.0:
            sa_steps = 50
            
        chain, energy = _simulated_annealing(sequence, chain, rng, max_steps=sa_steps)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain[:]
            
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
