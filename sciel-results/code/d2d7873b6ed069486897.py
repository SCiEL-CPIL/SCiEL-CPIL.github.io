"""HP lattice protein folding: Beam Search growth + Optimized Local Search."""
# EVOLVE-BLOCK-START
import random
import time
import math
import heapq

_T0 = time.time()
_DEADLINE = 195.0  # Wall-clock limit with safety margin (210s kill, leave 15s buffer)

def _remaining():
    return _DEADLINE - (time.time() - _T0)

def _compute_energy(sequence, chain):
    """Compute energy: -number of H-H non-bonded contacts."""
    n = len(chain)
    if n < 3:
        return 0
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    contacts = 0
    # Only check H residues to save time
    for i in range(n):
        if sequence[i] != 'H':
            continue
        x, y = chain[i]
        # Check 4 neighbors
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nb = (x + dx, y + dy)
            j = pos_to_idx.get(nb)
            if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                # Count each contact once (i < j)
                if i < j:
                    contacts += 1
    return -contacts

def _grow_beam_search(sequence, beam_width=50, rng=None):
    """
    Generate initial self-avoiding walk using Beam Search.
    State: (energy, chain, occupied_set, pos_to_idx)
    We keep the top 'beam_width' states at each step.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    # Initial state
    # energy is 0, chain is [(0,0)]
    # We use a list of tuples: (energy, chain, occupied, pos_to_idx)
    # Since we can't easily put sets in a heap for comparison, we manage the beam manually
    beam = []
    initial_chain = [(0, 0)]
    initial_occ = {(0, 0)}
    initial_idx = {(0, 0): 0}
    beam.append((0, initial_chain, initial_occ, initial_idx))
    
    for step in range(1, n):
        new_beam = []
        for energy, chain, occ, idx_map in beam:
            last_pos = chain[-1]
            x, y = last_pos
            
            # Try 4 neighbors
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nb = (x + dx, y + dy)
                if nb in occ:
                    continue
                
                # Calculate energy delta for this step
                # Only H-H contacts matter
                delta_e = 0
                if sequence[step] == 'H':
                    # Check neighbors of nb for H residues
                    for ddx, ddy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        n_nb = (nb[0] + ddx, nb[1] + ddy)
                        j = idx_map.get(n_nb)
                        if j is not None and sequence[j] == 'H':
                            # Contact formed
                            delta_e -= 1
                            
                new_energy = energy + delta_e
                new_chain = chain + [nb]
                new_occ = occ | {nb}
                new_idx = idx_map | {nb: step}
                
                new_beam.append((new_energy, new_chain, new_occ, new_idx))
        
        if not new_beam:
            # Dead end, should not happen for reasonable lengths if we don't restrict too tight
            return None
            
        # Sort by energy (lowest is best) and keep top beam_width
        # To handle ties or slight variations, we can just sort by energy
        new_beam.sort(key=lambda x: x[0])
        beam = new_beam[:beam_width]
        
        # Optimization: If we have very many beams, we might want to prune based on diversity
        # but for now, simple top-K is standard.
        
    # Return the best chain from the final beam
    if not beam:
        return None
    best_state = beam[0]
    return best_state[1]

def _pivot_move(chain, pos_to_idx, rng):
    """
    Pivot move: Rotate a suffix of the chain around a pivot point.
    Returns new_chain or None.
    """
    n = len(chain)
    if n < 5:
        return None
    
    # Choose pivot index (1 to n-2 to leave some prefix and suffix)
    pivot_idx = rng.randint(1, n - 2)
    pivot_pos = chain[pivot_idx]
    
    # The part to rotate is chain[pivot_idx+1:]
    suffix = chain[pivot_idx+1:]
    prefix_set = set(chain[:pivot_idx+1])
    
    # Try 90-degree CCW and CW, and 180-degree rotation
    # Relative coordinates
    rel_suffix = [(x - pivot_pos[0], y - pivot_pos[1]) for x, y in suffix]
    
    rotations = [
        lambda rx, ry: (-ry, rx),    # 90 CCW
        lambda rx, ry: (ry, -rx),    # 90 CW
        lambda rx, ry: (-rx, -ry)    # 180
    ]
    
    for rot_func in rotations:
        new_suffix = []
        valid = True
        for rx, ry in rel_suffix:
            nx, ny = rot_func(rx, ry)
            pos = (pivot_pos[0] + nx, pivot_pos[1] + ny)
            if pos in prefix_set:
                valid = False
                break
            new_suffix.append(pos)
        
        if valid:
            # Check self-avoidance within the rotated suffix
            if len(set(new_suffix)) == len(new_suffix):
                new_chain = chain[:pivot_idx+1] + new_suffix
                # Final check for self-avoidance (should be covered by prefix_set and internal check)
                if len(set(new_chain)) == n:
                    return new_chain
                    
    return None

def _reversal_move(chain, pos_to_idx, rng):
    """
    Reversal move: Reverse a segment of the chain.
    """
    n = len(chain)
    if n < 5:
        return None
    
    # Choose segment length and start
    max_len = n // 2
    if max_len < 3:
        return None
    seg_len = rng.randint(3, max_len)
    start = rng.randint(0, n - seg_len - 1)
    end = start + seg_len - 1
    
    # Check connectivity at boundaries
    # Left boundary: chain[start-1] to reversed chain[start] (which is chain[end])
    # Right boundary: reversed chain[end] (which is chain[start]) to chain[end+1]
    
    if start > 0:
        left_pos = chain[start - 1]
        new_left_pos = chain[end]
        if abs(left_pos[0] - new_left_pos[0]) + abs(left_pos[1] - new_left_pos[1]) != 1:
            return None
            
    if end < n - 1:
        right_pos = chain[end + 1]
        new_right_pos = chain[start]
        if abs(right_pos[0] - new_right_pos[0]) + abs(right_pos[1] - new_right_pos[1]) != 1:
            return None
            
    new_chain = chain[:start] + chain[start:end+1][::-1] + chain[end+1:]
    
    # Check self-avoidance
    if len(set(new_chain)) == n:
        return new_chain
        
    return None

def _segment_relocation_move(chain, pos_to_idx, rng):
    """
    Segment relocation: Cut a segment and insert it elsewhere.
    This is powerful but expensive.
    """
    n = len(chain)
    if n < 10:
        return None
        
    # Choose segment length (small for efficiency)
    seg_len = rng.randint(2, min(6, n // 4))
    if seg_len >= n:
        return None
        
    start = rng.randint(1, n - seg_len - 1)
    end = start + seg_len - 1
    
    # Extract segment
    segment = chain[start:end+1]
    
    # Gap is the rest of the chain
    gap = chain[:start] + chain[end+1:]
    gap_len = len(gap)
    
    # Try to insert at a random position in the gap
    # We need to check connectivity and self-avoidance
    # To speed up, try a few random insertion points
    for _ in range(5):
        k = rng.randint(0, gap_len)
        
        # Check connectivity
        # Left: gap[k-1] to segment[0]
        if k > 0:
            lp = gap[k-1]
            sf = segment[0]
            if abs(lp[0] - sf[0]) + abs(lp[1] - sf[1]) != 1:
                continue
        # Right: segment[-1] to gap[k]
        if k < gap_len:
            rp = gap[k]
            sl = segment[-1]
            if abs(rp[0] - sl[0]) + abs(rp[1] - sl[1]) != 1:
                continue
                
        new_chain = gap[:k] + segment + gap[k:]
        
        if len(set(new_chain)) == n:
            return new_chain
            
    return None

def _local_search(sequence, chain, rng, time_limit, max_iters):
    """
    Refine the chain using local search moves.
    """
    n = len(chain)
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    T = 2.0
    cooling = 0.999
    min_T = 0.1
    
    start_time = time.time()
    it = 0
    
    while it < max_iters:
        if _remaining() < 5.0 or (time.time() - start_time) > time_limit:
            break
            
        # Choose move
        r = rng.random()
        new_chain = None
        
        if r < 0.4:
            new_chain = _pivot_move(chain, None, rng)
        elif r < 0.7:
            new_chain = _reversal_move(chain, None, rng)
        else:
            new_chain = _segment_relocation_move(chain, None, rng)
            
        if new_chain is None:
            it += 1
            continue
            
        # Verify validity (unit steps and self-avoidance)
        # Self-avoidance is checked in move generation mostly, but let's be safe
        if len(set(new_chain)) != n:
            it += 1
            continue
            
        valid = True
        for i in range(n - 1):
            if abs(new_chain[i][0] - new_chain[i+1][0]) + abs(new_chain[i][1] - new_chain[i+1][1]) != 1:
                valid = False
                break
        if not valid:
            it += 1
            continue
            
        new_energy = _compute_energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta <= 0 or rng.random() < math.exp(-delta / T):
            chain = new_chain
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(chain)
                
        T = max(T * cooling, min_T)
        it += 1
        
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
        
    rng = random.Random(42 + len(sequence) * 7)
    
    best_chain = None
    best_energy = float('inf')
    
    start_time = time.time()
    time_budget = min(_remaining() - 5.0, 180.0)
    
    # Adaptive parameters based on length
    if n <= 25:
        beam_width = 100
        num_restarts = 50
        iters_per = 5000
    elif n <= 40:
        beam_width = 80
        num_restarts = 40
        iters_per = 4000
    elif n <= 60:
        beam_width = 60
        num_restarts = 30
        iters_per = 3000
    elif n <= 80:
        beam_width = 40
        num_restarts = 20
        iters_per = 2000
    else:
        beam_width = 30
        num_restarts = 15
        iters_per = 1500
        
    for restart in range(num_restarts):
        if _remaining() < 10.0:
            break
        if (time.time() - start_time) > time_budget:
            break
            
        # Generate initial conformation using Beam Search
        # Vary beam width slightly for diversity? Or just use fixed.
        # Let's try slightly varying the seed for the beam search if we had one, 
        # but here we just use the main rng.
        chain = _grow_beam_search(sequence, beam_width, rng)
        
        if chain is None or len(chain) != n:
            # Fallback to simple greedy if beam search fails (rare)
            chain = [(0,0)]
            occ = {(0,0)}
            for i in range(1, n):
                x, y = chain[-1]
                moved = False
                for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                    nb = (x+dx, y+dy)
                    if nb not in occ:
                        chain.append(nb)
                        occ.add(nb)
                        moved = True
                        break
                if not moved:
                    chain = None
                    break
            if chain is None or len(chain) != n:
                continue
                
        # Local Search
        time_left = min(time_budget - (time.time() - start_time), _remaining() - 5.0)
        if time_left < 1.0:
            break
            
        chain, energy = _local_search(sequence, chain, rng, time_left, max_iters=iters_per)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = list(chain)
            
    if best_chain is None:
        # Fallback: straight line (invalid for n>1 in terms of self-avoidance if it loops, but straight line is always valid SAW)
        best_chain = [(i, 0) for i in range(n)]
        
    return [[x, y] for x, y in best_chain]
# EVOLVE-BLOCK-END
