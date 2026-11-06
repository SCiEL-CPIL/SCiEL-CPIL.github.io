"""HP lattice protein folding using fragment-based Monte Carlo search."""
# EVOLVE-BLOCK-START
import random
import time
import math

_START_TIME = None

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_remaining():
    _init_time()
    # Leave 15s safety margin before 210s wall-clock limit
    return 195.0 - (time.time() - _START_TIME)

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _is_adjacent(p, q):
    return sum(abs(a - b) for a, b in zip(p, q)) == 1

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

def _compute_contact_map(sequence, chain):
    """
    Returns a set of tuples (i, j) representing H-H contacts (i < j).
    """
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    contacts = set()
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts.add((i, j))
    return contacts

def _energy_delta(sequence, chain, start, end, new_pos_start, old_contacts, new_contacts):
    """
    Calculate energy delta for moving a segment.
    This is a simplified check: we recompute energy for the whole chain if the segment move is accepted,
    but for the move proposal we can estimate or just compute fully if it's fast enough.
    For robustness and simplicity in the move logic, we will compute the full energy of the candidate 
    if the move is geometrically valid. Given N<=100, O(N^2) is ~10k ops, which is fast.
    """
    pass

def _generate_random_saw(sequence, dimension, rng):
    """Generate a random self-avoiding walk."""
    n = len(sequence)
    if n == 0:
        return []
    
    # Try to grow from center or start
    # Simple random growth
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        options = []
        for site in _neighbours(chain[-1]):
            if site not in occupied:
                options.append(site)
        
        if not options:
            return None # Dead end
        
        site = rng.choice(options)
        occupied[site] = i
        chain.append(site)
        
    return chain

def _segment_move(chain, start, end, rng, max_attempts=50):
    """
    Attempt to move the segment chain[start:end+1] to a new location.
    Returns (new_chain, success)
    """
    n = len(chain)
    if start == 0 and end == n - 1:
        return chain, False
        
    # The segment to move
    seg_len = end - start + 1
    segment = chain[start:end+1]
    
    # The fixed parts
    left_part = chain[:start]
    right_part = chain[end+1:]
    
    # We need to find a new position for the segment such that:
    # 1. It doesn't overlap with left_part or right_part
    # 2. It connects to left_part at the start (if left_part exists)
    # 3. It connects to right_part at the end (if right_part exists)
    
    # Strategy: Pick a random "anchor" point in the free space near the original location?
    # Or just pick a random valid position for the whole segment?
    # A better strategy: Random walk the segment from a new starting point.
    
    # If the segment is at the beginning (start==0), it must connect to right_part at the end.
    # If the segment is at the end (end==n-1), it must connect to left_part at the start.
    # If internal, it must connect to both.
    
    # Let's try to re-attach the segment to the left part (or right part if start==0).
    # Actually, a common move is to pick a random site in the grid, and try to grow the segment 
    # from there, ensuring it connects to the rest of the chain.
    
    # Simpler approach: 
    # 1. Remove segment.
    # 2. Pick a random neighbor of the "left" end (or "right" end if no left) as the new start of the segment.
    # 3. Try to walk the segment from there.
    
    # Determine connection point
    if start > 0:
        # Connect to left_part[-1]
        anchor = left_part[-1]
        # The new segment must start at a neighbor of anchor
        # And end at a neighbor of right_part[0] (if right_part exists)
        
        # Let's try to grow the segment from a random neighbor of anchor
        # and see if it can reach a neighbor of right_part[0]
        
        target_anchor = right_part[0] if right_part else None
        
        for _ in range(max_attempts):
            # Pick a random neighbor of anchor to start the segment
            candidates = [q for q in _neighbours(anchor) if q not in set(left_part) and q not in set(right_part)]
            if not candidates:
                break
            new_start = rng.choice(candidates)
            
            # Try to grow the segment from new_start
            # We need to place seg_len residues starting from new_start
            # such that the last residue is adjacent to target_anchor (if exists)
            
            # This is a constrained random walk.
            # Let's use a simple backtracking or just random walk with restarts.
            
            new_segment = [new_start]
            occupied = set(left_part) | set(right_part)
            occupied.add(new_start)
            
            success = True
            for k in range(1, seg_len):
                is_last = (k == seg_len - 1)
                options = []
                for q in _neighbours(new_segment[-1]):
                    if q in occupied:
                        continue
                    # If it's the last step, we must be adjacent to target_anchor
                    if is_last and target_anchor is not None:
                        if not _is_adjacent(q, target_anchor):
                            continue
                    options.append(q)
                
                if not options:
                    success = False
                    break
                
                # Heuristic: If not last step, prefer moves that keep options open? 
                # For now, random choice.
                next_pos = rng.choice(options)
                new_segment.append(next_pos)
                occupied.add(next_pos)
            
            if success:
                # Check if the last position is actually adjacent to target_anchor if it exists
                if target_anchor is not None:
                    if not _is_adjacent(new_segment[-1], target_anchor):
                        success = False
                if success:
                    # Construct new chain
                    new_chain = left_part + new_segment + right_part
                    return new_chain, True
                    
    elif start == 0:
        # Segment is at the beginning. It must connect to right_part at the end.
        # So we grow the segment backwards? Or just pick a new start and ensure end connects.
        # Actually, the segment order is fixed. 
        # So we need to find a path of length seg_len such that:
        # - Start is anywhere (free space)
        # - End is adjacent to right_part[0]
        
        target_anchor = right_part[0]
        
        for _ in range(max_attempts):
            # Pick a random end point adjacent to target_anchor
            candidates = [q for q in _neighbours(target_anchor) if q not in set(right_part)]
            if not candidates:
                break
            new_end = rng.choice(candidates)
            
            # Now we need to grow the segment BACKWARDS from new_end to a free start
            # Or just grow forwards from a random start?
            # Growing backwards is symmetric.
            
            new_segment_rev = [new_end]
            occupied = set(right_part)
            occupied.add(new_end)
            
            success = True
            for k in range(1, seg_len):
                is_last = (k == seg_len - 1) # This is the start of the segment in forward order
                options = []
                for q in _neighbours(new_segment_rev[-1]):
                    if q in occupied:
                        continue
                    options.append(q)
                
                if not options:
                    success = False
                    break
                
                next_pos = rng.choice(options)
                new_segment_rev.append(next_pos)
                occupied.add(next_pos)
            
            if success:
                new_segment = new_segment_rev[::-1]
                # The start of new_segment is now free.
                new_chain = new_segment + right_part
                return new_chain, True
                
    return chain, False

def _reversal_move(chain, start, end):
    """Reverse the segment chain[start:end+1]."""
    if start >= end:
        return chain
    
    new_chain = chain[:]
    new_chain[start:end+1] = chain[start:end+1][::-1]
    return new_chain

def _local_search_fragment(sequence, chain, rng, time_limit=5.0):
    """
    Local search using fragment moves and reversals.
    """
    n = len(chain)
    if n < 3:
        return chain
        
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    # Temperature schedule
    T_start = 2.0
    T_end = 0.1
    # Estimate iterations based on time
    # We'll just loop until time limit
    
    T = T_start
    
    # Fragment lengths to try
    # Start with large fragments, then smaller
    # For N=100, try lengths 20, 10, 5, 2
    frag_lengths = []
    max_frag = n // 3
    if max_frag >= 2:
        frag_lengths.append(max_frag)
    if n // 6 >= 2:
        frag_lengths.append(n // 6)
    if n // 10 >= 2:
        frag_lengths.append(n // 10)
    frag_lengths.append(2)
    frag_lengths.append(3)
    
    # Shuffle fragment lengths for variety? No, progressive is better for "polish"
    # But we should mix them.
    
    while _time_remaining() > 10: # Global safety
        # Pick a fragment length
        frag_len = rng.choice(frag_lengths)
        if frag_len >= n:
            frag_len = n // 2
            
        # Pick start position
        max_start = n - frag_len - 1
        if max_start < 0:
            continue
            
        start = rng.randint(0, max_start)
        end = start + frag_len
        
        # Decide move type: 80% reinsertion, 20% reversal
        if rng.random() < 0.8:
            new_chain, success = _segment_move(chain, start, end, rng, max_attempts=10)
            if not success:
                continue
        else:
            new_chain = _reversal_move(chain, start, end)
            
        new_energy = _energy(sequence, new_chain)
        
        # Acceptance criterion
        if new_energy < current_energy or rng.random() < math.exp((current_energy - new_energy) / T):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
        
        # Cool down
        T *= 0.999
        if T < T_end:
            T = T_end
            
    return best_chain

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

def fold(sequence, dimension):
    """
    Best conformation using fragment-based Monte Carlo search.
    """
    _init_time()
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 0
    
    n = len(sequence)
    
    # Determine number of restarts based on sequence length and time
    # Fragment moves are more expensive than kink moves, so fewer restarts
    if n <= 20:
        num_restarts = 50
    elif n <= 50:
        num_restarts = 20
    else:
        num_restarts = 10
        
    # Initial chains: mix of random SAWs and greedy growths
    for restart in range(num_restarts):
        if _time_remaining() < 15:
            break
            
        # 50% greedy, 50% random SAW
        if rng.random() < 0.5:
            chain = _grow(sequence, dimension, rng)
        else:
            chain = _generate_random_saw(sequence, dimension, rng)
            
        if chain is None:
            continue
            
        # Local search with fragment moves
        # Allocate time per restart
        # Total time budget ~180s. 
        # If 10 restarts, 18s each.
        # If 50 restarts, 3.6s each.
        
        # Let's just run until global time limit is close
        # But we need to ensure we don't run out of time in the middle of a restart
        # and leave the best result from the previous restart.
        
        chain = _local_search_fragment(sequence, chain, rng)
        
        value = _energy(sequence, chain)
        if value < best_energy:
            best, best_energy = chain, value
            
    if best is None:
        # Fallback: straight line
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
