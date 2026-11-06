"""Improved HP lattice folding with segment relocation local search."""
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
    # Leave 10s safety margin before 120s budget (or 210s wall-clock limit)
    # The evaluator kills at 210s, but the task contract mentions 120s CPU budget.
    # We aim for 110s to be safe against any overhead.
    return 110.0 - (time.time() - _START_TIME)

def _neighbours(point):
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

def _energy(sequence, chain):
    """Compute HP contact energy: -1 for each H-H pair on neighbouring sites not adjacent in chain."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _is_self_avoiding(chain):
    """Check if chain is self-avoiding with unit steps."""
    if not chain:
        return True
    seen = set()
    for i, p in enumerate(chain):
        if p in seen:
            return False
        seen.add(p)
        if i > 0:
            prev = chain[i-1]
            if abs(p[0]-prev[0]) + abs(p[1]-prev[1]) != 1:
                return False
    return True

def _grow_heuristic(sequence, dimension, rng):
    """
    Greedy chain growth with lookahead: prefer sites that maximize potential H-H contacts
    and leave room for future growth.
    """
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        current = chain[-1]
        options = []
        
        for site in _neighbours(current):
            if site in occupied:
                continue
            
            # Count immediate H-H contacts this placement would make
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            
            # Count free neighbours (for future growth flexibility)
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            
            # Score: high gain, then high free space, then random tiebreak
            score = gain * 10.0 + free * 0.5 + rng.random() * 0.1
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick best option
        options.sort(key=lambda x: x[0], reverse=True)
        site = options[0][1]
        occupied[site] = i
        chain.append(site)
    
    return chain

def _try_relocate_segment(chain, start_idx, end_idx, occupied_by_others, rng):
    """
    Attempt to relocate the segment chain[start_idx:end_idx+1] to a new position.
    Returns the new segment path (list of tuples) if successful, else None.
    """
    n = len(chain)
    seg_len = end_idx - start_idx + 1
    left_neighbor = chain[start_idx - 1]
    right_neighbor = chain[end_idx + 1]
    
    # Get candidate start positions (neighbors of left_neighbor)
    start_candidates = [p for p in _neighbours(left_neighbor) if p not in occupied_by_others]
    if not start_candidates:
        return None
    
    # Shuffle to randomize
    rng.shuffle(start_candidates)
    
    # For efficiency, we limit the number of random walk attempts per start candidate.
    # Since seg_len is small (3-8), a simple randomized walk with backtracking
    # or just a few random tries is sufficient.
    
    for start_pos in start_candidates:
        # If the segment is length 1, we just check if start_pos is adjacent to right_neighbor
        if seg_len == 1:
            if abs(start_pos[0] - right_neighbor[0]) + abs(start_pos[1] - right_neighbor[1]) == 1:
                return [start_pos]
            continue
        
        # Try a limited number of random extensions
        max_walks = 10
        for _ in range(max_walks):
            path = [start_pos]
            visited = {start_pos}
            current = start_pos
            
            valid = True
            for step in range(1, seg_len):
                # Get neighbors of current not in visited and not in occupied_by_others
                neighbors = [p for p in _neighbours(current) if p not in visited and p not in occupied_by_others]
                if not neighbors:
                    valid = False
                    break
                
                # Choose a neighbor
                next_pos = rng.choice(neighbors)
                path.append(next_pos)
                visited.add(next_pos)
                current = next_pos
            
            if not valid:
                continue
            
            # Check if we ended at a position adjacent to right_neighbor
            if abs(current[0] - right_neighbor[0]) + abs(current[1] - right_neighbor[1]) == 1:
                return path
        
    return None

def _local_search_segment(sequence, chain, rng, max_iter=100):
    """
    Local search via segment relocation.
    """
    n = len(chain)
    if n < 4:
        return chain
    
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    
    # We will try to improve the chain by relocating segments
    for _ in range(max_iter):
        if _time_remaining() < 2.0:
            break
        
        # Pick a random segment
        # Longer segments are harder to relocate but offer more energy change potential.
        # Shorter segments (3-5) are easier to relocate.
        # We bias towards shorter segments for higher success rate, occasionally trying longer ones.
        if rng.random() < 0.7:
            seg_len = rng.randint(3, min(5, n - 2))
        else:
            seg_len = rng.randint(5, min(8, n - 2))
        
        if seg_len > n - 2:
            seg_len = n - 2
        if seg_len < 3:
            continue
            
        start_idx = rng.randint(1, n - seg_len - 1)
        end_idx = start_idx + seg_len - 1
        
        # Occupied by others (excluding the segment itself)
        occupied_by_others = {p: i for i, p in enumerate(chain) if i < start_idx or i > end_idx}
        
        new_segment = _try_relocate_segment(chain, start_idx, end_idx, occupied_by_others, rng)
        
        if new_segment is None:
            continue
        
        # Construct new chain
        new_chain = list(chain)
        for i, p in enumerate(new_segment):
            new_chain[start_idx + i] = p
        
        # Verify self-avoidance just in case (should be guaranteed by logic, but safe)
        # And compute energy
        new_energy = _energy(sequence, new_chain)
        
        if new_energy < best_energy:
            best_chain = new_chain
            best_energy = new_energy
            # Update the reference chain for the next iteration's "others" calculation
            chain = best_chain
            
    return best_chain

def fold(sequence, dimension):
    """
    Fold an HP sequence on the square lattice.
    Returns list of [x, y] coordinates.
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
    
    # Strategy:
    # 1. Generate multiple diverse growth candidates.
    # 2. Apply segment relocation local search to the top candidates.
    
    # Determine number of growth attempts based on sequence length and time
    # For short sequences, we can afford more attempts
    if n <= 30:
        num_growth = 200
    elif n <= 50:
        num_growth = 100
    elif n <= 70:
        num_growth = 50
    else:
        num_growth = 20
    
    # Adjust for time
    while num_growth > 10 and _time_remaining() < 30:
        num_growth //= 2
    
    candidates = []
    
    for attempt in range(num_growth):
        if _time_remaining() < 15: # Leave time for local search
            break
        
        chain = _grow_heuristic(sequence, dimension, rng)
        if chain is None:
            continue
        
        # Verify self-avoidance
        if not _is_self_avoiding(chain):
            continue
        
        energy = _energy(sequence, chain)
        
        # Keep top candidates
        if len(candidates) < 10:
            candidates.append((energy, chain))
            if len(candidates) == 10:
                candidates.sort(key=lambda x: x[0])
        else:
            if energy < candidates[-1][0]:
                candidates[-1] = (energy, chain)
                candidates.sort(key=lambda x: x[0])
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
        candidates = [(best_energy, best_chain)]
    
    # Phase 2: Local search refinement on the best candidates
    # We sort candidates by energy and refine the best ones first
    candidates.sort(key=lambda x: x[0])
    
    # Refine the top 3 candidates
    for energy, chain in candidates[:3]:
        if _time_remaining() < 5.0:
            break
        
        # Allocate more iterations for the best candidate
        if energy == candidates[0][0]:
            max_iter = 500
        else:
            max_iter = 200
            
        improved_chain = _local_search_segment(sequence, chain, rng, max_iter=max_iter)
        
        if improved_chain is not None:
            improved_energy = _energy(sequence, improved_chain)
            if improved_energy < best_energy:
                best_energy = improved_energy
                best_chain = improved_chain
    
    # Final check: if we have time, do one more pass on the absolute best
    if _time_remaining() > 10.0:
        final_chain = _local_search_segment(sequence, best_chain, rng, max_iter=200)
        if final_chain is not None:
            final_energy = _energy(sequence, final_chain)
            if final_energy < best_energy:
                best_chain = final_chain
                best_energy = final_energy

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
