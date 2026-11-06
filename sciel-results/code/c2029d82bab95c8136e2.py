"""HP lattice protein folding: beam search growth + segment relocation local search."""
# EVOLVE-BLOCK-START
import random
import time

_T0 = time.time()
_DEADLINE = 195.0  # 210s wall-clock limit, 15s safety margin


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute energy from scratch using position->index map."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _grow_beam(sequence, dimension, rng, beam_width=8):
    """
    Beam search growth: keeps top partial chains based on contact gain.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    start = (0,) * dimension
    initial_chain = [start]
    initial_occupied = {start: 0}
    # Beam items: (chain, occupied_dict, score)
    # score is negative contact count so far (lower is better)
    beam = [(initial_chain, initial_occupied, 0)]
    
    for i in range(1, n):
        next_beam = []
        for chain, occupied, score in beam:
            current = chain[-1]
            for site in _neighbours(current):
                if site in occupied:
                    continue
                
                gain = 0
                if sequence[i] == 'H':
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == 'H':
                            gain += 1
                
                new_score = score - gain  # energy is -contacts, so adding contact lowers score
                
                new_chain = chain + [site]
                new_occupied = dict(occupied)
                new_occupied[site] = i
                
                next_beam.append((new_chain, new_occupied, new_score))
        
        if not next_beam:
            return None
        
        # Sort by score (lower is better), tie-break with random for diversity
        next_beam.sort(key=lambda x: (x[2], rng.random()))
        
        # Keep top beam_width
        beam = next_beam[:beam_width]
        
        # Time check
        if _remaining() < 2.0:
            break
    
    if not beam:
        return None
    
    # Return the best chain
    return beam[0][0]


def _local_search_segments(sequence, chain, rng, time_budget):
    """
    Local search using segment relocation moves.
    Moves a contiguous segment of the chain to a new location.
    This is a structural change from single-residue moves.
    """
    n = len(chain)
    if n < 3:
        return chain
    
    best_chain = list(chain)
    best_energy = _compute_energy(sequence, best_chain)
    current_chain = list(chain)
    current_energy = best_energy
    
    start_time = time.time()
    max_time = time_budget
    
    # Try segment moves
    while (time.time() - start_time) < max_time and _remaining() > 2.0:
        # Pick a random segment length (1 to min(15, n//2))
        max_seg_len = min(15, n // 2)
        if max_seg_len < 1:
            break
        seg_len = rng.randint(1, max_seg_len)
        
        # Pick a random start position for the segment (internal residues only)
        # Segment must not include endpoints
        if n - seg_len - 1 < 2:
            break
        start_idx = rng.randint(1, n - seg_len - 2)
        end_idx = start_idx + seg_len - 1
        
        # Extract the segment
        segment = current_chain[start_idx:end_idx + 1]
        
        # The segment is connected to:
        # - prev_residue = current_chain[start_idx - 1]
        # - next_residue = current_chain[end_idx + 1]
        
        prev_res = current_chain[start_idx - 1]
        next_res = current_chain[end_idx + 1]
        
        # Remove the segment from the chain temporarily
        # We need to find a new position for the segment
        # The new position must be such that:
        # 1. The first residue of the segment is adjacent to prev_res
        # 2. The last residue of the segment is adjacent to next_res
        # 3. No residue in the segment occupies a position already occupied by non-segment residues
        
        # Try multiple random walks to find a valid new position
        found_new_pos = False
        new_segment = None
        
        # Try up to 5 random walks
        for _ in range(5):
            # Start from prev_res, do a random walk of length seg_len
            # The walk must end at a position adjacent to next_res
            # And must not collide with non-segment residues
            
            # First, build the set of occupied positions excluding the segment
            occupied_set = set()
            for i in range(n):
                if i < start_idx or i > end_idx:
                    occupied_set.add(current_chain[i])
            
            # Random walk from prev_res
            pos = prev_res
            walk = [pos]
            
            # We need to walk seg_len-1 steps (since we start at prev_res, which is not part of the segment)
            # Actually, the first residue of the segment must be adjacent to prev_res
            # So we pick a neighbor of prev_res for the first segment residue
            
            # Pick a random neighbor of prev_res for the first segment position
            candidates = [q for q in _neighbours(prev_res) if q not in occupied_set]
            if not candidates:
                continue
            
            pos = rng.choice(candidates)
            walk = [pos]
            
            # Continue the random walk
            valid = True
            for step in range(seg_len - 1):
                candidates = [q for q in _neighbours(pos) if q not in occupied_set and q not in walk]
                if not candidates:
                    valid = False
                    break
                pos = rng.choice(candidates)
                walk.append(pos)
            
            if not valid:
                continue
            
            # Check if the last position is adjacent to next_res
            if len(walk) != seg_len:
                continue
            last_pos = walk[-1]
            if last_pos not in _neighbours(next_res):
                continue
            
            # Valid new position found
            new_segment = walk
            found_new_pos = True
            break
        
        if not found_new_pos:
            continue
        
        # Compute the energy delta
        # Remove old segment contacts, add new segment contacts
        old_energy = current_energy
        
        # Build where map for current chain
        where = {p: i for i, p in enumerate(current_chain)}
        
        # Count contacts involving the segment in the old position
        old_contacts = 0
        for i in range(start_idx, end_idx + 1):
            if sequence[i] == 'H':
                for q in _neighbours(current_chain[i]):
                    j = where.get(q)
                    if j is not None and j > i + 1 and sequence[j] == 'H':
                        # Check if j is not in the segment
                        if not (start_idx <= j <= end_idx):
                            old_contacts += 1
                    elif j is not None and j < i - 1 and sequence[j] == 'H':
                        # This would be counted when we process j, so skip
                        pass
        
        # Actually, let's just compute the energy of the new chain directly
        # It's simpler and less error-prone
        
        # Build the new chain
        new_chain = list(current_chain)
        for k in range(seg_len):
            new_chain[start_idx + k] = new_segment[k]
        
        # Verify self-avoidance
        if len(set(new_chain)) != n:
            continue
        
        # Verify unit steps
        valid = True
        for i in range(1, n):
            dx = abs(new_chain[i][0] - new_chain[i-1][0])
            dy = abs(new_chain[i][1] - new_chain[i-1][1])
            if dx + dy != 1:
                valid = False
                break
        
        if not valid:
            continue
        
        # Compute new energy
        new_energy = _compute_energy(sequence, new_chain)
        
        # Accept if better
        if new_energy < current_energy:
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
    
    return best_chain


def fold(sequence, dimension):
    """Best conformation using beam growth + segment relocation local search."""
    rng = random.Random(12345 + len(sequence))
    n = len(sequence)
    
    if n <= 1:
        return [[0]]
    
    best = None
    best_energy = 0  # worst possible (no contacts)
    
    # Phase 1: Generate initial conformations via beam growth
    num_growth = min(50, max(10, int(2.0 * n)))  # scale with sequence length
    growth_time_budget = min(5.0, max(1.0, 0.05 * n))
    
    growth_deadline = time.time() + growth_time_budget
    for _ in range(num_growth):
        if time.time() > growth_deadline or _remaining() < 5.0:
            break
        chain = _grow_beam(sequence, dimension, rng, beam_width=8)
        if chain is None:
            continue
        value = _compute_energy(sequence, chain)
        if value < best_energy:
            best_energy = value
            best = chain
    
    if best is None:
        # Fallback: straight line
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _compute_energy(sequence, best)
    
    # Phase 2: Local search with segment relocation on best and some other good solutions
    # Keep top few solutions for local search
    solutions = []
    for _ in range(min(20, num_growth)):
        if _remaining() < 3.0:
            break
        chain = _grow_beam(sequence, dimension, rng, beam_width=8)
        if chain is None:
            continue
        value = _compute_energy(sequence, chain)
        solutions.append((value, chain))
    
    solutions.sort(key=lambda x: x[0])
    
    # Local search on top solutions
    top_k = min(5, len(solutions))
    for k in range(top_k):
        if _remaining() < 2.0:
            break
        _, chain = solutions[k]
        # If this is already the best, skip duplicate work
        ls_time = min(3.0, max(0.5, _remaining() * 0.3 / (top_k - k)))
        improved = _local_search_segments(sequence, chain, rng, ls_time)
        imp_energy = _compute_energy(sequence, improved)
        if imp_energy < best_energy:
            best_energy = imp_energy
            best = improved
    
    # Also do local search on the initial best
    if _remaining() > 1.0:
        ls_time = min(2.0, _remaining() * 0.5)
        improved = _local_search_segments(sequence, best, rng, ls_time)
        imp_energy = _compute_energy(sequence, improved)
        if imp_energy < best_energy:
            best_energy = imp_energy
            best = improved
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
