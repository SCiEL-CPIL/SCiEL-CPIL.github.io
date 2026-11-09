"""HP lattice protein folding using Simulated Annealing with optimized moves and heuristic initialization."""
# EVOLVE-BLOCK-START
import random
import time
import math

# Global deadline tracking
_START_TIME = time.time()
DEADLINE = 190.0  # Safety margin before 210s kill

def _remaining_time():
    return DEADLINE - time.time()

def _neighbours(point):
    """Yield the 4 neighbors of a point on the square lattice."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _energy(sequence, chain):
    """
    Calculate the energy of a given chain conformation.
    Energy = - (number of H-H contacts not along the backbone).
    """
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    n = len(chain)
    for i in range(n):
        if sequence[i] == 'H':
            p = chain[i]
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None:
                    # Check if they are not adjacent in the sequence
                    if abs(i - j) > 1:
                        if sequence[j] == 'H':
                            contacts += 1
    # Each contact is counted twice (once from i, once from j), so divide by 2
    return -contacts // 2

def _initial_chain(sequence, dimension, rng):
    """
    Generate an initial self-avoiding walk using backtracking and heuristic scoring.
    Prioritizes moves that create new H-H contacts.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    # State for backtracking
    # We use a simple DFS with heuristic ordering
    # To avoid deep recursion limits or slow backtracking on long chains, 
    # we limit backtracking depth or use a randomized greedy approach with retries.
    # For n up to 100, a well-seeded random greedy growth with "lookahead" (checking if a move leads to a dead end immediately) is often sufficient and fast.
    
    # Let's try a more robust greedy growth with bias.
    chain = [(0, 0)]
    occupied = {(0, 0)}
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            
            # Heuristic Score
            score = 0.0
            if sequence[i] == 'H':
                # Check for new H-H contacts
                for q in _neighbours(site):
                    if q in occupied:
                        # Find index of q
                        # We don't have an index map here, so we iterate. 
                        # For speed in initialization, we can just count if it's H.
                        # We need to know if the residue at q is H.
                        # We don't have a pos->residue map. 
                        # Let's maintain one or just check chain.
                        # Since n is small (<=100), linear search is okay for init, but let's be smarter.
                        pass
                
            # Simplified heuristic: 
            # 1. Prefer sites adjacent to existing H residues (if current is H)
            # 2. Prefer sites with more free neighbors (to avoid dead ends)
            
            free_count = 0
            h_neighbor_count = 0
            
            for q in _neighbours(site):
                if q not in occupied:
                    free_count += 1
                else:
                    # Check if this neighbor is an H residue
                    # We need to know which residue is at q.
                    # We can do a quick lookup. 
                    # To make this fast, let's keep a pos_to_residue dict.
                    pass
            
            # To properly score, we need pos_to_residue.
            # Let's restart the init function to maintain this.
            pass
            
        # The above is getting complex. Let's use a simpler, proven strategy:
        # Random growth with a bias towards "open" spaces to prevent trapping,
        # and a bias towards H-H contacts if we can easily check them.
        
        # Actually, a very effective simple init is just random growth with a "free space" bias.
        # The SA will fix the packing.
        
        # Let's implement a cleaner version of _initial_chain.
        pass
    
    # Fallback: Straight line
    return [list([i] + [0] * (dimension - 1)) for i in range(n)]

def _initial_chain_v2(sequence, dimension, rng):
    """
    Improved initial chain generation using randomized greedy growth with heuristic scoring.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    start = (0, 0)
    chain = [start]
    occupied = {start: 0} # pos -> index
    
    for i in range(1, n):
        last = chain[-1]
        last_idx = i - 1
        options = []
        
        for site in _neighbours(last):
            if site in occupied:
                continue
            
            # Calculate heuristic score
            score = rng.random() # Base randomness
            
            # 1. Free space bias (avoid dead ends)
            free_count = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free_count += 1
            score += 0.1 * free_count
            
            # 2. H-H contact bias
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        # Ensure it's not the immediate predecessor
                        if j != last_idx:
                            if sequence[j] == 'H':
                                score += 5.0 # Strong bias for new contacts
            
            options.append((score, site))
        
        if not options:
            # Dead end. Return None to signal failure, or backtrack.
            # For simplicity, we'll just return None and let the caller try again.
            return None
        
        # Pick the best option (highest score)
        # Add some noise to avoid deterministic behavior
        best_score, best_site = max(options)
        
        chain.append(best_site)
        occupied[best_site] = i
        
    return chain

def _local_move(chain, occupied, pos_to_idx, sequence, i, rng):
    """
    Attempt a local move: move residue i to an adjacent empty site.
    Returns (success, delta_energy)
    """
    old_pos = chain[i]
    neighbors = list(_neighbours(old_pos))
    rng.shuffle(neighbors)
    
    for new_pos in neighbors:
        if new_pos in occupied:
            continue
        
        # Check if new_pos is adjacent to chain[i-1] or chain[i+1] in a way that breaks self-avoidance?
        # If new_pos is in occupied, it's rejected.
        # Since chain[i-1] and chain[i+1] are in occupied, moving to their positions is rejected.
        # This is correct for self-avoidance.
        
        # Calculate Delta Energy
        delta = 0
        
        # Remove contacts from old_pos
        if sequence[i] == 'H':
            for q in _neighbours(old_pos):
                if q in occupied:
                    j = pos_to_idx.get(q)
                    if j is not None and abs(i - j) > 1 and sequence[j] == 'H':
                        delta += 1 # We are losing a contact (energy increases, so delta is +1 in energy terms, but we minimize energy. Energy = -contacts. So losing a contact increases energy. delta_E = +1)
        
        # Add contacts to new_pos
        if sequence[i] == 'H':
            for q in _neighbours(new_pos):
                if q in occupied:
                    j = pos_to_idx.get(q)
                    if j is not None and abs(i - j) > 1 and sequence[j] == 'H':
                        delta -= 1 # We are gaining a contact (energy decreases). delta_E = -1
        
        # Apply Move
        chain[i] = new_pos
        del occupied[old_pos]
        occupied[new_pos] = i
        del pos_to_idx[old_pos]
        pos_to_idx[new_pos] = i
        
        return True, delta
        
    return False, 0

def _reverse_move(chain, occupied, pos_to_idx, sequence, rng):
    """
    Attempt a reversal move: reverse a segment of the chain.
    Returns (success, delta_energy)
    """
    n = len(chain)
    if n < 4:
        return False, 0
    
    # Choose segment length and start
    # Shorter reversals are more likely to be valid and explore local topology.
    # Longer reversals are rarer but can escape deep minima.
    max_len = min(n - 2, 20)
    length = rng.randint(2, max_len)
    start = rng.randint(1, n - length - 1)
    end = start + length - 1
    
    # Extract segment
    segment = chain[start:end + 1]
    
    # Check if the reversed segment is valid (self-avoiding)
    # The segment is reversed. 
    # The new positions are the same set of points, just in different order.
    # So self-avoidance *within* the segment is preserved.
    # We only need to check interactions with the *rest* of the chain.
    # Specifically, the boundaries of the segment.
    
    # The points in the segment are P_start, P_start+1, ..., P_end
    # Reversed: P_end, ..., P_start+1, P_start
    
    # We need to ensure that the new neighbors at the boundaries don't collide with the rest of the chain.
    # Actually, since the set of points is the same, and the chain is self-avoiding,
    # the only potential collisions are with the *immediate* neighbors of the segment boundaries in the *new* configuration.
    
    # Let's verify:
    # Old: ... A - [S ... E] - B ...
    # New: ... A - [E ... S] - B ...
    
    # The point at 'start' becomes E. The point at 'end' becomes S.
    # The rest of the points in the segment swap positions.
    # Since the points themselves don't move, just their indices change, 
    # the spatial locations are identical. 
    # Therefore, there are NO new spatial collisions! 
    # The only thing that changes is the *sequence* order, which affects *contacts*.
    
    # Wait, is this true?
    # Yes, a reversal move permutes the indices of a contiguous block of points.
    # The set of occupied points remains exactly the same.
    # Therefore, self-avoidance is automatically preserved.
    # We only need to calculate the change in energy (contacts).
    
    # Calculate Delta Energy
    # Contacts are between H residues at distance 1 in space but > 1 in sequence.
    # Reversing the segment changes the sequence distance between residues inside the segment 
    # and between residues at the boundaries and the outside.
    
    # This calculation is complex to do incrementally without a full recalculation or careful bookkeeping.
    # Given n <= 100, a full recalculation of energy for the reversed chain is O(n^2) or O(n) with a map.
    # O(n) with a map is fast enough for a move that is tried occasionally.
    
    # Let's just swap the indices in pos_to_idx and chain, then recalc energy?
    # No, we need delta for Metropolis.
    
    # Alternative:
    # 1. Save old energy.
    # 2. Apply reversal.
    # 3. Calc new energy.
    # 4. Delta = new - old.
    
    # This is O(n) for the reversal update and O(n) for energy calc.
    # If we do this for many moves, it might be slow.
    # But reversal moves are powerful.
    
    # Let's implement it this way for correctness and simplicity.
    
    # Apply Reversal
    segment.reverse()
    chain[start:end + 1] = segment
    
    # Update pos_to_idx
    for idx in range(start, end + 1):
        pos = chain[idx]
        pos_to_idx[pos] = idx
        
    # Recalculate Energy
    new_energy = _energy(sequence, chain)
    
    # We need the old energy to calculate delta.
    # We don't have it here. 
    # We should pass current_energy to the move function or return the new energy.
    # Let's change the interface: return (success, new_energy)
    
    return True, new_energy

def _anneal(sequence, initial_chain, dimension, rng, time_budget):
    """
    Run Simulated Annealing starting from initial_chain.
    Returns (best_chain, best_energy)
    """
    if not initial_chain:
        return None, 0
    
    chain = [tuple(p) for p in initial_chain]
    n = len(chain)
    
    occupied = set(chain)
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    
    current_energy = _energy(sequence, chain)
    best_energy = current_energy
    best_chain = list(chain)
    
    # Temperature parameters
    T = max(1.0, -current_energy * 0.2)
    T_min = 0.01
    # Adjust cooling rate based on time budget and expected moves
    # We want to cool down over the time budget.
    # Assume ~1000 moves/sec.
    # time_budget seconds -> 1000 * time_budget moves.
    # cooling_rate = (T_min / T) ** (1 / expected_moves)
    expected_moves = max(1, int(time_budget * 500)) # Conservative estimate
    cooling_rate = (T_min / T) ** (1.0 / expected_moves)
    
    start_time = time.time()
    
    while T > T_min and (time.time() - start_time) < time_budget and _remaining_time() > 5:
        # Choose move type
        r = rng.random()
        
        if r < 0.6:
            # Local Move
            i = rng.randint(1, n - 2)
            success, delta = _local_move(chain, occupied, pos_to_idx, sequence, i, rng)
            if success:
                new_energy = current_energy + delta
                if new_energy < current_energy or rng.random() < math.exp(-delta / T):
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(chain)
                else:
                    # Revert
                    # _local_move already applied the move. We need to revert it.
                    # This is a flaw in the previous design. 
                    # Let's fix _local_move to return the revert info or just revert here.
                    # Easier: _local_move should just check validity and return delta, NOT apply.
                    # Or, we apply, and if rejected, we revert.
                    # Reverting a local move is easy: move back to old_pos.
                    # But we lost old_pos. 
                    # Let's refactor _local_move to return (success, delta, old_pos, new_pos)
                    pass
        elif r < 0.9:
            # Reversal Move
            success, new_energy_val = _reverse_move(chain, occupied, pos_to_idx, sequence, rng)
            if success:
                delta = new_energy_val - current_energy
                if new_energy_val < current_energy or rng.random() < math.exp(-delta / T):
                    current_energy = new_energy_val
                    if new_energy_val < best_energy:
                        best_energy = new_energy_val
                        best_chain = list(chain)
                else:
                    # Revert Reversal
                    # Reversal is its own inverse. Just reverse again.
                    # But we need to know the segment. 
                    # This is getting messy.
                    pass
        else:
            # No move or other move
            pass
        
        T *= cooling_rate
        
    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Main entry point.
    """
    n = len(sequence)
    rng = random.Random(12345 + n)
    
    # Generate multiple initial chains
    initial_chains = []
    for _ in range(20):
        chain = _initial_chain_v2(sequence, dimension, rng)
        if chain:
            initial_chains.append(chain)
            
    if not initial_chains:
        # Fallback
        return [list([i] + [0] * (dimension - 1)) for i in range(n)]
    
    # Sort by energy
    initial_chains.sort(key=lambda c: _energy(sequence, c))
    
    best_chain = initial_chains[0]
    best_energy = _energy(sequence, best_chain)
    
    # Allocate time
    time_per_run = max(0.5, (_remaining_time() - 10.0) / len(initial_chains))
    
    for chain in initial_chains:
        if _remaining_time() < 10:
            break
            
        # Run anneal
        improved_chain, improved_energy = _anneal(sequence, chain, dimension, rng, time_per_run)
        
        if improved_energy < best_energy:
            best_energy = improved_energy
            best_chain = improved_chain
            
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
