import random
import time
import sys

# EVOLVE-BLOCK-START

# Global deadline to ensure we don't exceed the 330s hard limit
# The evaluator kills at 330s. We set a conservative internal deadline.
_DEADLINE = time.time() + 320.0

def _time_left():
    return _DEADLINE - time.time()

def _neighbors(p):
    """Yield 6 neighbors of point p in 3D cubic lattice."""
    yield (p[0] + 1, p[1], p[2])
    yield (p[0] - 1, p[1], p[2])
    yield (p[0], p[1] + 1, p[2])
    yield (p[0], p[1] - 1, p[2])
    yield (p[0], p[1], p[2] + 1)
    yield (p[0], p[1], p[2] - 1)

def _compute_energy(sequence, chain):
    """Compute the energy of a given conformation.
    Energy = - (number of non-consecutive H-H contacts).
    """
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbors(p):
                j = where.get(q)
                if j is not None and abs(i - j) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _greedy_grow(sequence, dimension, rng):
    """
    Generate a self-avoiding walk using a greedy heuristic.
    Prioritizes moves that create H-H contacts, then moves that leave space.
    """
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        options = []
        last = chain[-1]
        for site in _neighbors(last):
            if site in occupied:
                continue
            
            # Heuristic score
            score = rng.random()
            
            if sequence[i] == 'H':
                # Check for immediate H-H contacts
                for q in _neighbors(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == 'H':
                            score += 10.0
            
            # Check for free space (avoid trapping)
            free_space = 0
            for q in _neighbors(site):
                if q not in occupied:
                    free_space += 1
            score += 0.1 * free_space
            
            options.append((score, site))
        
        if not options:
            return None # Dead end
        
        # Pick best option (with some randomness via score)
        # Sort by score descending
        options.sort(key=lambda x: x[0], reverse=True)
        # Pick from top few to add diversity
        k = min(3, len(options))
        site = options[rng.randint(0, k - 1)][1]
        
        chain.append(site)
        occupied[site] = i
        
    return chain

def _local_search(sequence, chain, time_limit):
    """
    Simulated Annealing local search.
    Returns the best chain found.
    """
    n = len(sequence)
    if n < 2:
        return chain
    
    current_chain = list(chain)
    current_energy = _compute_energy(sequence, current_chain)
    
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Temperature parameters
    # Start high enough to accept bad moves, cool down
    T = 2.0
    cooling_rate = 0.999
    min_T = 0.01
    
    # Precompute neighbors for speed? No, dynamic neighbors are fine for N<140.
    # But we need to be fast.
    
    # Track occupied set for O(1) lookup
    occupied = {p: i for i, p in enumerate(current_chain)}
    
    start_time = time.time()
    steps = 0
    
    while T > min_T and (time.time() - start_time) < time_limit:
        steps += 1
        
        # 1. Pick a random segment to move (flip or slide)
        # For simplicity and robustness, we use "move a single residue" or "reverse a segment".
        # Reversing a segment is a standard move in protein folding.
        
        # Let's try reversing a segment of length L.
        # L can be 2 to n-1.
        # To keep it fast, we pick random i, j with j > i+1.
        
        # Optimization: We'll try a mix of moves.
        # Move 1: Reverse segment [i, j]
        # Move 2: Rotate a segment? (Too complex to check self-avoidance easily without full re-check)
        
        # Let's stick to Segment Reversal.
        # i, j are indices in the chain.
        # We need to ensure the new positions are valid (not occupied by other parts of the chain).
        
        # Pick random length for reversal? Or random i, j?
        # Random i, j is uniform over all segments.
        if n < 3:
            break
            
        i = rng.randint(0, n - 2)
        j = rng.randint(i + 1, n - 1)
        
        # The segment is chain[i...j]
        # New segment will be chain[j...i]
        # We need to check if the new positions for the internal nodes (i+1...j-1) 
        # and the endpoints i, j (which swap) are valid.
        # Actually, the endpoints i and j swap positions.
        # The internal nodes reverse their order.
        
        # Let's extract the segment
        segment = current_chain[i:j+1]
        
        # The new positions for indices i..j will be:
        # new_chain[i] = segment[-1]
        # new_chain[i+1] = segment[-2]
        # ...
        # new_chain[j] = segment[0]
        
        # We need to check if these new positions are occupied by nodes OUTSIDE the segment.
        # Nodes inside the segment will vacate their old positions.
        
        # Build set of positions occupied by the segment
        seg_positions = set(segment)
        
        # Check validity of the new configuration for the segment
        valid = True
        new_segment = list(reversed(segment))
        
        # The nodes that are NOT in the segment are fixed.
        # We need to check if any node in new_segment (at its new index)
        # conflicts with a node NOT in the segment.
        
        # Also, we must ensure the new segment itself is self-avoiding (which it is, being a reversal of a valid segment? 
        # Wait. If the original segment was self-avoiding, the reversed segment occupies the SAME set of positions.
        # So self-avoidance WITHIN the segment is preserved.
        # We only need to check for conflicts with the REST of the chain.
        
        # However, we must also ensure that the "new" positions of the segment nodes
        # are not currently occupied by the REST of the chain.
        
        # Let's identify the "rest" of the chain
        # Rest indices: 0..i-1 and j+1..n-1
        
        # For each index k in [i, j], the new position is new_segment[k-i].
        # We check if new_segment[k-i] is in the occupied set of the REST.
        
        # To do this efficiently:
        # 1. Remove segment positions from a copy of occupied? No, too slow.
        # 2. Check if new_segment[k-i] is in `occupied` AND its owner index is NOT in [i, j].
        
        for k in range(i, j + 1):
            new_pos = new_segment[k - i]
            owner_idx = occupied.get(new_pos)
            if owner_idx is not None:
                if owner_idx < i or owner_idx > j:
                    valid = False
                    break
        
        if not valid:
            continue
            
        # Calculate Delta Energy
        # Energy changes only for contacts involving the segment nodes and their neighbors.
        # This is complex to calculate exactly without rebuilding.
        # For N < 140, rebuilding energy is O(N * 6) ~ 840 ops.
        # Steps per second? If we do 10k steps/sec, that's 8.4M ops/sec. Python might be slow.
        # Let's try to optimize energy calculation or just accept the cost if steps are fewer.
        
        # Alternative: Use a simpler move that is easier to delta?
        # "Move a single residue" is hard because it breaks the chain.
        # "Flip a single residue" (if it has 2 free neighbors) is rare.
        
        # Let's stick to Reversal and just compute full energy.
        # If it's too slow, we reduce steps.
        
        # Apply move
        current_chain[i:j+1] = new_segment
        
        # Update occupied set
        # This is O(L).
        for k in range(i, j + 1):
            # Remove old mapping? We can just overwrite.
            # But we need to remove the OLD positions of the segment nodes that are no longer there.
            # Since we reversed, the set of positions is the same.
            # So `occupied` keys for positions in seg_positions are still valid, 
            # but the VALUES (indices) have changed.
            
            # Wait, `occupied` maps Position -> Index.
            # Old: pos -> idx
            # New: pos -> new_idx
            # Since the set of positions is the same, we just need to update the values.
            
            # But we also need to ensure we don't have stale entries.
            # Since the set of positions is identical, we just update the index for each position in the segment.
            
            # However, it's easier to just rebuild `occupied` if we are doing full energy calc anyway?
            # Rebuilding occupied is O(N).
            
            pass
        
        # Rebuild occupied to be safe and simple
        occupied = {p: idx for idx, p in enumerate(current_chain)}
        
        new_energy = _compute_energy(sequence, current_chain)
        
        delta_E = new_energy - current_energy
        
        if delta_E < 0 or rng.random() < pow(2.71828, -delta_E / T):
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(current_chain)
        else:
            # Revert
            current_chain[i:j+1] = segment
            occupied = {p: idx for idx, p in enumerate(current_chain)}
            # current_energy remains unchanged
        
        T *= cooling_rate
        
    return best_chain

def fold(sequence, dimension):
    """
    Best conformation using Multi-Start Simulated Annealing.
    """
    # We have a global deadline.
    # We should allocate time wisely.
    # The evaluator calls fold 26 times.
    # We don't know which sequence is which, but we can track total time used.
    
    # Since we can't easily pass state between calls in a stateless function,
    # we rely on the global _DEADLINE.
    
    # If we have very little time left, return a fast greedy result.
    if _time_left() < 1.0:
        rng = random.Random(12345 + len(sequence))
        chain = _greedy_grow(sequence, dimension, rng)
        if chain is None:
            return [list(p) for p in [(i, 0, 0) for i in range(len(sequence))]]
        return [list(p) for p in chain]
    
    # How much time to spend on this sequence?
    # Ideally, we want to spend more on harder sequences.
    # But we don't know difficulty a priori.
    # Let's assume we want to finish all sequences by the deadline.
    # We don't know how many sequences are left.
    # Heuristic: Spend a fixed fraction of remaining time?
    # Or just spend until time is low?
    
    # Let's try: Spend up to 50% of remaining time on each sequence, 
    # but at least 0.5s.
    # This is risky if there are many sequences.
    
    # Better: The task says "240 s CPU budget for all sequences together" in the text, 
    # but the contract says "330s wall-clock".
    # Let's assume we have ~12s per sequence on average (330/26).
    
    # Let's allocate 8 seconds per sequence to be safe.
    time_per_seq = 8.0
    
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0 # 0 is unfolded (energy 0). We want negative.
    
    # Initial greedy
    init_chain = _greedy_grow(sequence, dimension, rng)
    if init_chain:
        best_chain = init_chain
        best_energy = _compute_energy(sequence, best_chain)
    else:
        # Fallback
        best_chain = [(i, 0, 0) for i in range(len(sequence))]
        best_energy = 0
        
    # Multi-start SA
    # Number of restarts depends on time.
    # Each SA run takes some time.
    
    # We'll run SA until time runs out for this sequence.
    # But we must leave time for future sequences.
    
    # Let's just run one long SA or multiple short ones?
    # Multiple short ones with different inits is often better.
    
    # Let's do 5 restarts if time permits.
    num_restarts = 5
    
    for r in range(num_restarts):
        if _time_left() < 1.0:
            break
            
        # Generate a new initial chain
        # Vary the greedy randomness
        rng_sa = random.Random(12345 + len(sequence) + r * 100)
        init = _greedy_grow(sequence, dimension, rng_sa)
        
        if init is None:
            # Try a different start or simple line
            init = [(i, 0, 0) for i in range(len(sequence))]
            
        # Run SA
        # Allocate time for this SA run
        # Remaining time for this sequence:
        # We want to finish this sequence in ~8s.
        # Time elapsed for this sequence so far:
        # We don't track per-sequence time easily.
        
        # Let's just give SA a fixed time budget, say 1.5s per restart.
        sa_time = 1.5
        if r == 0:
            # Give the first one more time
            sa_time = 3.0
            
        # Check global deadline
        if _time_left() < sa_time + 0.5:
            sa_time = max(0.1, _time_left() - 0.5)
            
        if sa_time <= 0:
            break
            
        sa_chain = _local_search(sequence, init, sa_time)
        
        sa_energy = _compute_energy(sequence, sa_chain)
        
        if sa_energy < best_energy:
            best_energy = sa_energy
            best_chain = sa_chain
            
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(len(sequence))]
        
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
