"""
Strategy: Multi-start local search with pivot moves and simulated annealing.
Instead of just growing chains, we initialize with a fast greedy method, then
improve each conformation using local search (pivot moves) with a simulated
annealing schedule. We track time strictly to ensure we finish before the limit.
"""
# EVOLVE-BLOCK-START
import random
import time
import math
import sys

# Global deadline management
_START_TIME = time.time()
DEADLINE = _START_TIME + 320.0  # 320s to be safe within 330s limit

def _time_left():
    return DEADLINE - time.time()

def _neighbours(point):
    """Generate 6 neighbors on cubic lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _compute_energy(sequence, chain):
    """Compute energy of a conformation."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _greedy_grow(sequence, dimension, rng):
    """
    Fast greedy initialization.
    Prioritizes positions that create H-H contacts, then maximizes free space.
    """
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        options = []
        last = chain[-1]
        for site in _neighbours(last):
            if site in occupied:
                continue
            
            # Score: 100 * new_contacts + free_space + random_noise
            score = rng.random()
            
            if sequence[i] == 'H':
                contacts = 0
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == 'H':
                            contacts += 1
                score += 100 * contacts
            
            # Free space heuristic: prefer sites with more empty neighbors
            free = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free += 1
            score += free
            
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick best option
        options.sort(reverse=True)
        best_site = options[0][1]
        chain.append(best_site)
        occupied[best_site] = i
        
    return chain

def _local_search(sequence, chain, time_budget, rng):
    """
    Local search using pivot moves and simulated annealing.
    Pivot move: choose two indices i < j and reverse the segment chain[i:j+1].
    """
    n = len(chain)
    if n < 3:
        return chain, _compute_energy(sequence, chain)
    
    current_chain = chain
    current_energy = _compute_energy(sequence, chain)
    best_chain = current_chain
    best_energy = current_energy
    
    # Simulated Annealing parameters
    # Start temperature high enough to accept bad moves, cool down
    T_start = 1.0
    T_end = 0.01
    # Number of steps depends on time budget, but we check time in loop
    # Let's estimate steps per second: ~1000-5000 simple ops in Python
    # We will just run until time runs out or T is too low
    
    # We need to be careful with pivot moves as they can be expensive to validate
    # For small N, we can try more moves.
    
    start_t = time.time()
    
    # Simple cooling schedule
    # We don't know exact step count, so we check time every few steps
    T = T_start
    
    while time.time() < start_t + time_budget and T > T_end:
        # Generate a random pivot move
        # Pick i and j
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i > j:
            i, j = j, i
        
        # Skip trivial moves
        if j - i < 2:
            continue
            
        # Apply pivot
        # Note: Reversing a segment in a list is O(N)
        new_chain = current_chain[:i] + current_chain[i:j+1][::-1] + current_chain[j+1:]
        
        # Check self-avoidance
        # This is O(N^2) if done naively with set, but we can be smarter or just accept the cost
        # For N=48, O(N^2) is ~2300 ops, which is fast.
        # For N=136, O(N^2) is ~18000 ops, still fast in C-optimized set operations.
        
        # Optimization: Only check the moved segment against the rest and itself
        # But for simplicity and correctness first, let's use a set.
        # To speed up, we can pre-compute the set of the whole chain? No, it changes.
        # Let's just check the new segment.
        
        # Actually, checking the whole chain for self-avoidance is safer.
        # If N is large, this becomes a bottleneck.
        # Let's try to optimize the self-avoidance check.
        
        # We can keep a set of occupied points for the current chain.
        # But updating it for a pivot is tricky.
        # Let's just do the set check on the new chain.
        # For N=136, creating a set of 136 tuples is fast.
        
        try:
            if len(set(new_chain)) != n:
                continue # Collision
        except TypeError:
            continue
            
        new_energy = _compute_energy(sequence, new_chain)
        
        # Acceptance criteria
        if new_energy < current_energy:
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_chain = current_chain
                best_energy = current_energy
        else:
            # Accept with probability exp(-delta_E / T)
            delta_E = new_energy - current_energy # Positive
            prob = math.exp(-delta_E / T)
            if rng.random() < prob:
                current_chain = new_chain
                current_energy = new_energy
                
        # Cool down
        # Geometric cooling
        T *= 0.999 # Slow cooling to explore more
        
        # Check time periodically
        if time.time() > _DEADLINE - 5.0:
            break

    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Fold the sequence using multi-start local search.
    """
    global _DEADLINE
    # Initialize deadline on first call if not set (though module level should handle it)
    if not hasattr(fold, '_initialized'):
        fold._initialized = True
        # Recalculate deadline based on current time just in case import was delayed
        # But _START_TIME is set at import.
        pass
        
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    
    rng = random.Random(42 + n)
    
    # Allocate time per sequence
    # We have ~320s total for 26 sequences. ~12s per sequence on average.
    # But we should allocate dynamically based on remaining time.
    
    best_chain_overall = None
    best_energy_overall = 0 # Energy is negative, so we want min energy (most negative)
    
    # First, try to get a valid initial solution quickly
    # If greedy fails, we might need a simpler fallback or just return a line?
    # Greedy usually works for these sizes.
    
    # We will run multiple restarts.
    # Each restart: Greedy Init -> Local Search
    
    # How many restarts?
    # Let's say we have a time budget for this sequence.
    # We don't know how many sequences are left.
    # Simple heuristic: Use 1/26th of the total time for this sequence?
    # Or just run until global deadline?
    # The problem says "240s CPU budget for all sequences together" in the prompt text,
    # but the directive says "KILLS the program after 330s wall-clock".
    # We should assume the 330s is the hard limit for the entire process.
    # So we must manage time globally.
    
    # Let's estimate remaining sequences? No, we don't know.
    # Let's just use a fraction of the remaining time.
    # If this is the first sequence, we have ~320s.
    # If we use all of it, we might not finish others.
    # The prompt implies fold is called once per sequence.
    # We should probably limit this call to a reasonable amount, say 10-15 seconds,
    # and leave the rest for others.
    # However, if we are the last one, we should use more.
    # Since we don't know the order, a fixed budget per call is safer.
    # 26 sequences * 12s = 312s. This is close to 320s.
    # Let's use 11s per sequence to be safe.
    
    seq_time_budget = 11.0
    
    # If we are running low on global time, reduce budget
    global_time_left = _time_left()
    if global_time_left < seq_time_budget + 1.0:
        seq_time_budget = max(0.1, global_time_left - 1.0)
        
    if seq_time_budget <= 0:
        # Out of time, return a straight line (valid but bad)
        return [[i, 0, 0] for i in range(n)]
    
    best_chain = None
    best_energy = 0 # Init to 0 (unfolded)
    
    # Run restarts until time budget for this sequence is exhausted
    start_seq_time = time.time()
    
    while time.time() - start_seq_time < seq_time_budget:
        # 1. Initialize
        init_chain = _greedy_grow(sequence, dimension, rng)
        
        if init_chain is None:
            # Try again with different rng? Or fallback?
            # Greedy failing is rare for these sizes but possible.
            # Let's try a simple random walk or just skip this restart.
            continue
            
        init_energy = _compute_energy(sequence, init_chain)
        
        # 2. Improve
        # Allocate remaining time for this restart to local search
        time_spent = time.time() - start_seq_time
        remaining_for_ls = seq_time_budget - time_spent - 0.5 # 0.5s safety margin
        
        if remaining_for_ls <= 0:
            break
            
        improved_chain, improved_energy = _local_search(sequence, init_chain, remaining_for_ls, rng)
        
        # 3. Update best
        if improved_energy < best_energy:
            best_energy = improved_energy
            best_chain = improved_chain
            
        # Check if we hit global deadline
        if _time_left() < 1.0:
            break
            
    if best_chain is None:
        # Fallback: straight line
        best_chain = [[i, 0, 0] for i in range(n)]
        
    # Ensure format is list of lists of ints
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
