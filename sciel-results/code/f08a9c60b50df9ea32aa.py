"""HP lattice protein folding using simulated annealing with reversals and local moves."""
# EVOLVE-BLOCK-START
import random
import math
import time

# Global deadline guard
_TOTAL_BUDGET = 115.0  # Safety margin for 120s limit
_start_time = None

def _neighbours(point):
    """Yield the 4 neighbors of a 2D point."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _init_chain(sequence):
    """Generate a random self-avoiding walk to initialize the chain."""
    n = len(sequence)
    if n == 0:
        return []
    
    # Start with a straight line
    chain = [(i, 0) for i in range(n)]
    
    # Apply a few random reversals to scramble it
    rng = random.Random(42)
    for _ in range(10):
        i = rng.randint(0, n - 2)
        j = rng.randint(i + 1, n - 1)
        # Check if reversal is valid (unit steps at boundaries)
        if i > 0:
            p_prev = chain[i-1]
            p_new_start = chain[j]
            if abs(p_prev[0] - p_new_start[0]) + abs(p_prev[1] - p_new_start[1]) != 1:
                continue
        if j < n - 1:
            p_next = chain[j+1]
            p_new_end = chain[i]
            if abs(p_next[0] - p_new_end[0]) + abs(p_next[1] - p_new_end[1]) != 1:
                continue
        
        # Reverse
        chain[i:j+1] = chain[i:j+1][::-1]
        
    return chain

def _compute_energy(sequence, chain):
    """Calculate the energy of a given chain."""
    n = len(sequence)
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i in range(n):
        if sequence[i] == 'H':
            x, y = chain[i]
            # Check 4 neighbors
            for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
                j = where.get((nx, ny))
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _sa_fold(sequence, time_limit):
    """Simulated Annealing for a single sequence."""
    n = len(sequence)
    if n == 0:
        return []
    
    # Initialize
    chain = _init_chain(sequence)
    
    # Initial Energy
    current_energy = _compute_energy(sequence, chain)
    best_energy = current_energy
    best_chain = chain[:]
    
    # SA Parameters
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.9999 # Slow cooling
    
    rng = random.Random(12345 + len(sequence))
    
    start_t = time.time()
    
    while time.time() - start_t < time_limit:
        # Check global deadline
        if _start_time is not None and time.time() - _start_time > _TOTAL_BUDGET - 5:
            break
            
        # Choose move type
        # 2-opt Reversal
        i = rng.randint(0, n - 2)
        j = rng.randint(i + 1, n - 1)
        
        # Extract segment
        seg = chain[i:j+1]
        rev_seg = seg[::-1]
        
        # Check validity of reversal
        valid = True
        if i > 0:
            p_prev = chain[i-1]
            p_new_start = rev_seg[0]
            if abs(p_prev[0] - p_new_start[0]) + abs(p_prev[1] - p_new_start[1]) != 1:
                valid = False
        if valid and j < n - 1:
            p_next = chain[j+1]
            p_new_end = rev_seg[-1]
            if abs(p_next[0] - p_new_end[0]) + abs(p_next[1] - p_new_end[1]) != 1:
                valid = False
        
        if valid:
            # Apply reversal
            new_chain = chain[:i] + rev_seg + chain[j+1:]
            
            # Calculate Delta Energy
            new_energy = _compute_energy(sequence, new_chain)
            delta_e = new_energy - current_energy
            
            # Acceptance
            if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                chain = new_chain
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = chain[:]
            
            # Cooling
            T *= cooling_rate
            if T < T_min:
                T = T_min

    return best_chain

def fold(sequence, dimension):
    """Best conformation using Simulated Annealing."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Allocate time for this sequence
    # 10 sequences total. 115s budget.
    # Let's give each sequence 10s.
    time_limit = 10.0
    
    # Check remaining time
    elapsed = time.time() - _start_time
    remaining = _TOTAL_BUDGET - elapsed
    if remaining <= 0:
        # Fallback to straight line
        return [[i, 0] for i in range(len(sequence))]
    
    # Use min(10, remaining)
    actual_limit = min(time_limit, remaining)
    
    chain = _sa_fold(sequence, actual_limit)
    
    # Convert to list of lists
    return [list(p) for p in chain]

# EVOLVE-BLOCK-END
