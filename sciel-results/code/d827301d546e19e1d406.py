"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math
from collections import defaultdict

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _initial_random_chain(n, dimension, rng):
    """Generate a random self-avoiding walk of length n."""
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]
    
    # Use a simple random walk with backtracking or rejection
    # For n up to ~136, a simple random walk with backtracking is fast enough
    # We'll try to generate a chain that is "random" but valid.
    # A simple random walk often hits a dead end. 
    # Strategy: Start at 0, try to grow. If stuck, backtrack.
    # This is effectively a depth-first search for a random path.
    
    chain = [(0, 0, 0)]
    occupied = {chain[0]}
    
    # To ensure we get a chain of length n, we might need to restart if we fail
    # But for small n, random walk usually works or we can just use a simple heuristic:
    # Grow randomly, if stuck, restart.
    
    max_attempts = 100
    for _ in range(max_attempts):
        chain = [(0, 0, 0)]
        occupied = {chain[0]}
        success = True
        for i in range(1, n):
            last = chain[-1]
            options = [q for q in _neighbours(last) if q not in occupied]
            if not options:
                success = False
                break
            next_site = rng.choice(options)
            chain.append(next_site)
            occupied.add(next_site)
        if success:
            return chain
    # Fallback: straight line (rarely needed for n < 136 in 3D)
    return [(i, 0, 0) for i in range(n)]

def _calculate_energy(sequence, chain):
    """Calculate energy of a given chain."""
    if len(chain) < 2:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _monte_carlo_fold(sequence, n, dimension, rng, time_limit):
    """Perform Monte Carlo folding with incremental energy updates."""
    if n < 2:
        return _initial_random_chain(n, dimension, rng)
    
    # Initialize
    chain = _initial_random_chain(n, dimension, rng)
    if len(chain) < n:
        # Fallback if random walk failed (should be rare)
        chain = [(i, 0, 0) for i in range(n)]
        
    occupied = set(chain)
    where = {p: i for i, p in enumerate(chain)}
    
    # Precompute H positions for faster check
    is_h = [res == 'H' for res in sequence]
    
    # Initial energy
    current_energy = _calculate_energy(sequence, chain)
    
    best_chain = chain[:]
    best_energy = current_energy
    
    start_time = time.time()
    t = 1.0 # Temperature
    t_min = 0.01
    steps_per_second = 100000 # Estimate
    
    # We will do moves until time limit
    # Move types: 
    # 1. Pivot: Rotate a segment of the chain
    # 2. Single step: Move the end of the chain
    
    # For simplicity and robustness, let's use a mix of "move end" and "pivot"
    # However, pivots are complex to implement incrementally.
    # Let's stick to "move end" (single step) and "crank" (2-step rotation) for now.
    # Actually, a simple Metropolis with single-step moves on the end is very slow to explore.
    # Let's use a "pivot" move: pick a random point i, and rotate the chain from i to end.
    # This is hard to do incrementally.
    
    # Alternative: Use "single step" moves but with a high acceptance rate for random walks.
    # Better: Use "2-opt" or "crank" moves.
    
    # Let's implement a simple "move end" strategy first.
    # If the end is at P, try to move it to a neighbor Q.
    # This requires Q to be free.
    # This is very local.
    
    # Let's try a different approach: "Slithering" or "Reptation" is too slow.
    # Let's use "Pivot" moves implemented via full recalculation for simplicity first, 
    # but optimized by only checking the affected parts.
    
    # Actually, for N=136, full recalculation is O(N) ~ 136 ops. 
    # 10,000 moves/sec -> 1.3M ops/sec. 
    # In 300s, we can do 300,000,000 moves. 
    # This is plenty if we are smart.
    
    # Let's use a hybrid:
    # 1. Random pivot: pick i, try to rotate chain[i:] by 90 degrees around chain[i-1] axis? No, that's not a lattice move.
    # Lattice moves:
    # A valid move is changing the direction of a bond.
    # Let's pick a random index i (1 to n-2).
    # The bond i-1 -> i has a direction d.
    # We can try to change it to a perpendicular direction d'.
    # This shifts the entire tail i...n-1.
    # We need to check if the new positions are free.
    # And update energy.
    
    # This is a "pivot" move.
    
    def try_pivot(i):
        """Try to pivot the chain at index i. Returns (new_chain, new_energy) or None."""
        # i is the index of the start of the tail. 1 <= i <= n-2
        # The segment is chain[i-1] -> chain[i] -> ... -> chain[n-1]
        # We rotate the vector chain[i] - chain[i-1] to a new direction.
        
        p_prev = chain[i-1]
        p_curr = chain[i]
        
        # Vector from prev to curr
        dx = p_curr[0] - p_prev[0]
        dy = p_curr[1] - p_prev[1]
        dz = p_curr[2] - p_prev[2]
        
        # Find a perpendicular direction
        # There are 4 possible directions for the next step from p_prev.
        # One is -d (backwards, invalid).
        # The other 3 are valid candidates for the new direction.
        
        # Generate all 6 neighbors of p_prev
        candidates = []
        for axis in range(3):
            for step in (-1, 1):
                q = list(p_prev)
                q[axis] += step
                q = tuple(q)
                # Exclude the current direction (p_curr) and the one before (if i>1)
                if q == p_curr:
                    continue
                if i > 1:
                    p_prev_prev = chain[i-2]
                    if q == p_prev_prev:
                        continue
                candidates.append(q)
        
        if not candidates:
            return None
            
        # Pick a random candidate
        new_p_curr = rng.choice(candidates)
        
        # The rest of the chain shifts by the vector (new_p_curr - p_curr)
        shift_x = new_p_curr[0] - p_curr[0]
        shift_y = new_p_curr[1] - p_curr[1]
        shift_z = new_p_curr[2] - p_curr[2]
        
        # Check if the new tail collides with the head (0 to i-1)
        # Head is chain[0] to chain[i-1]
        # Tail is chain[i] to chain[n-1]
        # New tail is chain[i] shifted, etc.
        
        # We need to check if any new tail position is in the head.
        # And if any new tail position collides with the old tail (excluding the first point which is moving).
        # Actually, since it's a rigid rotation/translation of the tail, we just need to check against the Head.
        # The internal collisions of the tail remain valid (self-avoidance within tail is preserved).
        
        # Check collision with Head
        head_set = set(chain[:i]) # chain[0] to chain[i-1]
        
        # Generate new tail positions
        new_tail = []
        for j in range(i, n):
            old_p = chain[j]
            new_p = (old_p[0] + shift_x, old_p[1] + shift_y, old_p[2] + shift_z)
            new_tail.append(new_p)
            if new_p in head_set:
                return None # Collision
        
        # Also check if new_p_curr (which is new_tail[0]) collides with anything else? 
        # It's checked in head_set.
        
        # Construct new chain
        new_chain = chain[:i] + new_tail
        
        # Calculate energy change
        # Only contacts involving the tail (i to n-1) or the bond i-1->i can change.
        # Contacts are between H-H non-adjacent.
        
        # This is getting complex to do incrementally correctly without a robust framework.
        # Given the time limit, let's just recalculate energy for the new chain.
        # O(N) per move.
        
        new_energy = _calculate_energy(sequence, new_chain)
        
        return new_chain, new_energy

    moves = 0
    while time.time() - start_time < time_limit:
        # Pick a random pivot point
        # Bias towards smaller i? Or uniform?
        # Uniform is fine.
        i = rng.randint(1, n - 2)
        
        res = try_pivot(i)
        if res:
            new_chain, new_energy = res
            delta_e = new_energy - current_energy
            
            # Metropolis criterion
            if delta_e < 0 or rng.random() < math.exp(-delta_e / t):
                # Accept
                chain = new_chain
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = chain[:]
                
                moves += 1
                # Cool down
                t = max(t_min, t * 0.999)
                
    return best_chain

def fold(sequence, dimension):
    """Best conformation using Monte Carlo folding."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
        
    # Time budget for all sequences is 240s.
    # We don't know how many sequences will be called in this process.
    # The contract says "fold is called once per benchmark sequence in a sandboxed subprocess".
    # This implies `fold` is called multiple times, but the *process* has a 240s budget.
    # However, the function signature is `fold(sequence, dimension)`.
    # We must assume that we might be called multiple times.
    # But we don't have a global timer accessible across calls if they are separate invocations?
    # "fold is called once per benchmark sequence in a sandboxed subprocess"
    # This phrasing is ambiguous. Does it mean one subprocess per sequence? Or one subprocess for all?
    # "with a 240 s CPU budget for all sequences together"
    # This implies the *total* time for the script is 240s.
    # If `fold` is called in a loop in the same process, we can track time.
    # If it's called in separate processes, we can't.
    # Given "sandboxed subprocess" (singular) and "budget for all sequences", it's likely one process.
    # We will use a global start time.
    
    global _start_time
    if not hasattr(fold, '_start_time'):
        fold._start_time = time.time()
        
    elapsed = time.time() - fold._start_time
    # Reserve 10s for safety
    remaining = max(0, 230 - elapsed)
    
    if remaining <= 0:
        # Fallback: straight line
        return [[i, 0, 0] for i in range(n)]
    
    # Allocate time based on sequence length?
    # Longer sequences are harder.
    # Let's just use the remaining time for this call.
    # If this is the first call, we have ~230s.
    # If it's the 26th call, we might have very little.
    # We should divide time evenly?
    # We don't know the total number of calls.
    # Let's assume 26 calls.
    # If we have 230s left, and this is call k (unknown), it's risky.
    
    # Strategy: Use a fixed time per call, e.g., 5 seconds.
    # 26 * 5 = 130s. Safe.
    time_limit = min(remaining, 5.0)
    
    rng = random.Random(12345 + len(sequence))
    
    # Run MC
    best_chain = _monte_carlo_fold(sequence, n, dimension, rng, time_limit)
    
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
