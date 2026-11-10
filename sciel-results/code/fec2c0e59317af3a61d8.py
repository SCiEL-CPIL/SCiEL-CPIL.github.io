"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math
import sys

# Global state for time management
_start_time = None
_TOTAL_BUDGET = 230.0  # Leave 10s buffer for 240s limit

def _get_remaining_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return max(0.0, _TOTAL_BUDGET - (time.time() - _start_time))

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
    
    # Simple random walk with restarts
    max_attempts = 50
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
    # Fallback: straight line
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

def _replica_exchange_fold(sequence, n, dimension, rng, time_limit):
    """
    Replica Exchange Monte Carlo.
    Uses multiple replicas at different temperatures.
    """
    if n < 2:
        return _initial_random_chain(n, dimension, rng)
    
    is_h = [res == 'H' for res in sequence]
    
    # Number of replicas
    n_reps = 10
    # Temperature range: T_low to T_high
    # Energy scale is 1 per contact. 
    # T_low should be low enough to favor low energy, T_high high enough to explore.
    t_low = 0.5
    t_high = 5.0
    temps = [t_low * (t_high/t_low) ** (i / (n_reps - 1)) for i in range(n_reps)]
    
    # Initialize replicas
    replicas = []
    best_energy = float('inf')
    best_chain = None
    
    for i in range(n_reps):
        chain = _initial_random_chain(n, dimension, rng)
        if len(chain) < n:
            chain = [(j, 0, 0) for j in range(n)]
        energy = _calculate_energy(sequence, chain)
        replicas.append({'chain': chain, 'energy': energy, 'temp': temps[i]})
        if energy < best_energy:
            best_energy = energy
            best_chain = chain[:]
            
    start_time = time.time()
    moves_per_rep = 50 # Adjust based on speed
    swap_interval = 20 # Swap every N moves per replica
    
    # Precompute neighbor offsets for speed
    offsets = [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]
    
    while time.time() - start_time < time_limit:
        for rep_idx in range(n_reps):
            rep = replicas[rep_idx]
            chain = rep['chain']
            energy = rep['energy']
            temp = rep['temp']
            
            # Perform local moves
            for _ in range(moves_per_rep):
                # Move: Pivot
                # Pick a random pivot point i (1 to n-2)
                i = rng.randint(1, n - 2)
                
                p_prev = chain[i-1]
                p_curr = chain[i]
                
                # Vector from prev to curr
                dx = p_curr[0] - p_prev[0]
                dy = p_curr[1] - p_prev[1]
                dz = p_curr[2] - p_prev[2]
                
                # Generate candidates for new p_curr
                candidates = []
                for axis in range(3):
                    for step in (-1, 1):
                        q = list(p_prev)
                        q[axis] += step
                        q = tuple(q)
                        if q == p_curr:
                            continue
                        if i > 1:
                            p_prev_prev = chain[i-2]
                            if q == p_prev_prev:
                                continue
                        candidates.append(q)
                        
                if not candidates:
                    continue
                    
                new_p_curr = rng.choice(candidates)
                
                # Shift vector
                shift_x = new_p_curr[0] - p_curr[0]
                shift_y = new_p_curr[1] - p_curr[1]
                shift_z = new_p_curr[2] - p_curr[2]
                
                # Check collisions
                # Head: 0 to i-1
                # Tail: i to n-1
                # New tail is shifted.
                
                # Optimization: Instead of building new chain and checking set,
                # we can check if any (old_p + shift) is in head_set.
                # And if any (old_p + shift) is in tail_set (excluding the moving part? No, rigid body).
                # Actually, for a pivot, the tail moves rigidly.
                # Internal collisions are preserved.
                # We only need to check if the new tail intersects the head.
                
                head_set = set(chain[:i])
                
                # Check intersection
                collision = False
                for j in range(i, n):
                    old_p = chain[j]
                    new_p = (old_p[0] + shift_x, old_p[1] + shift_y, old_p[2] + shift_z)
                    if new_p in head_set:
                        collision = True
                        break
                
                if collision:
                    continue
                
                # Calculate energy change
                # Only contacts involving indices i-1, i, or j where j is in tail and neighbor is in head/tail?
                # Actually, contacts are between any two H residues at distance 1.
                # Moving the tail changes:
                # 1. Contacts between tail and head.
                # 2. Contacts within tail are preserved (rigid body).
                # 3. Contact between i-1 and i is broken/changed.
                
                # To be safe and simple, recalculate energy.
                # For N=136, O(N) is fast.
                new_chain = chain[:i] + [(chain[j][0]+shift_x, chain[j][1]+shift_y, chain[j][2]+shift_z) for j in range(i, n)]
                new_energy = _calculate_energy(sequence, new_chain)
                
                delta_e = new_energy - energy
                
                # Metropolis
                if delta_e < 0 or rng.random() < math.exp(-delta_e / temp):
                    rep['chain'] = new_chain
                    rep['energy'] = new_energy
                    energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = new_chain[:]
            
            # Replica Exchange
            if rep_idx > 0 and rep_idx % swap_interval == 0:
                # Try to swap with rep_idx - 1
                rep_a = replicas[rep_idx - 1]
                rep_b = replicas[rep_idx]
                
                e_a = rep_a['energy']
                e_b = rep_b['energy']
                t_a = rep_a['temp']
                t_b = rep_b['temp']
                
                # Delta = (e_b - e_a) * (1/t_a - 1/t_b)
                delta = (e_b - e_a) * (1.0/t_a - 1.0/t_b)
                
                if delta < 0 or rng.random() < math.exp(-delta):
                    # Swap chains and energies
                    rep_a['chain'], rep_b['chain'] = rep_b['chain'], rep_a['chain']
                    rep_a['energy'], rep_b['energy'] = rep_b['energy'], rep_a['energy']
    
    return best_chain

def fold(sequence, dimension):
    """Best conformation using Replica Exchange Monte Carlo."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
        
    remaining = _get_remaining_time()
    
    if remaining <= 0:
        # Fallback: straight line
        return [[i, 0, 0] for i in range(n)]
    
    # Allocate time
    # If we assume 26 calls, and we have 230s, we can't just use all of it for one call if it's the first.
    # But the contract says "fold is called once per benchmark sequence in a sandboxed subprocess".
    # This usually implies the *process* is the sandbox, and `fold` is called in a loop or similar.
    # If it's a single process for all 26, we must budget.
    # Let's assume we don't know the call index.
    # A safe heuristic: Use a fixed chunk of time, e.g., 5 seconds per call.
    # 26 * 5 = 130s. This leaves plenty of margin.
    # If the process is killed at 240s, and we've used 130s, we are safe.
    # If we try to be smarter and use more time for later calls, we risk timeout.
    # Let's stick to a fixed 5s per call to ensure completion.
    
    time_limit = min(remaining, 5.0)
    
    rng = random.Random(12345 + len(sequence))
    
    best_chain = _replica_exchange_fold(sequence, n, dimension, rng, time_limit)
    
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
