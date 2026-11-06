# EVOLVE-BLOCK-START
import time
import math
import random
import sys

# Global deadline for the entire process (240s budget, keep 30s safety margin)
GLOBAL_DEADLINE = time.time() + 210.0

def _neighbours(point):
    """Yield the 6 neighbors of a 3D point."""
    for axis in range(3):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy_from_contacts(sequence, contacts):
    """Calculate energy from a set of H-H contacts (i, j) where i < j."""
    # Energy is -1 for each H-H contact
    return -len(contacts)

def _compute_contacts(sequence, chain):
    """
    Compute the set of H-H contacts for a given chain.
    Returns a set of tuples (i, j) where i < j, sequence[i]=='H', sequence[j]=='H',
    and positions are adjacent.
    """
    n = len(chain)
    if n == 0:
        return set()
    
    pos_to_idx = {}
    for i, p in enumerate(chain):
        pos_to_idx[p] = i
        
    contacts = set()
    for i, p in enumerate(chain):
        if sequence[i] != 'H':
            continue
        for q in _neighbours(p):
            j = pos_to_idx.get(q)
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts.add((i, j))
                
    return contacts

def _local_search(sequence, chain, time_limit, rng):
    """
    Fast local search using pivot moves.
    Optimized for speed: 
    1. Precompute H indices.
    2. Maintain contact set and update incrementally if possible, 
       or recompute if faster for small N.
       For N up to 136, recomputing contacts is O(N * 6) which is ~800 ops.
       Very fast.
    3. Use aggressive cooling.
    """
    n = len(chain)
    if n < 4:
        return chain, _energy_from_contacts(sequence, _compute_contacts(sequence, chain))
    
    current_chain = [tuple(p) for p in chain]
    current_contacts = _compute_contacts(sequence, current_chain)
    current_energy = -len(current_contacts)
    
    best_chain = list(current_chain)
    best_energy = current_energy
    
    start_time = time.time()
    T = 1.5
    alpha = 0.985
    
    # Precompute H indices for quick check
    h_indices = [i for i, res in enumerate(sequence) if res == 'H']
    is_h = [res == 'H' for res in sequence]
    
    while time.time() < time_limit:
        # Choose pivot
        # Random pivot, biased towards the middle
        pivot_idx = rng.randint(1, n - 2)
        
        p_prev = current_chain[pivot_idx - 1]
        p_pivot = current_chain[pivot_idx]
        
        # Determine bond axis
        bond_vector = [p_pivot[i] - p_prev[i] for i in range(3)]
        bond_axis = -1
        for i in range(3):
            if bond_vector[i] != 0:
                bond_axis = i
                break
        if bond_axis == -1:
            continue
            
        rot_axes = [i for i in range(3) if i != bond_axis]
        r_axis = rot_axes[rng.randint(0, 1)]
        angle = 1 if rng.random() < 0.5 else -1
        
        # Rotate tail
        head_set = set(current_chain[:pivot_idx + 1])
        new_tail = []
        valid = True
        occupied_tail = set()
        
        # Optimization: Instead of rotating every point and checking collision,
        # we can just try the move. If it's invalid, we skip.
        
        for i in range(pivot_idx + 1, n):
            orig = current_chain[i]
            dx = orig[0] - p_pivot[0]
            dy = orig[1] - p_pivot[1]
            dz = orig[2] - p_pivot[2]
            
            if r_axis == 0:
                if angle == 1: nx, ny, nz = dx, -dz, dy
                else: nx, ny, nz = dx, dz, -dy
            elif r_axis == 1:
                if angle == 1: nx, ny, nz = dz, dy, -dx
                else: nx, ny, nz = -dz, dy, dx
            else:
                if angle == 1: nx, ny, nz = -dy, dx, dz
                else: nx, ny, nz = dy, -dx, dz
                
            new_point = (p_pivot[0] + nx, p_pivot[1] + ny, p_pivot[2] + nz)
            
            if new_point in head_set or new_point in occupied_tail:
                valid = False
                break
                
            occupied_tail.add(new_point)
            new_tail.append(new_point)
            
        if not valid:
            continue
            
        new_chain = current_chain[:pivot_idx + 1] + new_tail
        
        # Calculate new energy
        # For small N, full recompute is often faster than incremental due to overhead
        new_contacts = _compute_contacts(sequence, new_chain)
        new_energy = -len(new_contacts)
        
        delta_e = new_energy - current_energy
        
        if delta_e < 0:
            current_chain = new_chain
            current_contacts = new_contacts
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        else:
            if delta_e == 0:
                # Accept neutral moves to escape plateaus
                if rng.random() < 0.5:
                    current_chain = new_chain
                    current_contacts = new_contacts
                    current_energy = new_energy
            else:
                # Metropolis
                # T decays
                prob = math.exp(-delta_e / T)
                if rng.random() < prob:
                    current_chain = new_chain
                    current_contacts = new_contacts
                    current_energy = new_energy
                    
        T *= alpha
        
    return best_chain, best_energy

def _greedy_fold(sequence, dimension, rng, greedy_factor=0.0):
    """
    Greedy construction with some randomness.
    At each step, pick the neighbor that maximizes immediate contact gain.
    Ties broken by random or by 'openness' (number of empty neighbors).
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]
    
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0): 0}
    
    # Precompute H indices
    is_h = [res == 'H' for res in sequence]
    
    for i in range(1, n):
        last_pos = chain[-1]
        neighbors = list(_neighbours(last_pos))
        # Filter occupied
        free_neighbors = [nb for nb in neighbors if nb not in occupied]
        
        if not free_neighbors:
            return None # Failed
        
        if len(free_neighbors) == 1:
            next_pos = free_neighbors[0]
        else:
            # Score each neighbor
            scores = []
            for nb in free_neighbors:
                score = 0
                if is_h[i]:
                    # Check for H neighbors in occupied
                    for q in _neighbours(nb):
                        if q in occupied:
                            j = occupied[q]
                            if j < i - 1 and is_h[j]:
                                score += 1
                # Add small bonus for openness (number of free neighbors of nb)
                # This helps avoid trapping the chain
                openness = 0
                for q in _neighbours(nb):
                    if q not in occupied:
                        openness += 1
                score += 0.1 * openness
                
                # Add randomness
                score += rng.random() * greedy_factor
                
                scores.append((score, nb))
            
            # Sort by score descending
            scores.sort(key=lambda x: x[0], reverse=True)
            next_pos = scores[0][1]
            
        chain.append(next_pos)
        occupied[next_pos] = i
        
    return chain

def fold(sequence, dimension):
    """
    Multi-Start Greedy + Local Search.
    """
    start_time = time.time()
    n = len(sequence)
    
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    
    # Allocate time based on chain length
    # Longer chains need more search time per start, but fewer starts might be needed
    # Short chains can afford many starts
    
    # Total budget for this sequence
    # We don't know exactly how much time is left for other sequences, 
    # but we must finish all within 240s.
    # Strategy: Allocate a fixed slice of time per sequence, or dynamic.
    # Since we are in a single process, we must be careful.
    # Let's aim to use ~90% of the remaining global time for the current sequence?
    # No, that's dangerous if we have many sequences.
    # Better: Estimate cost per sequence.
    # 26 sequences. 240s total. ~9s per sequence on average.
    # But some are short (48), some long (136).
    # Let's allocate time proportional to N^1.5 or similar?
    # Or just a fixed chunk, say 8 seconds per sequence, ensuring we finish.
    
    # Let's use a dynamic slice: 
    # If we are early in the run, we can spend more.
    # But we don't know the order.
    # Safe bet: 8.5 seconds per sequence. 26 * 8.5 = 221s. Leaves 19s margin.
    
    seq_time_limit = 8.5
    # Adjust for very short chains to allow more starts
    if n <= 30:
        seq_time_limit = 6.0
    elif n > 100:
        seq_time_limit = 10.0
        
    deadline = start_time + seq_time_limit
    
    best_chain = None
    best_energy = 0 # Most negative is best, so start with 0 (worst)
    
    # Number of starts
    # For short chains, we can do many fast greedy starts
    # For long chains, greedy is slower, so fewer starts
    
    if n <= 30:
        num_starts = 50
    elif n <= 60:
        num_starts = 20
    else:
        num_starts = 10
        
    # We will run starts in a loop, checking time.
    # Each start: Greedy Fold + Local Search
    # We should split time between starts.
    # Time per start = seq_time_limit / num_starts
    
    time_per_start = seq_time_limit / num_starts
    
    rng = random.Random(42) # Fixed seed for reproducibility? 
    # The task doesn't specify seed, but deterministic is safer for debugging.
    # However, randomization helps exploration. Let's use a fixed seed for consistency.
    
    for k in range(num_starts):
        if time.time() > deadline - 0.5:
            break
            
        # Vary the greediness
        # Early starts: very greedy (low randomness)
        # Later starts: more random (high randomness)
        # greedy_factor controls the weight of random noise in scoring
        # 0.0 = pure greedy, 1.0 = significant noise
        
        # For short chains, pure greedy often works well if diversified by local search
        # Let's vary the initial direction or noise.
        
        # Actually, let's just vary the noise.
        # Noise level from 0.0 to 2.0
        noise = k * (2.0 / max(1, num_starts - 1))
        
        # Greedy Fold
        chain = _greedy_fold(sequence, dimension, rng, greedy_factor=noise)
        
        if chain is None:
            continue
            
        # Local Search
        # Allocate remaining time for this start to local search
        # Leave 0.1s margin for loop overhead
        ls_deadline = min(deadline - 0.1, time.time() + time_per_start - (time.time() - start_time - k*time_per_start))
        # Simpler: just use a fixed fraction of time_per_start for LS
        ls_time = time_per_start * 0.8
        if time.time() + ls_time > deadline:
            ls_time = deadline - time.time() - 0.1
        if ls_time <= 0:
            break
            
        opt_chain, opt_energy = _local_search(sequence, chain, ls_time, rng)
        
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_chain = opt_chain
            
    if best_chain is None:
        # Fallback: straight line
        best_chain = [[i, 0, 0] for i in range(n)]
    else:
        best_chain = [list(p) for p in best_chain]
        
    return best_chain

# EVOLVE-BLOCK-END
