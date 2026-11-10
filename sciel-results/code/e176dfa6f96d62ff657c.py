"""HP lattice protein folding on the square lattice using Monte Carlo E2T moves."""
# EVOLVE-BLOCK-START
import random
import time
import math

# Deadline guard: The evaluator kills after 210s. We stop searching at 180s to be safe.
_GLOBAL_START = time.time()
_DEADLINE = _GLOBAL_START + 175.0

def _neighbours(p):
    """Yield 4 neighbors of a point p."""
    x, y = p
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _compute_energy(sequence, chain):
    """Compute energy: -1 for each non-adjacent H-H contact."""
    n = len(chain)
    if n < 2:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(i - j) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _random_growth(sequence, dimension, rng, max_retries=100):
    """Generate a random self-avoiding walk using greedy growth."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    for _ in range(max_retries):
        chain = [(0, 0)]
        occupied = {(0, 0): 0}
        success = True
        for i in range(1, n):
            last = chain[-1]
            neighbors = list(_neighbours(last))
            rng.shuffle(neighbors)
            placed = False
            for site in neighbors:
                if site not in occupied:
                    chain.append(site)
                    occupied[site] = i
                    placed = True
                    break
            if not placed:
                success = False
                break
        if success:
            return chain
    # Fallback: straight line (always valid)
    return [(i, 0) for i in range(n)]

def _try_e2t_move(chain, occupied, sequence, rng, tail_len):
    """
    Attempt an End-to-Tail move.
    Detach the last 'tail_len' residues and reattach them.
    Returns (new_chain, new_occupied) or None if failed.
    """
    n = len(chain)
    if tail_len >= n:
        return None
    
    # Detach tail
    tail = chain[n - tail_len:]
    head = chain[:n - tail_len]
    
    # Remove tail from occupied
    new_occupied = occupied.copy()
    for p in tail:
        del new_occupied[p]
    
    # Pick a random site on the head to attach to? 
    # Standard E2T usually attaches to the end of the head or a random valid spot.
    # Attaching to the end of the head is simpler but less flexible.
    # Let's try attaching to a random free neighbor of a random position in the head.
    # This allows "internal" reconfiguration if we consider the cut point mobile, 
    # but here we fixed the cut. So we reattach the tail to the end of the head 
    # or a neighbor of the end of the head.
    
    # Let's try reattaching to a random free neighbor of the LAST residue of the head.
    head_end = head[-1]
    candidates = [p for p in _neighbours(head_end) if p not in new_occupied]
    
    if not candidates:
        return None
    
    start_site = rng.choice(candidates)
    
    # Grow the tail from start_site
    new_chain = head + [start_site]
    new_occupied[start_site] = n - tail_len
    
    current_pos = start_site
    for k in range(1, tail_len):
        neighbors = list(_neighbours(current_pos))
        rng.shuffle(neighbors)
        placed = False
        for site in neighbors:
            if site not in new_occupied:
                new_chain.append(site)
                new_occupied[site] = n - tail_len + k
                current_pos = site
                placed = True
                break
        if not placed:
            return None
            
    return new_chain, new_occupied

def _try_local_move(chain, occupied, sequence, rng):
    """
    Attempt a local move: move the last residue to a different neighbor.
    """
    n = len(chain)
    if n < 2:
        return None
    
    last = chain[-1]
    prev = chain[-2]
    
    # Remove last
    new_occupied = occupied.copy()
    del new_occupied[last]
    
    # Try other neighbors of prev
    candidates = [p for p in _neighbours(prev) if p not in new_occupied and p != last]
    
    if not candidates:
        return None
    
    new_pos = rng.choice(candidates)
    new_chain = chain[:-1] + [new_pos]
    new_occupied[new_pos] = n - 1
    
    return new_chain, new_occupied

def fold(sequence, dimension):
    """
    Fold the sequence using Monte Carlo with Simulated Annealing and E2T moves.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    rng = random.Random(12345 + len(sequence))
    
    # Initial chain
    chain = _random_growth(sequence, dimension, rng)
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _compute_energy(sequence, chain)
    
    best_chain = chain[:]
    best_energy = current_energy
    
    # Simulated Annealing parameters
    start_temp = 1.0
    end_temp = 0.01
    # Estimate steps based on time budget
    # We have roughly 175s total for 10 sequences. ~17s per sequence.
    # But we should check time dynamically.
    
    steps = 0
    max_steps = 50000  # Cap steps to avoid infinite loop if time check fails
    
    while time.time() < _DEADLINE:
        steps += 1
        if steps > max_steps:
            break
            
        # Temperature decay
        # Linear decay based on steps or time? 
        # Let's use time-based decay for consistency.
        elapsed = time.time() - _GLOBAL_START
        # Assume 175s total budget.
        # For this specific sequence, we don't know exactly how much time is left for others.
        # Simple approach: decay temp over the lifetime of this call.
        # Let's define a local start time for this fold call.
        # Actually, _GLOBAL_START is global. 
        # Let's just use a fixed number of steps per sequence if time allows.
        # If we are fast, do more. If slow, do less.
        
        # Let's use a simple cooling schedule based on steps within this call
        # to ensure we always finish.
        # Total steps allocated per sequence: 20000.
        # If we finish 20000 steps in 5s, we have time for more? 
        # The prompt says "Spare time does not ... automatically increase fixed restart ... counts".
        # So we stick to a fixed number of steps, but check deadline.
        
        if steps > 20000:
            break
            
        # Choose move type
        # 70% E2T, 30% Local
        if rng.random() < 0.7:
            # Tail length between 5 and min(20, n//2)
            max_tail = min(20, n // 2)
            if max_tail < 5:
                max_tail = 5
            tail_len = rng.randint(5, max_tail)
            move = _try_e2t_move(chain, occupied, sequence, rng, tail_len)
        else:
            move = _try_local_move(chain, occupied, sequence, rng)
            
        if move is None:
            continue
            
        new_chain, new_occupied = move
        new_energy = _compute_energy(sequence, new_chain)
        
        # Metropolis Acceptance
        delta = new_energy - current_energy
        if delta < 0:
            # Always accept improvement
            chain = new_chain
            occupied = new_occupied
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain[:]
        else:
            # Accept with probability exp(-delta / T)
            # T needs to be defined. 
            # Let's use a temperature that cools down.
            # T = start_temp * (1 - steps / max_steps)
            T = start_temp * (1.0 - steps / 20000.0)
            if T > 0:
                prob = math.exp(-delta / T)
                if rng.random() < prob:
                    chain = new_chain
                    occupied = new_occupied
                    current_energy = new_energy
            # Else T is 0, so only accept if delta < 0 (handled above)
            
    # Final polish: Local search with T=0 (Hill Climbing)
    # Run for a few more steps or until no improvement
    polish_steps = 1000
    for _ in range(polish_steps):
        if time.time() > _DEADLINE:
            break
        move = _try_local_move(chain, occupied, sequence, rng)
        if move is None:
            continue
        new_chain, new_occupied = move
        new_energy = _compute_energy(sequence, new_chain)
        if new_energy < current_energy:
            chain = new_chain
            occupied = new_occupied
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain[:]
        else:
            # In polish, we can also try E2T if stuck? 
            # Let's just stick to local for polish to be fast and precise.
            pass

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
