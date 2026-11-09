"""HP lattice protein folding: Fast Reversal Local Search with Simulated Annealing."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
# The task contract says 120s CPU budget. The directive says 210s wall clock.
# To be safe, we aim to finish well within 120s CPU.
_GLOBAL_DEADLINE = _GLOBAL_START + 110.0


def _remaining_time():
    return max(0.0, _GLOBAL_DEADLINE - time.time())


_DELTAS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _energy(sequence, chain):
    """Compute energy: -1 for each H-H contact on non-adjacent lattice neighbours."""
    if len(chain) < 2:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            x, y = p
            for dx, dy in _DELTAS:
                q = (x + dx, y + dy)
                j = where.get(q)
                if j is not None and abs(i - j) > 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts // 2


def _grow(sequence, rng, bias_hh=5.0, bias_free=1.0):
    """
    Greedy growth with randomization.
    Prefers sites that create H-H contacts and have free space.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        last_x, last_y = chain[-1]
        options = []
        
        for dx, dy in _DELATAS if False else _DELTAS:
            nx, ny = last_x + dx, last_y + dy
            site = (nx, ny)
            if site in occupied:
                continue
            
            # Calculate score
            score = 0.0
            
            # 1. Immediate H-H contact gain
            if sequence[i] == "H":
                for edx, edy in _DELTAS:
                    q = (nx + edx, ny + edy)
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        score += bias_hh
            
            # 2. Free space (to avoid trapping)
            free = 0
            for edx, edy in _DELTAS:
                q = (nx + edx, ny + edy)
                if q not in occupied:
                    free += 1
            score += bias_free * free
            
            # 3. Random noise
            score += rng.random() * 2.0
            
            options.append((score, site))
        
        if not options:
            # Dead end, return None to signal failure
            return None
        
        # Sort by score descending
        options.sort(key=lambda x: x[0], reverse=True)
        
        # Select top option with some probability, or random from top 3
        if len(options) > 1 and rng.random() < 0.3:
            # Pick from top 3
            top_k = min(3, len(options))
            weights = [1.0 / (k + 1) for k in range(top_k)]
            total_w = sum(weights)
            r = rng.random() * total_w
            cumulative = 0
            selected_site = options[0][1]
            for k in range(top_k):
                cumulative += weights[k]
                if r <= cumulative:
                    selected_site = options[k][1]
                    break
        else:
            selected_site = options[0][1]
            
        occupied[selected_site] = i
        chain.append(selected_site)
        
    return chain


def _try_reversal(chain, i, j):
    """
    Attempt to reverse the segment chain[i..j].
    Returns new chain if valid, else None.
    """
    n = len(chain)
    if i >= j:
        return None
    
    # Check boundary connections
    # 1. Left boundary: chain[i-1] must connect to chain[j]
    if i > 0:
        p_prev = chain[i - 1]
        p_end = chain[j]
        if abs(p_prev[0] - p_end[0]) + abs(p_prev[1] - p_end[1]) != 1:
            return None
    else:
        # If i==0, no left boundary check needed for connection, 
        # but we must ensure the new start (chain[j]) doesn't collide with suffix?
        # Actually, if i==0, the segment is the prefix.
        # The new start is chain[j]. The rest is chain[j+1:].
        # We need to check if chain[j] collides with chain[j+1:]? No, chain[j] was adjacent to chain[j+1] originally?
        # Wait. Original: ... chain[j] - chain[j+1] ...
        # Reversed: ... chain[j] becomes start of reversed part.
        # The connection is between chain[i-1] (none) and chain[j] (new start).
        # And between chain[i] (new end of reversed part) and chain[j+1].
        pass

    # 2. Right boundary: chain[i] must connect to chain[j+1]
    if j < n - 1:
        p_start = chain[i]
        p_next = chain[j + 1]
        if abs(p_start[0] - p_next[0]) + abs(p_start[1] - p_next[1]) != 1:
            return None
    
    # Check self-avoidance
    # The reversed segment is chain[i..j][::-1]
    # We need to ensure no point in the reversed segment is in the prefix (0..i-1) or suffix (j+1..n-1)
    # Note: The internal points of the segment are fine as they were valid before.
    
    prefix_set = set(chain[:i])
    suffix_set = set(chain[j+1:])
    
    # Combine prefix and suffix for fast lookup
    forbidden = prefix_set | suffix_set
    
    for k in range(i, j + 1):
        if chain[k] in forbidden:
            return None
            
    # Construct new chain
    # Optimization: Avoid slicing the whole chain if possible, but for N<=100 it's fine
    new_chain = chain[:i] + chain[i:j+1][::-1] + chain[j+1:]
    
    return new_chain


def _anneal(sequence, initial_chain, time_limit, rng):
    """
    Simulated Annealing with Reversal moves.
    """
    n = len(initial_chain)
    current_chain = initial_chain[:]
    current_energy = _energy(sequence, current_chain)
    best_chain = current_chain[:]
    best_energy = current_energy
    
    T = 2.0
    T_end = 0.05
    cooling_rate = 0.999
    
    deadline = time.time() + time_limit
    steps = 0
    stagnation = 0
    
    while time.time() < deadline:
        if _remaining_time() < 0.5:
            break
            
        # Generate random reversal
        # Bias towards shorter segments? Or random?
        # Shorter segments are more likely to be valid.
        # Let's pick i and j such that length is small.
        max_len = min(n, 20) # Limit segment length for validity probability
        if n < 4:
             break
             
        i = rng.randint(0, n - 2)
        # j is i + length
        # Pick length 1 to max_len
        length = rng.randint(1, min(max_len, n - i - 1))
        if length < 1:
            continue
        j = i + length
        
        new_chain = _try_reversal(current_chain, i, j)
        if new_chain is None:
            # Move invalid, count as stagnation? Or just retry?
            # Let's just retry to keep time uniform
            continue
            
        new_energy = _energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        # Metropolis criterion
        if delta_e < 0 or rng.random() < math.exp(-delta_e / max(T, 0.01)):
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = current_chain[:]
                stagnation = 0
            else:
                stagnation += 1
        else:
            stagnation += 1
            
        # Adaptive cooling
        if stagnation > 50:
            # Reheat
            T = min(T * 1.5, 5.0)
            stagnation = 0
        else:
            T *= cooling_rate
            if T < T_end:
                T = T_end
                
        steps += 1
        
    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Fold the sequence into a low-energy conformation.
    """
    rng = random.Random(42 + len(sequence))
    n = len(sequence)
    
    if n < 2:
        return [[0, 0]] * n
        
    best_chain = None
    best_energy = 0 # Energy is negative
    
    # Strategy: Multiple restarts with different biases
    # We allocate time to generate initial chains and then anneal them.
    
    total_time_budget = _remaining_time()
    # Reserve some time for the last anneal to finish cleanly
    effective_budget = total_time_budget - 5.0
    
    if effective_budget <= 0:
        # Fallback: simple straight line
        return [[i, 0] for i in range(n)]
        
    # Define restart configs
    # (bias_hh, bias_free)
    configs = [
        (10.0, 0.5),
        (5.0, 1.0),
        (2.0, 2.0),
        (15.0, 0.2),
        (8.0, 0.8),
        (1.0, 3.0),
        (20.0, 0.1),
        (4.0, 1.5),
    ]
    
    # We will run a loop. In each iteration, we pick a config, grow, and anneal.
    # We divide the time among these restarts.
    # Let's do dynamic allocation: give each restart a chunk of time.
    # If we run out of time, we stop.
    
    restarts = 0
    max_restarts = 20 # Cap to prevent infinite loop if time is large
    
    while _remaining_time() > 10.0 and restarts < max_restarts:
        restarts += 1
        config = configs[restarts % len(configs)]
        bias_hh, bias_free = config
        
        # Generate initial chain
        initial_chain = _grow(sequence, rng, bias_hh, bias_free)
        
        if initial_chain is None:
            # Try a few more times with this config? Or just skip?
            # Let's try 3 times
            for _ in range(3):
                initial_chain = _grow(sequence, rng, bias_hh, bias_free)
                if initial_chain is not None:
                    break
            if initial_chain is None:
                continue
                
        # Calculate time for this anneal
        # We want to use up the remaining time, but leave some for the next restart
        # Heuristic: If we have many restarts left, use less time.
        # Simple: Use 50% of remaining time for this anneal?
        # Better: Estimate how many restarts we can do.
        # Let's just assign a fixed fraction of the *original* effective budget per restart?
        # No, dynamic is better.
        
        # Let's allocate time such that we can do at least 5 restarts if possible.
        # time_per_restart = effective_budget / 5
        # But we should use more time if we are close to the end.
        
        # Strategy:
        # If restarts < 5, use effective_budget / 5
        # Else, use remaining_time / 2 (greedy endgame)
        
        if restarts < 5:
            time_for_anneal = effective_budget / 5.0
        else:
            time_for_anneal = _remaining_time() / 2.0 - 1.0 # Leave 1s for next check
            
        time_for_anneal = max(0.5, min(time_for_anneal, _remaining_time() - 1.0))
        
        if time_for_anneal <= 0:
            break
            
        best_c, best_e = _anneal(sequence, initial_chain, time_for_anneal, rng)
        
        if best_e < best_energy:
            best_energy = best_e
            best_chain = best_c
            
    if best_chain is None:
        # Fallback if all growths failed or no time left
        # Try one last simple growth
        initial_chain = _grow(sequence, rng, 5.0, 1.0)
        if initial_chain is not None:
            best_chain = initial_chain
            best_energy = _energy(sequence, best_chain)
        else:
            # Straight line
            best_chain = [(i, 0) for i in range(n)]
            best_energy = _energy(sequence, best_chain)
            
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
