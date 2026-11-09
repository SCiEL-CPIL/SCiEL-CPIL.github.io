"""HP lattice protein folding: diverse greedy growth + simulated annealing."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_GLOBAL_DEADLINE = _GLOBAL_START + 170.0


def _remaining_time():
    return max(0.0, _GLOBAL_DEADLINE - time.time())


_DELTAS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbours(point):
    x, y = point
    for dx, dy in _DELTAS:
        yield (x + dx, y + dy)


def _is_valid_chain(chain):
    """Check self-avoidance and unit steps."""
    seen = set()
    for i, p in enumerate(chain):
        if p in seen:
            return False
        seen.add(p)
        if i > 0:
            if abs(chain[i][0] - chain[i-1][0]) + abs(chain[i][1] - chain[i-1][1]) != 1:
                return False
    return True


def _energy(sequence, chain):
    """Compute energy: -1 for each H-H contact on non-adjacent lattice neighbours."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(i - j) > 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts // 2


def _grow(sequence, dimension, rng, bias=1.0):
    """One randomized greedy chain growth.
    bias > 1 favors contacts, bias < 1 favors free space/randomness.
    """
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    n = len(sequence)
    for i in range(1, n):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            # Weight the score
            score = bias * gain + 0.1 * free + rng.random()
            options.append((score, site))
        if not options:
            return None
        # Pick the best option
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _apply_reversal(chain, i, j):
    """Apply reversal from i to j. Returns None if invalid, else new chain."""
    n = len(chain)
    if i < 1 or j >= n - 1 or i >= j:
        return None
    
    p_prev = chain[i-1]
    p_next = chain[j+1]
    p_start = chain[i]
    p_end = chain[j]
    
    # Check connectivity
    if abs(p_prev[0] - p_end[0]) + abs(p_prev[1] - p_end[1]) != 1:
        return None
    if abs(p_start[0] - p_next[0]) + abs(p_start[1] - p_next[1]) != 1:
        return None
        
    # Check collisions in the reversed segment
    # The segment is chain[i..j]
    # We need to check if any point in the reversed segment collides with the rest of the chain (excluding the segment itself)
    # And if the segment is self-avoiding (it was, so it remains).
    # We just need to check collisions with prefix [0..i-1] and suffix [j+1..n-1]
    
    prefix_set = set(chain[:i])
    suffix_set = set(chain[j+1:])
    
    new_seg = chain[i:j+1][::-1]
    for p in new_seg:
        if p in prefix_set or p in suffix_set:
            return None
            
    new_chain = chain[:i] + new_seg + chain[j+1:]
    return new_chain


def _apply_pivot(chain, i, k):
    """Apply pivot at i with rotation k (1, 2, or 3). Returns None if invalid, else new chain."""
    n = len(chain)
    if i < 1 or i >= n - 1:
        return None
        
    pivot = chain[i]
    tail = chain[i+1:]
    prefix_set = set(chain[:i+1]) # Include pivot? No, pivot is fixed. Tail moves.
    # Actually, prefix is chain[0..i-1]. Pivot is chain[i].
    # The new tail must not collide with chain[0..i-1].
    # It must also connect to chain[i].
    
    prefix_set = set(chain[:i])
    
    new_tail = []
    valid = True
    for p in tail:
        x = p[0] - pivot[0]
        y = p[1] - pivot[0] # Wait, y = p[1] - pivot[1]
        y = p[1] - pivot[1]
        
        for _ in range(k):
            x, y = -y, x
            
        new_p = (pivot[0] + x, pivot[1] + y)
        new_tail.append(new_p)
        
        if new_p in prefix_set:
            valid = False
            break
            
    if not valid:
        return None
        
    # Check connection: new_tail[0] must be adjacent to chain[i-1]? No, chain[i] is the pivot.
    # The tail is attached to chain[i]. So new_tail[0] must be adjacent to chain[i].
    # It always is, because we rotated around chain[i].
    # But we must ensure new_tail[0] != chain[i-1] (self-avoidance at the junction)
    # And new_tail[0] != chain[i+1] (original position, which is now empty, so fine)
    
    if new_tail[0] == chain[i-1]:
        return None
        
    new_chain = chain[:i+1] + new_tail
    return new_chain


def _simulated_annealing(sequence, chain, time_limit, rng):
    """Run SA on a given chain."""
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
        
    best_chain = chain[:]
    best_energy = _energy(sequence, chain)
    current_energy = best_energy
    
    # Cooling schedule
    # Start T=1.0, decay to 0.01
    T = 1.0
    T_end = 0.01
    # Estimate steps: let's do as many as we can in time_limit
    # Or fix steps based on n? 
    # Let's use a time-based loop.
    
    deadline = time.time() + time_limit
    steps = 0
    
    while time.time() < deadline:
        # Pick a move
        if rng.random() < 0.5:
            # Reversal
            i = rng.randint(1, n - 2)
            j = rng.randint(i + 1, n - 1)
            new_chain = _apply_reversal(chain, i, j)
        else:
            # Pivot
            i = rng.randint(1, n - 2)
            k = rng.choice([1, 2, 3])
            new_chain = _apply_pivot(chain, i, k)
            
        if new_chain is None:
            continue
            
        new_energy = _energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain[:]
                
        # Cool down
        # Exponential cooling
        # We need to know total steps to calculate decay rate.
        # Since we don't know total steps, we can use a simple decay per step.
        # Or use a time-based decay.
        # Let's assume we want to reach T_end in ~1000 steps for small proteins.
        # For larger proteins, we might need more steps.
        # Let's just decay T by a factor.
        T *= 0.999
        
        steps += 1
        
        # If T is very low, we might stop early or restart?
        # For simplicity, let it run.
        
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation from many randomized greedy growths followed by simulated annealing."""
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    
    # Phase 1: Generate diverse initial chains
    initial_chains = []
    # Use different biases
    biases = [0.5, 1.0, 1.5, 2.0, 0.1]
    for bias in biases:
        for _ in range(5): # 5 chains per bias
            chain = _grow(sequence, dimension, rng, bias=bias)
            if chain:
                initial_chains.append(chain)
                
    if not initial_chains:
        # Fallback
        initial_chains = [[(i, 0) for i in range(n)]]
        
    # Evaluate and sort
    scored_chains = []
    for c in initial_chains:
        e = _energy(sequence, c)
        scored_chains.append((e, c))
        
    scored_chains.sort(key=lambda x: x[0])
    
    # Take top 5
    top_chains = [c for _, c in scored_chains[:5]]
    
    best_chain = top_chains[0]
    best_energy = scored_chains[0][0]
    
    # Phase 2: Simulated Annealing on top chains
    # Allocate time
    time_per_sa = _remaining_time() / len(top_chains)
    
    for chain in top_chains:
        if _remaining_time() < 1.0:
            break
        # Give each SA a slice of time
        # But we want to be fair. 
        # If we have 5 chains, and 100s left, each gets 20s.
        t_limit = min(_remaining_time() / len(top_chains), 30.0) # Cap at 30s per SA to avoid one taking too long
        new_chain, new_energy = _simulated_annealing(sequence, chain, t_limit, rng)
        if new_energy < best_energy:
            best_energy = new_energy
            best_chain = new_chain
            
    # Final validation
    if not _is_valid_chain(best_chain):
        # Fallback to a simple valid chain
        best_chain = [(i, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
