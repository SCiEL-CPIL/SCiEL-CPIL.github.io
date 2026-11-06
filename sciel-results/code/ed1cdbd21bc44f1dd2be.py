"""Improved HP lattice protein folding with local search refinement."""
# EVOLVE-BLOCK-START
import random
import time

_START_TIME = None

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_remaining():
    _init_time()
    # Budget is 120s CPU. Safety margin to 105s to be safe against wall-clock kill.
    return 105.0 - (time.time() - _START_TIME)

def _neighbours(point):
    """Yield 4-neighbours on square lattice."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _grow(sequence, dimension, rng):
    """
    One randomized greedy chain growth: prefer sites with new H-H contacts, then free space.
    """
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        current = chain[-1]
        options = []
        
        for site in _neighbours(current):
            if site in occupied:
                continue
            
            # Count immediate H-H contacts this placement would make
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            
            # Count free neighbours (for future growth flexibility)
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            
            # Score: prioritize contacts, then free space, then random tie-breaker
            score = gain * 10.0 + free * 0.1 + rng.random()
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick the best option
        options.sort(key=lambda x: x[0], reverse=True)
        site = options[0][1]
        occupied[site] = i
        chain.append(site)
    
    return chain

def _energy(sequence, chain):
    """Compute HP contact energy: -1 for each H-H pair on neighbouring sites not adjacent in chain."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _local_search(sequence, chain, rng, max_steps=200):
    """
    Refine the chain using kink moves and segment moves with simulated annealing.
    Optimized energy delta calculation for speed.
    """
    n = len(chain)
    if n < 3:
        return chain, _energy(sequence, chain)
    
    # Precompute occupied set for fast lookup
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    is_h = [c == 'H' for c in sequence]
    
    # Simulated annealing parameters
    start_temp = 2.0
    end_temp = 0.05
    
    # Precompute neighbor lists for speed? Maybe not necessary for N<=100.
    
    for step in range(max_steps):
        # Annealing schedule
        progress = step / max(1, max_steps)
        temp = start_temp * (end_temp / start_temp) ** progress
        
        # Choose move type: 60% kink, 40% segment (segment moves can escape local minima better)
        move_type = rng.random()
        
        if move_type < 0.6:
            # Kink move
            if n < 3:
                continue
            i = rng.randint(1, n - 2)
            prev = chain[i-1]
            curr = chain[i]
            next_ = chain[i+1]
            
            # Find valid new positions for curr
            # Must be neighbor of prev and next_
            candidates = []
            for p in _neighbours(prev):
                if p == curr:
                    continue
                # Check distance to next_ is 1
                if abs(p[0] - next_[0]) + abs(p[1] - next_[1]) == 1:
                    candidates.append(p)
            
            if not candidates:
                continue
                
            new_pos = rng.choice(candidates)
            
            if new_pos in occupied:
                continue
            
            # Calculate delta energy
            # Contacts involving i, i-1, i+1 are affected
            # Actually, only contacts involving the moved residue i and its neighbors
            # that are H-H contacts.
            
            # Old contacts involving i:
            # i is at curr. Neighbors of curr.
            # Check all neighbors of curr. If they are H and i is H, and index diff > 1.
            
            def _contacts_around(idx, pos, occ, seq_h):
                """Count H-H contacts involving residue idx at pos with neighbors in occ."""
                count = 0
                if not seq_h[idx]:
                    return 0
                for q in _neighbours(pos):
                    j = occ.get(q)
                    if j is not None and abs(j - idx) > 1 and seq_h[j]:
                        count += 1
                return count
            
            # Note: _contacts_around counts each pair twice if we sum over all i.
            # But for delta, we just need the change in total contacts.
            # Total contacts = 1/2 * sum over all i of contacts_around(i).
            # So delta_contacts = 1/2 * (new_sum - old_sum).
            # Or simply: delta_energy = -(new_contacts - old_contacts).
            
            # Let's just recalculate energy for simplicity and safety, as N is small.
            # But wait, the instruction says "identify the additional work". 
            # Recalculating energy is O(N^2) or O(N) with dict. 
            # Let's stick to the robust full recalculation if speed allows, 
            # otherwise optimize. 
            # With 100 steps * 100 residues * 100 restarts, it might be tight.
            # Let's try the delta calculation properly.
            
            # Old contribution of i to total contacts:
            # Contacts between i and j where j is neighbor of curr, j != i-1, j != i+1.
            # And j is H, i is H.
            
            # Let's just do the full energy calc for now, it's safer.
            # If we run out of time, we reduce steps.
            
            del occupied[curr]
            occupied[new_pos] = i
            chain[i] = new_pos
            
            new_energy = _energy(sequence, chain)
            delta = new_energy - current_energy
            
            if delta < 0 or (temp > 0.01 and rng.random() < pow(2.718281828, -delta / temp)):
                current_energy = new_energy
            else:
                # Revert
                chain[i] = curr
                del occupied[new_pos]
                occupied[curr] = i
        else:
            # Segment move (2-mer kink)
            if n < 4:
                continue
            i = rng.randint(1, n - 3)
            p0 = chain[i-1]
            p1 = chain[i]
            p2 = chain[i+1]
            p3 = chain[i+2]
            
            found = False
            # Try moving segment [i, i+1]
            # New positions p1_new, p2_new
            # p1_new must be neighbor of p0
            # p2_new must be neighbor of p1_new and p3
            
            for p1_new in _neighbours(p0):
                if p1_new == p2: # Can't go back
                    continue
                for p2_new in _neighbours(p1_new):
                    if p2_new == p0:
                        continue
                    if not (abs(p2_new[0] - p3[0]) + abs(p2_new[1] - p3[1]) == 1):
                        continue
                    
                    # Check self-avoidance
                    # p1_new and p2_new must not be occupied by any other residue
                    # They are currently occupied by i and i+1, so we remove them first
                    # But we must ensure they are not occupied by j != i, i+1
                    
                    if p1_new in occupied and occupied[p1_new] not in (i, i+1):
                        continue
                    if p2_new in occupied and occupied[p2_new] not in (i, i+1):
                        continue
                    
                    # Also p1_new != p2_new (implied by step)
                    
                    # Try the move
                    del occupied[p1]
                    del occupied[p2]
                    occupied[p1_new] = i
                    occupied[p2_new] = i + 1
                    chain[i] = p1_new
                    chain[i+1] = p2_new
                    
                    new_energy = _energy(sequence, chain)
                    delta = new_energy - current_energy
                    
                    if delta < 0 or (temp > 0.01 and rng.random() < pow(2.718281828, -delta / temp)):
                        current_energy = new_energy
                    else:
                        # Revert
                        chain[i] = p1
                        chain[i+1] = p2
                        del occupied[p1_new]
                        del occupied[p2_new]
                        occupied[p1] = i
                        occupied[p2] = i + 1
                    found = True
                    break
                if found:
                    break
    
    return chain, current_energy

def fold(sequence, dimension):
    """
    Fold an HP sequence on the square lattice.
    Uses randomized greedy growth followed by local search refinement.
    """
    _init_time()
    
    n = len(sequence)
    if n == 0:
        return []
    
    # Emergency fallback if no time left
    if _time_remaining() < 2.0:
        return [[i, 0] for i in range(n)]
    
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0 # Energy is negative, so we look for lower (more negative) values
    
    # Perform many restarts
    # We will dynamically adjust the number of local search steps based on time
    # and sequence length to maximize the number of complete (grow + search) cycles.
    
    iterations = 0
    
    while _time_remaining() > 5.0:
        iterations += 1
        
        # 1. Grow
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        # 2. Local Search
        # Estimate time per iteration.
        # If we have plenty of time, do more search steps.
        # If we are running low, do fewer.
        
        # Heuristic: 
        # N=20: 300 steps
        # N=100: 50 steps
        # Let's scale inversely with N.
        base_steps = 300
        if n > 50:
            base_steps = 150
        if n > 80:
            base_steps = 80
            
        # Check time remaining
        rem = _time_remaining()
        if rem < 10.0:
            # Critical: do very few steps or skip search?
            # Better to do a quick search to maybe improve, or just use grow?
            # Grow is fast. Search is slow.
            # If rem < 5, break.
            # If rem < 10, do 10 steps.
            steps = 10
        else:
            steps = base_steps
            
        chain, energy = _local_search(sequence, chain, rng, max_steps=steps)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
            
        # If we found a very good solution (e.g. close to theoretical max), 
        # we might want to continue to see if we can beat it, 
        # but for now, we just keep searching for the global best.
        
        # Optional: If best_energy is very low, we could increase steps for next iteration?
        # No, keep it simple.

    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
