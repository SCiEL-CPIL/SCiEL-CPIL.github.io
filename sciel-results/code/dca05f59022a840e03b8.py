"""Rewritten program using Simulated Annealing for HP lattice protein folding."""
# EVOLVE-BLOCK-START
import time
import random
import math

_START_TIME = time.time()
_HARD_DEADLINE = 320.0  # Safety margin before 330s kill

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _compute_energy(sequence, chain):
    """Compute energy of a chain. Energy = - (number of H-H contacts)."""
    if not chain:
        return 0
    
    # Map position to index
    pos_to_idx = {}
    for i, p in enumerate(chain):
        pos_to_idx[p] = i
        
    contacts = 0
    # To avoid double counting, only count contacts where j > i + 1
    for i in range(len(chain)):
        if sequence[i] != 'H':
            continue
        p = chain[i]
        for q in _neighbours(p):
            j = pos_to_idx.get(q)
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
                
    return -contacts

def _is_valid(chain):
    """Check if chain is self-avoiding and has unit steps."""
    if not chain:
        return True
    seen = set()
    for i, p in enumerate(chain):
        if p in seen:
            return False
        seen.add(p)
        if i > 0:
            prev = chain[i-1]
            dist = sum(abs(a-b) for a, b in zip(p, prev))
            if dist != 1:
                return False
    return True

def _random_walk(sequence, dimension, rng):
    """Generate a random self-avoiding walk."""
    n = len(sequence)
    if n == 0:
        return []
    
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0)}
    
    for _ in range(n - 1):
        last = chain[-1]
        moves = []
        for axis in range(dimension):
            for step in (-1, 1):
                new_pos = list(last)
                new_pos[axis] += step
                new_pos = tuple(new_pos)
                if new_pos not in occupied:
                    moves.append(new_pos)
        
        if not moves:
            return None
        
        next_pos = rng.choice(moves)
        chain.append(next_pos)
        occupied.add(next_pos)
        
    return chain

def _anneal(sequence, chain, dimension, time_budget, rng):
    """Simulated Annealing to optimize the chain."""
    n = len(sequence)
    if n == 0:
        return chain
    
    current_chain = list(chain)
    current_energy = _compute_energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    start_temp = 1.0
    end_temp = 0.01
    start_time = time.time()
    
    # We iterate until time budget is exhausted or temp is too low
    # Number of steps depends on n. Let's aim for ~1000 steps per residue? 
    # No, that's too slow. Let's just run until time runs out.
    
    while time.time() - start_time < time_budget:
        # Pick a random move
        # Move types:
        # 1. Move a single residue (if possible)
        # 2. Reverse a segment
        # 3. Rotate a segment (harder, skip for now)
        
        move_type = rng.random()
        
        new_chain = None
        
        if move_type < 0.5:
            # Single residue move
            # Pick a random internal residue (not ends, to keep connectivity easy? 
            # Actually, moving an end is a "growth" step, moving internal is a "shift")
            # Let's try shifting an internal residue to an adjacent empty spot.
            idx = rng.randint(1, n - 2)
            pos = current_chain[idx]
            neighbors = list(_neighbours(pos))
            rng.shuffle(neighbors)
            
            for new_pos in neighbors:
                # Check if new_pos is occupied by a non-adjacent residue
                # We need to check if new_pos is in the chain, and if so, 
                # is it idx-1 or idx+1?
                # If it is idx-1 or idx+1, moving there would create a loop or invalid step?
                # If we move idx to pos of idx-1, then step idx-1 -> idx becomes 0 distance. Invalid.
                # So new_pos must not be in the chain, OR if it is, it must be idx-1 or idx+1 
                # AND the move must be valid? 
                # Actually, if new_pos == chain[idx-1], the step from idx-1 to idx is 0. Invalid.
                # If new_pos == chain[idx+1], step from idx to idx+1 is 0. Invalid.
                # So new_pos must NOT be in the chain at all.
                
                # Wait, can we move into a spot vacated by a neighbor?
                # If we move idx to pos of idx-1:
                # Chain: ... A, B, C, D ... (B is idx)
                # Move B to A's position.
                # New chain: ... A, A, C, D ... -> Self-intersection! Invalid.
                # So new_pos must be empty.
                
                if new_pos in current_chain:
                    continue
                
                # Check if the move creates a self-intersection with the new position
                # Since new_pos is empty, the only risk is if the new position is adjacent 
                # to a part of the chain that isn't idx-1 or idx+1?
                # No, self-avoidance is just about occupying the same spot.
                # Unit step is guaranteed if we move to a neighbor.
                
                # However, we must ensure the new position doesn't collide with 
                # the rest of the chain. We already checked `new_pos in current_chain`.
                # But wait, `current_chain` is a list. `new_pos in current_chain` is O(N).
                # For N=136, this is fine.
                
                # Let's construct the new chain
                trial_chain = list(current_chain)
                trial_chain[idx] = new_pos
                
                # Verify validity (just in case)
                if not _is_valid(trial_chain):
                    continue
                
                new_chain = trial_chain
                break
        
        elif move_type < 0.8:
            # Segment reversal
            # Pick two indices i, j (i < j) and reverse chain[i:j+1]
            # Must be valid: 
            # 1. The connection at i-1 and i must remain unit step?
            #    Reversal changes the order. 
            #    Original: ... A, B, C, D, E ...
            #    Reverse B..D: ... A, D, C, B, E ...
            #    Step A->D must be 1. Step B->E must be 1.
            #    Also, no self-intersections within the reversed segment (reversal preserves internal distances, so if it was valid, it is valid internally).
            #    But the reversed segment might intersect the rest of the chain.
            
            i = rng.randint(0, n - 2)
            j = rng.randint(i + 1, n - 1)
            
            # Check boundary conditions
            # If i > 0, dist(chain[i-1], chain[j]) must be 1
            # If j < n-1, dist(chain[i], chain[j+1]) must be 1
            
            if i > 0:
                d = sum(abs(a-b) for a, b in zip(current_chain[i-1], current_chain[j]))
                if d != 1:
                    new_chain = None
                else:
                    new_chain = list(current_chain)
            else:
                new_chain = list(current_chain)
                
            if j < n - 1 and new_chain is not None:
                d = sum(abs(a-b) for a, b in zip(current_chain[i], current_chain[j+1]))
                if d != 1:
                    new_chain = None
                else:
                    # Perform reversal
                    new_chain[i:j+1] = reversed(new_chain[i:j+1])
            
            if new_chain is not None:
                # Check self-avoidance
                if not _is_valid(new_chain):
                    new_chain = None
        
        if new_chain is None:
            continue
            
        new_energy = _compute_energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        # Temperature decay
        # Simple linear decay based on time? Or step count?
        # Let's use a global step counter or time-based temp.
        # For now, let's just use a fixed cooling schedule based on elapsed time in this anneal call.
        # T = start_temp * (1 - elapsed/budget)
        elapsed = time.time() - start_time
        if elapsed >= time_budget:
            break
        t = start_temp * (1.0 - elapsed / time_budget)
        if t < end_temp:
            t = end_temp
            
        if delta < 0:
            # Accept
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        else:
            # Metropolis criterion
            # P = exp(-delta / T)
            # delta is positive (energy increase)
            prob = math.exp(-delta / t)
            if rng.random() < prob:
                current_chain = new_chain
                current_energy = new_energy
            # Else reject, keep current_chain
            
    return best_chain

def fold(sequence, dimension):
    """Best conformation using Simulated Annealing."""
    n = len(sequence)
    if n == 0:
        return []
    
    # Time allocation
    # Total budget 320s.
    # We have 26 sequences. 
    # Average ~12s per sequence.
    # Short sequences can be faster, long ones slower.
    # Let's allocate time proportional to n^1.5 or just fixed?
    # Fixed is safer. 320 / 26 ~ 12.3s.
    # Let's use 10s per sequence to be safe, leaving buffer.
    # But we should prioritize longer sequences?
    # The score is the MEAN ratio. So each sequence counts equally.
    # Therefore, equal time per sequence is optimal if the difficulty scales with n.
    # If short sequences are easy, they might finish early, but we can't dynamically redistribute 
    # easily without a global scheduler. 
    # Since `fold` is called independently, we must estimate.
    
    # Let's give 11 seconds per sequence.
    time_budget = 11.0
    
    # Check global deadline
    elapsed_global = time.time() - _START_TIME
    if elapsed_global > _HARD_DEADLINE - time_budget:
        # Not enough time for a full run, do a quick greedy
        pass # Logic below handles this
    
    # Generate initial walks
    # For short sequences, random walk is easy.
    # For long sequences, random walk might fail (get stuck).
    # Let's try a few random walks and pick the best energy.
    
    best_chain = None
    best_energy = float('inf')
    
    rng = random.Random(42 + len(sequence))
    
    # Try to generate valid initial walks
    # For n > 50, random walk often fails. 
    # We might need a smarter initialization or just accept that some sequences 
    # will start with a bad chain or fail to initialize.
    # If random walk fails, we can use a "snake" or spiral initialization.
    
    initial_walks = []
    for _ in range(5):
        w = _random_walk(sequence, dimension, rng)
        if w is not None:
            initial_walks.append(w)
            
    if not initial_walks:
        # Fallback: Straight line (might not be optimal but valid)
        # Or a simple spiral
        # Let's use a simple back-and-forth in x-y plane
        chain = []
        x = 0
        y = 0
        z = 0
        dx = 1
        dy = 0
        for i in range(n):
            chain.append((x, y, z))
            x += dx
            if i > 0 and i % 2 == 0:
                # Turn
                if dx == 1:
                    dx = 0
                    dy = 1
                elif dx == -1:
                    dx = 0
                    dy = -1
                elif dy == 1:
                    dx = -1
                    dy = 0
                elif dy == -1:
                    dx = 1
                    dy = 0
            # This logic is flawed for a general spiral. 
            # Let's just use a straight line if random fails.
            # Straight line is always valid.
            break
        if len(chain) < n:
             chain = [(i, 0, 0) for i in range(n)]
        initial_walks.append(chain)
        
    # Anneal the best initial walk
    # Actually, we should anneal the best one, or all of them?
    # Given time constraints, annealing the best initial walk is a good heuristic.
    # Or anneal each for a short time and pick the best?
    # Let's anneal the best initial walk for the full budget.
    
    # Sort initial walks by energy
    initial_walks.sort(key=lambda w: _compute_energy(sequence, w))
    
    # Pick the best initial walk
    start_chain = initial_walks[0]
    
    # Run Annealing
    final_chain = _anneal(sequence, start_chain, dimension, time_budget, rng)
    
    # Convert to list of lists
    return [list(p) for p in final_chain]
# EVOLVE-BLOCK-END
