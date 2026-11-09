"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

# Global time management
_TOTAL_DEADLINE = 205.0  # Safety margin before 210s kill
_START_TIME = time.time()

def _get_remaining_time():
    return _TOTAL_DEADLINE - (time.time() - _START_TIME)

def _neighbours(point):
    """Yield the 4 neighbors of a 2D point."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _rotate_around(point, center, rot):
    """
    Rotate point around center by rot degrees (0, 90, 180, 270).
    Returns the new point.
    """
    x, y = point[0] - center[0], point[1] - center[1]
    if rot == 0:
        nx, ny = x, y
    elif rot == 90:
        nx, ny = -y, x
    elif rot == 180:
        nx, ny = -x, -y
    elif rot == 270:
        nx, ny = y, -x
    else:
        nx, ny = x, y
    return (center[0] + nx, center[1] + ny)

def _initial_grow(sequence, rng):
    """
    Fast randomized greedy growth to get a valid initial SAW.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    for i in range(1, n):
        last = chain[-1]
        # Get unoccupied neighbors
        options = []
        for nb in _neighbours(last):
            if nb not in occupied:
                # Heuristic: prefer H-H contacts
                gain = 0
                if sequence[i] == 'H':
                    for nn in _neighbours(nb):
                        if nn in occupied and sequence[occupied[nn]] == 'H' and occupied[nn] < i - 1:
                            gain += 1
                # Add some randomness
                score = gain + rng.random()
                options.append((score, nb))
        
        if not options:
            # If stuck, return None (caller will handle fallback)
            return None
        
        # Pick the best option
        options.sort(reverse=True)
        next_pt = options[0][1]
        chain.append(next_pt)
        occupied[next_pt] = i
        
    return chain

def _calculate_energy(sequence, chain):
    """
    Calculate the energy of the conformation.
    Energy = - (number of H-H contacts)
    """
    n = len(chain)
    if n < 2:
        return 0
        
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    contacts = 0
    
    for i in range(n):
        if sequence[i] == 'H':
            p = chain[i]
            for nb in _neighbours(p):
                if nb in pos_to_idx:
                    j = pos_to_idx[nb]
                    if j > i + 1: # Avoid double counting and adjacent chain bonds
                        if sequence[j] == 'H':
                            contacts += 1
                            
    return -contacts

def _simulated_annealing(sequence, initial_chain, time_budget):
    """
    Run Simulated Annealing starting from initial_chain.
    Uses Pivot Moves.
    """
    n = len(sequence)
    if n <= 2:
        return initial_chain, _calculate_energy(sequence, initial_chain)
        
    # Deep copy
    current_chain = list(initial_chain)
    current_energy = _calculate_energy(sequence, current_chain)
    
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Occupied set for O(1) lookups
    occupied = {p: i for i, p in enumerate(current_chain)}
    
    # SA Parameters
    # Adjust temperature based on sequence length?
    # Standard cooling: T = T0 * alpha^step
    T = 2.0
    alpha = 0.999
    min_T = 0.01
    
    start_time = time.time()
    rng = random.Random()
    
    while T > min_T and (time.time() - start_time) < time_budget:
        # Check global deadline
        if _get_remaining_time() < 1.0:
            break
            
        # Pick a random pivot index
        # Pivot around chain[i-1], moving chain[i:]
        # i must be at least 1
        i = rng.randint(1, n - 1)
        if i == 1:
             # Pivoting the whole tail is risky for self-avoidance but allowed
             pass
             
        center = current_chain[i-1]
        tail = current_chain[i:]
        
        # Try 3 rotations: 90, 180, 270
        # We pick one at random to save time, or try all?
        # Trying all 3 is better for exploration.
        best_move = None
        best_delta_e = float('inf')
        best_new_chain = None
        best_new_occupied = None
        
        for rot in (90, 180, 270):
            new_tail = [_rotate_around(p, center, rot) for p in tail]
            
            # Check self-avoidance for the new tail
            # 1. New tail must not collide with itself
            # 2. New tail must not collide with the fixed head (chain[0:i])
            # Note: The first point of the new tail is adjacent to center, which is in head.
            # But we must ensure new_tail[0] is not in head (except center itself, but center is i-1, new_tail[0] is i).
            # Actually, new_tail[0] is the new position of chain[i].
            # It must not be in {chain[0], ..., chain[i-2]}
            
            valid = True
            seen_in_tail = set()
            
            # Check against head
            # We can optimize by just checking if any point in new_tail is in occupied_head
            # But occupied includes the tail we are moving.
            # Let's create a set of head points: {chain[0] ... chain[i-1]}
            # Actually, we can just check if any point in new_tail is in occupied
            # BUT we must exclude the points that are currently in the tail (which are being moved).
            # So, remove tail points from occupied, check new_tail, then add back?
            # That's O(N) per rotation.
            
            # Efficient check:
            # 1. Remove old tail from occupied
            for p in tail:
                del occupied[p]
            
            # 2. Check new_tail for collisions with remaining occupied (head)
            # and self-collisions
            for p in new_tail:
                if p in occupied:
                    valid = False
                    break
                if p in seen_in_tail:
                    valid = False
                    break
                seen_in_tail.add(p)
            
            if valid:
                # Calculate Delta E
                # Only contacts involving the moved segment or its neighbors can change.
                # For simplicity and given N<=100, let's just recalculate energy if valid?
                # No, let's try to be smart.
                # But recalculation is O(N). 3 rotations * O(N) * Steps.
                # If steps are ~10k, 3 * 100 * 10k = 3M ops. Fine.
                
                # Add new tail to occupied for energy calc?
                # We need the full chain to calc energy.
                new_chain = current_chain[:i] + new_tail
                new_energy = _calculate_energy(sequence, new_chain)
                delta_e = new_energy - current_energy
                
                if delta_e < best_delta_e:
                    best_delta_e = delta_e
                    best_new_chain = new_chain
                    best_new_occupied = {p: idx for idx, p in enumerate(new_chain)}
                    # We don't update occupied yet, we do it if accepted
            else:
                # Invalid, restore occupied
                for p in tail:
                    occupied[p] = current_chain.index(p) # This is O(N^2), bad.
                
            # Restore occupied for next rotation attempt
            # Since we deleted them, we must put them back
            for p in tail:
                 # Find index? We know indices are i to n-1
                 pass
            
            # This manual set management is error-prone and slow in Python.
            # Let's simplify: Just check validity using a temporary set.
            
        # The above loop is messy. Let's rewrite the move generation to be cleaner.
        
        # Restart the move selection logic with a cleaner approach
        # 1. Identify head and tail
        head = current_chain[:i]
        tail = current_chain[i:]
        
        # 2. Create a set of head points for collision check
        head_set = set(head)
        
        for rot in (90, 180, 270):
            new_tail = [_rotate_around(p, center, rot) for p in tail]
            
            # Check self-avoidance of new_tail
            new_tail_set = set()
            valid = True
            for p in new_tail:
                if p in new_tail_set:
                    valid = False
                    break
                new_tail_set.add(p)
                if p in head_set:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Calculate Energy
            new_chain = head + new_tail
            new_energy = _calculate_energy(sequence, new_chain)
            delta_e = new_energy - current_energy
            
            if delta_e < best_delta_e:
                best_delta_e = delta_e
                best_new_chain = new_chain
                # We will update occupied if accepted
        
        if best_new_chain is not None:
            # Acceptance criterion
            if best_delta_e < 0:
                accept = True
            else:
                # Boltzmann probability
                # Avoid overflow
                if T < 1e-5:
                    accept = False
                else:
                    accept = rng.random() < math.exp(-best_delta_e / T)
            
            if accept:
                current_chain = best_new_chain
                current_energy = current_energy + best_delta_e
                # Update occupied
                occupied = {p: idx for idx, p in enumerate(current_chain)}
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
        
        T *= alpha
        
    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Best conformation using multiple restarts of Greedy + Simulated Annealing.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    # Distribute time budget
    # We have ~12s per sequence on average.
    # Let's allocate 10s for this sequence to be safe.
    time_budget = min(10.0, _get_remaining_time() / 2.0) # Conservative
    
    if time_budget <= 0:
        # Fallback to straight line
        return [[i, 0] for i in range(n)]
        
    best_chain = None
    best_energy = 0 # Energy is negative, so 0 is worse than -1
    
    rng = random.Random(12345 + n)
    
    # Number of restarts depends on time
    # Let's do a few quick restarts
    restarts = 5
    
    for _ in range(restarts):
        if _get_remaining_time() < 1.0:
            break
            
        init_chain = _initial_grow(sequence, rng)
        if init_chain is None:
            # Try a simple spiral or line if greedy fails
            # For small N, line is safe
            init_chain = [[i, 0] for i in range(n)]
            init_chain = [tuple(p) for p in init_chain]
            
        # Run SA with a fraction of the time
        sa_time = time_budget / restarts
        chain, energy = _simulated_annealing(sequence, init_chain, sa_time)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
            
    if best_chain is None:
        best_chain = [[i, 0] for i in range(n)]
    else:
        best_chain = [list(p) for p in best_chain]
        
    return best_chain
# EVOLVE-BLOCK-END
