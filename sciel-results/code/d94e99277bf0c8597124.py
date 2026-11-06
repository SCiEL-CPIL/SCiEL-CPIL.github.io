"""
HP lattice protein folding using fast Simulated Annealing with incremental energy updates.
"""
# EVOLVE-BLOCK-START
import random
import time
import math

def fold(sequence, dimension):
    start_time = time.time()
    deadline = start_time + 320  # Safety margin before 330s kill
    
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    # 1. Initial Chain: Straight line
    chain = [(i, 0, 0) for i in range(n)]
    occupied = {p: i for i, p in enumerate(chain)}
    
    # 2. Initial Energy Calculation
    # Energy = - (number of H-H contacts)
    # Contacts are pairs (i, j) such that |i - j| > 1, sequence[i]=='H', sequence[j]=='H', and dist(chain[i], chain[j]) == 1
    # We store contacts as a set of tuples (i, j) with i < j
    contacts = set()
    for i in range(n):
        if sequence[i] == 'H':
            p = chain[i]
            for axis in range(dimension):
                for step in (-1, 1):
                    q = list(p)
                    q[axis] += step
                    q = tuple(q)
                    j = occupied.get(q)
                    if j is not None and j > i + 1 and sequence[j] == 'H':
                        contacts.add((i, j))
    
    current_energy = -len(contacts)
    best_chain = list(chain)
    best_energy = current_energy
    
    # Precompute H positions for quick lookup? No, sequence is string, index is fast.
    
    # Helper: Check if a point is a valid neighbor for a move
    # A move is valid if:
    # 1. The new position is not occupied by another residue (except the moving one).
    # 2. The chain remains connected (unit steps).
    
    # We will use a set of valid move types:
    # 1. Rotation (corner move): i-1, i, i+1 form an L. Move i to the 4th corner of the square.
    # 2. Slide: i-1, i, i+1 form a straight line. Move i to a side.
    # 3. Pivot (end move): Move residue 0 or n-1 to a free neighbor.
    
    def get_valid_moves(i):
        """Yield valid new positions for residue i."""
        if i == 0 or i == n - 1:
            # Pivot move
            p = chain[i]
            for axis in range(dimension):
                for step in (-1, 1):
                    q = list(p)
                    q[axis] += step
                    q = tuple(q)
                    if q not in occupied:
                        # Check connectivity: must be adjacent to the other end
                        other_idx = 1 if i == 0 else n - 2
                        other_p = chain[other_idx]
                        if sum(abs(a-b) for a, b in zip(q, other_p)) == 1:
                            yield q
        else:
            p_prev = chain[i-1]
            p_curr = chain[i]
            p_next = chain[i+1]
            
            # Check if straight or corner
            # Vector prev->curr
            v1 = [p_curr[a] - p_prev[a] for a in range(dimension)]
            # Vector curr->next
            v2 = [p_next[a] - p_curr[a] for a in range(dimension)]
            
            # If straight (v1 == -v2)
            if all(v1[a] == -v2[a] for a in range(dimension)):
                # Slide move: move curr perpendicular to the line
                for axis in range(dimension):
                    if v1[axis] == 0:
                        for step in (-1, 1):
                            q = list(p_curr)
                            q[axis] += step
                            q = tuple(q)
                            if q not in occupied:
                                yield q
            else:
                # Corner move: v1 != -v2 (must be orthogonal)
                # Find the 4th corner
                # p_curr is the corner. p_prev and p_next are the adjacent sides.
                # The 4th corner is p_prev + v2 (or p_next + v1)
                # Actually, if p_prev is (0,0,0), p_curr is (1,0,0), p_next is (1,1,0).
                # v1 = (1,0,0), v2 = (0,1,0).
                # The square is (0,0,0), (1,0,0), (1,1,0), (0,1,0).
                # The missing corner is (0,1,0).
                # q = p_prev + v2
                q = tuple(p_prev[a] + v2[a] for a in range(dimension))
                if q not in occupied:
                    yield q
    
    def calc_delta_energy(i, old_p, new_p):
        """Calculate change in energy when moving residue i from old_p to new_p."""
        delta = 0
        if sequence[i] == 'H':
            # Remove contacts with old neighbors
            for axis in range(dimension):
                for step in (-1, 1):
                    q = list(old_p)
                    q[axis] += step
                    q = tuple(q)
                    j = occupied.get(q)
                    if j is not None and j != i-1 and j != i+1 and sequence[j] == 'H':
                        # Check if this contact is in the set
                        pair = (min(i, j), max(i, j))
                        if pair in contacts:
                            delta += 1 # Losing a contact increases energy (less negative)
            
            # Add contacts with new neighbors
            for axis in range(dimension):
                for step in (-1, 1):
                    q = list(new_p)
                    q[axis] += step
                    q = tuple(q)
                    j = occupied.get(q)
                    if j is not None and j != i-1 and j != i+1 and sequence[j] == 'H':
                        # Check if this contact is NOT already in the set
                        pair = (min(i, j), max(i, j))
                        if pair not in contacts:
                            delta -= 1 # Gaining a contact decreases energy (more negative)
        
        return delta
    
    # Simulated Annealing Parameters
    # Time budget allocation:
    # We want to run many steps.
    # Let's estimate steps/sec. A step involves:
    # - Generating moves (fast)
    # - Delta energy (fast, 6 neighbors * 2)
    # - Updating state (fast)
    # We can likely do 50k-100k steps per second in Python.
    # 300 seconds -> ~3M steps.
    
    # We will use a time-based loop instead of step count to be robust.
    
    # Cooling schedule:
    # T_start = 1.0 (since delta E is integer, T=1 allows ~37% uphill moves for dE=1)
    # T_end = 0.05
    
    t_start = 1.0
    t_end = 0.05
    # We don't know total steps in advance, so we use a time-based cooling?
    # Or we can just run a fixed number of steps and check time.
    # Better: Run until deadline.
    # Use exponential decay based on elapsed time?
    # Or just a fixed number of steps per "phase" and restart if stuck?
    
    # Strategy: Single long SA run with time-based cooling.
    # T(t) = T_start * (T_end/T_start)^(t/t_total)
    # But t_total is unknown.
    # Alternative: Fixed step count per "epoch", reduce T, repeat.
    
    # Let's do:
    # While time < deadline:
    #   Run 10000 steps at current T
    #   T *= 0.95
    #   If T < T_end: T = T_end (or restart with higher T if stuck?)
    
    # To avoid getting stuck in local minima, we can occasionally restart from best_chain if no improvement for a long time.
    
    current_T = t_start
    no_improve_steps = 0
    improvement_threshold = 50000 # If no improvement in 50k steps, consider restart
    
    while time.time() < deadline:
        # Run a batch of steps
        steps_in_batch = 5000
        for _ in range(steps_in_batch):
            if time.time() > deadline:
                break
            
            # Pick a random residue to move
            i = random.randint(0, n - 1)
            
            # Get valid moves
            moves = list(get_valid_moves(i))
            if not moves:
                continue
            
            new_p = random.choice(moves)
            old_p = chain[i]
            
            # Calculate delta E
            dE = calc_delta_energy(i, old_p, new_p)
            
            # Metropolis criterion
            if dE < 0 or random.random() < math.exp(-dE / current_T):
                # Accept move
                # Update state
                del occupied[old_p]
                chain[i] = new_p
                occupied[new_p] = i
                
                # Update contacts
                if sequence[i] == 'H':
                    # Remove old contacts
                    for axis in range(dimension):
                        for step in (-1, 1):
                            q = list(old_p)
                            q[axis] += step
                            q = tuple(q)
                            j = occupied.get(q)
                            if j is not None and j != i-1 and j != i+1 and sequence[j] == 'H':
                                pair = (min(i, j), max(i, j))
                                if pair in contacts:
                                    contacts.remove(pair)
                    
                    # Add new contacts
                    for axis in range(dimension):
                        for step in (-1, 1):
                            q = list(new_p)
                            q[axis] += step
                            q = tuple(q)
                            j = occupied.get(q)
                            if j is not None and j != i-1 and j != i+1 and sequence[j] == 'H':
                                pair = (min(i, j), max(i, j))
                                if pair not in contacts:
                                    contacts.add(pair)
                
                current_energy += dE
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
            else:
                no_improve_steps += 1
        
        # Cool down
        current_T *= 0.98
        if current_T < t_end:
            current_T = t_end
            
        # Restart logic
        if no_improve_steps > improvement_threshold:
            # Restart from best_chain
            chain = list(best_chain)
            occupied = {p: i for i, p in enumerate(chain)}
            
            # Recalculate contacts for best_chain
            contacts = set()
            for i in range(n):
                if sequence[i] == 'H':
                    p = chain[i]
                    for axis in range(dimension):
                        for step in (-1, 1):
                            q = list(p)
                            q[axis] += step
                            q = tuple(q)
                            j = occupied.get(q)
                            if j is not None and j > i + 1 and sequence[j] == 'H':
                                contacts.add((i, j))
            
            current_energy = best_energy
            current_T = t_start * 0.5 # Restart with slightly lower T
            no_improve_steps = 0
            
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
