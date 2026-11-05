import random
import time
import math

def _neighbours(point):
    for axis in range(3):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _generate_random_saw(n, max_attempts=1000):
    for _ in range(max_attempts):
        chain = [(0, 0, 0)]
        occupied = {(0, 0, 0)}
        success = True
        for i in range(1, n):
            last = chain[-1]
            options = [p for p in _neighbours(last) if p not in occupied]
            if not options:
                success = False
                break
            next_pos = random.choice(options)
            chain.append(next_pos)
            occupied.add(next_pos)
        if success:
            return chain
    # Fallback
    side = int(math.ceil(n ** (1.0 / 3.0)))
    chain = []
    occupied = set()
    x = y = z = 0
    for i in range(n):
        pos = (x, y, z)
        if pos in occupied:
            break
        chain.append(pos)
        occupied.add(pos)
        x += 1
        if x > side:
            x = 0
            y += 1
            if y > side:
                y = 0
                z += 1
    while len(chain) < n:
        pos = (0, 0, z)
        if pos in occupied:
            z += 1
            continue
        chain.append(pos)
        occupied.add(pos)
    return chain

def _calculate_energy(sequence, chain, pos_to_idx):
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = pos_to_idx.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    
    start_time = time.time()
    deadline = start_time + 320.0
    
    best_chain = None
    best_energy = 0
    
    initial_chain = _generate_random_saw(n)
    pos_to_idx = {p: i for i, p in enumerate(initial_chain)}
    occupied = set(initial_chain)
    current_energy = _calculate_energy(sequence, initial_chain, pos_to_idx)
    
    best_chain = list(initial_chain)
    best_energy = current_energy
    
    T_start = 5.0
    T_end = 0.05
    T = T_start
    
    # Precompute H indices for faster check?
    h_indices = [i for i, s in enumerate(sequence) if s == 'H']
    
    while time.time() < deadline:
        # Decay T
        T = max(T_end, T * 0.9995)
        
        # Choose move type: 80% Local, 20% Pivot
        if random.random() < 0.8:
            # Local Move
            i = random.randint(0, n - 1)
            pos = initial_chain[i]
            
            # Find valid neighbors
            candidates = []
            for nb in _neighbours(pos):
                if nb not in occupied:
                    candidates.append(nb)
            
            if not candidates:
                continue
                
            new_pos = random.choice(candidates)
            
            # Calculate Delta E
            # Remove contacts from pos
            delta_e = 0
            for nb in _neighbours(pos):
                if nb in occupied:
                    j = pos_to_idx.get(nb)
                    if j is not None and abs(i - j) > 1:
                        if sequence[i] == 'H' and sequence[j] == 'H':
                            delta_e += 1 # Removing a contact increases energy (less negative)
            
            # Add contacts at new_pos
            for nb in _neighbours(new_pos):
                if nb in occupied:
                    j = pos_to_idx.get(nb)
                    if j is not None and abs(i - j) > 1:
                        if sequence[i] == 'H' and sequence[j] == 'H':
                            delta_e -= 1 # Adding a contact decreases energy (more negative)
            
            # Acceptance
            if delta_e < 0 or random.random() < math.exp(-delta_e / T):
                # Execute move
                occupied.remove(pos)
                occupied.add(new_pos)
                pos_to_idx.pop(pos)
                pos_to_idx[new_pos] = i
                initial_chain[i] = new_pos
                
                current_energy += delta_e
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(initial_chain)
        else:
            # Pivot Move
            if n < 5: continue
            i = random.randint(0, n - 2) # Pivot at i, move i+1..n-1
            # Choose rotation
            # 12 rotations
            axis = random.randint(0, 2)
            sign = random.choice([-1, 1])
            # Rotation logic:
            # We rotate the vector (p - pivot) for all p in chain[i+1:]
            # Let's define the rotation function
            def rotate(p, pivot, axis, sign):
                dx = p[0] - pivot[0]
                dy = p[1] - pivot[1]
                dz = p[2] - pivot[2]
                if axis == 0:
                    # Rotate around X
                    # (dy, dz) -> (sign*dz, -sign*dy)
                    new_dy = sign * dz
                    new_dz = -sign * dy
                    return (pivot[0], pivot[1] + new_dy, pivot[2] + new_dz)
                elif axis == 1:
                    # Rotate around Y
                    # (dx, dz) -> (sign*dz, -sign*dx)
                    new_dx = sign * dz
                    new_dz = -sign * dx
                    return (pivot[0] + new_dx, pivot[1], pivot[2] + new_dz)
                else:
                    # Rotate around Z
                    # (dx, dy) -> (sign*dy, -sign*dx)
                    new_dx = sign * dy
                    new_dy = -sign * dx
                    return (pivot[0] + new_dx, pivot[1] + new_dy, pivot[2])
            
            pivot = initial_chain[i]
            # Check collisions
            # We need to check if any new position is in occupied (excluding the moved part)
            # To do this efficiently, we can just simulate the move and check.
            
            # Get the moved part
            moved_indices = range(i + 1, n)
            new_positions = [rotate(initial_chain[j], pivot, axis, sign) for j in moved_indices]
            
            # Check if any new position is in occupied
            # We must exclude the positions of the moved part from the check
            # So we temporarily remove them? Or check against occupied - set(old_moved)
            
            old_moved_positions = {initial_chain[j] for j in moved_indices}
            valid = True
            for p in new_positions:
                if p in occupied and p not in old_moved_positions:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Calculate Delta E
            # Remove contacts involving moved part with fixed part
            # Add contacts involving moved part with fixed part
            
            # To do this, we can just calculate the energy of the new chain?
            # No, that's O(N).
            # Let's just do O(N) for pivot moves since they are less frequent.
            
            # Construct new chain
            new_chain = initial_chain[:i+1] + new_positions
            
            # Update pos_to_idx and occupied?
            # It's safer to just calculate energy from scratch for pivot moves
            # and only update state if accepted.
            
            new_pos_to_idx = {p: idx for idx, p in enumerate(new_chain)}
            new_energy = _calculate_energy(sequence, new_chain, new_pos_to_idx)
            
            delta_e = new_energy - current_energy
            
            if delta_e < 0 or random.random() < math.exp(-delta_e / T):
                # Accept
                initial_chain = new_chain
                pos_to_idx = new_pos_to_idx
                occupied = set(new_chain)
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(initial_chain)

    return [list(p) for p in best_chain]
