import random
import time
import math

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _random_saw(n, dimension, rng):
    """Generate a random self-avoiding walk."""
    if n == 0:
        return []
    chain = [(0,) * dimension]
    occupied = {chain[0]: 0}
    for i in range(1, n):
        options = []
        for site in _neighbours(chain[-1]):
            if site not in occupied:
                options.append(site)
        if not options:
            return None
        site = rng.choice(options)
        chain.append(site)
        occupied[site] = i
    return chain

def _pivot_move(chain, k, axis, angle, dimension):
    """
    Perform a pivot move.
    Cut at k. Rotate segment k..N-1 around the bond k-1 -> k.
    axis: unit vector of the bond (e.g., (1,0,0))
    angle: 90, 180, 270
    Returns new chain or None if collision.
    """
    n = len(chain)
    if k <= 0 or k >= n:
        return None
    
    # The bond is from chain[k-1] to chain[k]
    # The axis of rotation is this bond direction.
    # Since we rotate around the bond, chain[k] stays in place relative to chain[k-1].
    # Actually, chain[k] is the first point of the moving segment.
    # The axis passes through chain[k-1] and chain[k].
    # So chain[k] is on the axis. It does not move.
    
    # We need to rotate points chain[k+1] ... chain[n-1]
    # around the line passing through chain[k-1] and chain[k].
    
    # For simplicity, we can translate the segment so chain[k-1] is at origin?
    # No, let's just use the rotation logic directly.
    
    # Axis is one of (1,0,0), (-1,0,0), etc.
    # Let's determine the axis index.
    axis_idx = -1
    for i in range(dimension):
        if axis[i] != 0:
            axis_idx = i
            break
            
    if axis_idx == -1:
        return None
        
    # Rotation matrices for 90, 180, 270 degrees around axis_idx
    # We only need to rotate the coordinates perpendicular to the axis.
    
    new_chain = list(chain)
    
    # Points k and k-1 are fixed (k-1 is in fixed part, k is on axis)
    # Wait, is k fixed?
    # The axis is the bond k-1 -> k.
    # Point k is on the axis. So it stays put.
    # Points > k are rotated.
    
    # Let's extract the moving part
    moving = chain[k+1:]
    
    # We need to rotate these points around the line defined by chain[k-1] and chain[k].
    # Let's translate the system so chain[k-1] is at origin.
    # Then the axis is along the vector chain[k] - chain[k-1].
    # Since it's a lattice bond, this vector is a unit vector along an axis.
    
    # Let's just apply the rotation to the relative coordinates.
    # Relative to chain[k-1]:
    # p_rel = p - chain[k-1]
    # Rotate p_rel
    # p_new = chain[k-1] + p_rot
    
    # Rotation around axis_idx by angle:
    # If axis_idx == 0 (x-axis):
    #   90: (x, y, z) -> (x, -z, y)
    #   180: (x, y, z) -> (x, -y, -z)
    #   270: (x, y, z) -> (x, z, -y)
    # Similar for y and z.
    
    # We need to be careful with the direction of the axis.
    # The bond direction is `axis`.
    # If axis is (1,0,0), rotation is standard.
    # If axis is (-1,0,0), rotation is inverted?
    # Actually, rotating around a line is independent of the direction of the vector defining the line,
    # EXCEPT for the sign of the angle.
    # 90 deg around +x is different from 90 deg around -x.
    # But since we try all angles (90, 180, 270), and we try all cuts,
    # we can just pick a canonical axis direction (e.g., positive) and adjust angle sign?
    # Or simpler: just define rotation based on the actual axis vector.
    
    # Let's implement a generic rotation around a lattice axis.
    
    def rotate_point(p, origin, axis_vec, angle):
        # p, origin, axis_vec are tuples of length dimension
        # Translate p relative to origin
        rel = tuple(p[i] - origin[i] for i in range(dimension))
        
        # Determine the two perpendicular axes
        # If axis is x (idx 0), perpendicular are y(1), z(2)
        # If axis is y (idx 1), perpendicular are x(0), z(2)
        # If axis is z (idx 2), perpendicular are x(0), y(1)
        
        # Let's map to a local coordinate system where axis is z? No, too complex.
        # Just hardcode for dimension 3.
        
        if axis_vec[0] != 0:
            # Axis is X
            x, y, z = rel
            if angle == 90:
                # Rotation direction depends on axis sign?
                # Standard right hand rule around +X: Y->Z, Z->-Y
                # If axis is -X, Y->-Z, Z->Y
                # Let's just try both 90 and 270.
                # So we don't need to care about sign if we try all angles.
                new_y = -z
                new_z = y
            elif angle == 180:
                new_y = -y
                new_z = -z
            elif angle == 270:
                new_y = z
                new_z = -y
            else:
                return None
            new_rel = (x, new_y, new_z)
            
        elif axis_vec[1] != 0:
            # Axis is Y
            x, y, z = rel
            if angle == 90:
                # Around +Y: Z->X, X->-Z
                new_x = z
                new_z = -x
            elif angle == 180:
                new_x = -x
                new_z = -z
            elif angle == 270:
                new_x = -z
                new_z = x
            else:
                return None
            new_rel = (new_x, y, new_z)
            
        elif axis_vec[2] != 0:
            # Axis is Z
            x, y, z = rel
            if angle == 90:
                # Around +Z: X->Y, Y->-X
                new_x = -y
                new_y = x
            elif angle == 180:
                new_x = -x
                new_y = -y
            elif angle == 270:
                new_x = y
                new_y = -x
            else:
                return None
            new_rel = (new_x, new_y, z)
        else:
            return None
            
        # Translate back
        new_p = tuple(origin[i] + new_rel[i] for i in range(dimension))
        return new_p

    # Fixed part: chain[0..k-1]
    # Moving part: chain[k..n-1]
    # Note: chain[k] is on the axis, so it should remain unchanged.
    # Let's verify: rotate(chain[k], chain[k-1], axis, angle)
    # rel = chain[k] - chain[k-1] = axis_vec (unit vector)
    # If axis is X, rel = (1,0,0) or (-1,0,0).
    # If rel = (1,0,0), rotate 90 around X: (1,0,0) -> (1,0,0). Correct.
    
    new_moving = []
    for p in moving:
        new_p = rotate_point(p, chain[k-1], axis, angle)
        if new_p is None:
            return None
        new_moving.append(new_p)
        
    new_chain[k+1:] = new_moving
    
    # Check collisions
    # Fixed positions: chain[0..k-1]
    # Moving positions: new_chain[k..n-1]
    # Note: new_chain[k] is same as chain[k].
    
    occupied_fixed = {chain[i] for i in range(k)}
    
    # Check if any new_moving point (plus chain[k]) is in occupied_fixed
    # And check if any new_moving point collides with another new_moving point?
    # No, rotation preserves distances, so no self-collision in moving part.
    # But we must check collision with fixed part.
    
    # Also, chain[k] is in new_chain. Is it in occupied_fixed?
    # chain[k] was in the moving part. It is adjacent to chain[k-1] (which is in fixed part).
    # It should not be in occupied_fixed unless k-1 == k (impossible) or self-intersection earlier.
    # Since original chain was valid, chain[k] is not in occupied_fixed.
    # And it stays in the same place. So it's safe.
    
    for p in new_moving:
        if p in occupied_fixed:
            return None
            
    return new_chain

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0]*dimension]
        
    rng = random.Random(12345 + len(sequence))
    
    # Time budget
    start_time = time.time()
    deadline = start_time + 230.0  # Leave 10s margin
    
    best_chain = None
    best_energy = 0
    
    # Initial random SAW
    for _ in range(5):
        chain = _random_saw(n, dimension, rng)
        if chain is None:
            continue
        e = _energy(sequence, chain)
        if best_chain is None or e < best_energy:
            best_chain = chain
            best_energy = e
            
    if best_chain is None:
        # Fallback: straight line
        return [[i, 0, 0] for i in range(n)]
        
    # Simulated Annealing
    # We run SA on the best chain found so far, and also try restarting if stuck?
    # Let's do multiple independent SA runs.
    
    # Number of restarts depends on time
    # Let's say we do up to 20 restarts, or until time runs out.
    
    for restart in range(20):
        if time.time() > deadline - 5.0:
            break
            
        # Initialize a new chain
        # Sometimes use the best chain found so far, sometimes random
        if restart == 0:
            current_chain = list(best_chain)
            current_energy = best_energy
        else:
            # Random start or perturbed best
            if rng.random() < 0.5:
                current_chain = _random_saw(n, dimension, rng)
                if current_chain is None:
                    continue
                current_energy = _energy(sequence, current_chain)
            else:
                current_chain = list(best_chain)
                # Perturb
                for _ in range(10):
                    k = rng.randint(1, n-1)
                    axis_vec = tuple(current_chain[k][i] - current_chain[k-1][i] for i in range(dimension))
                    angle = rng.choice([90, 180, 270])
                    new_c = _pivot_move(current_chain, k, axis_vec, angle, dimension)
                    if new_c:
                        current_chain = new_c
                        current_energy = _energy(sequence, current_chain)
                        break
                
        # SA Loop
        T = 5.0
        T_min = 0.1
        alpha = 0.999
        steps = 5000
        
        for step in range(steps):
            if time.time() > deadline:
                return [list(p) for p in best_chain]
                
            # Pick a move
            k = rng.randint(1, n-1)
            axis_vec = tuple(current_chain[k][i] - current_chain[k-1][i] for i in range(dimension))
            angle = rng.choice([90, 180, 270])
            
            new_chain = _pivot_move(current_chain, k, axis_vec, angle, dimension)
            
            if new_chain is None:
                continue
                
            new_energy = _energy(sequence, new_chain)
            delta_e = new_energy - current_energy
            
            if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                current_chain = new_chain
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = current_chain
                    
            T *= alpha
            if T < T_min:
                T = T_min
                
    return [list(p) for p in best_chain]
