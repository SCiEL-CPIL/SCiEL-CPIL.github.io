# EVOLVE-BLOCK-START
import time
import random
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


def _grow(sequence, dimension, rng):
    n = len(sequence)
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, n):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            # Heuristic: prefer high gain, then high free space, then random
            options.append((gain * 10.0 + free * 0.1 + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _local_optimize(sequence, chain, dimension, rng, time_limit):
    n = len(chain)
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    # Precompute neighbors for speed? No, let's keep it simple first.
    # We will use pivot moves and 1-moves.
    
    def _get_energy(chain):
        return _energy(sequence, chain)
    
    def _pivot_move(chain, pivot_idx, axis, angle):
        # Rotate the tail (pivot_idx+1 to end) around the bond (pivot_idx-1, pivot_idx)
        # Bond vector is from chain[pivot_idx-1] to chain[pivot_idx]
        p_prev = chain[pivot_idx - 1]
        p_pivot = chain[pivot_idx]
        
        v = [p_pivot[k] - p_prev[k] for k in range(dimension)]
        bond_axis = -1
        for k in range(dimension):
            if v[k] != 0:
                bond_axis = k
                break
        
        if bond_axis == -1:
            return None
            
        # The rotation axis must be perpendicular to the bond axis
        # In 3D, there are 2 axes perpendicular to bond_axis
        # We rotate around one of them by 90 or 270 degrees (or 180? 180 is same as rotating around other axis?)
        # Actually, rotating a vector perpendicular to the bond by 90, 180, 270 degrees.
        # The bond vector itself is invariant.
        
        # Let's define the rotation.
        # We need to rotate all points in chain[pivot_idx+1:] around the pivot point p_pivot
        # such that the bond (p_pivot - p_prev) is preserved?
        # No, the pivot is at pivot_idx. The bond (pivot_idx-1, pivot_idx) is fixed.
        # The tail is attached to p_pivot.
        # We rotate the tail around the axis defined by the bond vector?
        # No, that would move the first tail atom.
        
        # Standard pivot move:
        # Pick a bond (i, i+1). Rotate the tail (i+1...N) around the bond axis.
        # Here we pick pivot_idx as the fixed point?
        # Let's pick a bond (k, k+1) where k < n-1.
        # Rotate the tail k+1...n-1 around the axis passing through chain[k] and chain[k+1].
        
        # Let's redefine _pivot_move to take the bond index k.
        pass

    def _pivot_move_chain(chain, k):
        # Rotate tail k+1..n-1 around bond k-k+1
        p1 = chain[k]
        p2 = chain[k+1]
        
        v = [p2[i] - p1[i] for i in range(dimension)]
        # Find the axis of the bond
        bond_axis = -1
        for i in range(dimension):
            if v[i] != 0:
                bond_axis = i
                break
        
        if bond_axis == -1:
            return None
            
        # The rotation is around the axis parallel to v.
        # We rotate the vector from p2 to each tail point.
        # Since we are on a cubic lattice, the possible rotations are 90, 180, 270 degrees.
        # There are 3 possible axes perpendicular to the bond? No.
        # In 3D, given a bond axis, there are 4 possible orientations for the tail if we consider the "frame".
        # Actually, the tail is a rigid body attached to p2.
        # The vector from p2 to the first tail atom (p3) is perpendicular to the bond? Not necessarily.
        # But in a self-avoiding walk, p3 cannot be p1.
        
        # Let's try all 3 rotations of 90 degrees around the two axes perpendicular to the bond?
        # No, it's easier to just try all 8 possible neighbors of p2 that are not p1 and not occupied by the head,
        # and see if the rest of the tail can be placed? No, that's not a rigid rotation.
        
        # Correct pivot move:
        # The tail is rigid. We rotate it around the bond axis.
        # In 3D, there are 4 distinct rotations: 0, 90, 180, 270 degrees.
        # 0 is the identity.
        # We need to define the rotation matrix.
        
        # Let's identify the two axes perpendicular to the bond axis.
        perp_axes = [i for i in range(dimension) if i != bond_axis]
        
        # We can rotate around one of the perp axes?
        # If we rotate around a perp axis, the bond vector (along bond_axis) stays fixed?
        # Yes, if the rotation axis is perpendicular to the bond vector and passes through p2?
        # No, the rotation axis must be the bond line itself for the bond to stay fixed?
        # Wait. If we rotate around the bond line (axis passing through p1 and p2), the bond stays fixed.
        # But the bond line is not a coordinate axis necessarily?
        # In cubic lattice, bonds are along coordinate axes.
        # So the bond line IS a coordinate axis (shifted).
        
        # So we rotate around the bond axis (coordinate axis bond_axis).
        # The rotation is in the plane of the other two axes.
        # There are 3 non-identity rotations: 90, 180, 270.
        
        rot_axes = perp_axes # These are the axes that get swapped/changed
        
        # Let's define the rotation of a point p around p2, around the axis bond_axis.
        def rotate_point(p, center, axis, angle):
            # p, center are tuples of length 3
            # axis is the bond axis (0, 1, or 2)
            # angle is 1 (90 deg), 2 (180 deg), or 3 (270 deg)
            
            # Translate to origin
            dp = [p[i] - center[i] for i in range(dimension)]
            
            # Rotate dp
            if axis == 0:
                # Rotate in y-z plane
                if angle == 1: # 90 deg: y -> z, z -> -y
                    dp[1], dp[2] = -dp[2], dp[1]
                elif angle == 2: # 180 deg
                    dp[1], dp[2] = -dp[1], -dp[2]
                elif angle == 3: # 270 deg: y -> -z, z -> y
                    dp[1], dp[2] = dp[2], -dp[1]
            elif axis == 1:
                # Rotate in x-z plane
                if angle == 1:
                    dp[0], dp[2] = -dp[2], dp[0]
                elif angle == 2:
                    dp[0], dp[2] = -dp[0], -dp[2]
                elif angle == 3:
                    dp[0], dp[2] = dp[2], -dp[0]
            elif axis == 2:
                # Rotate in x-y plane
                if angle == 1:
                    dp[0], dp[1] = -dp[1], dp[0]
                elif angle == 2:
                    dp[0], dp[1] = -dp[0], -dp[1]
                elif angle == 3:
                    dp[0], dp[1] = dp[1], -dp[0]
            
            # Translate back
            new_p = tuple(center[i] + dp[i] for i in range(dimension))
            return new_p

        head_set = set(chain[:k+1])
        
        for angle in [1, 2, 3]:
            new_tail = []
            valid = True
            tail_occupied = set()
            
            for i in range(k+1, n):
                orig_point = chain[i]
                new_point = rotate_point(orig_point, p2, bond_axis, angle)
                
                if new_point in head_set:
                    valid = False
                    break
                if new_point in tail_occupied:
                    valid = False
                    break
                if new_tail:
                    prev_new = new_tail[-1]
                    dist = sum(abs(new_point[i] - prev_new[i]) for i in range(dimension))
                    if dist != 1:
                        valid = False
                        break
                tail_occupied.add(new_point)
                new_tail.append(new_point)
                
            if valid:
                candidate = chain[:k+1] + new_tail
                return candidate
        
        return None

    def _one_move(chain, idx):
        # Try to move chain[idx] to a neighbor
        # Must not collide with chain[idx-1] or chain[idx+1] or any other occupied site
        # And the new position must be free
        
        p = chain[idx]
        head = set(chain[:idx])
        tail = set(chain[idx+1:])
        
        for neighbor in _neighbours(p):
            if neighbor in head or neighbor in tail:
                continue
            # Check if neighbor is p_prev or p_next?
            # head includes p_prev (idx-1). tail includes p_next (idx+1).
            # So if neighbor is in head or tail, it's invalid.
            # This covers collisions.
            
            candidate = list(chain)
            candidate[idx] = neighbor
            
            # Check self-avoidance of the new chain?
            # Since we removed p and added neighbor, and neighbor was not in head/tail,
            # and p was in the chain, we just need to ensure neighbor doesn't collide with itself? No.
            # We need to ensure the new chain is self-avoiding.
            # The only new collision could be if neighbor is in the chain?
            # We checked neighbor not in head and not in tail.
            # head = chain[0..idx-1], tail = chain[idx+1..n-1].
            # So neighbor is not in the chain.
            # Thus the new chain is self-avoiding.
            
            return candidate
        return None

    start_time = time.time()
    step = 0
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        step += 1
        
        if rng.random() < 0.5:
            # Pivot move
            k = rng.randint(0, n - 2)
            candidate = _pivot_move_chain(chain, k)
            if candidate is None:
                continue
        else:
            # 1-move
            idx = rng.randint(1, n - 2) # Avoid ends? Ends are harder to move without breaking connectivity?
            # Actually, moving an end is just a 1-move.
            # Let's allow any idx.
            idx = rng.randint(0, n - 1)
            candidate = _one_move(chain, idx)
            if candidate is None:
                continue
        
        new_energy = _energy(sequence, candidate)
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < math.exp(delta / T):
            chain = candidate
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
        
        T *= cooling
        if T < T_min:
            T = T_min
            
        # Occasionally reheat?
        if step % 1000 == 0:
            T = max(T, 0.5)
            
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation of a fixed number of randomized greedy growths and local optimizations."""
    start_time = time.time()
    deadline = start_time + 300  # Safety margin
    
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    
    best_chain = None
    best_energy = 0
    
    # Allocate time budget
    # We have ~280s to use.
    # Let's do multiple restarts.
    
    # Estimate time per restart?
    # Just loop until time is up.
    
    max_restarts = 100
    for r in range(max_restarts):
        if time.time() - start_time > deadline - 10:
            break
            
        rng = random.Random(42 + r)
        
        # Generate initial chain
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        # Optimize
        # Allocate more time for longer chains?
        # Let's just give a fixed time slice per restart, or until deadline.
        # If we have many short chains, we can do more restarts.
        # Let's say 2s per restart for short chains, more for long?
        # Simple: use remaining time, but cap it?
        # If we cap it, we might do many restarts.
        
        # Let's use a dynamic time slice.
        # If n < 50, use 5s. If n > 100, use 10s.
        time_slice = 5.0
        if n > 80:
            time_slice = 10.0
        if n > 120:
            time_slice = 20.0
            
        opt_start = time.time()
        opt_limit = min(opt_start + time_slice, deadline - 5)
        
        optimized_chain, opt_energy = _local_optimize(sequence, chain, dimension, rng, opt_limit)
        
        if optimized_chain is not None:
            if best_chain is None or opt_energy < best_energy:
                best_chain = optimized_chain
                best_energy = opt_energy

    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
