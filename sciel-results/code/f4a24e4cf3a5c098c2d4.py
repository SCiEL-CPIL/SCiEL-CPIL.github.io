import random
import math
import time
import numpy as np

# Global time management
_START_TIME = time.time()
# The task says 240s CPU budget. Evaluator kills at 300s wall clock.
# We should aim to finish well before 240s to be safe.
# Let's use 220s as our hard stop.
_TOTAL_BUDGET = 220.0

def _get_remaining_time():
    return _TOTAL_BUDGET - (time.time() - _START_TIME)

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _calculate_energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _init_chain(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0,)*dimension]
    
    rng = random.Random(42)
    best_init = None
    best_init_energy = 0
    
    # Try a few greedy initializations
    for _ in range(10):
        chain = [(0,)*dimension]
        occupied = {chain[0]: 0}
        success = True
        for i in range(1, n):
            last = chain[-1]
            candidates = [q for q in _neighbours(last) if q not in occupied]
            if not candidates:
                success = False
                break
            # Prefer H-H contacts if possible
            best_cand = None
            best_score = -1
            for c in candidates:
                score = 0
                if sequence[i] == "H":
                    for nb in _neighbours(c):
                        if nb in occupied and occupied[nb] < i-1 and sequence[occupied[nb]] == "H":
                            score += 1
                score += rng.random() * 0.1
                if score > best_score:
                    best_score = score
                    best_cand = c
            chain.append(best_cand)
            occupied[best_cand] = i
        
        if success and len(chain) == n:
            e = _calculate_energy(sequence, chain)
            if best_init is None or e < best_init_energy:
                best_init = chain
                best_init_energy = e
                
    if best_init is None:
        # Fallback
        best_init = [(i, 0, 0) for i in range(n)]
        
    return best_init

def _get_rotation_matrix(v1, v2):
    """
    Returns a 3x3 rotation matrix that rotates unit vector v1 to v2.
    v1 and v2 must be numpy arrays of length 3.
    """
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)
    
    # Normalize
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return np.eye(3)
    v1 = v1 / n1
    v2 = v2 / n2
    
    # If same direction
    if np.allclose(v1, v2):
        return np.eye(3)
        
    # If opposite directions
    if np.allclose(v1, -v2):
        # Pick an arbitrary axis perpendicular to v1
        # v1 = (x, y, z)
        # If x is not dominant, use (0, 1, 0) cross v1?
        # Simpler: find any vector not parallel to v1
        ref = np.array([1.0, 0.0, 0.0])
        if abs(v1[0]) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        axis = np.cross(v1, ref)
        axis = axis / np.linalg.norm(axis)
        # Rotate 180 degrees around axis
        c = -1.0 # cos(180)
        s = 0.0  # sin(180)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = c * np.eye(3) + s * K + (1 - c) * np.outer(axis, axis)
        return R

    # General case
    u = np.cross(v1, v2)
    s = np.linalg.norm(u)
    c = np.dot(v1, v2)
    
    if s == 0:
        return np.eye(3)
        
    u = u / s
    K = np.array([
        [0, -u[2], u[1]],
        [u[2], 0, -u[0]],
        [-u[1], u[0], 0]
    ])
    R = c * np.eye(3) + s * K + (1 - c) * np.outer(u, u)
    return R

def _apply_pivot_move(chain, pivot_idx, new_dir, dimension):
    """
    Returns new_chain or None if invalid.
    pivot_idx is the index of the pivot residue.
    The bond pivot_idx -> pivot_idx+1 is rotated to new_dir.
    The tail pivot_idx+1 .. N-1 is rotated rigidly.
    """
    n = len(chain)
    if pivot_idx >= n - 1:
        return None
        
    pivot_point = np.array(chain[pivot_idx], dtype=float)
    old_tail_start = np.array(chain[pivot_idx + 1], dtype=float)
    
    v_old = old_tail_start - pivot_point
    v_new = np.array(new_dir, dtype=float)
    
    # If no change
    if np.allclose(v_old, v_new):
        return list(chain)
        
    R = _get_rotation_matrix(v_old, v_new)
    
    # New tail
    new_tail = [tuple(pivot_point + v_new)]
    
    # Apply rotation to subsequent points
    # C_{k+2} = C_{k+1} + R(C_{k+2} - C_{k+1})
    # General: C'_j = C'_j-1 + R(C_j - C_j-1)
    
    prev_new = pivot_point + v_new
    for j in range(pivot_idx + 2, n):
        c_old_prev = np.array(chain[j-1], dtype=float)
        c_old_curr = np.array(chain[j], dtype=float)
        v_step = c_old_curr - c_old_prev
        v_rotated = R @ v_step
        curr_new = prev_new + v_rotated
        new_tail.append(tuple(curr_new))
        prev_new = curr_new
        
    # Check collisions
    # Head is chain[0..pivot_idx]
    head_set = set(chain[:pivot_idx+1])
    
    # Check self-collision in tail
    tail_set = set()
    for p in new_tail:
        if p in head_set:
            return None
        if p in tail_set:
            return None
        tail_set.add(p)
        
    # Combine
    new_chain = chain[:pivot_idx+1] + new_tail
    return new_chain

def fold(sequence, dimension):
    """
    Best conformation using Monte Carlo with Pivot moves.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0]*dimension]
        
    # Initialize
    chain = _init_chain(sequence, dimension)
    best_chain = list(chain)
    best_energy = _calculate_energy(sequence, best_chain)
    current_energy = best_energy
    
    # Annealing parameters
    start_temp = 1.0
    end_temp = 0.01
    # We will run until time runs out
    
    rng = random.Random(123)
    
    # Directions
    directions = []
    for axis in range(dimension):
        for step in (-1, 1):
            d = [0]*dimension
            d[axis] = step
            directions.append(d)
            
    while _get_remaining_time() > 0.1:
        # Pick pivot
        # Avoid pivot 0 and N-1
        pivot_idx = rng.randint(1, n - 2)
        new_dir = rng.choice(directions)
        
        new_chain = _apply_pivot_move(chain, pivot_idx, new_dir, dimension)
        if new_chain is None:
            continue
            
        new_energy = _calculate_energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        # Metropolis
        if delta_e < 0:
            accept = True
        else:
            # Temperature schedule: linear decrease
            # We don't know total steps, so use time-based?
            # Or just fixed low temp?
            # Let's use a simple cooling schedule based on iteration count
            # But we don't have a fixed number of iterations.
            # Let's just use a constant temp that allows some uphill moves.
            temp = 0.5
            if delta_e == 0:
                accept = True
            else:
                accept = rng.random() < math.exp(-delta_e / temp)
                
        if accept:
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
                
    return [list(p) for p in best_chain]
