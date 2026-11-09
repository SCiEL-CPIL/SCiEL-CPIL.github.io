"""Rewritten program using Simulated Annealing on Self-Avoiding Walks."""
# EVOLVE-BLOCK-START
import time
import math
import random


def _spiral(N):
    """Generate a spiral self-avoiding walk of length N."""
    if N == 0:
        return []
    chain = [(0, 0)]
    if N == 1:
        return chain
    
    dx, dy = 1, 0
    step_len = 1
    steps_in_current_dir = 0
    turns = 0
    
    while len(chain) < N:
        x, y = chain[-1]
        chain.append((x + dx, y + dy))
        steps_in_current_dir += 1
        
        if steps_in_current_dir == step_len:
            steps_in_current_dir = 0
            turns += 1
            # Rotate 90 degrees clockwise
            if dx == 1:
                dx, dy = 0, 1
            elif dy == 1:
                dx, dy = -1, 0
            elif dx == -1:
                dx, dy = 0, -1
            else:
                dx, dy = 1, 0
            
            # Increase step length every 2 turns (after moving vertically)
            if turns % 2 == 0:
                step_len += 1
                
    return chain


def _calc_energy(sequence, chain):
    """Calculate energy of a given chain configuration."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    # Directions: Right, Left, Up, Down
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                q = (p[0] + dx, p[1] + dy)
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def fold(sequence, dimension):
    """Fold sequence using Simulated Annealing."""
    N = len(sequence)
    if N == 0:
        return []
    if N == 1:
        return [[0, 0]]
    
    start_time = time.time()
    # 120s budget for 10 sequences, so ~12s each. 
    # Use 11s to be safe.
    deadline = start_time + 11.0
    
    # 1. Initial Conformation
    chain = _spiral(N)
    occupied = set(chain)
    current_energy = _calc_energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    
    rng = random.Random(42)
    
    # 2. Simulated Annealing
    T = 1.0
    cooling = 0.995
    min_T = 0.01
    
    while time.time() < deadline:
        # Choose move
        # 70% Pivot, 30% Slither
        if rng.random() < 0.7:
            # Pivot Move
            i = rng.randint(1, N - 2)
            rot = rng.choice([1, 2, 3])
            
            pivot = chain[i - 1]
            new_seg = []
            valid = True
            prev = pivot
            
            # Build new segment
            for j in range(i, N):
                vec = (chain[j][0] - prev[0], chain[j][1] - prev[1])
                if rot == 1:
                    new_vec = (-vec[1], vec[0])
                elif rot == 2:
                    new_vec = (-vec[0], -vec[1])
                else:
                    new_vec = (vec[1], -vec[0])
                
                new_point = (prev[0] + new_vec[0], prev[1] + new_vec[1])
                new_seg.append(new_point)
                
                # Check collision with fixed part (0..i-1)
                # We need to ensure new_point is not in occupied 
                # except if it is in the old segment (which we are replacing)
                # But we don't have old segment as a set readily.
                # However, we know the old segment positions are chain[i:]
                # We can check: if new_point in occupied AND new_point not in chain[i:]
                # This is O(N) check per point, making it O(N^2).
                # For N=100, 100*100 = 10,000 ops. 
                # 10,000 ops * 10,000 moves = 100M ops. Might be too slow in Python.
                
                # Optimization: Pre-compute set of old segment positions?
                # That's O(N) per pivot.
                # Then check is O(1).
                # Total O(N) per pivot.
                
                prev = new_point
            
            # Re-do with optimization
            old_seg_set = set(chain[i:])
            new_seg = []
            valid = True
            prev = pivot
            for j in range(i, N):
                vec = (chain[j][0] - prev[0], chain[j][1] - prev[1])
                if rot == 1:
                    new_vec = (-vec[1], vec[0])
                elif rot == 2:
                    new_vec = (-vec[0], -vec[1])
                else:
                    new_vec = (vec[1], -vec[0])
                
                new_point = (prev[0] + new_vec[0], prev[1] + new_vec[1])
                new_seg.append(new_point)
                
                if new_point in occupied and new_point not in old_seg_set:
                    valid = False
                    break
                prev = new_point
            
            if valid:
                candidate = chain[:i] + new_seg
                new_energy = _calc_energy(sequence, candidate)
                delta_e = new_energy - current_energy
                
                if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                    # Update
                    # Remove old segment from occupied
                    for p in old_seg_set:
                        occupied.remove(p)
                    # Add new segment
                    for p in new_seg:
                        occupied.add(p)
                    
                    chain = candidate
                    current_energy = new_energy
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = chain[:]
        else:
            # Slither Move
            # Move last residue
            last = chain[-1]
            neighbors = [(last[0] + 1, last[1]), (last[0] - 1, last[1]),
                         (last[0], last[1] + 1), (last[0], last[1] - 1)]
            rng.shuffle(neighbors)
            
            for n in neighbors:
                if n not in occupied:
                    # Valid move
                    # Update chain
                    old_last = chain[-1]
                    chain[-1] = n
                    
                    # Update occupied
                    occupied.remove(old_last)
                    occupied.add(n)
                    
                    new_energy = _calc_energy(sequence, chain)
                    delta_e = new_energy - current_energy
                    
                    if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                        current_energy = new_energy
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = chain[:]
                    else:
                        # Reject, revert
                        chain[-1] = old_last
                        occupied.remove(n)
                        occupied.add(old_last)
                    break # Only try one neighbor per slither attempt
        
        T *= cooling
        if T < min_T:
            T = 1.0 # Restart temperature

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
