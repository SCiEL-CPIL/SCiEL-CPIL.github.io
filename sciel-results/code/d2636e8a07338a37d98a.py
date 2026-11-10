"""Rewritten HP lattice protein folding using Monte Carlo with Pivot Moves."""
import random
import math


def _compute_energy(sequence, chain):
    """Compute the negative H-H contact energy."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            x, y = p
            for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
                j = where.get((nx, ny))
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _initial_chain(sequence):
    """Create a random self-avoiding walk as initial state."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    # Start at 0,0, random direction
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    for i in range(1, n):
        candidates = []
        x, y = chain[-1]
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if (nx, ny) not in occupied:
                candidates.append((nx, ny))
        
        if not candidates:
            # Dead end, fallback to spiral
            chain = [(0, 0)]
            occupied = {(0, 0): 0}
            direction = (1, 0)
            for j in range(1, n):
                placed = False
                for turn in (1, 0, -1):
                    if turn == 1:
                        new_dir = (direction[1], -direction[0])
                    elif turn == -1:
                        new_dir = (-direction[1], direction[0])
                    else:
                        new_dir = direction
                    
                    new_pos = (chain[-1][0] + new_dir[0], chain[-1][1] + new_dir[1])
                    if new_pos not in occupied:
                        chain.append(new_pos)
                        occupied[new_pos] = j
                        direction = new_dir
                        placed = True
                        break
                if not placed:
                    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        new_pos = (chain[-1][0] + dx, chain[-1][1] + dy)
                        if new_pos not in occupied:
                            chain.append(new_pos)
                            occupied[new_pos] = j
                            direction = (dx, dy)
                            placed = True
                            break
                if not placed:
                    for k in range(j, n):
                        chain.append((chain[-1][0], chain[-1][1] + 1))
                    break
            return chain
            
        new_pos = random.choice(candidates)
        chain.append(new_pos)
        occupied[new_pos] = i
        
    return chain


def _pivot_move(chain, k, angle):
    """
    Perform a pivot move at index k with given angle (90, 180, 270).
    Returns new chain if valid, None if collision.
    """
    n = len(chain)
    if k >= n - 1:
        return None
        
    head = chain[:k+1]
    tail = chain[k+1:]
    
    pivot = chain[k]
    px, py = pivot
    
    new_tail = []
    occupied_head = set(head)
    
    for p in tail:
        vx = p[0] - px
        vy = p[1] - py
        
        if angle == 90:
            nvx, nvy = -vy, vx
        elif angle == 180:
            nvx, nvy = -vx, -vy
        elif angle == 270:
            nvx, nvy = vy, -vx
        else:
            return None
            
        new_p = (px + nvx, py + nvy)
        
        if new_p in occupied_head:
            return None
            
        new_tail.append(new_p)
        
    return head + new_tail


def _simulate_annealing_pivot(sequence, dimension, rng, num_iterations=10000):
    n = len(sequence)
    if n <= 1:
        return _initial_chain(sequence)
    
    chain = _initial_chain(sequence)
    current_energy = _compute_energy(sequence, chain)
    
    best_chain = list(chain)
    best_energy = current_energy
    
    T_start = 1.0
    T_end = 0.01
    cooling_rate = (T_end / T_start) ** (1.0 / num_iterations)
    
    for it in range(num_iterations):
        T = T_start * (cooling_rate ** it)
        
        k = rng.randint(0, n - 2)
        angle = rng.choice([90, 180, 270])
        
        new_chain = _pivot_move(chain, k, angle)
        
        if new_chain is None:
            continue
            
        new_energy = _compute_energy(sequence, new_chain)
        
        if new_energy < current_energy or rng.random() < math.exp((new_energy - current_energy) / T):
            current_energy = new_energy
            chain = new_chain
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
                
    return best_chain


def fold(sequence, dimension):
    """Best conformation using Monte Carlo with Pivot Moves."""
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    best_chain = None
    best_energy = 0
    
    num_restarts = 3
    
    for restart in range(num_restarts):
        rng = random.Random(54321 + len(sequence) * 7 + restart)
        
        if n <= 30:
            num_iter = 15000
        elif n <= 60:
            num_iter = 10000
        else:
            num_iter = 5000
            
        chain = _simulate_annealing_pivot(sequence, dimension, rng, num_iter)
        energy = _compute_energy(sequence, chain)
        
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy
            
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]
