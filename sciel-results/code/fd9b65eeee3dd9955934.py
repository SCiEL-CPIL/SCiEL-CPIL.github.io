import random
import time

# EVOLVE-BLOCK-START

_T0 = time.time()
_DEADLINE = 200.0  # Safety margin before 210s kill

def _remaining():
    return _DEADLINE - (time.time() - _T0)

def _energy(seq, chain):
    """
    Compute energy: -number of H-H contacts.
    """
    n = len(chain)
    if n == 0:
        return 0
    
    h_positions = set()
    pos_to_idx = {}
    for i, p in enumerate(chain):
        pos_to_idx[p] = i
        if seq[i] == 'H':
            h_positions.add(p)
            
    contacts = 0
    for i, p in enumerate(chain):
        if seq[i] == 'H':
            x, y = p
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                q = (x + dx, y + dy)
                j = pos_to_idx.get(q)
                if j is not None:
                    if abs(i - j) > 1:
                        if q in h_positions:
                            contacts += 1
    return -contacts

def _grow(seq):
    """
    Generate a random self-avoiding walk (SAW).
    """
    n = len(seq)
    if n == 0:
        return []
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    for i in range(1, n):
        last_x, last_y = chain[-1]
        valid_moves = []
        for dx, dy in directions:
            nx, ny = last_x + dx, last_y + dy
            if (nx, ny) not in occupied:
                valid_moves.append((nx, ny))
        
        if not valid_moves:
            return None
        
        nx, ny = random.choice(valid_moves)
        chain.append((nx, ny))
        occupied[(nx, ny)] = i
        
    return chain

def _local_search(seq, chain, max_steps):
    """
    Local search using pivot moves.
    """
    n = len(chain)
    if n < 3:
        return chain
    
    current_chain = list(chain)
    current_energy = _energy(seq, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    rng = random.Random()
    steps = 0
    
    while steps < max_steps:
        if _remaining() < 5:
            break
            
        steps += 1
        i = rng.randint(1, n - 2)
        
        pivot_x, pivot_y = current_chain[i-1]
        tail = current_chain[i:]
        head_set = set(current_chain[:i])
        
        dx0 = tail[0][0] - pivot_x
        dy0 = tail[0][1] - pivot_y
        
        rotations = [
            (dy0, -dx0),
            (-dx0, -dy0),
            (-dy0, dx0)
        ]
        
        rng.shuffle(rotations)
        
        moved = False
        for ndx, ndy in rotations:
            new_tail = []
            cur_x = pivot_x + ndx
            cur_y = pivot_y + ndy
            new_tail.append((cur_x, cur_y))
            
            valid = True
            if (cur_x, cur_y) in head_set:
                valid = False
            
            if valid:
                for k in range(1, len(tail)):
                    ddx = tail[k][0] - tail[k-1][0]
                    ddy = tail[k][1] - tail[k-1][1]
                    cur_x += ddx
                    cur_y += ddy
                    new_tail.append((cur_x, cur_y))
                    
                    if (cur_x, cur_y) in head_set:
                        valid = False
                        break
            
            if valid:
                new_chain = current_chain[:i] + new_tail
                new_energy = _energy(seq, new_chain)
                
                if new_energy <= current_energy:
                    current_chain = new_chain
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
                    moved = True
                    break
        
    return best_chain

def fold(sequence, dimension):
    """
    Best conformation using randomized growth and local search with time guard.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    best_chain = None
    best_energy = 0
    
    if n <= 2:
        return [[i, 0] for i in range(n)]
        
    while _remaining() > 2:
        chain = _grow(sequence)
        if chain is None:
            continue
            
        improved_chain = _local_search(sequence, chain, 50)
        
        e = _energy(sequence, improved_chain)
        
        if best_chain is None or e < best_energy:
            best_chain = improved_chain
            best_energy = e
            
    if best_chain is None:
        best_chain = [[i, 0] for i in range(n)]
    else:
        best_chain = [list(p) for p in best_chain]
        
    return best_chain

# EVOLVE-BLOCK-END
