"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

_DEADLINE = time.time() + 325  # Safety margin before 330s kill


def _energy(sequence, chain):
    """
    Calculate the energy of the chain.
    Energy = - (number of H-H contacts)
    """
    n = len(chain)
    if n == 0:
        return 0
    
    # Map position to index
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            x, y, z = p
            # Check 6 neighbors
            neighbors = [
                (x+1, y, z), (x-1, y, z),
                (x, y+1, z), (x, y-1, z),
                (x, y, z+1), (x, y, z-1)
            ]
            for q in neighbors:
                if q in pos_to_idx:
                    j = pos_to_idx[q]
                    # Must be H, and not adjacent in chain
                    if sequence[j] == 'H' and abs(i - j) > 1:
                        contacts += 1
                        
    return -contacts


def _grow(sequence, dimension, rng):
    """
    Generate a random self-avoiding walk using a greedy approach with H-preference.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    start = (0, 0, 0)
    chain = [start]
    occupied = {start}
    
    for i in range(1, n):
        last = chain[-1]
        x, y, z = last
        moves = [
            (x+1, y, z), (x-1, y, z),
            (x, y+1, z), (x, y-1, z),
            (x, y, z+1), (x, y, z-1)
        ]
        
        # Filter out occupied
        valid_moves = [m for m in moves if m not in occupied]
        
        if not valid_moves:
            return None
        
        # Prefer moves that are close to existing H residues if current is H
        if sequence[i] == 'H':
            # Score moves by proximity to H
            def score_move(m):
                sx, sy, sz = m
                neighbors = [
                    (sx+1, sy, sz), (sx-1, sy, sz),
                    (sx, sy+1, sz), (sx, sy-1, sz),
                    (sx, sy, sz+1), (sx, sy, sz-1)
                ]
                score = 0
                for nb in neighbors:
                    if nb in occupied:
                        # Check if the residue at nb is H
                        # We need index of nb
                        # This is slow. Let's just use random for init.
                        pass
                return rng.random()
            
            # Simple random pick for now to keep init fast
            next_move = rng.choice(valid_moves)
        else:
            next_move = rng.choice(valid_moves)
            
        chain.append(next_move)
        occupied.add(next_move)
        
    return chain


def _sa_fold(sequence, dimension, rng, time_budget):
    """
    Simulated Annealing refinement.
    """
    n = len(sequence)
    if n <= 1:
        return [[0]*dimension], 0
    
    # Initialize with best of a few grows
    best_init = None
    best_init_energy = 0
    for _ in range(50):
        c = _grow(sequence, dimension, rng)
        if c is not None:
            e = _energy(sequence, c)
            if best_init is None or e < best_init_energy:
                best_init = c
                best_init_energy = e
                
    if best_init is None:
        best_init = [(i, 0, 0) for i in range(n)]
        best_init_energy = _energy(sequence, best_init)
        
    chain = list(best_init)
    current_energy = best_init_energy
    best_chain = list(chain)
    best_energy = current_energy
    
    start_temp = 1.0
    min_temp = 0.05
    cooling_rate = 0.995
    
    end_time = time.time() + time_budget
    
    # Track no-improvement steps
    no_improve_steps = 0
    max_no_improve = 5000
    
    while time.time() < end_time:
        if start_temp < min_temp:
            # Reheat from best
            if best_energy < current_energy:
                chain = list(best_chain)
                current_energy = best_energy
            start_temp = 0.5
            no_improve_steps = 0
            continue
            
        # Choose move
        move_type = rng.random()
        new_chain = None
        
        if move_type < 0.7:
            # Reversal Move
            if n > 2:
                i = rng.randint(0, n - 2)
                j = rng.randint(i + 1, n - 1)
                
                segment = chain[i:j+1]
                reversed_segment = segment[::-1]
                
                static_set = set(chain[:i]) | set(chain[j+1:])
                
                valid = True
                for p in reversed_segment:
                    if p in static_set:
                        valid = False
                        break
                
                if valid:
                    new_chain = chain[:i] + reversed_segment + chain[j+1:]
        else:
            # Local Move
            idx = rng.randint(0, n - 1)
            p = chain[idx]
            x, y, z = p
            
            moves = [
                (x+1, y, z), (x-1, y, z),
                (x, y+1, z), (x, y-1, z),
                (x, y, z+1), (x, y, z-1)
            ]
            
            occupied = set(chain)
            occupied.remove(p)
            
            valid_moves = [m for m in moves if m not in occupied]
            if valid_moves:
                new_p = rng.choice(valid_moves)
                new_chain = list(chain)
                new_chain[idx] = new_p
        
        if new_chain is None:
            continue
            
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta <= 0 or rng.random() < min(1.0, max(1e-10, 2.718281828 ** (-delta / start_temp))):
            chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
                no_improve_steps = 0
            else:
                no_improve_steps += 1
                
        start_temp *= cooling_rate
        
    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Best conformation via Simulated Annealing with time budget.
    """
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
        
    global _DEADLINE
    if time.time() > _DEADLINE:
        return [[i, 0, 0] for i in range(n)]
    
    # Allocate time
    # 26 sequences. 330s total.
    # 12s per sequence.
    time_budget = 12.0
    
    rng = random.Random(42 + n)
    
    chain, energy = _sa_fold(sequence, dimension, rng, time_budget)
    
    return [list(p) for p in chain]

# EVOLVE-BLOCK-END
