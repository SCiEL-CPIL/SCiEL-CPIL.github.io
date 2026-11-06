import random
import time

_GLOBAL_START = time.time()
_DEADLINE = _GLOBAL_START + 230.0

def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for axis in range(len(p)):
                for step in (-1, 1):
                    q = list(p)
                    q[axis] += step
                    q = tuple(q)
                    j = where.get(q)
                    if j is not None and j > i + 1 and sequence[j] == "H":
                        contacts += 1
    return -contacts

def _beam_search(sequence, dimension, beam_width=200):
    start = (0,) * dimension
    beam = [([start], {start}, {start: 0}, 0)]
    
    for i in range(1, len(sequence)):
        new_beam = []
        for chain, occupied, pos_to_idx, energy in beam:
            last = chain[-1]
            for axis in range(dimension):
                for step in (-1, 1):
                    new_pos = list(last)
                    new_pos[axis] += step
                    new_pos = tuple(new_pos)
                    
                    if new_pos in occupied:
                        continue
                    
                    inc = 0
                    if sequence[i] == "H":
                        for axis2 in range(dimension):
                            for step2 in (-1, 1):
                                neighbor = list(new_pos)
                                neighbor[axis2] += step2
                                neighbor = tuple(neighbor)
                                j = pos_to_idx.get(neighbor)
                                if j is not None and j < i - 1:
                                    if sequence[j] == "H":
                                        inc += 1
                    new_energy = energy - inc
                    
                    new_chain = chain + [new_pos]
                    new_occupied = occupied | {new_pos}
                    new_pos_to_idx = pos_to_idx.copy()
                    new_pos_to_idx[new_pos] = i
                    
                    free_space = 0
                    for axis2 in range(dimension):
                        for step2 in (-1, 1):
                            neighbor = list(new_pos)
                            neighbor[axis2] += step2
                            if tuple(neighbor) not in new_occupied:
                                free_space += 1
                            
                    new_beam.append((new_chain, new_occupied, new_pos_to_idx, new_energy, free_space))
        
        if not new_beam:
            return None
        
        new_beam.sort(key=lambda x: (x[3], -x[4]))
        beam = new_beam[:beam_width]
        
    if not beam:
        return None
    return beam[0][0]

def _local_search_pivot(sequence, chain, rng, max_iters=1000):
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    
    for _ in range(max_iters):
        if time.time() > _DEADLINE:
            break
        if len(current_chain) < 3:
            break
        i = rng.randint(0, len(current_chain) - 2)
        pivot_pos = current_chain[i]
        old_next = current_chain[i+1]
        
        candidates = []
        for axis in range(len(pivot_pos)):
            for step in (-1, 1):
                new_next = list(pivot_pos)
                new_next[axis] += step
                new_next = tuple(new_next)
                if new_next == old_next:
                    continue
                if i > 0 and new_next == current_chain[i-1]:
                    continue
                candidates.append(new_next)
        
        if not candidates:
            continue
            
        new_next = rng.choice(candidates)
        
        new_chain = current_chain[:i+1] + [new_next]
        
        vec = tuple(n - o for n, o in zip(new_next, old_next))
        
        tail = current_chain[i+2:]
        new_tail = [tuple(p[0] + vec[0], p[1] + vec[1], p[2] + vec[2]) for p in tail]
        
        new_chain.extend(new_tail)
        
        if len(new_chain) != len(set(new_chain)):
            continue
            
        valid = True
        for k in range(1, len(new_chain)):
            diff = sum(abs(a-b) for a,b in zip(new_chain[k], new_chain[k-1]))
            if diff != 1:
                valid = False
                break
        if not valid:
            continue
            
        new_energy = _energy(sequence, new_chain)
        
        if new_energy < current_energy:
            current_chain = new_chain
            current_energy = new_energy
            
    return current_chain, current_energy

def fold(sequence, dimension):
    if time.time() > _DEADLINE:
        return [list([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
    
    rng = random.Random(12345 + len(sequence))
    
    best_chain = _beam_search(sequence, dimension, beam_width=200)
    
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        best_energy = _energy(sequence, best_chain)
    else:
        best_energy = _energy(sequence, best_chain)
        
    if time.time() < _DEADLINE - 1.0:
        refined_chain, refined_energy = _local_search_pivot(sequence, best_chain, rng, max_iters=1000)
        if refined_energy < best_energy:
            best_chain = refined_chain
            best_energy = refined_energy
            
    return [list(p) for p in best_chain]
