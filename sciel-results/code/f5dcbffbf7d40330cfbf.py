"""
HP lattice protein folding using Simulated Annealing with Smart Regrowth moves.
"""
# EVOLVE-BLOCK-START
import random
import time

def _neighbours(point):
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _energy(sequence, chain):
    if len(chain) < 2:
        return 0
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = pos_to_idx.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _grow_tail(start_point, sequence, start_idx, length, head_occupied, head_residue_map, rng):
    """
    Grow a tail of given length starting from start_point.
    start_idx is the index of the first residue in the tail in the original sequence.
    head_occupied: set of points occupied by the head.
    head_residue_map: dict mapping point -> residue char for the head.
    Returns list of new points or None.
    """
    if length <= 0:
        return []
    
    new_chain = [start_point]
    new_occupied = set(head_occupied)
    new_occupied.add(start_point)
    # Extend residue map for the new tail as we grow? 
    # Or just check against head_residue_map and new_chain residues.
    # Let's maintain a map for the new tail too.
    new_residue_map = dict(head_residue_map)
    new_residue_map[start_point] = sequence[start_idx - 1] # The point before start_idx is the last of the head
    
    current = start_point
    
    for k in range(length):
        idx = start_idx + k
        res = sequence[idx]
        
        candidates = []
        for site in _neighbours(current):
            if site in new_occupied:
                continue
            
            score = rng.random() * 0.5 # Jitter
            
            # Check contacts with H's in the current occupied set (head + new tail so far)
            if res == 'H':
                for q in _neighbours(site):
                    if q in new_occupied:
                        # Check if the residue at q is H
                        res_q = new_residue_map.get(q)
                        if res_q == 'H':
                            score += 1.0 # Strong preference for H-H contact
            
            # Prefer sites with more free space
            free = sum(1 for q in _neighbours(site) if q not in new_occupied)
            score += 0.1 * free
            
            candidates.append((score, site))
            
        if not candidates:
            return None
            
        best_site = max(candidates, key=lambda x: x[0])[1]
        new_chain.append(best_site)
        new_occupied.add(best_site)
        new_residue_map[best_site] = res
        current = best_site
        
    return new_chain[1:]

def fold(sequence, dimension):
    N = len(sequence)
    if N == 0:
        return []
    if N == 1:
        return [[0, 0]]
    
    deadline = time.time() + 110.0
    rng = random.Random(42)
    
    # 1. Initial Conformation
    chain = [(0, 0)]
    occupied = {chain[0]}
    residue_map = {chain[0]: sequence[0]}
    
    for i in range(1, N):
        best_site = None
        best_score = -1
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            score = rng.random()
            # Heuristic: prefer H-H contacts
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in occupied and residue_map.get(q) == 'H':
                        score += 1.0
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score += 0.1 * free
            if score > best_score:
                best_score = score
                best_site = site
        
        if best_site is None:
            chain = [(i, 0) for i in range(N)]
            occupied = set(chain)
            residue_map = {p: sequence[i] for i, p in enumerate(chain)}
            break
        
        chain.append(best_site)
        occupied.add(best_site)
        residue_map[best_site] = sequence[i]
    
    if len(chain) < N:
        chain = [(i, 0) for i in range(N)]
        occupied = set(chain)
        residue_map = {p: sequence[i] for i, p in enumerate(chain)}
        
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    
    T = 1.0
    T_min = 0.01
    alpha = 0.995
    
    while time.time() < deadline:
        cut = rng.randint(N // 4, 3 * N // 4)
        head = chain[:cut]
        tail_len = N - cut
        start_point = head[-1]
        head_occupied = set(head)
        # Build head residue map
        head_residue_map = {p: sequence[i] for i, p in enumerate(head)}
        
        new_tail = _grow_tail(start_point, sequence, cut, tail_len, head_occupied, head_residue_map, rng)
        
        if new_tail is None:
            T *= alpha
            continue
            
        new_chain = head + new_tail
        
        if len(set(new_chain)) != N:
            T *= alpha
            continue
            
        new_energy = _energy(sequence, new_chain)
        delta_e = new_energy - best_energy
        
        if delta_e < 0 or rng.random() < pow(2.718281828, -delta_e / T):
            chain = new_chain
            if new_energy < best_energy:
                best_chain = new_chain
                best_energy = new_energy
        
        T *= alpha
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
