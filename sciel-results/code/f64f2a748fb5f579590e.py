"""HP lattice protein folding on the cubic lattice."""
# EVOLVE-BLOCK-START
import time
import random
import heapq

_DEADLINE = time.time() + 320  # safety margin before 330s kill


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


def _max_remaining_contacts(sequence, current_index):
    """Admissible heuristic: max possible additional H-H contacts from remaining residues."""
    h_count_remaining = sum(1 for i in range(current_index + 1, len(sequence)) if sequence[i] == 'H')
    h_count_placed = sum(1 for i in range(current_index + 1) if sequence[i] == 'H')
    # Each remaining H can contact at most 2 placed H's
    return min(h_count_placed * h_count_remaining, h_count_remaining * 2)


def _a_star_fold(sequence, dimension, time_limit=30.0):
    """A* search for short chains (<=30 residues)."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0,) * dimension]
    
    start = (0,) * dimension
    counter = 0
    initial_state = (0, counter, (start,), {start: 0}, 0)
    counter += 1
    open_heap = [initial_state]
    
    best_chain = None
    best_contacts = -1
    
    start_time = time.time()
    nodes_expanded = 0
    max_nodes = 200000
    
    while open_heap:
        if time.time() - start_time > time_limit:
            break
        
        priority, _, chain, pos_to_idx, contacts = heapq.heappop(open_heap)
        current_index = len(chain) - 1
        
        if current_index == n - 1:
            if contacts > best_contacts:
                best_contacts = contacts
                best_chain = chain
            continue
        
        nodes_expanded += 1
        if nodes_expanded > max_nodes:
            break
        
        last = chain[-1]
        
        for site in _neighbours(last):
            if site in pos_to_idx:
                continue
            
            new_chain = chain + (site,)
            new_pos_to_idx = dict(pos_to_idx)
            new_pos_to_idx[site] = current_index + 1
            new_contacts = contacts
            
            if sequence[current_index + 1] == 'H':
                for q in _neighbours(site):
                    if q in new_pos_to_idx:
                        j = new_pos_to_idx[q]
                        if j < current_index and sequence[j] == 'H':
                            new_contacts += 1
            
            heuristic = _max_remaining_contacts(sequence, current_index + 1)
            priority = -(new_contacts + heuristic)
            
            heapq.heappush(open_heap, (priority, counter, new_chain, new_pos_to_idx, new_contacts))
            counter += 1
    
    if best_chain is not None:
        return list(best_chain)
    
    return [tuple([i] + [0] * (dimension - 1)) for i in range(n)]


def _beam_fold(sequence, dimension, beam_width=200, time_limit=60.0):
    """Beam search with incremental contact tracking for medium chains."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0,) * dimension]
    
    start = (0,) * dimension
    beam = [((start,), {start: 0}, 0)]
    
    start_time = time.time()
    
    for step in range(1, n):
        if time.time() - start_time > time_limit:
            break
        
        candidates = []
        
        for chain, pos_to_idx, contacts in beam:
            last = chain[-1]
            for site in _neighbours(last):
                if site in pos_to_idx:
                    continue
                
                new_chain = chain + (site,)
                new_pos_to_idx = dict(pos_to_idx)
                new_pos_to_idx[site] = step
                new_contacts = contacts
                
                if sequence[step] == 'H':
                    for q in _neighbours(site):
                        if q in new_pos_to_idx:
                            j = new_pos_to_idx[q]
                            if j < step - 1 and sequence[j] == 'H':
                                new_contacts += 1
                
                candidates.append((new_chain, new_pos_to_idx, new_contacts))
        
        if not candidates:
            break
        
        # Sort by contacts (descending) and keep top beam_width
        candidates.sort(key=lambda x: -x[2])
        beam = candidates[:beam_width]
    
    if not beam:
        return [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    best_chain = max(beam, key=lambda x: x[2])
    return list(best_chain[0])


def _greedy_grow(sequence, dimension, rng, time_limit=None):
    """One randomized greedy chain growth."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        if time_limit is not None and time.time() > time_limit:
            return None
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _local_refine(sequence, chain, dimension, max_iters=3000, rng=None, time_limit=None):
    """Local search with kink moves, swaps, and reversal moves."""
    if len(chain) < 3:
        return chain
    
    if rng is None:
        rng = random.Random()
        
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    
    pos_to_idx = {p: i for i, p in enumerate(best_chain)}
    
    for it in range(max_iters):
        if time_limit is not None and time.time() > time_limit:
            break
        if time.time() > _DEADLINE - 15:
            break
        
        improved = False
        n = len(best_chain)
        
        # Try multiple random moves
        for _ in range(80):
            move_type = rng.random()
            
            if move_type < 0.5:
                # Kink move: pick two non-adjacent residues and move one to neighbor of other
                i = rng.randint(0, n - 1)
                j = rng.randint(0, n - 1)
                if abs(i - j) < 3:
                    continue
                
                old_pos = best_chain[j]
                target_pos = best_chain[i]
                
                for neighbor in _neighbours(target_pos):
                    if neighbor == old_pos:
                        continue
                    if neighbor in pos_to_idx:
                        occ = pos_to_idx[neighbor]
                        if occ == j or abs(occ - j) <= 1:
                            continue
                        continue
                    
                    # Tentative move
                    best_chain[j] = neighbor
                    new_pos_to_idx = {p: idx for idx, p in enumerate(best_chain)}
                    
                    if len(new_pos_to_idx) != n:
                        best_chain[j] = old_pos
                        continue
                    
                    valid = True
                    for k in range(n - 1):
                        p1, p2 = best_chain[k], best_chain[k + 1]
                        dist = sum(abs(a - b) for a, b in zip(p1, p2))
                        if dist != 1:
                            valid = False
                            break
                    
                    if not valid:
                        best_chain[j] = old_pos
                        continue
                    
                    e = _energy(sequence, best_chain)
                    if e < best_energy:
                        best_energy = e
                        pos_to_idx = new_pos_to_idx
                        improved = True
                        break
                    else:
                        best_chain[j] = old_pos
                if improved:
                    break
            
            elif move_type < 0.8:
                # Swap move: swap two non-adjacent residues
                i = rng.randint(0, n - 1)
                j = rng.randint(0, n - 1)
                if abs(i - j) < 2 or abs(i - j) > 3:
                    continue
                
                p_i, p_j = best_chain[i], best_chain[j]
                
                # Check if swap maintains unit steps
                valid = True
                for k in (i - 1, i, j - 1, j):
                    if 0 <= k < n - 1:
                        p1, p2 = best_chain[k], best_chain[k + 1]
                        dist = sum(abs(a - b) for a, b in zip(p1, p2))
                        if dist != 1:
                            valid = False
                            break
                    elif k == n - 1:
                        if k > 0:
                            p1, p2 = best_chain[k - 1], best_chain[k]
                            dist = sum(abs(a - b) for a, b in zip(p1, p2))
                            if dist != 1:
                                valid = False
                                break
                
                if not valid:
                    continue
                
                # Perform swap
                best_chain[i], best_chain[j] = p_j, p_i
                new_pos_to_idx = {p: idx for idx, p in enumerate(best_chain)}
                
                if len(new_pos_to_idx) != n:
                    best_chain[i], best_chain[j] = p_i, p_j
                    continue
                
                e = _energy(sequence, best_chain)
                if e < best_energy:
                    best_energy = e
                    pos_to_idx = new_pos_to_idx
                    improved = True
                    break
                else:
                    best_chain[i], best_chain[j] = p_i, p_j
            else:
                # Reversal move: reverse a segment of the chain
                i = rng.randint(0, n - 2)
                j = rng.randint(i + 2, n - 1)
                
                # Check if reversal maintains unit steps at boundaries
                valid = True
                if i > 0:
                    p1, p2 = best_chain[i - 1], best_chain[j]
                    dist = sum(abs(a - b) for a, b in zip(p1, p2))
                    if dist != 1:
                        valid = False
                if j < n - 1:
                    p1, p2 = best_chain[i], best_chain[j + 1]
                    dist = sum(abs(a - b) for a, b in zip(p1, p2))
                    if dist != 1:
                        valid = False
                
                if not valid:
                    continue
                
                # Perform reversal
                segment = best_chain[i:j + 1]
                segment.reverse()
                best_chain[i:j + 1] = segment
                new_pos_to_idx = {p: idx for idx, p in enumerate(best_chain)}
                
                if len(new_pos_to_idx) != n:
                    segment.reverse()
                    best_chain[i:j + 1] = segment
                    continue
                
                e = _energy(sequence, best_chain)
                if e < best_energy:
                    best_energy = e
                    pos_to_idx = new_pos_to_idx
                    improved = True
                    break
                else:
                    segment.reverse()
                    best_chain[i:j + 1] = segment
            
            if improved:
                break
        
        if not improved:
            break
    
    return best_chain


def fold(sequence, dimension):
    """Best conformation using A*/beam search + local refinement."""
    rng = random.Random(12345 + len(sequence))
    best = None
    best_energy = 0
    
    seq_len = len(sequence)
    seq_deadline = _DEADLINE - 10
    
    # Choose search strategy based on sequence length
    if seq_len <= 30:
        # Use A* search for short chains
        chain = _a_star_fold(sequence, dimension, time_limit=30.0)
        if chain:
            refined = _local_refine(sequence, chain, dimension, max_iters=2000, rng=rng, time_limit=seq_deadline)
            value = _energy(sequence, refined)
            if value < best_energy:
                best = refined
                best_energy = value
    elif seq_len <= 64:
        # Use beam search for medium chains
        chain = _beam_fold(sequence, dimension, beam_width=200, time_limit=45.0)
        if chain:
            refined = _local_refine(sequence, chain, dimension, max_iters=3000, rng=rng, time_limit=seq_deadline)
            value = _energy(sequence, refined)
            if value < best_energy:
                best = refined
                best_energy = value
    else:
        # Use beam search with smaller width for long chains
        chain = _beam_fold(sequence, dimension, beam_width=100, time_limit=30.0)
        if chain:
            refined = _local_refine(sequence, chain, dimension, max_iters=2000, rng=rng, time_limit=seq_deadline)
            value = _energy(sequence, refined)
            if value < best_energy:
                best = refined
                best_energy = value
    
    # Additional greedy restarts if time remains
    remaining_time = seq_deadline - time.time()
    if remaining_time > 5:
        n_restarts = min(50, int(remaining_time * 10))
        for r in range(n_restarts):
            if time.time() > seq_deadline:
                break
            
            chain = _greedy_grow(sequence, dimension, rng, time_limit=seq_deadline)
            if chain is None:
                continue
            
            refined = _local_refine(sequence, chain, dimension, max_iters=min(1000, 20 * seq_len), rng=rng, time_limit=seq_deadline)
            value = _energy(sequence, refined)
            
            if value < best_energy:
                best = refined
                best_energy = value
    
    if best is None:
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(seq_len)]
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
