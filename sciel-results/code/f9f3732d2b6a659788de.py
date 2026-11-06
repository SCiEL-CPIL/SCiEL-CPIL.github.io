"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import sys

_GLOBAL_START = time.time()
_GLOBAL_LIMIT = 195.0  # Safety margin for 210s kill


def _time_left():
    return _GLOBAL_LIMIT - (time.time() - _GLOBAL_START)


def _neighbours(point):
    """Yield the 4 neighbours of a 2D lattice point."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _energy(sequence, chain):
    """Compute the HP contact energy of a conformation."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _build_contact_priority(sequence):
    """Pre-identify high-priority H-H pairs based on sequence distance.
    Pairs that are far apart in sequence but both H are candidates for contacts.
    Returns a dict mapping (i, j) with i < j to a priority score."""
    h_indices = [i for i, c in enumerate(sequence) if c == "H"]
    priority = {}
    for a_idx in range(len(h_indices)):
        for b_idx in range(a_idx + 1, len(h_indices)):
            i, j = h_indices[a_idx], h_indices[b_idx]
            dist = j - i
            # Higher priority for pairs that are far apart (more likely to form contacts in folded state)
            # but not too far (must be feasible on lattice)
            if 3 <= dist <= len(sequence) - 2:
                # Score: prefer medium-distance pairs, penalize very close (already bonded) and very far
                if dist <= 10:
                    score = 2.0
                elif dist <= 20:
                    score = 3.0
                elif dist <= 40:
                    score = 4.0
                else:
                    score = 3.0
                priority[(i, j)] = score
    return priority


def _count_priority_contacts(sequence, chain, priority_map):
    """Count how many high-priority H-H contacts are satisfied in the conformation."""
    where = {p: i for i, p in enumerate(chain)}
    count = 0
    for (i, j), _ in priority_map.items():
        pi = chain[i]
        pj = chain[j]
        if abs(pi[0] - pj[0]) + abs(pi[1] - pj[1]) == 1:
            count += 1
    return count


def _grow_with_priority(sequence, dimension, rng, priority_map):
    """Greedy chain growth that prioritizes satisfying high-priority H-H contacts."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    
    # Pre-compute for each position i, which future H positions have high priority with i
    h_indices = [i for i, c in enumerate(sequence) if c == "H"]
    
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            
            gain = 0
            if sequence[i] == "H":
                # Count new H-H contacts with previously placed H residues
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        # Check if this is a high-priority pair
                        pair_key = (min(j, i), max(j, i))
                        gain += priority_map.get(pair_key, 1.0)
            
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _pivot_move(chain, rng):
    """Perform a single pivot move: rotate a subchain by 90, 180, or 270 degrees around a pivot point."""
    if len(chain) < 4:
        return None
    
    # Choose a random pivot index (not the first or last)
    pivot_idx = rng.randint(1, len(chain) - 2)
    pivot = chain[pivot_idx]
    
    # Get the segment to rotate (from pivot_idx to end)
    segment = chain[pivot_idx:]
    
    # Try different rotations
    for angle in [90, 180, 270, -90]:
        # Rotate segment around pivot
        rotated = []
        for p in segment:
            dx, dy = p[0] - pivot[0], p[1] - pivot[1]
            if angle == 90:
                ndx, ndy = -dy, dx
            elif angle == -90:
                ndx, ndy = dy, -dx
            elif angle == 180:
                ndx, ndy = -dx, -dy
            else:
                ndx, ndy = dx, dy
            rotated.append((pivot[0] + ndx, pivot[1] + ndy))
        
        # Check if the rotated segment is self-avoiding and doesn't collide with the fixed part
        new_chain = chain[:pivot_idx] + rotated
        new_occupied = set(new_chain)
        if len(new_occupied) == len(new_chain):
            # Check unit steps
            valid = True
            for i in range(1, len(new_chain)):
                dx = abs(new_chain[i][0] - new_chain[i-1][0])
                dy = abs(new_chain[i][1] - new_chain[i-1][1])
                if dx + dy != 1:
                    valid = False
                    break
            if valid:
                return new_chain
    
    return None


def _crankshaft_move(chain, rng):
    """Perform a crankshaft move: rotate an internal segment of length 3 or 4."""
    if len(chain) < 6:
        return None
    
    # Choose a random segment start (at least 2 from start and end)
    seg_start = rng.randint(2, len(chain) - 4)
    seg_len = rng.choice([3, 4])
    
    if seg_start + seg_len > len(chain):
        return None
    
    segment = chain[seg_start:seg_start + seg_len]
    pivot = chain[seg_start]
    
    # Try rotations of the segment
    for angle in [90, 180, 270, -90]:
        rotated = []
        for p in segment:
            dx, dy = p[0] - pivot[0], p[1] - pivot[1]
            if angle == 90:
                ndx, ndy = -dy, dx
            elif angle == -90:
                ndx, ndy = dy, -dx
            elif angle == 180:
                ndx, ndy = -dx, -dy
            else:
                ndx, ndy = dx, dy
            rotated.append((pivot[0] + ndx, pivot[1] + ndy))
        
        new_chain = chain[:seg_start] + rotated + chain[seg_start + seg_len:]
        
        # Check self-avoidance
        new_occupied = set(new_chain)
        if len(new_occupied) != len(new_chain):
            continue
        
        # Check unit steps at boundaries
        valid = True
        # Check connection before segment
        if seg_start > 0:
            dx = abs(new_chain[seg_start][0] - new_chain[seg_start - 1][0])
            dy = abs(new_chain[seg_start][1] - new_chain[seg_start - 1][1])
            if dx + dy != 1:
                valid = False
        # Check connection after segment
        if seg_start + seg_len < len(new_chain):
            dx = abs(new_chain[seg_start + seg_len][0] - new_chain[seg_start + seg_len - 1][0])
            dy = abs(new_chain[seg_start + seg_len][1] - new_chain[seg_start + seg_len - 1][1])
            if dx + dy != 1:
                valid = False
        
        if valid:
            return new_chain
    
    return None


def _targeted_refinement(sequence, chain, priority_map, time_limit=3.0):
    """Targeted refinement that specifically tries to satisfy unsatisfied high-priority contacts."""
    if chain is None:
        return None
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    start_time = time.time()
    rng = random.Random(hash(tuple(current_chain)) % (2**32))
    
    max_iters = 2000
    for iteration in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 15:
            break
        
        # Find unsatisfied high-priority contacts
        where = {p: i for i, p in enumerate(current_chain)}
        unsatisfied = []
        for (i, j), score in priority_map.items():
            pi = current_chain[i]
            pj = current_chain[j]
            if abs(pi[0] - pj[0]) + abs(pi[1] - pj[1]) != 1:
                unsatisfied.append((i, j, score))
        
        if not unsatisfied:
            break
        
        # Pick a random unsatisfied contact to try to satisfy
        target_i, target_j, _ = rng.choice(unsatisfied)
        
        # Try to move one of the two residues to be adjacent to the other
        # We'll try pivot moves centered near the target residues
        success = False
        for _attempt in range(10):
            if _time_left() < 15:
                break
            
            # Try pivot at a position between target_i and target_j
            mid_idx = (target_i + target_j) // 2
            pivot_idx = max(1, min(mid_idx, len(current_chain) - 2))
            
            new_chain = _pivot_move(current_chain, rng)
            if new_chain is None:
                continue
            
            new_energy = _energy(sequence, new_chain)
            delta = new_energy - current_energy
            
            # Accept if it improves energy or satisfies a priority contact
            if delta < 0:
                current_chain = new_chain
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
                success = True
                break
        
        # If no targeted move succeeded, do a random move
        if not success:
            if rng.random() < 0.6:
                new_chain = _pivot_move(current_chain, rng)
            else:
                new_chain = _crankshaft_move(current_chain, rng)
            
            if new_chain is not None:
                new_energy = _energy(sequence, new_chain)
                delta = new_energy - current_energy
                if delta < 0 or rng.random() < 0.3:
                    current_chain = new_chain
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
    
    return best_chain


def _local_search(sequence, chain, priority_map, max_iterations=1000, time_limit=5.0):
    """Simulated annealing local search with priority-aware acceptance."""
    if chain is None:
        return None
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    start_time = time.time()
    temperature = 3.0
    cooling = 0.997
    
    rng = random.Random(hash(tuple(current_chain)) % (2**32))
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 10:
            break
        
        # Choose move type: pivot or crankshaft
        if rng.random() < 0.6:
            new_chain = _pivot_move(current_chain, rng)
        else:
            new_chain = _crankshaft_move(current_chain, rng)
        
        if new_chain is None:
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        # Compute change in priority contacts satisfied
        old_priority = _count_priority_contacts(sequence, current_chain, priority_map)
        new_priority = _count_priority_contacts(sequence, new_chain, priority_map)
        priority_delta = new_priority - old_priority
        
        # Accept if energy improves, or if priority contacts improve significantly
        accept = False
        if delta < 0:
            accept = True
        elif priority_delta > 0 and delta < 2:
            accept = True
        elif rng.random() < pow(2.718281828, -delta / max(temperature, 0.1)):
            accept = True
        
        if accept:
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        
        temperature *= cooling
    
    return best_chain


def fold(sequence, dimension):
    """Best conformation using priority-guided growth + targeted refinement + local search."""
    seq_len = len(sequence)
    
    # Build contact priority map
    priority_map = _build_contact_priority(sequence)
    
    # Adjust strategy based on sequence length
    if seq_len <= 30:
        num_grow = 400
        local_search_iters = 3000
        local_search_time = 10.0
        refine_time = 5.0
    elif seq_len <= 50:
        num_grow = 300
        local_search_iters = 2000
        local_search_time = 8.0
        refine_time = 4.0
    elif seq_len <= 70:
        num_grow = 200
        local_search_iters = 1500
        local_search_time = 6.0
        refine_time = 3.0
    else:
        num_grow = 150
        local_search_iters = 1000
        local_search_time = 4.0
        refine_time = 2.0
    
    rng = random.Random(12345 + seq_len)
    best_chain = None
    best_energy = 0
    
    # First pass: priority-guided greedy growth, track top few
    top_chains = []
    for i in range(num_grow):
        if _time_left() < 30:
            break
        
        chain = _grow_with_priority(sequence, dimension, rng, priority_map)
        if chain is None:
            continue
        
        value = _energy(sequence, chain)
        if best_chain is None or value < best_energy:
            best_energy = value
            best_chain = chain
            top_chains = [(value, chain)]
        elif len(top_chains) < 5 and value < best_energy + 5:
            top_chains.append((value, chain))
            top_chains.sort(key=lambda x: x[0])
    
    # Second pass: targeted refinement on best chain
    if best_chain is not None and _time_left() > 40:
        refined = _targeted_refinement(sequence, best_chain, priority_map, time_limit=refine_time)
        if refined is not None:
            refined_energy = _energy(sequence, refined)
            if refined_energy < best_energy:
                best_energy = refined_energy
                best_chain = refined
    
    # Third pass: local search on top chains
    if best_chain is not None and _time_left() > 35:
        # Local search on the best chain with most iterations
        improved = _local_search(sequence, best_chain, priority_map, max_iterations=local_search_iters, 
                                  time_limit=local_search_time)
        if improved is not None:
            improved_energy = _energy(sequence, improved)
            if improved_energy < best_energy:
                best_energy = improved_energy
                best_chain = improved
        
        # Local search on other good chains
        for val, ch in top_chains[1:]:
            if _time_left() < 20:
                break
            if val >= best_energy + 3:
                continue
            
            improved = _local_search(sequence, ch, priority_map, max_iterations=local_search_iters // 2,
                                      time_limit=local_search_time / 2)
            if improved is not None:
                improved_energy = _energy(sequence, improved)
                if improved_energy < best_energy:
                    best_energy = improved_energy
                    best_chain = improved
    
    # Final fallback: straight line
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(seq_len)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
