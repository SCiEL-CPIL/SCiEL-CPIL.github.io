"""HP lattice protein folding on the cubic lattice."""
# EVOLVE-BLOCK-START
import random
import time
import heapq

_DEADLINE = time.time() + 320  # safety margin before 330s kill


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Energy = -(# of H-H non-adjacent lattice contacts)."""
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
    """One randomized greedy chain growth with trap penalty."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            trap_penalty = -0.5 if free == 1 else 0.0
            options.append((gain + 0.1 * free + trap_penalty + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _astar_fold(sequence, dimension, time_limit):
    """A* search for short chains (<=48 residues) using admissible H-H contact heuristic.
    Returns (chain, energy) or (None, None) if no solution found within time limit."""
    n = len(sequence)
    if n < 3:
        return None, None
    
    # Precompute H positions
    h_positions = [i for i, c in enumerate(sequence) if c == 'H']
    
    # State: (chain_tuple, occupied_frozenset, current_contacts)
    # For efficiency, we use a priority queue with (f_score, g_score, chain, occupied, contacts)
    # f = g + h, where g = -contacts (cost so far, negative is better), h = admissible lower bound on remaining cost
    
    start = (0,) * dimension
    initial_chain = (start,)
    initial_occupied = frozenset([start])
    initial_contacts = 0
    
    # Admissible heuristic: max possible additional H-H contacts from remaining residues
    # For each remaining H residue, it can contact at most 6 already-placed H residues
    # but we need a tighter bound. Simple admissible: 0 (always admissible but weak)
    # Better: count how many H residues remain and estimate max contacts
    def _admissible_h(placed_count, contacts_so_far):
        """Admissible heuristic: minimum possible additional cost (negative contacts).
        Since we can always get 0 more contacts, h = 0 is admissible.
        For better pruning, estimate max possible contacts remaining."""
        remaining_h = sum(1 for i in range(placed_count, n) if sequence[i] == 'H')
        # Each remaining H can contact at most (placed_h_count) existing H residues
        placed_h = sum(1 for i in range(placed_count) if sequence[i] == 'H')
        # Max additional contacts: each new H can contact up to min(6, placed_h) existing H
        # But this is an upper bound on contacts, so lower bound on cost is -max_contacts
        max_additional = 0
        for i in range(placed_count, n):
            if sequence[i] == 'H':
                # This H can contact at most min(6, placed_h) existing H residues
                # But some of those might be adjacent in chain, so subtract those
                max_additional += min(6, placed_h)
                placed_h += 1
        return -max_additional  # lower bound on additional cost (more negative = better)
    
    # Priority queue: (f_score, tie_breaker, chain, occupied, contacts)
    # f_score = -contacts + h (lower is better)
    start_h = _admissible_h(1, 0)
    start_f = 0 + start_h  # g=0, h=start_h
    counter = 0
    open_set = [(start_f, counter, initial_chain, initial_occupied, 0)]
    best_g = {initial_chain: 0}
    
    while open_set:
        if time.time() > time_limit:
            return None, None
        
        f, _, chain, occupied, contacts = heapq.heappop(open_set)
        
        if len(chain) == n:
            return list(chain), -contacts
        
        last_pos = chain[-1]
        i = len(chain) - 1  # index of last placed residue
        next_i = i + 1  # next residue to place
        
        for site in _neighbours(last_pos):
            if site in occupied:
                continue
            
            new_occupied = occupied | {site}
            new_chain = chain + (site,)
            new_contacts = contacts
            
            if sequence[next_i] == 'H':
                # Count new H-H contacts
                for q in _neighbours(site):
                    if q in occupied:
                        # Find which residue is at q
                        # This is O(n) per neighbor, but for short chains it's OK
                        for j, p in enumerate(chain):
                            if p == q and j < i and sequence[j] == 'H':
                                new_contacts += 1
                                break
            
            g = -new_contacts
            h = _admissible_h(next_i + 1, new_contacts)
            f_new = g + h
            
            # Check if we've seen this state with a better g
            if new_chain in best_g:
                if g >= best_g[new_chain]:
                    continue
            
            best_g[new_chain] = g
            counter += 1
            heapq.heappush(open_set, (f_new, counter, new_chain, new_occupied, new_contacts))
    
    return None, None


def _is_valid_move(chain, i, new_site, dimension, pos_to_idx):
    """Check that moving residue i to new_site keeps the chain self-avoiding and connected."""
    if new_site in pos_to_idx:
        return False
    prev = chain[i - 1]
    nxt = chain[i + 1]
    d_prev = sum(abs(new_site[a] - prev[a]) for a in range(dimension))
    d_next = sum(abs(new_site[a] - nxt[a]) for a in range(dimension))
    if d_prev != 1 or d_next != 1:
        return False
    return True


def _is_valid_kink(chain, i, site1, site2, dimension, pos_to_idx):
    """Check that a 2-residue kink (i -> site1, i+1 -> site2) is valid."""
    if site1 in pos_to_idx or site2 in pos_to_idx:
        return False
    if site1 == site2:
        return False
    prev = chain[i - 1]
    nxt = chain[i + 2] if i + 2 < len(chain) else None
    d_prev = sum(abs(site1[a] - prev[a]) for a in range(dimension))
    d_mid = sum(abs(site2[a] - site1[a]) for a in range(dimension))
    if d_prev != 1 or d_mid != 1:
        return False
    if nxt is not None:
        d_next = sum(abs(site2[a] - nxt[a]) for a in range(dimension))
        if d_next != 1:
            return False
    return True


def _local_refine(sequence, chain, dimension, rng, max_steps=60):
    """Local search with 1-residue moves, 2-residue kink moves, and SA-style escape.
    Uses incremental position map for efficiency."""
    current_energy = _energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    n = len(chain)
    if n < 3:
        return best_chain

    # Build position map
    pos_to_idx = {p: i for i, p in enumerate(best_chain)}
    
    stall_count = 0
    for step in range(max_steps):
        if time.time() > _DEADLINE - 5:
            break

        moved = False

        # --- 1-residue move at a random internal position ---
        i = rng.randint(1, n - 2)
        for site in _neighbours(best_chain[i]):
            if site in pos_to_idx:
                continue
            if not _is_valid_move(best_chain, i, site, dimension, pos_to_idx):
                continue
            # Tentative move
            old_site = best_chain[i]
            best_chain[i] = site
            # Update pos_to_idx
            del pos_to_idx[old_site]
            pos_to_idx[site] = i
            
            new_energy = _energy(sequence, best_chain)
            if new_energy < best_energy:
                best_energy = new_energy
                moved = True
                stall_count = 0
                break
            elif new_energy == best_energy and rng.random() < 0.3:
                moved = True
                stall_count = 0
                break
            else:
                # Revert
                best_chain[i] = old_site
                del pos_to_idx[site]
                pos_to_idx[old_site] = i

        # --- 2-residue kink move ---
        if not moved and n >= 4:
            i = rng.randint(1, n - 3)
            for site1 in _neighbours(best_chain[i]):
                if site1 in pos_to_idx:
                    continue
                for site2 in _neighbours(site1):
                    if site2 in pos_to_idx or site2 == site1:
                        continue
                    if not _is_valid_kink(best_chain, i, site1, site2, dimension, pos_to_idx):
                        continue
                    # Tentative move
                    old_site1 = best_chain[i]
                    old_site2 = best_chain[i + 1]
                    best_chain[i] = site1
                    best_chain[i + 1] = site2
                    # Update pos_to_idx
                    del pos_to_idx[old_site1]
                    del pos_to_idx[old_site2]
                    pos_to_idx[site1] = i
                    pos_to_idx[site2] = i + 1
                    
                    new_energy = _energy(sequence, best_chain)
                    if new_energy < best_energy:
                        best_energy = new_energy
                        moved = True
                        stall_count = 0
                        break
                    elif new_energy == best_energy and rng.random() < 0.2:
                        moved = True
                        stall_count = 0
                        break
                    else:
                        # Revert
                        best_chain[i] = old_site1
                        best_chain[i + 1] = old_site2
                        del pos_to_idx[site1]
                        del pos_to_idx[site2]
                        pos_to_idx[old_site1] = i
                        pos_to_idx[old_site2] = i + 1
                if moved:
                    break

        if not moved:
            # Simulated-annealing style: try a few random 1-moves even if they worsen energy
            escaped = False
            for _ in range(3):
                j = rng.randint(1, n - 2)
                for site in _neighbours(best_chain[j]):
                    if site in pos_to_idx:
                        continue
                    if not _is_valid_move(best_chain, j, site, dimension, pos_to_idx):
                        continue
                    old_site = best_chain[j]
                    best_chain[j] = site
                    del pos_to_idx[old_site]
                    pos_to_idx[site] = j
                    
                    new_energy = _energy(sequence, best_chain)
                    if new_energy <= best_energy + 2:
                        escaped = True
                        break
                    else:
                        # Revert
                        best_chain[j] = old_site
                        del pos_to_idx[site]
                        pos_to_idx[old_site] = j
                if escaped:
                    break
            if escaped:
                stall_count += 1
                if stall_count >= 5:
                    break
            else:
                break

    return best_chain


def fold(sequence, dimension):
    """Best conformation using A* for short chains, greedy growth + local refinement for longer."""
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))

    best_chain = None
    best_energy = 0  # energy is negative, so 0 means no contacts

    # For short chains, try A* search first
    if n <= 48:
        astar_time_limit = min(30, _DEADLINE - time.time() - 5)
        if astar_time_limit > 5:
            astar_result, astar_energy = _astar_fold(sequence, dimension, time.time() + astar_time_limit)
            if astar_result is not None:
                best_chain = astar_result
                best_energy = astar_energy
                # Still try greedy + refine to see if we can do better
                # But A* should be optimal or near-optimal, so limit greedy attempts
    
    # Scale number of attempts based on chain length
    if n <= 48:
        num_attempts = 800 if best_chain is None else 200
    elif n <= 64:
        num_attempts = 400
    elif n <= 96:
        num_attempts = 200
    else:
        num_attempts = 100

    seq_deadline = _DEADLINE - 5

    for attempt in range(num_attempts):
        if time.time() > seq_deadline:
            break

        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue

        value = _energy(sequence, chain)

        if value < best_energy:
            best_chain = chain
            best_energy = value

        # Do refinement on promising candidates (within 2 contacts of best)
        if value <= best_energy + 2:
            refined = _local_refine(sequence, chain, dimension, rng, max_steps=60)
            refined_energy = _energy(sequence, refined)
            if refined_energy < best_energy:
                best_chain = refined
                best_energy = refined_energy

    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
