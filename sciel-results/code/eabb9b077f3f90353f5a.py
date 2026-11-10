"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import sys

_DEADLINE = time.time() + 175.0


def _compute_energy(sequence, chain):
    """Compute the negative number of non-adjacent H-H contacts."""
    n = len(chain)
    if n < 2:
        return 0
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    contacts = 0
    for i in range(n):
        if sequence[i] == 'H':
            x, y = chain[i]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                j = where.get((x + dx, y + dy))
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts // 2


def _kink_energy_delta(sequence, chain, occupied, i, new_pos):
    """
    Compute the change in contact count when moving residue i to new_pos.
    Only considers contacts involving residue i.
    Returns delta in contacts (negative = fewer contacts = worse, positive = more contacts = better).
    """
    n = len(chain)
    if i < 1 or i > n - 2:
        return 0
    
    old_pos = chain[i]
    old_x, old_y = old_pos
    new_x, new_y = new_pos
    
    # Count contacts involving residue i at old position
    old_contacts = 0
    if sequence[i] == 'H':
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor_pos = (old_x + dx, old_y + dy)
            j = occupied.get(neighbor_pos)
            if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                old_contacts += 1
    
    # Count contacts involving residue i at new position
    # Temporarily remove old position from occupied for lookup
    del occupied[old_pos]
    occupied[new_pos] = i
    
    new_contacts = 0
    if sequence[i] == 'H':
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor_pos = (new_x + dx, new_y + dy)
            j = occupied.get(neighbor_pos)
            if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                new_contacts += 1
    
    # Restore
    del occupied[new_pos]
    occupied[old_pos] = i
    
    return new_contacts - old_contacts


def _pivot_energy_delta(sequence, chain, occupied, pivot, new_dir):
    """
    Compute the change in contact count for a pivot move.
    Rotate the segment from pivot+1 to end around pivot.
    new_dir is the new direction from pivot to pivot+1.
    Returns delta in contacts, or None if the move is invalid.
    """
    n = len(chain)
    if pivot < 1 or pivot >= n - 1:
        return None
    
    old_pivot_pos = chain[pivot]
    old_next_pos = chain[pivot + 1]
    
    # New position of pivot+1
    px, py = old_pivot_pos
    nx, ny = px + new_dir[0], py + new_dir[1]
    
    # Check if new position is free (except for the segment being moved)
    # Build set of positions in the moving segment
    moving_positions = set()
    for i in range(pivot + 1, n):
        moving_positions.add(chain[i])
    
    if (nx, ny) not in moving_positions and (nx, ny) in occupied and occupied[(nx, ny)] != pivot + 1:
        return None
    
    # Simulate the pivot move
    # The segment from pivot+1 to end gets rotated
    # We need to compute the new positions
    
    # Direction from pivot to pivot+1 in old chain
    old_dx = old_next_pos[0] - px
    old_dy = old_next_pos[1] - py
    
    # We need to rotate the entire segment
    # This is complex, so let's just compute the full energy difference
    
    # For now, return None to indicate we should compute full energy
    return None


def _fast_grow(sequence, rng, temperature=0.0):
    """
    Fast randomized greedy chain growth.
    Prefer sites that create new H-H contacts, then sites with more free space.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    res_type = [1 if c == 'H' else 0 for c in sequence]
    
    for i in range(1, n):
        last = chain[-1]
        x, y = last
        options = []
        
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            site = (x + dx, y + dy)
            if site in occupied:
                continue
            
            gain = 0
            if res_type[i]:
                sx, sy = site
                for ddx, ddy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    j = occupied.get((sx + ddx, sy + ddy))
                    if j is not None and j < i - 1 and res_type[j]:
                        gain += 1
            
            free = 0
            for ddx, ddy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if (site[0] + ddx, site[1] + ddy) not in occupied:
                    free += 1
            
            score = gain * 10.0 + free * 0.1 + rng.random() * temperature
            options.append((score, site))
        
        if not options:
            return None
        
        best_score = -1e18
        best_site = None
        for s, st in options:
            if s > best_score:
                best_score = s
                best_site = st
        
        occupied[best_site] = i
        chain.append(best_site)

    return chain


def _local_improve_fast(sequence, chain, rng, max_moves=5000):
    """
    Improve conformation via kink flips with incremental energy computation.
    """
    n = len(chain)
    if n < 3:
        return chain, _compute_energy(sequence, chain)

    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    no_improve_count = 0
    
    res_type = [1 if c == 'H' else 0 for c in sequence]
    
    for _ in range(max_moves):
        if time.time() > _DEADLINE - 5:
            break

        # Kink flip: rotate residue i around residue i-1
        i = rng.randint(1, n - 2)
        p = chain[i]
        prev = chain[i - 1]
        nxt = chain[i + 1]
        px, py = prev
        
        candidates = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            np = (px + dx, py + dy)
            if np == p:
                continue
            if abs(np[0] - nxt[0]) + abs(np[1] - nxt[1]) != 1:
                continue
            if np in occupied and occupied[np] != i:
                continue
            candidates.append(np)

        if candidates:
            np = candidates[rng.randint(0, len(candidates) - 1)]
            
            # Use incremental energy delta
            delta_contacts = _kink_energy_delta(sequence, chain, occupied, i, np)
            delta_energy = -delta_contacts // 2
            
            del occupied[p]
            chain[i] = np
            occupied[np] = i
            
            new_energy = current_energy + delta_energy

            if new_energy < current_energy:
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(chain)
                no_improve_count = 0
            else:
                del occupied[np]
                chain[i] = p
                occupied[p] = i
                no_improve_count += 1
                if no_improve_count > 500:
                    break

    return best_chain, best_energy


def _regrow_sa_fast(sequence, chain, rng, max_iters=5000, start_temp=6.0, cool_rate=0.996):
    """
    Simulated annealing using regrow moves: pick a split point, regrow the suffix.
    """
    n = len(sequence)
    if n < 4:
        return chain, _compute_energy(sequence, chain)

    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    temp = start_temp
    no_improve = 0
    stuck_count = 0

    for it in range(max_iters):
        if time.time() > _DEADLINE - 10:
            break
        if temp < 0.05:
            break

        min_split = 3
        max_split = max(min_split, n - 3)
        if max_split <= min_split:
            break
        
        mid = (min_split + max_split) // 2
        if rng.random() < 0.7:
            offset = int(rng.gauss(0, (max_split - min_split) / 4))
            split = min(max(min_split, mid + offset), max_split)
        else:
            split = rng.randint(min_split, max_split)

        prefix = chain[:split]
        
        new_suffix = []
        last = prefix[-1]
        occupied = {p: i for i, p in enumerate(prefix)}
        success = True
        
        for i in range(split, n):
            x, y = last
            options = []
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                site = (x + dx, y + dy)
                if site in occupied:
                    continue
                gain = 0
                if sequence[i] == 'H':
                    sx, sy = site
                    for ddx, ddy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        j = occupied.get((sx + ddx, sy + ddy))
                        if j is not None and j < i - 1 and sequence[j] == 'H':
                            gain += 1
                free = 0
                for ddx, ddy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    if (site[0] + ddx, site[1] + ddy) not in occupied:
                        free += 1
                score = gain * 10.0 + free * 0.1 + rng.random() * temp
                options.append((score, site))
            
            if not options:
                success = False
                break
            
            best_score = -1e18
            best_site = None
            for s, st in options:
                if s > best_score:
                    best_score = s
                    best_site = st
            
            occupied[best_site] = i
            last = best_site
            new_suffix.append(best_site)
        
        if not success:
            temp *= cool_rate
            no_improve += 1
            stuck_count += 1
            continue

        new_chain = prefix + new_suffix
        new_energy = _compute_energy(sequence, new_chain)

        delta = new_energy - current_energy
        if delta < 0 or (temp > 0 and rng.random() < (10 ** (-delta / max(temp, 0.1)))):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
                no_improve = 0
                stuck_count = 0
            else:
                no_improve += 1
                stuck_count += 1
        else:
            no_improve += 1
            stuck_count += 1

        if stuck_count > 150:
            temp *= cool_rate ** 1.5
            stuck_count = 0
        else:
            temp *= cool_rate

        if no_improve > 800:
            break

    return best_chain, best_energy


def _min_conflicts_fold(sequence, rng, max_iters=6000, grid_size=24):
    """
    Min-conflicts style search on a grid representation.
    Represent the conformation as a self-avoiding walk on a grid.
    Use random restarts and local search to minimize energy.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    best_chain = None
    best_energy = float('inf')
    
    for restart in range(15):
        if time.time() > _DEADLINE - 15:
            break
        
        # Initialize with a random self-avoiding walk
        # Start from the center of the grid
        cx, cy = grid_size // 2, grid_size // 2
        chain = [(cx, cy)]
        occupied = {(cx, cy): 0}
        
        success = True
        for i in range(1, n):
            last = chain[-1]
            x, y = last
            options = []
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                site = (x + dx, y + dy)
                if 0 <= site[0] < grid_size and 0 <= site[1] < grid_size:
                    if site not in occupied:
                        options.append(site)
            
            if not options:
                success = False
                break
            
            site = options[rng.randint(0, len(options) - 1)]
            occupied[site] = i
            chain.append(site)
        
        if not success:
            continue
        
        energy = _compute_energy(sequence, chain)
        
        # Local search: kink flips with incremental energy
        for _ in range(3000):
            if time.time() > _DEADLINE - 15:
                break
            
            i = rng.randint(1, n - 2)
            p = chain[i]
            prev = chain[i - 1]
            nxt = chain[i + 1]
            px, py = prev
            
            candidates = []
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                np = (px + dx, py + dy)
                if np == p:
                    continue
                if abs(np[0] - nxt[0]) + abs(np[1] - nxt[1]) != 1:
                    continue
                if np in occupied and occupied[np] != i:
                    continue
                if not (0 <= np[0] < grid_size and 0 <= np[1] < grid_size):
                    continue
                candidates.append(np)
            
            if candidates:
                np = candidates[rng.randint(0, len(candidates) - 1)]
                
                delta_contacts = _kink_energy_delta(sequence, chain, occupied, i, np)
                delta_energy = -delta_contacts // 2
                
                del occupied[p]
                chain[i] = np
                occupied[np] = i
                
                new_energy = energy + delta_energy
                
                if new_energy < energy:
                    energy = new_energy
                else:
                    del occupied[np]
                    chain[i] = p
                    occupied[p] = i
        
        if energy < best_energy:
            best_energy = energy
            best_chain = list(chain)
    
    if best_chain is None:
        # Fallback
        return _fast_grow(sequence, rng, temperature=0.0) or [[i, 0] for i in range(n)]
    
    return best_chain


def fold(sequence, dimension):
    """
    Fold a sequence into a low-energy conformation.
    """
    if dimension != 2:
        raise ValueError("Only dimension 2 is supported")

    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]

    rng = random.Random(12345 + len(sequence))

    best_chain = None
    best_energy = float('inf')

    # Phase 1: Generate diverse initial solutions with fast growth
    initial_chains = []
    
    temps = [0.0, 0.0, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    for temp in temps:
        for _ in range(15):
            if time.time() > _DEADLINE - 40:
                break
            chain = _fast_grow(sequence, rng, temperature=temp)
            if chain is not None:
                initial_chains.append(chain)
        
        if time.time() > _DEADLINE - 40:
            break

    # Evaluate all initial chains and keep the top K
    evaluated_chains = []
    for chain in initial_chains:
        energy = _compute_energy(sequence, chain)
        evaluated_chains.append((energy, chain))
    
    evaluated_chains.sort(key=lambda x: x[0])
    
    if not evaluated_chains:
        best_chain = [[i, 0] for i in range(n)]
        best_energy = _compute_energy(sequence, best_chain)
    else:
        top_k = 8
        for energy, chain in evaluated_chains[:top_k]:
            if time.time() > _DEADLINE - 35:
                break
            
            if time.time() < _DEADLINE - 30:
                improved_chain, improved_energy = _local_improve_fast(sequence, chain, rng, max_moves=5000)
                if improved_energy < energy:
                    energy = improved_energy
                    chain = improved_chain
            
            if time.time() < _DEADLINE - 25:
                sa_chain, sa_energy = _regrow_sa_fast(sequence, chain, rng, max_iters=5000, start_temp=6.0, cool_rate=0.996)
                if sa_energy < energy:
                    energy = sa_energy
                    chain = sa_chain
            
            if time.time() < _DEADLINE - 20:
                improved_chain2, improved_energy2 = _local_improve_fast(sequence, chain, rng, max_moves=3000)
                if improved_energy2 < energy:
                    energy = improved_energy2
                    chain = improved_chain2
            
            if energy < best_energy:
                best_energy = energy
                best_chain = chain

    # Phase 2: More restarts with local search + SA
    if time.time() < _DEADLINE - 25:
        restarts_done = 0
        max_restarts = 100
        while restarts_done < max_restarts:
            if time.time() > _DEADLINE - 25:
                break
            
            temp = [0.0, 0.3, 0.5, 1.0, 1.5, 2.0][restarts_done % 6]
            chain = _fast_grow(sequence, rng, temperature=temp)
            
            restarts_done += 1
            
            if chain is None:
                continue
            
            energy = _compute_energy(sequence, chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = chain
            
            if time.time() < _DEADLINE - 20:
                improved = _local_improve_fast(sequence, chain, rng, max_moves=3000)
                if improved[1] < best_energy:
                    best_energy = improved[1]
                    best_chain = improved[0]
            
            if time.time() < _DEADLINE - 15:
                sa_c, sa_e = _regrow_sa_fast(sequence, best_chain, rng, max_iters=3000, start_temp=5.0, cool_rate=0.997)
                if sa_e < best_energy:
                    best_energy = sa_e
                    best_chain = sa_c

    # Phase 3: Try min-conflicts approach
    if time.time() < _DEADLINE - 15:
        mc_chain = _min_conflicts_fold(sequence, rng, max_iters=6000, grid_size=24)
        if mc_chain:
            mc_energy = _compute_energy(sequence, mc_chain)
            if mc_energy < best_energy:
                best_energy = mc_energy
                best_chain = mc_chain

    # Final local polish
    if time.time() < _DEADLINE - 5:
        for _ in range(4):
            if time.time() > _DEADLINE - 5:
                break
            final_chain, final_energy = _local_improve_fast(sequence, best_chain, rng, max_moves=4000)
            if final_energy < best_energy:
                best_energy = final_energy
                best_chain = final_chain
            else:
                break

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
