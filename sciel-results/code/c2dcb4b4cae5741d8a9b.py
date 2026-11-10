"""HP lattice protein folding - improved algorithm with local search."""
import time
import random
import sys

# Global time tracking
_START_TIME = time.time()
_TIME_BUDGET = 280  # seconds, leaving 20s safety margin before 300s limit


def _remaining_time():
    return max(0.0, _TIME_BUDGET - (time.time() - _START_TIME))


def _neighbours(point):
    """Generate all 6 neighbours in 3D cubic lattice."""
    n = len(point)
    for axis in range(n):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute H-H contact energy for a chain."""
    n = len(chain)
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _energy_delta_add(sequence, chain, occupied, pos, i):
    """Compute energy gain if we add residue i at position pos."""
    gain = 0
    if sequence[i] == 'H':
        for q in _neighbours(pos):
            j = occupied.get(q)
            if j is not None and j < i - 1 and sequence[j] == 'H':
                gain += 1
    return gain


def _energy_delta_remove(sequence, chain, occupied, i):
    """Compute energy loss if we remove residue i from position chain[i]."""
    loss = 0
    if sequence[i] == 'H':
        p = chain[i]
        for q in _neighbours(p):
            j = occupied.get(q)
            if j is not None and j > i + 1 and sequence[j] == 'H':
                loss += 1
    return loss


def _random_grow(sequence, dimension, rng, bias=0.3):
    """Random chain growth with slight bias toward hydrophobic contacts."""
    n = len(sequence)
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        last = chain[-1]
        candidates = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Count potential H-H contacts
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            # Count free space (escape from dead ends)
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 2.0 + free * bias + rng.random()
            candidates.append((score, site))
        
        if not candidates:
            return None
        
        # Weighted selection
        total = sum(c[0] for c in candidates)
        r = rng.random() * total
        cum = 0
        chosen = candidates[-1][1]
        for score, site in candidates:
            cum += score
            if r <= cum:
                chosen = site
                break
        
        chain.append(chosen)
        occupied[chosen] = i
    
    return chain


def _local_search(sequence, chain, occupied, current_energy, time_limit=5.0):
    """Local search: try moving each residue to a nearby empty site."""
    n = len(chain)
    start_time = time.time()
    improved = True
    
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        rng_local = random.Random(int(time.time() * 1000) % 100000)
        
        # Randomize order of positions to search
        order = list(range(n))
        rng_local.shuffle(order)
        
        for i in order:
            if (time.time() - start_time) >= time_limit:
                break
            
            if i == 0:
                # Don't move the first residue (anchor)
                continue
            
            pos = chain[i]
            
            # Try moving to neighbours of pos, and neighbours of neighbours
            move_targets = set()
            for q in _neighbours(pos):
                move_targets.add(q)
                for r in _neighbours(q):
                    if r not in occupied and r != pos:
                        move_targets.add(r)
            
            # Also try the original position's neighbours directly
            for q in _neighbours(pos):
                if q not in occupied:
                    move_targets.add(q)
            
            best_move = None
            best_new_energy = current_energy
            
            for target in move_targets:
                if target in occupied:
                    continue
                
                # Check self-avoidance: target must not be adjacent to chain[i-1] or chain[i+1]
                # Actually, we need to check that the move doesn't create non-consecutive adjacency
                # that would violate self-avoidance... wait, self-avoidance just means no two
                # consecutive residues occupy the same site. Non-consecutive residues can be
                # adjacent (that's what creates contacts).
                
                # But we need to make sure the move is valid:
                # 1. target not occupied (checked)
                # 2. The new position doesn't create a conflict with immediate neighbors
                #    Actually in a chain, consecutive residues must be adjacent. If we move
                #    residue i, we need chain[i-1] and chain[i+1] to still be adjacent to target.
                #    Wait no - the chain is a path, so chain[i-1] must be adjacent to chain[i],
                #    and chain[i] must be adjacent to chain[i+1]. If we move chain[i], both
                #    constraints must hold.
                
                # Check adjacency with i-1
                if i > 0:
                    d1 = sum(abs(a - b) for a, b in zip(target, chain[i-1]))
                    if d1 != 1:
                        continue
                
                # Check adjacency with i+1
                if i < n - 1:
                    d2 = sum(abs(a - b) for a, b in zip(target, chain[i+1]))
                    if d2 != 1:
                        continue
                
                # Compute energy change
                # Remove old contacts at pos
                old_gain = 0
                if sequence[i] == 'H':
                    for q in _neighbours(pos):
                        j = occupied.get(q)
                        if j is not None and j != i and j > i + 1 and sequence[j] == 'H':
                            old_gain += 1
                        elif j is not None and j != i and j < i - 1 and sequence[j] == 'H':
                            old_gain += 1
                
                # Add new contacts at target
                new_gain = 0
                if sequence[i] == 'H':
                    for q in _neighbours(target):
                        j = occupied.get(q)
                        if j is not None and j != i and j > i + 1 and sequence[j] == 'H':
                            new_gain += 1
                        elif j is not None and j != i and j < i - 1 and sequence[j] == 'H':
                            new_gain += 1
                
                # Also need to check: moving i might break contacts between other residues
                # that were only connected through i's position? No, contacts are between
                # non-consecutive H residues. Moving i only affects contacts involving i.
                
                delta = new_gain - old_gain
                new_energy = current_energy - delta  # energy is negative of contacts
                
                if new_energy < best_new_energy:
                    best_new_energy = new_energy
                    best_move = target
            
            if best_move is not None and best_new_energy < current_energy:
                # Perform the move
                old_pos = chain[i]
                del occupied[old_pos]
                chain[i] = best_move
                occupied[best_move] = i
                current_energy = best_new_energy
                improved = True
    
    return chain, current_energy


def _anneal(sequence, chain, occupied, current_energy, time_limit=10.0):
    """Simulated annealing on the chain."""
    n = len(chain)
    start_time = time.time()
    rng = random.Random(int(time.time() * 1000) % 100000)
    
    T = 5.0
    T_min = 0.01
    cooling = 0.995
    
    while T > T_min and (time.time() - start_time) < time_limit:
        i = rng.randint(1, n - 1)
        pos = chain[i]
        
        # Generate candidate move
        targets = list(_neighbours(pos))
        rng.shuffle(targets)
        
        moved = False
        for target in targets:
            if target in occupied:
                continue
            
            # Check adjacency constraints
            if i > 0:
                d1 = sum(abs(a - b) for a, b in zip(target, chain[i-1]))
                if d1 != 1:
                    continue
            if i < n - 1:
                d2 = sum(abs(a - b) for a, b in zip(target, chain[i+1]))
                if d2 != 1:
                    continue
            
            # Compute energy delta
            old_gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(pos):
                    j = occupied.get(q)
                    if j is not None and j != i and j > i + 1 and sequence[j] == 'H':
                        old_gain += 1
                    elif j is not None and j != i and j < i - 1 and sequence[j] == 'H':
                        old_gain += 1
            
            new_gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(target):
                    j = occupied.get(q)
                    if j is not None and j != i and j > i + 1 and sequence[j] == 'H':
                        new_gain += 1
                    elif j is not None and j != i and j < i - 1 and sequence[j] == 'H':
                        new_gain += 1
            
            delta = old_gain - new_gain  # positive means energy increases (worse)
            new_energy = current_energy + delta
            
            if new_energy <= current_energy or rng.random() < pow(2.718281828, -delta / T):
                # Accept
                del occupied[pos]
                chain[i] = target
                occupied[target] = i
                current_energy = new_energy
                moved = True
                break
        
        T *= cooling
    
    return chain, current_energy


def fold(sequence, dimension):
    """Fold an HP sequence into a self-avoiding walk on the cubic lattice."""
    global _START_TIME
    n = len(sequence)
    
    if n <= 1:
        return [[0] * dimension]
    
    best_chain = None
    best_energy = 0  # 0 means unfolded (no contacts)
    
    # Phase 1: Multiple random growths to get diverse starting points
    num_growths = 30
    rng_main = random.Random(12345 + n)
    
    initial_chains = []
    for _ in range(num_growths):
        chain = _random_grow(sequence, dimension, rng_main, bias=0.3)
        if chain is not None:
            energy = _compute_energy(sequence, chain)
            occupied = {p: i for i, p in enumerate(chain)}
            initial_chains.append((chain, occupied, energy))
            if energy < best_energy:
                best_energy = energy
                best_chain = list(chain)
    
    if not initial_chains:
        # Fallback: straight line
        return [[i] + [0] * (dimension - 1) for i in range(n)]
    
    # Phase 2: Local search on the best few initial chains
    # Sort by energy and take top 5
    initial_chains.sort(key=lambda x: x[2])
    top_chains = initial_chains[:min(5, len(initial_chains))]
    
    for chain, occupied, energy in top_chains:
        if _remaining_time() < 5.0:
            break
        
        # Local search first
        time_for_ls = min(3.0, _remaining_time() * 0.3)
        chain, energy = _local_search(sequence, chain, occupied, energy, time_limit=time_for_ls)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = list(chain)
    
    # Phase 3: Simulated annealing on the best chain if time remains
    if _remaining_time() > 10.0 and best_chain is not None:
        occupied = {p: i for i, p in enumerate(best_chain)}
        sa_time = min(20.0, _remaining_time() * 0.5)
        chain, energy = _anneal(sequence, best_chain, occupied, best_energy, time_limit=sa_time)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = list(chain)
    
    # Phase 4: Final local search polish
    if _remaining_time() > 3.0 and best_chain is not None:
        occupied = {p: i for i, p in enumerate(best_chain)}
        ls_time = min(5.0, _remaining_time() * 0.8)
        chain, energy = _local_search(sequence, best_chain, occupied, best_energy, time_limit=ls_time)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = list(chain)
    
    # Convert to required format
    if best_chain is None:
        return [[i] + [0] * (dimension - 1) for i in range(n)]
    
    return [list(p) for p in best_chain]
