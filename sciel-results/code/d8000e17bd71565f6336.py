"""HP lattice protein folding on cubic lattice - bond-vector CMA-ES + SAW projection + pivot MC polish."""
import random
import time
import math

_GLOBAL_START = time.time()
_HARD_DEADLINE = _GLOBAL_START + 325.0  # 325s hard limit, 5s safety margin


def _neighbours(point):
    """Generate 6 neighbours in 3D cubic lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute energy: -number of H-H contacts (non-adjacent)."""
    n = len(chain)
    if n < 2:
        return 0
    occupied = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = occupied.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _delta_energy_move(sequence, chain, i, old_pos, new_pos, occupied):
    """Compute energy change when moving residue i from old_pos to new_pos.
    Returns delta such that new_energy = old_energy - delta (delta > 0 means improvement)."""
    if sequence[i] != 'H':
        return 0
    delta = 0
    # Lost contacts at old_pos
    for q in _neighbours(old_pos):
        j = occupied.get(q)
        if j is not None and j != i and sequence[j] == 'H':
            delta -= 1
    # Gained contacts at new_pos
    for q in _neighbours(new_pos):
        j = occupied.get(q)
        if j is not None and j != i and sequence[j] == 'H':
            delta += 1
    return delta


def _project_to_saw(bond_vectors, n, dimension=3):
    """Project a sequence of bond vectors onto the nearest valid self-avoiding walk.
    
    Each bond vector is a 3D continuous vector. We project each to the nearest
    unit lattice direction, then greedily build a SAW, skipping bonds that cause
    self-intersection or are invalid. If we can't place a bond, we try alternatives.
    
    Returns a list of n lattice points (tuples) or None if it fails completely.
    """
    # Unit lattice directions
    unit_dirs = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
    
    # Project each bond vector to nearest unit direction
    projected_bonds = []
    for bv in bond_vectors:
        best_dir = unit_dirs[0]
        best_dot = -2.0
        for d in unit_dirs:
            dot = bv[0]*d[0] + bv[1]*d[1] + bv[2]*d[2]
            if dot > best_dot:
                best_dot = dot
                best_dir = d
        projected_bonds.append(best_dir)
    
    # Build the chain greedily
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0): 0}
    
    for i in range(1, n):
        # Try the projected bond first
        placed = False
        for attempt in range(6):
            if attempt == 0:
                d = projected_bonds[i-1]
            else:
                # Try alternative directions (cyclic shift)
                idx = (unit_dirs.index(projected_bonds[i-1]) + attempt) % 6
                d = unit_dirs[idx]
            
            last = chain[-1]
            new_pos = (last[0] + d[0], last[1] + d[1], last[2] + d[2])
            
            if new_pos not in occupied:
                chain.append(new_pos)
                occupied[new_pos] = i
                placed = True
                break
        
        if not placed:
            return None
    
    return chain


def _grow_scaffold(sequence, dimension, rng, temperature=0.0):
    """Grow a self-avoiding walk scaffold with Metropolis-like acceptance."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)
    
    for i in range(1, n):
        candidates = []
        for site in _neighbours(chain[-1]):
            if site not in occupied:
                candidates.append(site)
        
        if not candidates:
            return None
        
        if len(candidates) == 1:
            best_site = candidates[0]
        elif temperature > 0:
            scores = []
            for site in candidates:
                gain = 0
                if sequence[i] == "H":
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == "H":
                            gain += 1
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                score = gain * 2.0 + free * 0.1
                scores.append(score)
            
            max_score = max(scores)
            weights = [math.exp((s - max_score) / max(temperature, 1e-6)) for s in scores]
            total = sum(weights)
            r = rng.random() * total
            cumsum = 0
            best_site = candidates[0]
            for (site, w) in zip(candidates, weights):
                cumsum += w
                if r <= cumsum:
                    best_site = site
                    break
        else:
            best_score = -1e10
            best_site = candidates[0]
            for site in candidates:
                gain = 0
                if sequence[i] == "H":
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == "H":
                            gain += 1
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                score = gain * 2.0 + free * 0.1
                if score > best_score:
                    best_score = score
                    best_site = site
        
        chain.append(best_site)
        occupied[best_site] = i
    
    return chain


def _grow_compact_scaffold(sequence, dimension, rng):
    """Growth that packs tightly, preferring sites with many occupied neighbours."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)
    
    for i in range(1, n):
        candidates = []
        for site in _neighbours(chain[-1]):
            if site not in occupied:
                candidates.append(site)
        
        if not candidates:
            return None
        
        best_score = -1e10
        best_site = candidates[0]
        for site in candidates:
            pack_score = 0
            for q in _neighbours(site):
                if q in occupied:
                    pack_score += 1
            contact_score = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        contact_score += 1
            score = contact_score * 2 + pack_score * 0.1 + rng.random() * 0.3
            if score > best_score:
                best_score = score
                best_site = site
        
        chain.append(best_site)
        occupied[best_site] = i
    
    return chain


def _cma_es_fold(sequence, rng, deadline, n):
    """Use a CMA-ES-like optimizer on bond vectors to find good folds.
    
    We represent the fold as n-1 bond vectors (each 3D). The objective is
    negative energy (minimize). We project to SAW and compute energy.
    """
    if n < 4:
        return None, 0
    
    # Initialize bond vectors: mostly straight, with some randomness
    def _init_bonds():
        bonds = []
        # Start with a random direction
        dirs = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
        prev_dir = rng.choice(dirs)
        for i in range(n - 1):
            # 70% chance to continue in same direction, 30% turn
            if rng.random() < 0.7:
                d = prev_dir
            else:
                # Pick a perpendicular direction
                perp = [dd for dd in dirs if dd != d and dd != (-d[0], -d[1], -d[2])]
                d = rng.choice(perp)
            # Add small noise
            noise = [rng.gauss(0, 0.3) for _ in range(3)]
            bonds.append([d[0] + noise[0], d[1] + noise[1], d[2] + noise[2]])
            prev_dir = d
        return bonds
    
    # CMA-ES parameters
    sigma = 0.5
    pop_size = min(8, max(4, n // 5))
    max_evals = 200  # Limit evaluations to stay within time budget
    
    # Initial population
    population = []
    for _ in range(pop_size):
        bonds = _init_bonds()
        chain = _project_to_saw(bonds, n)
        if chain is not None:
            energy = _compute_energy(sequence, chain)
            population.append((energy, bonds, chain))
    
    if not population:
        return None, 0
    
    population.sort(key=lambda x: x[0])
    best_energy, best_bonds, best_chain = population[0]
    
    # CMA-ES main loop
    mean = [sum(b[k] for _, b, _ in population) / pop_size for k in range(3)]
    for _ in range(n - 1):
        pass  # Simplified: just use mean of top solutions
    
    # More sophisticated: weighted mean
    weights = [math.exp(-i / 2.0) for i in range(pop_size)]
    wsum = sum(weights)
    mean = []
    for k in range(3):
        m = 0.0
        for i, (e, b, c) in enumerate(population):
            m += weights[i] * b[k]
        mean.append(m / wsum)
    
    eval_count = pop_size
    while time.time() < deadline and eval_count < max_evals:
        # Generate new candidates around the mean
        new_candidates = []
        for _ in range(pop_size):
            bonds = []
            for i in range(n - 1):
                # Sample around mean direction with noise
                base = mean
                noise = [rng.gauss(0, sigma) for _ in range(3)]
                bonds.append([base[0] + noise[0], base[1] + noise[1], base[2] + noise[2]])
            
            chain = _project_to_saw(bonds, n)
            if chain is not None:
                energy = _compute_energy(sequence, chain)
                new_candidates.append((energy, bonds, chain))
                eval_count += 1
        
        if not new_candidates:
            break
        
        # Combine with existing population, keep best
        population.extend(new_candidates)
        population.sort(key=lambda x: x[0])
        population = population[:pop_size]
        
        if population[0][0] < best_energy:
            best_energy, best_bonds, best_chain = population[0]
        
        # Update mean
        weights = [math.exp(-i / 2.0) for i in range(pop_size)]
        wsum = sum(weights)
        mean = []
        for k in range(3):
            m = 0.0
            for i, (e, b, c) in enumerate(population):
                m += weights[i] * b[k]
            mean.append(m / wsum)
        
        # Adapt sigma (simplified)
        sigma *= 0.95
    
    return best_chain, best_energy


def _pivot_move(sequence, chain, occupied, i, rng):
    """Try a pivot move at position i. Returns (new_chain, new_occupied, delta_e) or None."""
    n = len(chain)
    if i < 1 or i > n - 2:
        return None
    
    old = chain[i]
    prev = chain[i-1]
    next_pos = chain[i+1]
    
    candidates = []
    for site in _neighbours(prev):
        if site == old:
            continue
        if site in occupied and occupied[site] != i:
            continue
        dist = sum(abs(site[a] - next_pos[a]) for a in range(3))
        if dist == 1:
            candidates.append(site)
    
    if not candidates:
        return None
    
    site = rng.choice(candidates)
    delta = _delta_energy_move(sequence, chain, i, old, site, occupied)
    delta_e = -delta
    
    new_chain = list(chain)
    new_occupied = dict(occupied)
    new_occupied[old] = -1
    new_occupied[site] = i
    new_chain[i] = site
    
    return new_chain, new_occupied, delta_e


def _segment_move(sequence, chain, occupied, i, j, rng):
    """Try a segment reversal move [i, j]. Returns (new_chain, new_occupied, delta_e) or None."""
    n = len(chain)
    if i >= j or i < 1 or j > n - 2:
        return None
    
    segment = chain[i:j+1]
    new_segment = list(reversed(segment))
    
    # Validate connectivity
    valid = True
    if i > 0:
        prev_pos = chain[i-1]
        first_new = new_segment[0]
        if sum(abs(prev_pos[a] - first_new[a]) for a in range(3)) != 1:
            valid = False
    if valid and j < n - 1:
        next_pos = chain[j+1]
        last_new = new_segment[-1]
        if sum(abs(next_pos[a] - last_new[a]) for a in range(3)) != 1:
            valid = False
    
    if not valid:
        return None
    
    # Check self-avoidance
    occ_set = set()
    for idx, pos in enumerate(chain):
        if idx < i or idx > j:
            occ_set.add(pos)
    for pos in new_segment:
        if pos in occ_set:
            return None
    
    # Compute energy delta
    occ_map = {}
    for idx, pos in enumerate(chain):
        if idx < i or idx > j:
            occ_map[pos] = idx
    
    old_cc = 0
    for idx in range(i, j+1):
        if sequence[idx] == 'H':
            pos = chain[idx]
            for q in _neighbours(pos):
                oj = occ_map.get(q)
                if oj is not None and sequence[oj] == 'H':
                    old_cc += 1
    
    new_cc = 0
    for idx in range(i, j+1):
        if sequence[idx] == 'H':
            new_pos = chain[i+j-idx]
            for q in _neighbours(new_pos):
                oj = occ_map.get(q)
                if oj is not None and sequence[oj] == 'H':
                    new_cc += 1
    
    delta_e = -(new_cc - old_cc)
    
    # Apply move
    new_chain = list(chain)
    for k in range(j - i + 1):
        new_chain[i + k] = new_segment[k]
    
    new_occupied = {}
    for idx, pos in enumerate(new_chain):
        new_occupied[pos] = idx
    
    return new_chain, new_occupied, delta_e


def _local_search(sequence, chain, rng, deadline):
    """Simulated annealing with pivot and segment moves."""
    n = len(chain)
    if n < 4:
        return chain, _compute_energy(sequence, chain)
    
    best_chain = list(chain)
    best_energy = _compute_energy(sequence, best_chain)
    current_chain = list(best_chain)
    current_occupied = {p: i for i, p in enumerate(best_chain)}
    current_energy = best_energy
    
    T = 2.0
    T_min = 0.05
    cooling = 0.9995
    
    while time.time() < deadline and T > T_min:
        T *= cooling
        improved = False
        
        for _ in range(n // 2 + 1):
            if time.time() > deadline:
                break
            
            r = rng.random()
            if r < 0.5:
                # Pivot move
                i = rng.randint(1, n - 2)
                result = _pivot_move(sequence, current_chain, current_occupied, i, rng)
                if result is not None:
                    new_chain, new_occupied, delta_e = result
                    if delta_e <= 0 or rng.random() < math.exp(-delta_e / max(T, 1e-10)):
                        current_chain = new_chain
                        current_occupied = new_occupied
                        current_energy += delta_e
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = list(current_chain)
                        improved = True
            else:
                # Segment move
                i = rng.randint(1, n - 3)
                j = rng.randint(i + 1, n - 2)
                result = _segment_move(sequence, current_chain, current_occupied, i, j, rng)
                if result is not None:
                    new_chain, new_occupied, delta_e = result
                    if delta_e <= 0 or rng.random() < math.exp(-delta_e / max(T, 1e-10)):
                        current_chain = new_chain
                        current_occupied = new_occupied
                        current_energy += delta_e
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = list(current_chain)
                        improved = True
        
        if not improved and T < T_min * 2:
            break
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation via CMA-ES bond-vector optimization + scaffold growth + local search."""
    n = len(sequence)
    rng = random.Random(42 + len(sequence) * 7 + sum(ord(c) for c in sequence))
    
    # Time budget
    remaining_total = _HARD_DEADLINE - time.time()
    per_seq_time = min(12.0, max(2.0, remaining_total / 26.0))
    deadline = time.time() + per_seq_time
    
    best_chain = None
    best_energy = 0
    
    # Phase 1: Generate diverse initial chains
    initial_chains = []
    
    # Strategy 1: Scaffold growth
    for temp in [0.0, 0.2, 0.4]:
        if time.time() > deadline - 4.0:
            break
        chain = _grow_scaffold(sequence, dimension, rng, temperature=temp)
        if chain:
            initial_chains.append(chain)
    
    # Strategy 2: Compact scaffold
    chain = _grow_compact_scaffold(sequence, dimension, rng)
    if chain:
        initial_chains.append(chain)
    
    # Strategy 3: CMA-ES bond-vector optimization
    if time.time() < deadline - 3.0:
        cma_chain, cma_energy = _cma_es_fold(sequence, rng, deadline - 2.0, n)
        if cma_chain is not None:
            initial_chains.append(cma_chain)
    
    # Evaluate all initial chains
    for chain in initial_chains:
        e = _compute_energy(sequence, chain)
        if best_chain is None or e < best_energy:
            best_chain = chain
            best_energy = e
    
    if best_chain is None:
        # Fallback: simple snake pattern
        best_chain = []
        for i in range(n):
            x = i % 8
            y = (i // 8) % 8
            z = (i // 64) % 8
            best_chain.append((x, y, z))
        if len(set(best_chain)) < n:
            best_chain = [(i, 0, 0) for i in range(n)]
        best_energy = _compute_energy(sequence, best_chain)
    
    # Phase 2: Local search polish
    remaining = deadline - time.time()
    if remaining > 1.0:
        best_chain, best_energy = _local_search(
            sequence, best_chain, rng, time.time() + remaining - 0.5
        )
    
    return [list(p) for p in best_chain]
