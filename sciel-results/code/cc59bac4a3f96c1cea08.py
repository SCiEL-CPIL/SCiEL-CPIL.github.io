"""HP lattice protein folding on cubic lattice - replica exchange SA with diverse initializations."""
import random
import time
import math

_GLOBAL_START = time.time()
_HARD_DEADLINE = _GLOBAL_START + 315.0  # 315s hard limit, 15s safety margin


def _neighbours(point):
    """Generate 6 neighbours in 3D cubic lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain, occupied=None):
    """Compute energy: -number of H-H contacts (non-adjacent)."""
    n = len(chain)
    if occupied is None:
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


def _grow_greedy(sequence, dimension, rng, temperature=0.0, compact_bias=0.0):
    """Greedy growth with optional randomness and compactness bias. Returns chain or None."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)
    
    for i in range(1, n):
        best_score = -1e10
        best_site = None
        options = []
        
        # Compute centroid for compactness
        if compact_bias > 0 and len(chain) > 1:
            cx = sum(p[0] for p in chain) / len(chain)
            cy = sum(p[1] for p in chain) / len(chain)
            cz = sum(p[2] for p in chain) / len(chain)
        else:
            cx = cy = cz = 0
        
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            
            # Compactness: distance from centroid
            if compact_bias > 0:
                dist = abs(site[0] - cx) + abs(site[1] - cy) + abs(site[2] - cz)
                compact_score = -dist * compact_bias
            else:
                compact_score = 0
            
            score = gain + 0.05 * free + compact_score
            if temperature > 0:
                score += rng.gauss(0, temperature)
            options.append((score, site))
        
        if not options:
            return None
        
        if temperature > 0:
            # Boltzmann selection
            max_score = max(s for s, _ in options)
            weights = [math.exp((s - max_score) / max(temperature, 1e-6)) for s, _ in options]
            total = sum(weights)
            r = rng.random() * total
            cumsum = 0
            best_site = options[0][1]
            for (score, site), w in zip(options, weights):
                cumsum += w
                if r <= cumsum:
                    best_site = site
                    break
        else:
            best_score, best_site = max(options)
        
        chain.append(best_site)
        occupied[best_site] = i
    
    return chain


def _grow_compact(sequence, dimension, rng):
    """Growth that tries to pack tightly by preferring sites with many occupied neighbours."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)
    
    for i in range(1, n):
        best_score = -1e10
        best_site = None
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            # Count how many occupied neighbours this site has (packing preference)
            pack_score = 0
            for q in _neighbours(site):
                if q in occupied:
                    pack_score += 1
            # Also prefer H-H contacts
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
        
        if best_site is None:
            return None
        
        chain.append(best_site)
        occupied[best_site] = i
    
    return chain


def _local_search_sa(sequence, chain, rng, deadline, max_iters_per_residue=3):
    """Simulated annealing with single moves and 2-opt moves."""
    n = len(chain)
    if n < 4:
        return chain, _compute_energy(sequence, chain)
    
    best_chain = list(chain)
    best_energy = _compute_energy(sequence, best_chain)
    occupied = {p: i for i, p in enumerate(best_chain)}
    
    current_energy = best_energy
    current_chain = list(best_chain)
    current_occupied = dict(occupied)
    
    # Cooling schedule
    t0 = 5.0
    t_end = 0.1
    # Estimate iterations
    total_iters = n * max_iters_per_residue * 10
    alpha = (t_end / t0) ** (1.0 / max(total_iters, 1))
    
    T = t0
    improved = False
    
    while time.time() < deadline:
        improved = False
        T *= alpha
        if T < t_end:
            T = t_end
        
        for _ in range(n):
            if time.time() > deadline:
                break
            
            # Choose move type
            if rng.random() < 0.7 and n > 4:
                # 2-opt move
                i = rng.randint(1, n - 2)
                j = rng.randint(i + 1, n - 1)
                
                # Try the move
                new_chain = list(current_chain)
                segment = new_chain[i:j+1]
                new_segment = list(reversed(segment))
                
                # Validate
                valid = True
                if i > 0:
                    prev_pos = new_chain[i-1]
                    first_new = new_segment[0]
                    if sum(abs(prev_pos[a] - first_new[a]) for a in range(len(prev_pos))) != 1:
                        valid = False
                if valid and j < n - 1:
                    next_pos = new_chain[j+1]
                    last_new = new_segment[-1]
                    if sum(abs(next_pos[a] - last_new[a]) for a in range(len(next_pos))) != 1:
                        valid = False
                
                if valid:
                    # Check self-avoidance
                    occ_set = set()
                    for idx, pos in enumerate(new_chain):
                        if idx < i or idx > j:
                            occ_set.add(pos)
                    for pos in new_segment:
                        if pos in occ_set:
                            valid = False
                            break
                
                if valid:
                    # Compute energy delta
                    occ_map = {}
                    for idx, pos in enumerate(new_chain):
                        if idx < i or idx > j:
                            occ_map[pos] = idx
                    
                    delta = 0
                    for k in range(i, j + 1):
                        old_site = segment[k - i]
                        new_site = new_segment[k - i]
                        if sequence[k] == 'H':
                            # Lost contacts at old_site
                            for q in _neighbours(old_site):
                                jj = occ_map.get(q)
                                if jj is not None and sequence[jj] == 'H':
                                    delta -= 1
                            # Gained contacts at new_site
                            for q in _neighbours(new_site):
                                jj = occ_map.get(q)
                                if jj is not None and sequence[jj] == 'H':
                                    delta += 1
                    
                    accept = False
                    if delta <= 0:
                        accept = True
                    else:
                        if T > 0.001:
                            accept = rng.random() < math.exp(-delta / T)
                    
                    if accept:
                        # Apply reversal
                        for k in range(i, j + 1):
                            old_site = segment[k - i]
                            new_site = new_segment[k - i]
                            if old_site != new_site:
                                current_occupied[old_site] = -1
                                current_occupied[new_site] = k
                            current_chain[k] = new_site
                        
                        current_energy += delta
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = list(current_chain)
                            improved = True
            else:
                # Single-site move
                i = rng.randint(1, n - 2)
                candidates = list(_neighbours(current_chain[i]))
                if not candidates:
                    continue
                new_site = candidates[rng.randint(0, len(candidates) - 1)]
                
                if new_site in current_occupied:
                    continue
                old_site = current_chain[i]
                
                # Check adjacency
                if sum(abs(current_chain[i-1][a] - new_site[a]) for a in range(3)) != 1:
                    continue
                if sum(abs(current_chain[i+1][a] - new_site[a]) for a in range(3)) != 1:
                    continue
                
                delta = _delta_energy_move(sequence, current_chain, i, old_site, new_site, current_occupied)
                
                accept = False
                if delta <= 0:
                    accept = True
                else:
                    if T > 0.001:
                        accept = rng.random() < math.exp(-delta / T)
                
                if accept:
                    current_occupied[old_site] = -1
                    current_occupied[new_site] = i
                    current_chain[i] = new_site
                    current_energy += delta
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
                        improved = True
        
        if not improved and T <= t_end:
            break
    
    return best_chain, best_energy


def _replica_exchange(sequence, dimension, rng, deadline):
    """Replica exchange simulated annealing with multiple chains at different temperatures."""
    n = len(sequence)
    if n < 4:
        chain = _grow_compact(sequence, dimension, rng)
        if chain is None:
            chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        return [list(p) for p in chain]
    
    # Create multiple replicas with different initializations
    num_replicas = min(4, max(2, 60 // n))
    replicas = []
    for r in range(num_replicas):
        if r == 0:
            chain = _grow_compact(sequence, dimension, rng)
        elif r == 1:
            chain = _grow_greedy(sequence, dimension, rng, temperature=0.5, compact_bias=0.3)
        elif r == 2:
            chain = _grow_greedy(sequence, dimension, rng, temperature=1.0, compact_bias=0.1)
        else:
            chain = _grow_greedy(sequence, dimension, rng, temperature=1.5, compact_bias=0.0)
        
        if chain is None:
            chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        
        replicas.append({'chain': chain, 'energy': _compute_energy(sequence, chain)})
    
    # Sort replicas by energy
    replicas.sort(key=lambda x: x['energy'])
    
    best_chain = list(replicas[0]['chain'])
    best_energy = replicas[0]['energy']
    
    # Replica exchange SA
    temperatures = [5.0, 2.0, 0.5, 0.1][:num_replicas]
    
    while time.time() < deadline:
        # Update each replica
        for r in range(num_replicas):
            if time.time() >= deadline:
                break
            T = temperatures[r]
            chain = replicas[r]['chain']
            energy = replicas[r]['energy']
            
            # Perform some SA moves
            for _ in range(n // 2):
                if time.time() >= deadline:
                    break
                
                if rng.random() < 0.5:
                    # Single move
                    i = rng.randint(1, n - 2)
                    candidates = list(_neighbours(chain[i]))
                    if not candidates:
                        continue
                    new_site = candidates[rng.randint(0, len(candidates) - 1)]
                    
                    occupied = {p: idx for idx, p in enumerate(chain)}
                    if new_site in occupied:
                        continue
                    old_site = chain[i]
                    
                    if sum(abs(chain[i-1][a] - new_site[a]) for a in range(3)) != 1:
                        continue
                    if sum(abs(chain[i+1][a] - new_site[a]) for a in range(3)) != 1:
                        continue
                    
                    delta = _delta_energy_move(sequence, chain, i, old_site, new_site, occupied)
                    
                    accept = False
                    if delta <= 0:
                        accept = True
                    else:
                        if T > 0.001:
                            accept = rng.random() < math.exp(-delta / T)
                    
                    if accept:
                        occupied[old_site] = -1
                        occupied[new_site] = i
                        chain[i] = new_site
                        energy += delta
                else:
                    # 2-opt move
                    i = rng.randint(1, n - 2)
                    j = rng.randint(i + 1, n - 1)
                    
                    new_chain = list(chain)
                    segment = new_chain[i:j+1]
                    new_segment = list(reversed(segment))
                    
                    valid = True
                    if i > 0:
                        prev_pos = new_chain[i-1]
                        first_new = new_segment[0]
                        if sum(abs(prev_pos[a] - first_new[a]) for a in range(3)) != 1:
                            valid = False
                    if valid and j < n - 1:
                        next_pos = new_chain[j+1]
                        last_new = new_segment[-1]
                        if sum(abs(next_pos[a] - last_new[a]) for a in range(3)) != 1:
                            valid = False
                    
                    if valid:
                        occ_set = set()
                        for idx, pos in enumerate(new_chain):
                            if idx < i or idx > j:
                                occ_set.add(pos)
                        for pos in new_segment:
                            if pos in occ_set:
                                valid = False
                                break
                    
                    if valid:
                        occ_map = {}
                        for idx, pos in enumerate(new_chain):
                            if idx < i or idx > j:
                                occ_map[pos] = idx
                        
                        delta = 0
                        for k in range(i, j + 1):
                            old_site = segment[k - i]
                            new_site = new_segment[k - i]
                            if sequence[k] == 'H':
                                for q in _neighbours(old_site):
                                    jj = occ_map.get(q)
                                    if jj is not None and sequence[jj] == 'H':
                                        delta -= 1
                                for q in _neighbours(new_site):
                                    jj = occ_map.get(q)
                                    if jj is not None and sequence[jj] == 'H':
                                        delta += 1
                        
                        accept = False
                        if delta <= 0:
                            accept = True
                        else:
                            if T > 0.001:
                                accept = rng.random() < math.exp(-delta / T)
                        
                        if accept:
                            for k in range(i, j + 1):
                                old_site = segment[k - i]
                                new_site = new_segment[k - i]
                                if old_site != new_site:
                                    new_chain[k] = new_site
                            chain = new_chain
                            energy += delta
            
            replicas[r]['chain'] = chain
            replicas[r]['energy'] = energy
        
        # Sort replicas and track best
        replicas.sort(key=lambda x: x['energy'])
        if replicas[0]['energy'] < best_energy:
            best_energy = replicas[0]['energy']
            best_chain = list(replicas[0]['chain'])
    
    return [list(p) for p in best_chain]


def fold(sequence, dimension):
    """Best conformation via replica exchange SA with diverse initializations."""
    start_time = time.time()
    # 240s total for 26 sequences, so ~9.2s each. Use 8.5s per sequence for safety.
    total_deadline = start_time + 8.5
    
    rng = random.Random(12345 + len(sequence))
    
    # Use replica exchange SA
    result = _replica_exchange(sequence, dimension, rng, total_deadline)
    
    return result
