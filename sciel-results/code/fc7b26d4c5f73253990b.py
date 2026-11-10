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
                    
                    old_cc = 0
                    for idx in range(i, j+1):
                        if sequence[idx] == 'H':
                            pos = current_chain[idx]
                            for q in _neighbours(pos):
                                oj = occ_map.get(q)
                                if oj is not None and sequence[oj] == 'H':
                                    old_cc += 1
                    
                    new_cc = 0
                    for idx in range(i, j+1):
                        if sequence[idx] == 'H':
                            new_pos = new_chain[i+j-idx]
                            for q in _neighbours(new_pos):
                                oj = occ_map.get(q)
                                if oj is not None and sequence[oj] == 'H':
                                    new_cc += 1
                    
                    delta_e = -(new_cc - old_cc)  # Energy change: new - old
                    accept = delta_e < 0 or (T > 0 and rng.random() < math.exp(-delta_e / T))
                    
                    if accept:
                        # Apply
                        for k in range(j - i + 1):
                            new_chain[i + k] = new_segment[k]
                        # Rebuild occupied
                        for idx, pos in enumerate(new_chain):
                            current_occupied[pos] = idx
                        for pos in list(current_occupied.keys()):
                            if pos not in new_chain:
                                del current_occupied[pos]
                        
                        current_chain = new_chain
                        current_energy += delta_e
                        
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = list(current_chain)
                            occupied = dict(current_occupied)
                        improved = True
            else:
                # Single move
                idx = rng.randint(1, n - 2)
                old = current_chain[idx]
                prev = current_chain[idx-1]
                next_pos = current_chain[idx+1]
                
                candidates = []
                for site in _neighbours(prev):
                    if site == old:
                        continue
                    if site in current_occupied and current_occupied[site] != idx:
                        continue
                    dist = sum(abs(site[a] - next_pos[a]) for a in range(len(site)))
                    if dist == 1:
                        candidates.append(site)
                
                if candidates:
                    site = rng.choice(candidates)
                    delta = _delta_energy_move(sequence, current_chain, idx, old, site, current_occupied)
                    delta_e = -delta  # Energy change
                    
                    accept = delta_e < 0 or (T > 0 and rng.random() < math.exp(-delta_e / T))
                    
                    if accept:
                        current_occupied[old] = -1
                        current_occupied[site] = idx
                        current_chain[idx] = site
                        current_energy += delta_e
                        
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = list(current_chain)
                            occupied = dict(current_occupied)
                        improved = True
        
        if not improved and T < t_end * 1.1:
            break
    
    return best_chain, best_energy


def _replica_exchange_sa(sequence, chains, rng, deadline):
    """Replica exchange simulated annealing with multiple replicas at different temperatures."""
    n = len(chains[0])
    if n < 4:
        return chains[0], _compute_energy(sequence, chains[0])
    
    num_reps = len(chains)
    if num_reps < 2:
        return _local_search_sa(sequence, chains[0], rng, deadline)
    
    # Initialize replicas with different temperatures
    # Sort chains by energy (best first)
    scored = [(_compute_energy(sequence, c), c) for c in chains]
    scored.sort()
    
    replicas = [list(c) for _, c in scored]
    replica_energies = [e for e, _ in scored]
    replica_occupied = [{p: i for i, p in enumerate(c)} for c in replicas]
    
    # Temperature schedule: higher T for worse replicas, lower T for best
    base_T = 3.0
    T_scale = 0.7  # Each replica has T * T_scale^k
    temperatures = [base_T * (T_scale ** k) for k in range(num_reps)]
    
    best_chain = list(replicas[0])
    best_energy = replica_energies[0]
    
    total_iters = n * 50
    t0 = base_T
    t_end = 0.05
    alpha = (t_end / t0) ** (1.0 / max(total_iters, 1))
    
    iters = 0
    while time.time() < deadline:
        iters += 1
        progress = min(iters / total_iters, 1.0)
        global_T = t0 * (t_end / t0) ** progress
        
        # Pick two replicas to potentially swap
        r1 = rng.randint(0, num_reps - 1)
        r2 = rng.randint(0, num_reps - 1)
        if r1 == r2:
            r2 = (r1 + 1) % num_reps
        
        # Make a move on replica r1
        r = r1
        T_r = temperatures[r] * global_T / base_T  # Scale temperature
        
        # Choose move type
        if rng.random() < 0.7 and n > 4:
            # 2-opt move
            i = rng.randint(1, n - 2)
            j = rng.randint(i + 1, n - 1)
            
            new_chain = list(replicas[r])
            segment = new_chain[i:j+1]
            new_segment = list(reversed(segment))
            
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
                
                old_cc = 0
                for idx in range(i, j+1):
                    if sequence[idx] == 'H':
                        pos = replicas[r][idx]
                        for q in _neighbours(pos):
                            oj = occ_map.get(q)
                            if oj is not None and sequence[oj] == 'H':
                                old_cc += 1
                
                new_cc = 0
                for idx in range(i, j+1):
                    if sequence[idx] == 'H':
                        new_pos = new_chain[i+j-idx]
                        for q in _neighbours(new_pos):
                            oj = occ_map.get(q)
                            if oj is not None and sequence[oj] == 'H':
                                new_cc += 1
                
                delta_e = -(new_cc - old_cc)
                accept = delta_e < 0 or (T_r > 0 and rng.random() < math.exp(-delta_e / T_r))
                
                if accept:
                    for k in range(j - i + 1):
                        new_chain[i + k] = new_segment[k]
                    for idx, pos in enumerate(new_chain):
                        replica_occupied[r][pos] = idx
                    for pos in list(replica_occupied[r].keys()):
                        if pos not in new_chain:
                            del replica_occupied[r][pos]
                    
                    replicas[r] = new_chain
                    replica_energies[r] += delta_e
        else:
            # Single move
            idx = rng.randint(1, n - 2)
            old = replicas[r][idx]
            prev = replicas[r][idx-1]
            next_pos = replicas[r][idx+1]
            
            candidates = []
            for site in _neighbours(prev):
                if site == old:
                    continue
                if site in replica_occupied[r] and replica_occupied[r][site] != idx:
                    continue
                dist = sum(abs(site[a] - next_pos[a]) for a in range(len(site)))
                if dist == 1:
                    candidates.append(site)
            
            if candidates:
                site = rng.choice(candidates)
                delta = _delta_energy_move(sequence, replicas[r], idx, old, site, replica_occupied[r])
                delta_e = -delta
                
                accept = delta_e < 0 or (T_r > 0 and rng.random() < math.exp(-delta_e / T_r))
                
                if accept:
                    replica_occupied[r][old] = -1
                    replica_occupied[r][site] = idx
                    replicas[r][idx] = site
                    replica_energies[r] += delta_e
        
        # Try replica swap between r1 and r2
        if r1 != r2:
            T1 = temperatures[r1] * global_T / base_T
            T2 = temperatures[r2] * global_T / base_T
            E1 = replica_energies[r1]
            E2 = replica_energies[r2]
            
            # Swap acceptance criterion
            delta_swap = (E1 / T2 + E2 / T1) - (E1 / T1 + E2 / T2)
            if delta_swap < 0 or rng.random() < math.exp(-delta_swap):
                # Swap replicas
                replicas[r1], replicas[r2] = replicas[r2], replicas[r1]
                replica_energies[r1], replica_energies[r2] = replica_energies[r2], replica_energies[r1]
                replica_occupied[r1], replica_occupied[r2] = replica_occupied[r2], replica_occupied[r1]
        
        # Update best
        for r in range(num_reps):
            if replica_energies[r] < best_energy:
                best_energy = replica_energies[r]
                best_chain = list(replicas[r])
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation via diverse growth + replica exchange SA."""
    n = len(sequence)
    rng = random.Random(42 + len(sequence) * 7 + sum(ord(c) for c in sequence))
    
    # Time budget: adapt dynamically based on number of sequences
    # Assume ~26 sequences, so ~12s each, but use global deadline
    per_seq_time = min(12.0, max(2.0, (_HARD_DEADLINE - time.time()) / 26.0))
    deadline = time.time() + per_seq_time
    
    best_chain = None
    best_energy = 0  # Energy is negative, so 0 is "no contacts"
    
    # Phase 1: Generate diverse initial chains
    initial_chains = []
    
    # Strategy 1: Greedy with various temperatures and compactness
    for temp in [0.0, 0.1, 0.2, 0.3]:
        if time.time() > deadline - 6.0:
            break
        for compact in [0.0, 0.3]:
            chain = _grow_greedy(sequence, dimension, rng, temperature=temp, compact_bias=compact)
            if chain:
                initial_chains.append(chain)
    
    # Strategy 2: Compact growth
    for _ in range(2):
        if time.time() > deadline - 6.0:
            break
        chain = _grow_compact(sequence, dimension, rng)
        if chain:
            initial_chains.append(chain)
    
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
        # Ensure self-avoidance and unit steps
        if len(set(best_chain)) < n:
            best_chain = [(i, 0, 0) for i in range(n)]
        best_energy = _compute_energy(sequence, best_chain)
    
    # Phase 2: Replica exchange SA on top initial chains
    if time.time() < deadline - 0.5:
        # Select top 3-4 initial chains for replica exchange
        scored = [(_compute_energy(sequence, c), c) for c in initial_chains]
        scored.sort()
        top_chains = [c for _, c in scored[:min(4, len(scored))]]
        
        if len(top_chains) >= 2:
            best_chain, best_energy = _replica_exchange_sa(
                sequence, top_chains, rng, deadline - 0.5
            )
        else:
            # Fallback to single replica SA
            best_chain, best_energy = _local_search_sa(
                sequence, best_chain, rng, deadline - 0.5, max_iters_per_residue=2
            )
    
    # Phase 3: If time remains, try SA on a second-best initial chain
    remaining = deadline - time.time()
    if remaining > 2.0 and len(initial_chains) > 1:
        # Sort initial chains by energy
        scored = [(_compute_energy(sequence, c), c) for c in initial_chains]
        scored.sort()
        # Try SA on the second best if it's close to the best
        if len(scored) >= 2:
            e2, c2 = scored[1]
            if e2 > best_energy - 3:  # Within 3 contacts of best
                c2, e2 = _local_search_sa(sequence, c2, rng, time.time() + remaining - 0.5, max_iters_per_residue=1)
                if e2 < best_energy:
                    best_chain, best_energy = c2, e2
    
    return [list(p) for p in best_chain]
