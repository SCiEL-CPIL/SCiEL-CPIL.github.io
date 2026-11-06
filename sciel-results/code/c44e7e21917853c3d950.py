"""
HP lattice protein folding on the square lattice.
Strategy: Structure-First CSP.
1. Heuristically select a set of H-H contacts (non-adjacent) that are likely feasible.
2. Use backtracking search to find a self-avoiding walk on the square lattice 
   that realizes these contacts.
3. Use local search to polish the result.
This avoids the topological traps of greedy growth by enforcing the target structure early.
"""
import time
import random
import math

_DEADLINE = time.time() + 190.0


def _remaining():
    return _DEADLINE - time.time()


def _neighbours(point):
    """Yield the 4 neighbours of a 2D lattice point."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _energy(sequence, chain):
    """Compute the HP energy: -1 for each non-consecutive H-H contact."""
    if chain is None:
        return 1
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _select_contacts(sequence, rng, max_contacts=None):
    """
    Heuristically select a set of H-H contacts.
    We prioritize H residues that are far apart in sequence but likely to be close in space.
    We use a greedy approach: pick H residues, try to match them with other H residues
    that are not too close in sequence (to avoid impossible geometric constraints)
    but not too far (to ensure they can be packed).
    """
    n = len(sequence)
    h_indices = [i for i, c in enumerate(sequence) if c == 'H']
    if len(h_indices) < 2:
        return []
    
    # Sort H indices by "exposure" or just randomly? 
    # Let's use a random order to diversify, but bias towards high index differences
    # that are still geometrically plausible.
    
    contacts = []
    used = set()
    
    # Potential contacts: pairs (i, j) where i, j are H, |i-j| > 1
    # We want to form a graph.
    
    # Simple heuristic: 
    # 1. Identify "core" H residues (middle of sequence)
    # 2. Try to connect them to other H residues
    
    # A better heuristic: 
    # Estimate the maximum possible contacts based on sequence length.
    # For a compact blob, roughly n/4 or n/3 contacts are possible.
    target_contacts = min(len(h_indices) // 2, n // 4 + 2)
    
    # Generate candidate pairs
    candidates = []
    for i in h_indices:
        for j in h_indices:
            if i == j or abs(i - j) <= 1:
                continue
            if i > j:
                i, j = j, i
            # Penalize very close sequence distances (hard to fold)
            # Penalize very large distances (hard to bring together)
            dist = j - i
            if dist < 3 or dist > n - 3:
                continue
            # Score: prefer medium distances? Or random?
            # Let's just collect all valid pairs and pick randomly with bias
            candidates.append((i, j))
            
    if not candidates:
        return []
        
    # Shuffle candidates
    rng.shuffle(candidates)
    
    # Greedily add contacts if they don't conflict (simple check: no residue has >2 contacts? 
    # In a 2D lattice, a residue can have max 4 neighbors, but 2 are chain neighbors.
    # So max 2 non-chain contacts per residue is a good upper bound for stability, 
    # though 3 or 4 is possible if chain neighbors are far. 
    # Let's allow up to 3 contacts per residue to be safe, but prefer 2.
    
    contact_count = {i: 0 for i in h_indices}
    selected = []
    
    for i, j in candidates:
        if contact_count[i] < 3 and contact_count[j] < 3:
            # Check if this contact is "feasible" geometrically? 
            # Hard to know without folding. 
            # Just take it.
            selected.append((i, j))
            contact_count[i] += 1
            contact_count[j] += 1
            if len(selected) >= target_contacts:
                break
                
    return selected


def _csp_fold(sequence, dimension, rng, contacts, max_nodes=200000, time_limit=10.0):
    """
    Backtracking search to find a conformation that satisfies the given contacts.
    We prioritize moves that satisfy a contact if possible.
    """
    n = len(sequence)
    if n <= 2:
        chain = [(0, 0)]
        for i in range(1, n):
            chain.append((i, 0))
        return chain, _energy(sequence, chain)

    best_chain = None
    best_energy = 0
    satisfied_contacts = 0
    total_contacts = len(contacts)
    
    # Map contact index to the pair
    # We need to track which contacts are satisfied.
    # For efficiency, we'll just compute energy at the end, but use contacts for pruning/ordering.
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    nodes_visited = 0
    start_time = time.time()
    
    # Precompute for each position i, which contacts involve it
    # contact_map[i] = list of (other_idx, contact_id)
    contact_map = {i: [] for i in range(n)}
    for cid, (i, j) in enumerate(contacts):
        contact_map[i].append((j, cid))
        contact_map[j].append((i, cid))
        
    satisfied = [False] * total_contacts
    
    def _dfs(depth, current_chain, current_occupied, current_energy):
        nonlocal best_chain, best_energy, nodes_visited, satisfied
        
        if nodes_visited > max_nodes:
            return
        if time.time() - start_time > time_limit:
            return
            
        if depth == n:
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
            return

        last = current_chain[-1]
        res = sequence[depth]
        
        candidates = []
        
        for site in _neighbours(last):
            if site in current_occupied:
                continue
            
            # Dead end check
            if depth < n - 2:
                free_neighbors = 0
                for nb in _neighbours(site):
                    if nb not in current_occupied and nb != last:
                        free_neighbors += 1
                if free_neighbors == 0:
                    continue

            delta = 0
            new_satisfied = 0
            
            if res == 'H':
                # Check contacts with existing H's
                for nb in _neighbours(site):
                    j = current_occupied.get(nb)
                    if j is not None and j < depth - 1 and sequence[j] == 'H':
                        delta -= 1
                        # Check if this satisfies a target contact
                        for (other_idx, cid) in contact_map[depth]:
                            if other_idx == j and not satisfied[cid]:
                                new_satisfied += 1
                                # Mark as satisfied temporarily? 
                                # We need to be careful with backtracking.
                                # Since we're just scoring, we can count it.
                                # But we need to restore. 
                                # For simplicity in this heuristic, we just count potential.
                                # Actually, let's just use the energy delta.
                                # The contact list is mainly for *ordering* moves.
            
            # Score for ordering: 
            # 1. Prefer moves that create energy contacts (delta < 0)
            # 2. Prefer moves that satisfy target contacts
            # 3. Prefer moves that leave space
            
            free_count = sum(1 for nb in _neighbours(site) if nb not in current_occupied and nb != last)
            
            # Heuristic score:
            score = delta * 100 + new_satisfied * 10 + free_count
            
            candidates.append((score, site, delta))

        if not candidates:
            return

        # Sort by score (highest first, so most negative delta first)
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Limit branching
        max_branch = min(len(candidates), 4)
        
        for score, site, delta in candidates[:max_branch]:
            current_chain.append(site)
            current_occupied[site] = depth
            
            nodes_visited += 1
            _dfs(depth + 1, current_chain, current_occupied, current_energy + delta)
            
            current_chain.pop()
            del current_occupied[site]

    _dfs(1, chain, occupied, 0)

    if best_chain is None:
        # Fallback: try a simple greedy growth if CSP failed
        best_chain, best_energy = _greedy_growth(sequence, dimension, rng, temperature=0.5)
        if best_chain is None:
            best_chain = [(i, 0) for i in range(n)]
            best_energy = _energy(sequence, best_chain)

    return best_chain, best_energy


def _greedy_growth(sequence, dimension, rng, temperature=0.5):
    """
    Grow a chain using probabilistic greedy approach.
    """
    n = len(sequence)
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}

    for i in range(1, n):
        res = sequence[i]
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if res == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 3.0 + 0.3 * free + rng.random() * temperature
            options.append((score, site))

        if not options:
            return None

        options.sort(key=lambda x: -x[0])
        if temperature > 0 and len(options) > 1:
            top = options[0][0]
            probs = []
            total = 0.0
            for sc, _ in options[:min(4, len(options))]:
                p = math.exp((sc - top) / max(temperature, 0.1))
                probs.append(p)
                total += p
            if total > 0:
                probs = [p / total for p in probs]
                r = rng.random()
                cum = 0.0
                chosen_idx = len(probs) - 1
                for idx, p in enumerate(probs):
                    cum += p
                    if r <= cum:
                        chosen_idx = idx
                        break
                site = options[chosen_idx][1]
            else:
                site = options[0][1]
        else:
            site = options[0][1]

        occupied[site] = i
        chain.append(site)

    return chain


def _local_search(sequence, chain, dimension, rng, max_iter=300):
    """
    Local search with multiple move types and simulated annealing.
    Optimized for speed with precomputed structures.
    """
    if chain is None:
        return None, 1

    n = len(chain)
    current_chain = list(chain)
    current_energy = _energy(sequence, chain)
    pos_map = {p: i for i, p in enumerate(current_chain)}

    best_chain = list(current_chain)
    best_energy = current_energy

    # Simulated annealing parameters
    T = 3.0
    cool_rate = 0.995

    for it in range(max_iter):
        if _remaining() < 0.5:
            break

        # Choose move type
        r = rng.random()

        if r < 0.5:
            # Single residue move
            i = rng.randint(1, n - 2)
            p = current_chain[i]
            best_delta = 0
            best_q = None

            for q in _neighbours(p):
                if q == current_chain[i - 1] or q == current_chain[i + 1]:
                    continue
                if q in pos_map and pos_map[q] != i:
                    continue

                # Check unit steps
                prev = current_chain[i - 1]
                nxt = current_chain[i + 1]
                if abs(q[0] - prev[0]) + abs(q[1] - prev[1]) != 1:
                    continue
                if abs(q[0] - nxt[0]) + abs(q[1] - nxt[1]) != 1:
                    continue

                # Compute energy delta
                delta = 0
                if sequence[i] == 'H':
                    # Remove old contacts
                    for old_q in _neighbours(p):
                        j = pos_map.get(old_q)
                        if j is not None and j != i and sequence[j] == 'H' and abs(j - i) > 1:
                            delta += 1
                    # Add new contacts
                    for new_q in _neighbours(q):
                        j = pos_map.get(new_q)
                        if j is not None and j != i and sequence[j] == 'H' and abs(j - i) > 1:
                            delta -= 1

                if delta < best_delta:
                    best_delta = delta
                    best_q = q

            if best_q is not None:
                # Accept with SA
                if best_delta <= 0 or rng.random() < math.exp(-best_delta / max(T, 0.1)):
                    old_p = current_chain[i]
                    del pos_map[old_p]
                    current_chain[i] = best_q
                    pos_map[best_q] = i
                    current_energy += best_delta

                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)

        elif r < 0.8:
            # Segment shift
            seg_len = rng.choice([2, 3, 4])
            if n > seg_len + 2:
                i = rng.randint(1, n - seg_len - 2)
                # Direction from i-1 to i
                dx = current_chain[i][0] - current_chain[i - 1][0]
                dy = current_chain[i][1] - current_chain[i - 1][1]

                # Perpendicular directions
                perp_dirs = [(-dy, dx), (dy, -dx)]

                for px, py in perp_dirs:
                    new_seg = []
                    valid = True
                    for k in range(seg_len):
                        new_p = (current_chain[i + k][0] + px, current_chain[i + k][1] + py)
                        new_seg.append(new_p)

                    # Check occupancy
                    for new_p in new_seg:
                        j = pos_map.get(new_p)
                        if j is not None and not (i <= j < i + seg_len):
                            valid = False
                            break

                    if not valid:
                        continue

                    # Check connections
                    if abs(new_seg[0][0] - current_chain[i - 1][0]) + abs(new_seg[0][1] - current_chain[i - 1][1]) != 1:
                        continue
                    if abs(new_seg[-1][0] - current_chain[i + seg_len][0]) + abs(new_seg[-1][1] - current_chain[i + seg_len][1]) != 1:
                        continue

                    # Check internal steps
                    for k in range(seg_len - 1):
                        if abs(new_seg[k + 1][0] - new_seg[k][0]) + abs(new_seg[k + 1][1] - new_seg[k][1]) != 1:
                            valid = False
                            break
                    if not valid:
                        continue

                    # Compute energy delta
                    delta = 0
                    for k in range(seg_len):
                        old_p = current_chain[i + k]
                        new_p = new_seg[k]
                        if sequence[i + k] == 'H':
                            # Remove old contacts with outside
                            for q in _neighbours(old_p):
                                j = pos_map.get(q)
                                if j is not None and j != i + k and sequence[j] == 'H' and abs(j - (i + k)) > 1:
                                    if not (i <= j < i + seg_len):
                                        delta += 1
                            # Add new contacts with outside
                            for q in _neighbours(new_p):
                                j = pos_map.get(q)
                                if j is not None and j != i + k and sequence[j] == 'H' and abs(j - (i + k)) > 1:
                                    if not (i <= j < i + seg_len):
                                        delta -= 1

                    if delta < 0 or rng.random() < math.exp(-delta / max(T, 0.1)):
                        # Apply move
                        for k in range(seg_len):
                            old_p = current_chain[i + k]
                            del pos_map[old_p]
                            current_chain[i + k] = new_seg[k]
                            pos_map[new_seg[k]] = i + k
                        current_energy += delta

                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = list(current_chain)
                        break  # Only one segment move per iteration

        else:
            # Reverse a short segment
            seg_len = rng.choice([3, 4, 5])
            if n > seg_len + 2:
                i = rng.randint(1, n - seg_len - 2)
                new_seg = list(reversed(current_chain[i:i + seg_len]))

                # Check if positions are free
                valid = True
                for k, new_p in enumerate(new_seg):
                    j = pos_map.get(new_p)
                    if j is not None and not (i <= j < i + seg_len):
                        valid = False
                        break
                if not valid:
                    continue

                # Check connections
                if abs(new_seg[0][0] - current_chain[i - 1][0]) + abs(new_seg[0][1] - current_chain[i - 1][1]) != 1:
                    continue
                if abs(new_seg[-1][0] - current_chain[i + seg_len][0]) + abs(new_seg[-1][1] - current_chain[i + seg_len][1]) != 1:
                    continue

                # Check internal steps
                for k in range(seg_len - 1):
                    if abs(new_seg[k + 1][0] - new_seg[k][0]) + abs(new_seg[k + 1][1] - new_seg[k][1]) != 1:
                        valid = False
                        break
                if not valid:
                    continue

                # Compute energy delta
                delta = 0
                for k in range(seg_len):
                    old_p = current_chain[i + k]
                    new_p = new_seg[k]
                    if sequence[i + k] == 'H':
                        for q in _neighbours(old_p):
                            j = pos_map.get(q)
                            if j is not None and j != i + k and sequence[j] == 'H' and abs(j - (i + k)) > 1:
                                if not (i <= j < i + seg_len):
                                    delta += 1
                        for q in _neighbours(new_p):
                            j = pos_map.get(q)
                            if j is not None and j != i + k and sequence[j] == 'H' and abs(j - (i + k)) > 1:
                                if not (i <= j < i + seg_len):
                                    delta -= 1

                if delta < 0 or rng.random() < math.exp(-delta / max(T, 0.1)):
                    for k in range(seg_len):
                        old_p = current_chain[i + k]
                        del pos_map[old_p]
                        current_chain[i + k] = new_seg[k]
                        pos_map[new_seg[k]] = i + k
                    current_energy += delta

                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)

        T *= cool_rate

    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Find a low-energy conformation for the given HP sequence on a square lattice.
    Combines Structure-First CSP search with local search polishing.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    rng = random.Random(42 + len(sequence))

    best_chain = None
    best_energy = 0  # unfolded baseline

    # Strategy: 
    # 1. Try several different contact graphs
    # 2. For each, run CSP backtracking
    # 3. Polish with local search
    # 4. Also run standard greedy+local search as a baseline
    
    # Determine number of CSP attempts based on length
    # Short sequences: more attempts, longer search
    # Long sequences: fewer attempts, shorter search
    
    if n <= 30:
        csp_attempts = 5
        csp_nodes = 100000
        csp_time = 5.0
        local_iter = 150
    elif n <= 60:
        csp_attempts = 3
        csp_nodes = 50000
        csp_time = 3.0
        local_iter = 100
    else:
        csp_attempts = 2
        csp_nodes = 30000
        csp_time = 2.0
        local_iter = 80

    # Baseline: Greedy + Local Search
    # Run a few quick greedy restarts first to get a baseline
    for restart in range(5):
        if _remaining() < 5.0:
            break
        temp = 0.5
        chain = _greedy_growth(sequence, dimension, rng, temperature=temp)
        if chain is not None:
            energy = _energy(sequence, chain)
            if energy < best_energy:
                best_chain = chain
                best_energy = energy
            improved_chain, improved_energy = _local_search(sequence, chain, dimension, rng, max_iter=local_iter//2)
            if improved_chain is not None and improved_energy < best_energy:
                best_chain = improved_chain
                best_energy = improved_energy

    # CSP Attempts
    for attempt in range(csp_attempts):
        if _remaining() < 10.0:
            break
            
        # Select contacts
        # Vary the randomness of contact selection
        contacts = _select_contacts(sequence, rng)
        
        # Run CSP
        csp_chain, csp_energy = _csp_fold(sequence, dimension, rng, contacts, 
                                           max_nodes=csp_nodes, time_limit=csp_time)
        
        if csp_chain is not None:
            if csp_energy < best_energy:
                best_chain = csp_chain
                best_energy = csp_energy
            
            # Polish the CSP result
            improved_chain, improved_energy = _local_search(sequence, csp_chain, dimension, rng, max_iter=local_iter)
            if improved_chain is not None and improved_energy < best_energy:
                best_chain = improved_chain
                best_energy = improved_energy

    # Final polish on the best chain
    if best_chain is not None and _remaining() > 5.0:
        final_chain, final_energy = _local_search(sequence, best_chain, dimension, rng, max_iter=local_iter)
        if final_chain is not None and final_energy < best_energy:
            best_chain = final_chain
            best_energy = final_energy

    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]

    return [list(p) for p in best_chain]
