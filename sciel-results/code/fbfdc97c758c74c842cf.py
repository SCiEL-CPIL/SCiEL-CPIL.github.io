import random
import time

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

def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth."""
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
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _local_search(sequence, chain, dimension, rng, max_iters=1000):
    """Improve the chain using segment moves."""
    N = len(chain)
    occupied = {p: i for i, p in enumerate(chain)}
    best_energy = _energy(sequence, chain)
    
    for _ in range(max_iters):
        # Pick a random segment
        L = rng.randint(1, 5)
        if L > N:
            L = N
        i = rng.randint(0, N - L)
        
        # Segment is chain[i : i+L]
        left_anchor = chain[i-1] if i > 0 else None
        right_anchor = chain[i+L] if i+L < N else None
        
        # We only attempt moves if at least one anchor exists to constrain the path
        if left_anchor is None and right_anchor is None:
            continue
            
        orig_segment = chain[i : i+L]
        
        # Remove segment from occupied
        for p in orig_segment:
            del occupied[p]
        
        new_segment = None
        
        if left_anchor is not None:
            start_options = [q for q in _neighbours(left_anchor) if q not in occupied]
            if start_options:
                start_site = rng.choice(start_options)
                
                if right_anchor is not None:
                    target_neighbors = [q for q in _neighbours(right_anchor) if q not in occupied]
                    if target_neighbors:
                        # DFS to find path of length L from start_site to any target_neighbor
                        visited = set(occupied)
                        visited.add(start_site)
                        
                        def dfs(pos, steps, path):
                            if steps == 0:
                                if pos in target_neighbors:
                                    return path
                                return None
                            for nxt in _neighbours(pos):
                                if nxt not in visited:
                                    visited.add(nxt)
                                    res = dfs(nxt, steps - 1, path + [nxt])
                                    if res:
                                        return res
                                    visited.remove(nxt)
                            return None

                        new_segment = dfs(start_site, L - 1, [start_site])
                        
                else:
                    # Just grow random walk
                    visited = set(occupied)
                    visited.add(start_site)
                    new_segment = [start_site]
                    for _ in range(L - 1):
                        options = [q for q in _neighbours(new_segment[-1]) if q not in visited]
                        if not options:
                            new_segment = None
                            break
                        nxt = rng.choice(options)
                        visited.add(nxt)
                        new_segment.append(nxt)
                        
        elif right_anchor is not None:
            # Similar logic, but start from right and go left?
            # For simplicity, we can just pick a random start near right_anchor and grow left?
            # Or just skip. Let's skip for now to keep code simple and fast.
            pass

        if new_segment is not None and len(new_segment) == L:
            # Apply move
            chain[i : i+L] = new_segment
            for k, p in enumerate(new_segment):
                occupied[p] = i + k
            
            new_energy = _energy(sequence, chain)
            
            if new_energy < best_energy:
                best_energy = new_energy
            else:
                # Revert
                chain[i : i+L] = orig_segment
                for p in new_segment:
                    del occupied[p]
                for k, p in enumerate(orig_segment):
                    occupied[p] = i + k
        else:
            # Revert
            for k, p in enumerate(orig_segment):
                chain[i+k] = p
                occupied[p] = i+k

    return chain, best_energy

def fold(sequence, dimension):
    """Best conformation using randomized growth and local search."""
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 0 # Energy is negative
    
    start_time = time.time()
    deadline = start_time + 280 # 280s limit
    
    N = len(sequence)
    
    while time.time() < deadline:
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        improved_chain, energy = _local_search(sequence, chain, dimension, rng, max_iters=500)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = improved_chain
            
        if time.time() > deadline - 10:
            break

    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(N)]
        
    return [list(p) for p in best_chain]
