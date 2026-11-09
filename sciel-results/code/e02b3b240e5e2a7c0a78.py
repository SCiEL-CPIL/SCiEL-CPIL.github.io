"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import sys

_START_TIME = time.time()
_DEADLINE = 195.0  # Safety margin before 210s kill

def _remaining_time():
    return max(0.0, _DEADLINE - (time.time() - _START_TIME))

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    """Compute HP contact energy: -number of H-H non-bonded contacts."""
    n = len(chain)
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    
    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        p = chain[i]
        for q in _neighbours(p):
            j = where.get(q)
            if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                contacts += 1
    # Each contact is counted twice (once from each H residue), so divide by 2
    return -contacts // 2

def _heuristic_score(sequence, chain, occupied, i, site):
    """
    Score a candidate site for the i-th residue.
    Higher score is better.
    """
    if sequence[i] != 'H':
        return 0.0
    
    # Count potential H-H contacts with existing chain
    gain = 0
    for q in _neighbours(site):
        if q in occupied:
            j = occupied[q]
            if j < i - 1 and sequence[j] == 'H':
                gain += 1
                
    # Penalize sites with too few free neighbours (trapping)
    free_count = sum(1 for q in _neighbours(site) if q not in occupied)
    if free_count < 2:
        return -100.0 + gain # Strong penalty for trapping, but allow if high gain
    
    return gain * 10.0 + free_count * 0.5

def _backtrack_fold(sequence, dimension, rng, max_depth=None):
    """
    Depth-first search with backtracking.
    Returns (chain, energy) or (None, None) if stuck.
    """
    n = len(sequence)
    if max_depth is None:
        max_depth = n
        
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    def dfs(depth):
        if depth == n:
            return True
            
        current = chain[-1]
        candidates = []
        
        for site in _neighbours(current):
            if site in occupied:
                continue
            
            # Check if placing here would trap the chain (simple heuristic)
            # Count free neighbours of site
            free_neigh = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free_neigh += 1
            
            # If we place here, and free_neigh is 0 or 1, it's likely a dead end
            # unless it's the last residue
            if depth < n - 1 and free_neigh < 2:
                # Allow if it's the only option? No, prefer to avoid.
                # Actually, if free_neigh == 0, it's definitely a dead end for next step.
                # If free_neigh == 1, it forces a specific next step.
                pass 

            score = _heuristic_score(sequence, chain, occupied, depth, site)
            candidates.append((score, site))
            
        if not candidates:
            return False
        
        # Sort candidates by score (descending)
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Shuffle slightly to allow exploration if scores are close
        # But keep the best ones on top
        for score, site in candidates:
            chain.append(site)
            occupied[site] = depth
            
            if dfs(depth + 1):
                return True
            
            # Backtrack
            chain.pop()
            del occupied[site]
            
        return False

    if dfs(1):
        energy = _energy(sequence, chain)
        return chain, energy
    return None, None

def fold(sequence, dimension):
    """
    Find best conformation using iterative deepening backtracking.
    """
    n = len(sequence)
    best_chain = None
    best_energy = 0
    
    # For very short sequences, brute force or simple growth is fine
    if n <= 10:
        # Simple randomized growth is fast enough for small N
        rng = random.Random(12345 + len(sequence))
        for _ in range(100):
            start = (0,) * dimension
            chain, occupied = [start], {start: 0}
            success = True
            for i in range(1, n):
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
                    success = False
                    break
                site = max(options)[1]
                occupied[site] = i
                chain.append(site)
            if success:
                e = _energy(sequence, chain)
                if e < best_energy:
                    best_energy = e
                    best_chain = chain
        if best_chain:
            return [list(p) for p in best_chain]
        return [tuple([i] + [0] * (dimension - 1)) for i in range(n)]

    # For larger sequences, use backtracking with time limit
    # We will run multiple independent backtracking searches with different randomizations
    # The backtracking itself is deterministic given the sort order, so we add noise to scores
    
    num_trials = 20
    for trial in range(num_trials):
        if _remaining_time() < 5.0:
            break
            
        rng = random.Random(12345 + len(sequence) + trial)
        
        # Modify the heuristic to include randomness based on rng
        # We can't easily pass rng to the nested dfs without closure tricks or globals
        # Let's use a wrapper that injects noise
        
        def _noisy_heuristic(sequence, chain, occupied, i, site):
            base = _heuristic_score(sequence, chain, occupied, i, site)
            return base + rng.random() * 0.5 # Small noise to diversify paths

        # Monkey patch or pass function? 
        # For simplicity in this constrained environment, let's just run the standard 
        # backtracking but with a random start or random tie-breaking.
        
        # Actually, the current _backtrack_fold uses a fixed heuristic.
        # Let's create a version that accepts a noise factor.
        
        def dfs_with_noise(depth, noise_rng):
            if depth == n:
                return True
                
            current = chain[-1]
            candidates = []
            
            for site in _neighbours(current):
                if site in occupied:
                    continue
                score = _heuristic_score(sequence, chain, occupied, depth, site)
                score += noise_rng.random() * 1.0
                candidates.append((score, site))
                
            if not candidates:
                return False
                
            candidates.sort(key=lambda x: x[0], reverse=True)
            
            for score, site in candidates:
                chain.append(site)
                occupied[site] = depth
                
                if dfs_with_noise(depth + 1, noise_rng):
                    return True
                
                chain.pop()
                del occupied[site]
                
            return False

        chain = [ (0,) * dimension ]
        occupied = { (0,) * dimension : 0 }
        
        if dfs_with_noise(1, rng):
            e = _energy(sequence, chain)
            if e < best_energy:
                best_energy = e
                best_chain = chain
                
    if best_chain is None:
        # Fallback to straight line if no solution found (should not happen for valid HP)
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
