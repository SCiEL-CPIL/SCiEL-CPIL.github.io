"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time

# Explicit deadline: evaluator kills after 210s wall-clock; keep 30s safety margin.
_DEADLINE = time.time() + 180.0


def _neighbours(point):
    """Yield the 4 unit-step neighbours of a 2D point (tuple)."""
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _compute_energy(sequence, chain):
    """Compute the HP contact energy: -1 for each H-H non-bonded contact."""
    n = len(chain)
    pos_to_idx = {}
    for i, p in enumerate(chain):
        pos_to_idx[p] = i

    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        x, y = chain[i]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j_pos = (x + dx, y + dy)
            j = pos_to_idx.get(j_pos)
            if j is not None and abs(i - j) > 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts // 2  # Each contact counted twice


def _greedy_fold(sequence, dimension, rng, temperature=0.05):
    """
    Simulated annealing style chain growth.
    Returns a valid self-avoiding chain or None if stuck.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        candidates = []
        last = chain[-1]
        for site in _neighbours(last):
            if site not in occupied:
                # Calculate immediate gain (new H-H contacts)
                gain = 0
                if sequence[i] == 'H':
                    for nb in _neighbours(site):
                        idx = occupied.get(nb)
                        if idx is not None and idx < i - 1 and sequence[idx] == 'H':
                            gain += 1
                # Calculate free space (heuristic for avoiding dead ends)
                free = sum(1 for nb in _neighbours(site) if nb not in occupied)
                
                # Score: gain is primary, free space secondary, random noise
                score = gain * 10.0 + free * 0.5 + rng.random() * temperature
                candidates.append((score, site))
        
        if not candidates:
            return None
        
        # Pick best candidate (with some randomness if close scores)
        # Sort by score descending
        candidates.sort(key=lambda x: -x[0])
        
        # With some probability, pick a random top-3 to allow exploration
        if len(candidates) > 1 and rng.random() < 0.2:
            k = min(3, len(candidates))
            choice_idx = rng.randint(0, k - 1)
        else:
            choice_idx = 0
            
        best_site = candidates[choice_idx][1]
        chain.append(best_site)
        occupied[best_site] = i
        
    return chain


def _local_search(sequence, chain, rng, max_iterations=2000):
    """
    Try to improve the chain energy via local moves (reversals, 180-degree turns, etc).
    This is a simplified version; for strict self-avoidance, we only accept valid moves.
    """
    if not chain:
        return chain
    
    n = len(chain)
    current_energy = _compute_energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    
    occupied = {p: i for i, p in enumerate(chain)}
    
    # Simple local search: try to move a single residue (if possible) or small segments.
    # Given the complexity of self-avoiding walks, single residue moves are rare.
    # Instead, we'll rely on the greedy growth being decent and just return.
    # A full local search is complex and might be slow.
    
    return best_chain


def fold(sequence, dimension):
    """
    Find the best conformation for the given HP sequence.
    Returns a list of [x, y] coordinates.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    # Fallback: straight line
    straight = [[i, 0] for i in range(n)]
    
    best_chain = None
    best_energy = 0  # Energy is negative, so we want the most negative (lowest)
    
    rng = random.Random(42)
    
    # Number of restarts based on time budget
    # We have 180s total. Let's use a fixed number of restarts that is safe.
    # For n=100, greedy is O(n^2) roughly. 100 restarts should be fast.
    num_restarts = 500
    
    for r in range(num_restarts):
        if time.time() > _DEADLINE - 10:
            break
            
        # Vary temperature slightly for diversity
        temp = 0.01 + rng.random() * 0.1
        chain = _greedy_fold(sequence, dimension, rng, temperature=temp)
        
        if chain is None:
            continue
            
        energy = _compute_energy(sequence, chain)
        
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy
            
            # If we hit a very good energy, we might stop early
            # But we don't know the best possible, so keep going.
            
    if best_chain is None:
        return straight
        
    # Convert tuples to lists
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
