"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import time
import random
import sys

# Explicit deadline: evaluator kills after 210s wall-clock; keep 30s safety margin.
_DEADLINE = time.time() + 170.0


def _neighbours(point):
    """Yield the 4 unit-step neighbours of a 2D point (tuple)."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _compute_energy(sequence, chain):
    """Compute the HP energy: -1 per non-consecutive H-H contact."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts // 2  # each contact counted twice


def _grow_greedy(sequence, dimension, rng, temperature=0.5):
    """
    Randomized greedy chain growth with simulated-annealing-like acceptance.
    Prefers moves that create new H-H contacts but allows occasional non-optimal
    choices to escape local minima.
    """
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        current_site = chain[-1]
        options = []
        
        for site in _neighbours(current_site):
            if site in occupied:
                continue
            
            # Calculate gain: number of new H-H contacts if we place residue i here
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            
            # Free space heuristic: prefer sites with more open neighbours
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            
            # Combine gain, free space, and randomness
            score = gain * 2.0 + free * 0.1 + rng.random() * temperature
            options.append((score, site))
        
        if not options:
            return None
        
        # Sort by score descending and pick with some randomness
        options.sort(key=lambda x: -x[0])
        
        # Pick from top options with probability weighted by rank
        top_k = min(3, len(options))
        if rng.random() < 0.7:
            site = options[0][1]
        elif top_k > 1:
            site = options[rng.randint(0, top_k - 1)][1]
        else:
            site = options[0][1]
        
        occupied[site] = i
        chain.append(site)
    
    return chain


def _local_search(sequence, chain, max_steps=200):
    """
    Local search: try small perturbations (rotations, reversals of short segments)
    and keep if energy improves.
    """
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    n = len(chain)
    rng = random.Random(42)
    
    steps = 0
    while steps < max_steps and time.time() < _DEADLINE - 5:
        steps += 1
        new_chain = list(chain)
        
        # Try different perturbation types
        pert_type = rng.random()
        
        if pert_type < 0.4 and n > 4:
            # Reverse a short segment
            start = rng.randint(1, n - 3)
            length = rng.randint(2, min(4, n - start - 1))
            end = start + length - 1
            new_chain[start:end + 1] = reversed(new_chain[start:end + 1])
            
        elif pert_type < 0.7 and n > 3:
            # Rotate a short segment
            start = rng.randint(1, n - 3)
            length = rng.randint(2, min(3, n - start - 1))
            end = start + length - 1
            segment = new_chain[start:end + 1]
            shift = rng.randint(1, length - 1)
            segment = segment[shift:] + segment[:shift]
            new_chain[start:end + 1] = segment
            
        else:
            # Swap two non-adjacent positions (risky, may break self-avoidance)
            i = rng.randint(1, n - 2)
            j = rng.randint(1, n - 2)
            if abs(i - j) > 1:
                new_chain[i], new_chain[j] = new_chain[j], new_chain[i]
        
        # Check self-avoidance
        if len(set(tuple(p) for p in new_chain)) != n:
            continue
        
        # Check unit steps
        valid = True
        for k in range(1, n):
            dx = abs(new_chain[k][0] - new_chain[k-1][0])
            dy = abs(new_chain[k][1] - new_chain[k-1][1])
            if dx + dy != 1:
                valid = False
                break
        
        if not valid:
            continue
        
        new_energy = _compute_energy(sequence, new_chain)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_chain = new_chain
            chain = new_chain
        elif rng.random() < 0.1 and new_energy < current_energy:
            # Accept worse moves occasionally to escape local minima
            chain = new_chain
            current_energy = new_energy
        else:
            current_energy = _compute_energy(chain)
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Find the best conformation using multiple strategies:
    1. Multiple randomized greedy growths
    2. Local search refinement on promising solutions
    """
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 0
    
    # Phase 1: Generate diverse initial solutions via randomized growth
    num_growths = 50
    candidates = []
    
    for _ in range(num_growths):
        if time.time() > _DEADLINE - 10:
            break
        
        # Vary temperature for diversity
        temp = 0.3 + rng.random() * 0.7
        chain = _grow_greedy(sequence, dimension, rng, temperature=temp)
        if chain is None:
            continue
        
        energy = _compute_energy(sequence, chain)
        candidates.append((energy, chain))
    
    # Sort by energy (best first)
    candidates.sort(key=lambda x: x[0])
    
    # Phase 2: Local search on top candidates
    num_ls = min(10, len(candidates))
    for energy, chain in candidates[:num_ls]:
        if time.time() > _DEADLINE - 15:
            break
        
        refined_chain, refined_energy = _local_search(sequence, chain, max_steps=100)
        
        if refined_energy < best_energy or best_chain is None:
            best_energy = refined_energy
            best_chain = refined_chain
    
    # If no valid chain found, return a straight line
    if best_chain is None:
        best_chain = [(i, 0) for i in range(len(sequence))]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
