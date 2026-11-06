"""Rewritten HP lattice folding program using simulated annealing with chain moves."""
import random
import math


def _neighbours(point):
    """Yield the 4 lattice neighbours of a 2D point (tuple)."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute HP energy: -1 for each non-consecutive H-H contact."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    n = len(chain)
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _is_self_avoiding(chain):
    """Check that no two positions in the chain are the same."""
    return len(set(chain)) == len(chain)


def _valid_step(chain, new_pos, i):
    """Check if placing new_pos at index i keeps chain self-avoiding (excluding position i itself)."""
    if i > 0 and new_pos == chain[i - 1]:
        return False
    if i < len(chain) - 1 and new_pos == chain[i + 1]:
        return False
    # Check against all other positions
    for j, p in enumerate(chain):
        if j == i or j == i - 1 or j == i + 1:
            continue
        if new_pos == p:
            return False
    return True


def _initial_chain(sequence, dimension):
    """Create an initial self-avoiding walk using a simple spiral/greedy approach."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0,) * dimension]
    
    # Start with a straight line, then try to make it more compact
    chain = [(i, 0) for i in range(n)]
    return chain


def _make_move(chain, sequence, rng):
    """
    Perform a random move on the chain: pick a random interior residue and try to move it
    to one of its neighbours, if that keeps the chain self-avoiding.
    Returns (new_chain, delta_energy) or (None, 0) if no valid move found.
    """
    n = len(chain)
    if n <= 2:
        return None, 0
    
    # Pick a random interior position
    i = rng.randint(1, n - 2)
    old_pos = chain[i]
    
    # Try all 4 neighbours
    neighbours = list(_neighbours(old_pos))
    rng.shuffle(neighbours)
    
    for new_pos in neighbours:
        if _valid_step(chain, new_pos, i):
            # Compute delta energy
            old_energy = _energy(sequence, chain)
            new_chain = chain[:i] + [new_pos] + chain[i+1:]
            new_energy = _energy(sequence, new_chain)
            return new_chain, new_energy - old_energy
    
    return None, 0


def _simulated_annealing(sequence, dimension, rng, max_iter=5000, T0=1.0, T1=0.01):
    """
    Run simulated annealing to find a low-energy conformation.
    Returns the best chain found.
    """
    n = len(sequence)
    if n <= 1:
        return _initial_chain(sequence, dimension)
    
    # Start with a compact initial conformation
    chain = _initial_chain(sequence, dimension)
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    max_iter = min(max_iter, 5000)
    T0 = 1.0
    T1 = 0.01
    
    for it in range(max_iter):
        T = T0 * (T1 / T0) ** (it / max_iter)
        
        new_chain, delta = _make_move(chain, sequence, rng)
        if new_chain is None:
            continue
        
        if delta <= 0 or rng.random() < math.exp(-delta / T):
            chain = new_chain
            current_energy += delta
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
    
    return best_chain


def fold(sequence, dimension):
    """
    Find a low-energy self-avoiding conformation for the given HP sequence on a 2D lattice.
    Returns a list of len(sequence) integer lattice points (lists of length dimension).
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    # Use multiple restarts with simulated annealing
    best_chain = None
    best_energy = 0  # Start at 0 (unfolded)
    
    # Number of restarts based on sequence length to stay within time budget
    # Total budget is 120s for all sequences. Be conservative.
    num_restarts = 10 if n <= 40 else 5 if n <= 70 else 3
    
    base_seed = 12345 + len(sequence)
    
    for restart in range(num_restarts):
        rng = random.Random(base_seed + restart)
        
        # Create a varied initial chain
        if restart == 0:
            # Straight line
            chain = _initial_chain(sequence, dimension)
        else:
            # Try a different initial pattern
            if restart == 1:
                # Zigzag
                chain = []
                x, y = 0, 0
                for i in range(n):
                    chain.append((x, y))
                    if i % 2 == 0:
                        x += 1
                    else:
                        y += 1
            else:
                # Random walk initial
                chain = [(0, 0)]
                occupied = {chain[0]}
                directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                for i in range(1, n):
                    valid_dirs = [d for d in directions if (chain[-1][0] + d[0], chain[-1][1] + d[1]) not in occupied]
                    if not valid_dirs:
                        # Fallback: extend in a new direction
                        valid_dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                        valid_dirs = [d for d in valid_dirs if (chain[-1][0] + d[0], chain[-1][1] + d[1]) not in occupied]
                        if not valid_dirs:
                            break
                    d = rng.choice(valid_dirs)
                    new_pos = (chain[-1][0] + d[0], chain[-1][1] + d[1])
                    chain.append(new_pos)
                    occupied.add(new_pos)
                if len(chain) < n:
                    # Fallback to straight line
                    chain = _initial_chain(sequence, dimension)
        
        # Verify initial chain is valid
        if not _is_self_avoiding(chain):
            chain = _initial_chain(sequence, dimension)
        
        # Run simulated annealing
        result = _simulated_annealing(sequence, dimension, rng, max_iter=3000)
        
        if result is not None and len(result) == n:
            energy = _energy(sequence, result)
            if energy < best_energy:
                best_energy = energy
                best_chain = result
    
    # Fallback if no valid chain found
    if best_chain is None or len(best_chain) != n:
        best_chain = _initial_chain(sequence, dimension)
    
    # Convert to list of lists
    return [list(p) for p in best_chain]
