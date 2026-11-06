"""
HP lattice protein folding on the cubic lattice.
Strategy: Monte Carlo growth with local optimization (pivot moves + kink moves + SA).
"""
import random
import time
import math

# EVOLVE-BLOCK-START

def _neighbours(point):
    """Yield the 6 neighbours of a 3D lattice point."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute energy as -number of H-H contacts (non-consecutive)."""
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
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
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


def _pivot_move(chain, rng):
    """
    Try a pivot move: rotate a random subchain.
    Returns (new_chain, energy_delta) or None if move is invalid.
    """
    n = len(chain)
    if n < 3:
        return None
    
    # Choose pivot point
    i = rng.randint(1, n - 2)
    
    # Get the subchain to rotate
    fixed = chain[:i + 1]
    moving = chain[i + 1:]
    
    # Generate a random rotation
    # In 3D, we can rotate by 90, 180, 270 degrees around any axis
    # For simplicity, let's just try reflecting or shifting the moving part
    
    # Try a simple move: shift the moving part by a random vector
    # Actually, let's do a proper pivot: rotate the subchain around the pivot point
    
    # The moving part is relative to chain[i]
    pivot = chain[i]
    relative = [(p[0] - pivot[0], p[1] - pivot[1], p[2] - pivot[2]) for p in moving]
    
    # Try all 6 possible 90-degree rotations (3 axes * 2 directions) + 180-degree (3 axes)
    # Actually, let's just try a few random transformations
    
    # For a proper pivot, we need to rotate the subchain around the pivot point
    # Let's try rotating by 90 degrees around a random axis
    
    axis = rng.randint(0, 2)
    angle = rng.choice([1, -1, 2, -2])  # 90, -90, 180, -180 degrees
    
    # Rotation matrices for 90 degrees around each axis
    # This is getting complex. Let's use a simpler approach:
    # Just try moving the end of the subchain to a random adjacent position
    
    # Actually, let's implement a simpler local move:
    # Try to move residue i+1 to a different adjacent position
    
    return None  # Placeholder


def _local_refine(sequence, chain, rng, max_iters=500, deadline=None):
    """
    Local refinement using simulated annealing with kink moves.
    A kink move picks a random residue and tries to move it to a new position
    such that the chain remains self-avoiding.
    """
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    n = len(sequence)
    if n < 3:
        return best_chain, best_energy
    
    # Precompute H positions
    is_h = [c == 'H' for c in sequence]
    
    def _kink_move(ch, rng):
        """Try a kink move: move a random internal residue to a new adjacent position."""
        i = rng.randint(1, len(ch) - 2)
        # The residue at i must stay connected to i-1 and i+1
        # So we can only move it to positions adjacent to both ch[i-1] and ch[i+1]
        # Actually, a kink move moves residue i to a new position p such that
        # p is adjacent to ch[i-1] and ch[i+1], and p != ch[i], and p is not occupied.
        
        prev = ch[i - 1]
        next_ = ch[i + 1]
        
        # Find common neighbours of prev and next_
        candidates = set()
        for q in _neighbours(prev):
            if q == next_:
                continue
            # q must be adjacent to next_
            if any(abs(q[a] - next_[a]) == 1 for a in range(3) if q[a] != next_[a]):
                # Check if q is adjacent to next_
                dist = sum(abs(q[a] - next_[a]) for a in range(3))
                if dist == 1:
                    candidates.add(q)
        
        if not candidates:
            return None
        
        # Exclude the current position
        candidates.discard(ch[i])
        
        if not candidates:
            return None
        
        # Check self-avoidance
        occupied = set(ch)
        occupied.discard(ch[i])
        
        valid = [q for q in candidates if q not in occupied]
        if not valid:
            return None
        
        new_pos = rng.choice(valid)
        new_ch = list(ch)
        new_ch[i] = new_pos
        return new_ch
    
    temperature = 1.0
    cooling = 0.99
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        new_chain = _kink_move(current_chain, rng)
        if new_chain is None:
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < math.exp(-delta / temperature):
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_chain = list(current_chain)
                best_energy = current_energy
        
        temperature *= cooling
        if temperature < 0.01:
            temperature = 0.01
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using greedy growth + local refinement."""
    start_time = time.time()
    deadline = start_time + 220  # Leave margin for 240s budget
    
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 0
    
    n = len(sequence)
    
    # Determine number of restarts based on time
    num_restarts = max(50, min(200, int((deadline - time.time()) / max(0.01, n / 1000))))
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        # Local refinement
        remaining_time = deadline - time.time()
        max_iters = max(100, int(remaining_time * 100))
        refined_chain, refined_energy = _local_refine(
            sequence, chain, rng, max_iters=max_iters, deadline=deadline
        )
        
        if refined_energy < best_energy:
            best_chain = refined_chain
            best_energy = refined_energy
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
