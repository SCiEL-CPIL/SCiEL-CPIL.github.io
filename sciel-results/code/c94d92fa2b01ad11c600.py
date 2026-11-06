"""HP lattice protein folding using multi-start simulated annealing with reversals."""
# EVOLVE-BLOCK-START
import random
import time
import math
from collections import defaultdict


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _build_contact_map(sequence, chain):
    """Build mapping from lattice point to residue index and set of H lattice points."""
    where = {}
    h_points = set()
    for i, p in enumerate(chain):
        pt = tuple(p)
        where[pt] = i
        if sequence[i] == 'H':
            h_points.add(pt)
    return where, h_points


def _compute_energy(sequence, chain):
    """Compute energy as minus the number of non-consecutive H-H contacts."""
    where, h_points = _build_contact_map(sequence, chain)
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _move_reversal(chain, i, j):
    """Reverse the segment from index i to j (inclusive). Returns new chain or None if self-intersecting."""
    if j <= i:
        return None
    n = len(chain)
    new_chain = chain[:]
    new_chain[i:j+1] = reversed(chain[i:j+1])
    
    # Check self-avoidance for the reversed segment
    # The endpoints i and j stay connected to their neighbors, so we need to check:
    # 1. The reversed segment doesn't intersect itself (it doesn't since original was self-avoiding)
    # 2. The reversed segment doesn't intersect the parts outside [i, j]
    # 3. The connections at i-1->i and j->j+1 are still valid (they are, since endpoints don't move)
    
    # Build set of occupied points for quick lookup
    occupied = set()
    for k in range(n):
        if i <= k <= j:
            continue
        occupied.add(tuple(new_chain[k]))
    
    for k in range(i, j+1):
        if tuple(new_chain[k]) in occupied:
            return None
        occupied.add(tuple(new_chain[k]))
    
    return new_chain


def _move_kink(chain, i):
    """Try to move residue i to an adjacent empty site. Returns new chain or None."""
    if i <= 0 or i >= len(chain) - 1:
        return None
    
    current = tuple(chain[i])
    for neighbor in _neighbours(current):
        # Check if this neighbor is already occupied (excluding the residue itself and its neighbors in chain)
        # The residue at i is connected to i-1 and i+1, so those are fine
        occupied = set()
        for k in range(len(chain)):
            if k == i or k == i-1 or k == i+1:
                continue
            occupied.add(tuple(chain[k]))
        
        if neighbor not in occupied:
            new_chain = chain[:]
            new_chain[i] = list(neighbor)
            return new_chain
    return None


def _simulated_annealing(sequence, chain, rng, time_limit):
    """Run simulated annealing on the given chain."""
    start_time = time.time()
    n = len(chain)
    
    # Initial energy
    current_energy = _compute_energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    
    # Temperature schedule
    initial_temp = 2.0
    final_temp = 0.01
    cooling_rate = 0.999
    
    temp = initial_temp
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() - start_time < time_limit:
        if temp <= final_temp:
            break
        
        # Choose a move
        move_type = rng.random()
        new_chain = None
        
        if move_type < 0.7:
            # Reversal move
            i = rng.randint(0, n - 2)
            j = rng.randint(i + 1, n - 1)
            # Limit reversal length for efficiency
            if j - i > 20:
                j = min(i + 20, n - 1)
            new_chain = _move_reversal(chain, i, j)
        else:
            # Kink move
            i = rng.randint(1, n - 2)
            new_chain = _move_kink(chain, i)
        
        if new_chain is None:
            continue
        
        new_energy = _compute_energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < math.exp(-delta / temp):
            chain = new_chain
            current_energy = new_energy
            no_improve_count = 0
            if current_energy < best_energy:
                best_chain = chain[:]
                best_energy = current_energy
        else:
            no_improve_count += 1
        
        temp *= cooling_rate
        
        if no_improve_count > max_no_improve:
            # Restart with current best
            chain = best_chain[:]
            current_energy = best_energy
            no_improve_count = 0
            temp = initial_temp * 0.5
    
    return best_chain, best_energy


def _greedy_grow(sequence, dimension, rng):
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


def _polish(sequence, chain, rng, max_iterations=100):
    """Local search polish: try moves that strictly improve energy."""
    current_energy = _compute_energy(sequence, chain)
    n = len(chain)
    improved = True
    iterations = 0
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # Try reversals
        for i in range(n - 1):
            for j in range(i + 1, min(i + 15, n)):
                new_chain = _move_reversal(chain, i, j)
                if new_chain is None:
                    continue
                new_energy = _compute_energy(sequence, new_chain)
                if new_energy < current_energy:
                    chain = new_chain
                    current_energy = new_energy
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Try kinks
        for i in range(1, n - 1):
            new_chain = _move_kink(chain, i)
            if new_chain is None:
                continue
            new_energy = _compute_energy(sequence, new_chain)
            if new_energy < current_energy:
                chain = new_chain
                current_energy = new_energy
                improved = True
                break
    
    return chain, current_energy


def fold(sequence, dimension):
    """Fold the sequence using multi-start simulated annealing with polishing."""
    n = len(sequence)
    deadline = time.time() + 180.0  # Safety margin before 210s kill
    
    # Number of restarts based on sequence length
    if n <= 30:
        num_restarts = 30
        sa_time_per_restart = 3.0
    elif n <= 50:
        num_restarts = 20
        sa_time_per_restart = 5.0
    elif n <= 70:
        num_restarts = 15
        sa_time_per_restart = 7.0
    else:
        num_restarts = 10
        sa_time_per_restart = 10.0
    
    rng = random.Random(12345 + n)
    best_chain = None
    best_energy = 0
    
    for restart in range(num_restarts):
        # Check time
        if time.time() > deadline - 30:
            break
        
        # Generate initial conformation via greedy growth
        chain = _greedy_grow(sequence, dimension, rng)
        if chain is None:
            # Fallback: straight line
            chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        
        # Run simulated annealing
        sa_time = min(sa_time_per_restart, max(1.0, deadline - time.time() - 30))
        polished_chain, energy = _simulated_annealing(sequence, chain, rng, sa_time)
        
        # Polish the result
        polished_chain, energy = _polish(sequence, polished_chain, rng)
        
        if best_chain is None or energy < best_energy:
            best_chain = polished_chain
            best_energy = energy
    
    # Final polish if we have time
    if best_chain is not None and time.time() < deadline - 10:
        best_chain, best_energy = _polish(sequence, best_chain, rng, max_iterations=200)
    
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
