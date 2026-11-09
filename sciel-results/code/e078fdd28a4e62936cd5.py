"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from math import exp

_START_TIME = time.time()
_DEADLINE = 205.0  # Safety margin before 210s kill


def _remaining_time():
    return _DEADLINE - (time.time() - _START_TIME)


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute the energy of a conformation."""
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


def _local_move(sequence, chain, rng):
    """
    Perform a single local move (pivot or rotation) on the chain.
    Returns the new chain if the move is valid (self-avoiding), else None.
    """
    n = len(chain)
    if n < 3:
        return None
    
    # Choose a pivot point (not the first or last)
    pivot = rng.randint(1, n - 2)
    
    # Try all 3 possible rotations (90, 180, 270 degrees)
    # For 2D, we can rotate the tail segment
    dx = chain[pivot + 1][0] - chain[pivot][0]
    dy = chain[pivot + 1][1] - chain[pivot][1]
    
    # The tail is chain[pivot+1:]
    # We'll try rotating the tail by 90, 180, 270 degrees around the pivot
    # In 2D, rotation by 90 deg: (dx, dy) -> (-dy, dx)
    # Rotation by 180 deg: (dx, dy) -> (-dx, -dy)
    # Rotation by 270 deg: (dx, dy) -> (dy, -dx)
    
    # Actually, for a proper pivot move, we rotate the entire tail segment
    # Let's implement it properly:
    
    tail = chain[pivot + 1:]
    if not tail:
        return None
    
    # Build the set of occupied positions for the head part (chain[:pivot+1])
    # We need to check if the new tail positions collide with the head
    
    # Try the 3 rotations
    rotations = [
        (-tail[0][1], tail[0][0]),      # 90 deg: (dx, dy) -> (-dy, dx)
        (-tail[0][0], -tail[0][1]),     # 180 deg
        (tail[0][1], -tail[0][0]),      # 270 deg
    ]
    
    # Actually, let's do it more carefully. The direction from pivot to tail[0] is (dx, dy).
    # We rotate the entire tail by 90, 180, or 270 degrees around the pivot.
    
    # Let's just try shifting the tail in different directions
    # A simpler approach: try moving the tail in 3 new directions
    
    # Get the current direction of the first tail segment
    # Try 3 perpendicular/opposite directions
    
    # Actually, let's use a simpler local move: pick a random position and try to move it
    # by one step in a random direction, if it doesn't collide.
    
    # Let's do a "slide" move: pick a random position i (1 to n-2), try to move it
    # in one of 4 directions, checking for collisions with all other positions.
    
    i = rng.randint(1, n - 2)
    # Try 4 directions
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    rng.shuffle(directions)
    
    for dx, dy in directions:
        new_pos = (chain[i][0] + dx, chain[i][1] + dy)
        # Check if new_pos is not occupied by any other position
        # We need to check against all positions except i
        # Also, the move must be a unit step from both neighbors
        
        # Check unit step from previous
        if abs(chain[i][0] - chain[i-1][0]) + abs(chain[i][1] - chain[i-1][1]) != 1:
            continue  # Shouldn't happen
        
        # Check unit step to new position from previous
        if abs(new_pos[0] - chain[i-1][0]) + abs(new_pos[1] - chain[i-1][1]) != 1:
            continue
        
        # Check unit step to new position from next
        if abs(new_pos[0] - chain[i+1][0]) + abs(new_pos[1] - chain[i+1][1]) != 1:
            continue
        
        # Check no collision with other positions
        collides = False
        for j, pos in enumerate(chain):
            if j == i:
                continue
            if pos == new_pos:
                collides = True
                break
        
        if not collides:
            new_chain = chain.copy()
            new_chain[i] = new_pos
            return new_chain
    
    return None


def _anneal(sequence, chain, max_steps, rng):
    """
    Perform simulated annealing on the chain.
    Returns the best chain found.
    """
    n = len(chain)
    if n < 3:
        return chain
    
    current_energy = _energy(sequence, chain)
    best_chain = chain
    best_energy = current_energy
    
    # Temperature schedule
    T_start = 2.0
    T_end = 0.1
    cooling = (T_end / T_start) ** (1.0 / max_steps)
    
    for step in range(max_steps):
        if _remaining_time() < 0.5:
            break
        
        T = T_start * (cooling ** step)
        
        new_chain = _local_move(sequence, chain, rng)
        if new_chain is None:
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta <= 0 or rng.random() < exp(-delta / T):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain
        
        # Occasionally restart from a new growth to escape local minima
        if step % 500 == 499:
            new_growth = _grow(sequence, 2, rng)
            if new_growth is not None:
                ge = _energy(sequence, new_growth)
                if ge < current_energy:
                    chain = new_growth
                    current_energy = ge
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = chain
    
    return best_chain


def fold(sequence, dimension):
    """Best conformation using greedy growth + simulated annealing."""
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    
    # First, get initial solutions via greedy growth
    initial_chains = []
    for _ in range(50):
        chain = _grow(sequence, dimension, rng)
        if chain is not None:
            initial_chains.append(chain)
    
    if not initial_chains:
        # Fallback: straight line
        return [list(p) for p in [(i, 0) for i in range(n)]]
    
    # Find the best initial chain
    best_chain = min(initial_chains, key=lambda c: _energy(sequence, c))
    best_energy = _energy(sequence, best_chain)
    
    # Now apply simulated annealing
    # Distribute the remaining time across multiple annealing runs
    time_budget = _remaining_time() - 2.0  # Leave 2s safety margin
    
    # Each annealing run takes roughly n * max_steps * cost_per_step
    # Let's estimate: for n=100, max_steps=10000, that's ~1M operations
    # We'll do multiple short runs
    
    # Do up to 5 annealing runs, each with adaptive max_steps
    num_runs = 5
    steps_per_run = max(1000, int(10000 * (100 / n)))  # Scale with chain length
    
    for run in range(num_runs):
        if _remaining_time() < 1.0:
            break
        
        # Start from the current best or a new random growth
        if run == 0:
            start_chain = best_chain
        else:
            # Try starting from a new growth or the best
            if rng.random() < 0.5:
                start_chain = _grow(sequence, dimension, rng)
                if start_chain is None:
                    start_chain = best_chain
            else:
                start_chain = best_chain
        
        # Run annealing
        improved = _anneal(sequence, start_chain, steps_per_run, rng)
        improved_energy = _energy(sequence, improved)
        
        if improved_energy < best_energy:
            best_chain = improved
            best_energy = improved_energy
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
