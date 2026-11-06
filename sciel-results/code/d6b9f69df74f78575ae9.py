"""HP lattice protein folding on the cubic lattice.
Strategy: Beam search growth + simulated annealing with kink moves and deadline guard.
"""
# EVOLVE-BLOCK-START
import random
import time
import math

# Global deadline guard
_TOTAL_DEADLINE = 330.0  # seconds wall-clock
_start_time = None


def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def _time_remaining():
    return _TOTAL_DEADLINE - _get_elapsed() - 10.0  # 10s safety margin


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute energy: -number of H-H contacts (non-adjacent along chain)."""
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


def _beam_growth(sequence, dimension, beam_width=20):
    """Beam search growth: maintain top-k partial chains by contact gain."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0,) * dimension]
    
    start = (0,) * dimension
    initial_chain = [start]
    initial_occupied = {start: 0}
    
    # Beam: list of (score, chain, occupied) where score = -contacts (higher is better)
    beam = [(0, initial_chain, initial_occupied)]
    
    for i in range(1, n):
        new_beam = []
        for score, chain, occupied in beam:
            last = chain[-1]
            residue = sequence[i]
            for site in _neighbours(last):
                if site in occupied:
                    continue
                # Calculate gain
                gain = 0
                if residue == 'H':
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == 'H':
                            gain += 1
                # Penalty for reducing free space (optional, keep simple)
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                # Score: primarily gain, small bonus for free space
                new_score = score + gain + 0.05 * free
                new_chain = chain + [site]
                new_occupied = dict(occupied)
                new_occupied[site] = i
                new_beam.append((new_score, new_chain, new_occupied))
        
        if not new_beam:
            break
        
        # Sort by score descending and keep top beam_width
        new_beam.sort(key=lambda x: x[0], reverse=True)
        beam = new_beam[:beam_width]
        
        # Early exit if time is running low
        if _time_remaining() < 5.0:
            break
    
    if not beam:
        return None
    
    # Return the best chain from the beam
    best_entry = max(beam, key=lambda x: x[0])
    return best_entry[1]


def _kink_move(chain, rng):
    """Try a kink move: pick two non-adjacent positions and reverse the subchain."""
    n = len(chain)
    if n < 4:
        return None
    i = rng.randint(0, n - 4)
    j = rng.randint(i + 2, n - 1)
    segment = chain[i + 1:j + 1]
    new_segment = segment[::-1]
    
    # Check self-avoidance for the new positions
    occupied = set(chain)
    for p in segment:
        occupied.discard(p)
    
    valid = True
    for p in new_segment:
        if p in occupied:
            valid = False
            break
    if not valid:
        return None
    
    # Check connectivity at boundaries
    # chain[i] connects to new_segment[0]
    if abs(new_segment[0][0] - chain[i][0]) + abs(new_segment[0][1] - chain[i][1]) + abs(new_segment[0][2] - chain[i][2]) != 1:
        return None
    # new_segment[-1] connects to chain[j+1]
    if j + 1 < n:
        if abs(new_segment[-1][0] - chain[j + 1][0]) + abs(new_segment[-1][1] - chain[j + 1][1]) + abs(new_segment[-1][2] - chain[j + 1][2]) != 1:
            return None
    
    new_chain = chain[:i + 1] + new_segment + chain[j + 1:]
    return new_chain


def _simulated_annealing(sequence, chain, rng, time_budget):
    """Simulated annealing with kink moves."""
    n = len(chain)
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    # Temperature schedule
    T = max(1.0, abs(current_energy) * 0.5)
    T_min = 0.01
    cooling_rate = 0.995
    
    start_t = time.time()
    while time.time() - start_t < time_budget and T > T_min:
        new_chain = _kink_move(chain, rng)
        
        if new_chain is None:
            T *= cooling_rate
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
        
        T *= cooling_rate
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Fold a sequence using beam search growth + simulated annealing."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    n = len(sequence)
    rng = random.Random(12345 + n)
    
    # Allocate time based on sequence length
    if n <= 48:
        num_restarts = 10
        sa_budget_per_restart = 5.0
        beam_width = 30
    elif n <= 64:
        num_restarts = 8
        sa_budget_per_restart = 6.0
        beam_width = 25
    elif n <= 96:
        num_restarts = 6
        sa_budget_per_restart = 8.0
        beam_width = 20
    else:
        num_restarts = 4
        sa_budget_per_restart = 10.0
        beam_width = 15
    
    best_chain = None
    best_energy = 0
    
    for restart in range(num_restarts):
        if _time_remaining() < 15.0:
            break
        
        # Beam search growth
        chain = _beam_growth(sequence, dimension, beam_width)
        if chain is None:
            continue
        
        energy = _energy(sequence, chain)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = list(chain)
        
        # Simulated annealing to improve
        sa_budget = min(sa_budget_per_restart, _time_remaining() / 2.0)
        if sa_budget > 0.5:
            improved_chain, improved_energy = _simulated_annealing(sequence, chain, rng, sa_budget)
            if improved_energy < best_energy:
                best_energy = improved_energy
                best_chain = improved_chain
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
