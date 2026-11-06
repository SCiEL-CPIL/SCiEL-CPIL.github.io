"""HP lattice protein folding on the cubic lattice.
Strategy: Multi-start deterministic beam search with incremental energy tracking,
followed by a fast local search (kink moves) with strict time budgeting.
"""
# EVOLVE-BLOCK-START
import time
import math
import random

_TOTAL_DEADLINE = 320.0  # seconds wall-clock (10s safety margin from 330s limit)
_start_time = None


def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def _time_remaining():
    return _TOTAL_DEADLINE - _get_elapsed() - 5.0  # 5s extra safety margin


def _neighbours(point):
    """Yield 6 neighbours on the cubic lattice."""
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


class _BeamState:
    """A single beam entry with incremental energy tracking."""
    __slots__ = ['score', 'chain', 'occupied', 'h_positions']

    def __init__(self, score, chain, occupied, h_positions):
        self.score = score  # -contacts (higher is better, i.e., more negative energy)
        self.chain = chain
        self.occupied = occupied  # dict: point -> index
        self.h_positions = h_positions  # set of indices where H residues are


def _beam_growth(sequence, dimension, beam_width, time_limit):
    """Beam search growth with incremental energy. Returns best chain or None."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0,) * dimension]

    start = (0,) * dimension
    initial_chain = [start]
    initial_occupied = {start: 0}
    initial_h = {0} if sequence[0] == 'H' else set()

    beam = [_BeamState(0, initial_chain, initial_occupied, initial_h)]

    for i in range(1, n):
        new_beam = []
        residue = sequence[i]

        for state in beam:
            last = state.chain[-1]
            for site in _neighbours(last):
                if site in state.occupied:
                    continue

                # Calculate incremental gain
                gain = 0
                if residue == 'H':
                    for q in _neighbours(site):
                        j = state.occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == 'H':
                            gain += 1

                # Small bonus for free space to encourage exploration
                free = 0
                for q in _neighbours(site):
                    if q not in state.occupied:
                        free += 1

                new_score = state.score + gain + 0.02 * free
                new_chain = state.chain + [site]
                new_occupied = dict(state.occupied)
                new_occupied[site] = i
                new_h = state.h_positions | ({i} if residue == 'H' else set())

                new_beam.append(_BeamState(new_score, new_chain, new_occupied, new_h))

        if not new_beam:
            break

        # Sort by score descending and keep top beam_width
        new_beam.sort(key=lambda x: x.score, reverse=True)
        beam = new_beam[:beam_width]

        # Early exit if time is running low
        if _time_remaining() < 3.0:
            break

    if not beam:
        return None

    best_entry = max(beam, key=lambda x: x.score)
    return best_entry.chain


def _sa_move(chain, rng):
    """Generate a random move: either a kink or a pivot."""
    n = len(chain)
    if n < 4:
        return None
    
    # Choose move type
    if rng.random() < 0.5:
        # Kink move
        i = rng.randint(0, n - 4)
        j = rng.randint(i + 2, n - 1)
        segment = chain[i + 1:j + 1]
        new_segment = segment[::-1]
        
        # Check self-avoidance
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
        
        # Check connectivity
        if abs(new_segment[0][0] - chain[i][0]) + abs(new_segment[0][1] - chain[i][1]) + abs(new_segment[0][2] - chain[i][2]) != 1:
            return None
        if j + 1 < n:
            if abs(new_segment[-1][0] - chain[j + 1][0]) + abs(new_segment[-1][1] - chain[j + 1][1]) + abs(new_segment[-1][2] - chain[j + 1][2]) != 1:
                return None
                
        new_chain = chain[:i + 1] + new_segment + chain[j + 1:]
        return new_chain
    else:
        # Pivot move: rotate segment around chain[i]
        i = rng.randint(0, n - 4)
        j = rng.randint(i + 2, n - 1)
        segment = chain[i + 1:j + 1]
        
        # Determine the axis of the bond between chain[i] and chain[i+1]
        p0 = chain[i]
        p1 = chain[i + 1]
        
        # Find the axis of rotation (the dimension where p0 and p1 differ)
        axis = 0
        for k in range(3):
            if p0[k] != p1[k]:
                axis = k
                break
        
        # Determine the rotation angle (90, 180, 270)
        angle = rng.choice([1, 2, 3]) # 1:90, 2:180, 3:270
        
        # Define rotation around the axis through p0, perpendicular to the bond axis
        # This is a rotation in the plane perpendicular to the bond axis.
        # Let's identify the two other axes
        axes = [0, 1, 2]
        axes.remove(axis)
        a1, a2 = axes[0], axes[1]
        
        # Rotation matrix for the 2D plane (a1, a2)
        # We rotate the vector (p - p0) by the angle.
        # Since it's a cubic lattice, we can just swap/flip coordinates.
        # 90 deg: (x, y) -> (-y, x) or (y, -x)
        # 180 deg: (x, y) -> (-x, -y)
        # 270 deg: (x, y) -> (y, -x) or (-y, x)
        
        # Let's implement a generic 90-degree rotation in the plane spanned by a1 and a2.
        # The bond vector is along `axis`.
        # We need to rotate the rest of the segment around the axis passing through p0 and parallel to `axis`.
        
        new_segment = []
        for p in segment:
            new_p = list(p)
            # Translate to origin relative to p0
            dx = p[a1] - p0[a1]
            dy = p[a2] - p0[a2]
            
            # Rotate (dx, dy) by angle * 90 degrees
            if angle == 1: # 90 deg
                new_dx = -dy
                new_dy = dx
            elif angle == 2: # 180 deg
                new_dx = -dx
                new_dy = -dy
            else: # 270 deg
                new_dx = dy
                new_dy = -dx
            
            new_p[a1] = p0[a1] + new_dx
            new_p[a2] = p0[a2] + new_dy
            # axis coordinate remains unchanged
            new_segment.append(tuple(new_p))
            
        # Check self-avoidance
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
        
        # Check connectivity at start (chain[i] to new_segment[0])
        # The distance must be 1
        if abs(new_segment[0][0] - chain[i][0]) + abs(new_segment[0][1] - chain[i][1]) + abs(new_segment[0][2] - chain[i][2]) != 1:
            return None
            
        # Check connectivity at end (new_segment[-1] to chain[j+1])
        if j + 1 < n:
            if abs(new_segment[-1][0] - chain[j + 1][0]) + abs(new_segment[-1][1] - chain[j + 1][1]) + abs(new_segment[-1][2] - chain[j + 1][2]) != 1:
                return None
                
        new_chain = chain[:i + 1] + new_segment + chain[j + 1:]
        return new_chain


def _simulated_annealing(sequence, chain, rng, time_budget):
    """Simulated annealing with kink and pivot moves."""
    n = len(chain)
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    start_t = time.time()
    t = 1.0
    cooling = 0.995
    
    while time.time() - start_t < time_budget:
        new_chain = _sa_move(chain, rng)
        if new_chain is None:
            continue
            
        new_energy = _energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or rng.random() < math.exp(-delta_e / t):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
                
        t *= cooling
        if t < 0.01:
            t = 0.01

    return best_chain, best_energy


def fold(sequence, dimension):
    """Fold a sequence using multi-start beam search + simulated annealing."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    n = len(sequence)
    rng = random.Random(12345 + n)

    # Allocate time based on sequence length
    # SA is more expensive per step but finds better minima.
    # Reduce restarts, increase SA time.
    if n <= 48:
        num_restarts = 5
        beam_width = 20
        sa_budget = 3.0
    elif n <= 64:
        num_restarts = 4
        beam_width = 15
        sa_budget = 4.0
    elif n <= 96:
        num_restarts = 3
        beam_width = 12
        sa_budget = 5.0
    else:
        num_restarts = 2
        beam_width = 10
        sa_budget = 6.0

    best_chain = None
    best_energy = 0

    for restart in range(num_restarts):
        if _time_remaining() < 15.0:
            break

        # Beam search growth
        bw = beam_width
        chain = _beam_growth(sequence, dimension, bw, _time_remaining() / 4.0)
        if chain is None:
            continue

        energy = _energy(sequence, chain)

        if energy < best_energy:
            best_energy = energy
            best_chain = list(chain)

        # Simulated Annealing to improve
        sa_time = min(sa_budget, _time_remaining() / 2.0)
        if sa_time > 0.5:
            improved_chain, improved_energy = _simulated_annealing(sequence, chain, rng, sa_time)
            if improved_energy < best_energy:
                best_energy = improved_energy
                best_chain = improved_chain

    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
