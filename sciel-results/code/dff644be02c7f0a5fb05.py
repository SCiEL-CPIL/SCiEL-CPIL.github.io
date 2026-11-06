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
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _beam_growth(sequence, dimension, beam_width=50, rng=None):
    """Beam search chain growth: keep top beam_width partial chains by score."""
    if rng is None:
        rng = random.Random()
    
    start = (0,) * dimension
    # Each entry: (score, chain, occupied_set)
    # Score = energy so far (negative is better) + small bonus for free space
    beam = [(0.0, [start], {start})]
    
    for i in range(1, len(sequence)):
        new_beam = []
        for score, chain, occupied in beam:
            last = chain[-1]
            for site in _neighbours(last):
                if site in occupied:
                    continue
                # Calculate gain in energy
                gain = 0
                if sequence[i] == "H":
                    for q in _neighbours(site):
                        if q in occupied:
                            j = chain.index(q) if q in chain else -1
                            # Need to check if the occupied site corresponds to an H residue
                            # that is not adjacent in the chain
                            if j >= 0 and j < i - 1 and sequence[j] == "H":
                                gain += 1
                # Also count contacts with existing H's
                # Actually let's compute properly
                new_occupied = occupied | {site}
                # Gain = new H-H contacts created
                if sequence[i] == "H":
                    for q in _neighbours(site):
                        if q in occupied and q not in new_occupied:  # q was already occupied
                            # Find which residue is at q
                            # We need a mapping from position to index
                            pass
                
                # Simpler: just use a heuristic score
                # Count H-H contacts involving the new residue
                new_contacts = 0
                if sequence[i] == "H":
                    for q in _neighbours(site):
                        if q in occupied:
                            # Check if the residue at q is H and not adjacent
                            # We need to know which index is at q
                            # Let's store index mapping
                            pass
                
                # For efficiency, let's just use a simple heuristic
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                # Heuristic score: prefer sites with more H neighbors (if current is H)
                h_neighbors = 0
                if sequence[i] == "H":
                    for q in _neighbours(site):
                        if q in occupied:
                            h_neighbors += 1
                
                new_score = score + h_neighbors * 2.0 + free * 0.1 + rng.random() * 0.5
                new_chain = chain + [site]
                new_occupied = occupied | {site}
                new_beam.append((new_score, new_chain, new_occupied))
        
        if not new_beam:
            return None
        
        # Keep top beam_width
        new_beam.sort(key=lambda x: x[0], reverse=True)
        beam = new_beam[:beam_width]
    
    # Return the best complete chain
    best = max(beam, key=lambda x: x[0])
    return best[1]


def _kink_move(chain, dimension, rng):
    """Perform a kink move: move residue i to a neighboring unoccupied site."""
    n = len(chain)
    if n < 3:
        return None
    
    # Choose a random residue (not endpoints)
    i = rng.randint(1, n - 2)
    p = chain[i]
    
    for site in _neighbours(p):
        # Check if site is unoccupied (except by the residue itself and its neighbors in chain)
        if site in chain:
            # Allow if it's a valid position (not occupied by other residues)
            # Actually, site must not be occupied by any other residue
            # The residue at i is being moved, so its current position is freed
            # But we need to check if site is occupied by another residue
            if site == p:
                continue
            # Check if site is in the chain at a different index
            if chain.index(site) != i:
                continue
        
        # Check if moving to site is valid (no self-intersection)
        new_chain = list(chain)
        new_chain[i] = site
        
        # Verify self-avoidance
        if len(set(new_chain)) != n:
            continue
        
        # Verify unit steps
        valid = True
        for j in range(n - 1):
            a, b = new_chain[j], new_chain[j + 1]
            dist = sum(abs(a[k] - b[k]) for k in range(dimension))
            if dist != 1:
                valid = False
                break
        
        if valid:
            return new_chain
    
    return None


def _simulated_annealing(sequence, chain, dimension, time_budget, rng):
    """Simulated annealing with kink moves."""
    n = len(sequence)
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    t = 50.0  # initial temperature
    cooling = 0.999
    
    while time_budget > 0:
        new_chain = _kink_move(chain, dimension, rng)
        if new_chain is None:
            t *= cooling
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < math.exp(-delta / t):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
        
        t *= cooling
        if t < 0.01:
            break
        
        # Check time periodically
        if int(time.time() * 10) % 10 == 0:
            if _get_elapsed() > _TOTAL_DEADLINE - time_budget - 5:
                break
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using beam search growth + simulated annealing."""
    if _start_time is None:
        global _start_time
        _start_time = time.time()
    
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    
    # Phase 1: Beam search growth to get initial candidates
    initial_chains = []
    for i in range(10):
        chain = _beam_growth(sequence, dimension, beam_width=50, rng=rng)
        if chain is not None:
            initial_chains.append(chain)
    
    if not initial_chains:
        # Fallback: straight line
        return [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    # Phase 2: Simulated annealing on best initial chain
    # Pick the best initial chain by energy
    best_initial = min(initial_chains, key=lambda c: _energy(sequence, c))
    
    # Calculate time budget for annealing
    elapsed = _get_elapsed()
    time_budget = _TOTAL_DEADLINE - elapsed - 15.0  # 15s safety margin
    if time_budget < 1.0:
        return [list(p) for p in best_initial]
    
    # Run simulated annealing
    best_chain, best_energy = _simulated_annealing(
        sequence, best_initial, dimension, time_budget, rng
    )
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
