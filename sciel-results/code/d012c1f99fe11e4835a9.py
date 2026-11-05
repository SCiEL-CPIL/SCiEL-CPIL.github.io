"""HP lattice protein folding on the cubic lattice."""
# EVOLVE-BLOCK-START
import time
import random

# Precompute neighbor offsets for 3D cubic lattice
_NEIGHBOR_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # safety margin before 330s kill


def _time_left():
    return _DEADLINE - time.time()


def _energy_correct(sequence, chain):
    """Compute energy: -number of H-H contacts on neighboring non-consecutive sites."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] != 'H':
            continue
        for dx, dy, dz in _NEIGHBOR_OFFSETS:
            q = (p[0] + dx, p[1] + dy, p[2] + dz)
            j = where.get(q)
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts


def _beam_search(sequence, dimension, beam_width=200):
    """
    Beam search for HP protein folding.
    Maintains a beam of the best partial conformations at each step.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]

    # Beam: list of (energy_so_far, chain_tuple, occupied_dict)
    # We use tuples for chains to make them hashable for deduplication
    beam = [(0, ((0, 0, 0),), {(0, 0, 0): 0})]

    for i in range(1, n):
        if _time_left() < 10.0:
            break

        new_beam = []
        seen = set()

        for energy_so_far, chain_tuple, occupied in beam:
            last = chain_tuple[-1]
            seq_i = sequence[i]

            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                site = (last[0] + dx, last[1] + dy, last[2] + dz)
                if site in occupied:
                    continue

                # Compute new contacts gained
                gain = 0
                if seq_i == 'H':
                    for dx2, dy2, dz2 in _NEIGHBOR_OFFSETS:
                        q = (site[0] + dx2, site[1] + dy2, site[2] + dz2)
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == 'H':
                            gain += 1

                # New energy is more negative when gain is higher
                new_energy = energy_so_far - gain

                # Deduplicate: use the last position and a hash of the chain shape
                # For efficiency, we'll use a tuple of the last few positions as a proxy
                key = (site, chain_tuple[-3:] if len(chain_tuple) >= 3 else chain_tuple)
                if key in seen:
                    continue
                seen.add(key)

                new_chain = chain_tuple + (site,)
                new_occupied = dict(occupied)
                new_occupied[site] = i

                new_beam.append((new_energy, new_chain, new_occupied))

        if not new_beam:
            break

        # Sort by energy (most negative first) and keep top beam_width
        new_beam.sort(key=lambda x: x[0])
        beam = new_beam[:beam_width]

    if not beam:
        return None

    # Return the best complete chain
    best_energy, best_chain_tuple, _ = beam[0]
    return list(best_chain_tuple)


def _local_search(sequence, dimension, chain, rng, max_iterations=500):
    """
    Local search using simulated annealing with segment reversals.
    """
    n = len(chain)
    if n < 5:
        return chain

    current_chain = list(chain)
    current_energy = _energy_correct(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy

    # Simulated Annealing parameters
    T = 2.0
    T_min = 0.01
    alpha = 0.9995

    for iteration in range(max_iterations):
        if _time_left() < 5.0:
            break

        # Generate a move: Segment Reversal
        # Randomly choose segment length and start position
        max_seg_len = min(n - 2, 15)
        if max_seg_len < 2:
            continue

        seg_len = rng.randint(2, max_seg_len)
        start_idx = rng.randint(1, n - seg_len - 1)
        end_idx = start_idx + seg_len - 1

        # Perform reversal
        new_chain = current_chain[:]
        new_chain[start_idx:end_idx + 1] = new_chain[start_idx:end_idx + 1][::-1]

        # Check validity (Self-Avoidance and Connectivity)
        # Connectivity is preserved by reversal.
        # Check Self-Avoidance: 
        # We only need to check if the new positions of the segment collide with the rest of the chain.
        # The rest of the chain is fixed.
        # The segment itself is just reversed, so internal collisions are impossible if it was valid before?
        # No, reversing a segment changes the positions of the residues in the segment.
        # We need to check if any residue in the new segment position is occupied by a residue outside the segment.
        
        # Create a set of positions occupied by residues NOT in the segment
        outside_positions = set()
        for i in range(n):
            if i < start_idx or i > end_idx:
                outside_positions.add(tuple(current_chain[i]))
        
        valid = True
        for i in range(start_idx, end_idx + 1):
            if tuple(new_chain[i]) in outside_positions:
                valid = False
                break
        
        # Also check connectivity at the boundaries
        if valid:
            # Check link (start_idx-1, start_idx)
            if start_idx > 0:
                p1 = new_chain[start_idx-1]
                p2 = new_chain[start_idx]
                if sum(abs(a-b) for a,b in zip(p1,p2)) != 1:
                    valid = False
            # Check link (end_idx, end_idx+1)
            if valid and end_idx < n-1:
                p1 = new_chain[end_idx]
                p2 = new_chain[end_idx+1]
                if sum(abs(a-b) for a,b in zip(p1,p2)) != 1:
                    valid = False

        if not valid:
            continue

        new_energy = _energy_correct(sequence, new_chain)
        
        # Metropolis Acceptance
        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < pow(2.71828, -delta_e / max(T, 1e-9)):
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_chain = list(current_chain)
                best_energy = current_energy
        
        T *= alpha
        if T < T_min:
            T = T_min

    return best_chain


def fold(sequence, dimension):
    """
    Fold a protein sequence on the cubic lattice.
    Returns a list of len(sequence) integer lattice points (lists of length dimension).
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]

    best_chain = None
    best_energy = 0

    # Time budget for this sequence
    seq_start = time.time()
    time_budget = min(120.0, max(15.0, n * 0.8))

    # Phase 1: Beam search with different widths
    # Use multiple beam searches with varying widths for diversity
    beam_configs = [
        (100, 12345),
        (200, 54321),
        (300, 98765),
        (150, 11111),
        (250, 22222),
    ]

    for beam_width, seed in beam_configs:
        if time.time() - seq_start > time_budget * 0.6:
            break
        if _time_left() < 15.0:
            break

        chain = _beam_search(sequence, dimension, beam_width=beam_width)
        if chain is None or len(chain) != n:
            continue

        energy = _energy_correct(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy

    # Phase 2: Local search on the best chain
    # Run multiple local searches with different seeds and time budgets
    if best_chain is not None and _time_left() > 15.0:
        # Determine time left for local search
        ls_time_budget = _time_left() - 10.0 # Keep 10s buffer
        if ls_time_budget > 0:
            # Run local search with increasing iterations based on time
            # Estimate iterations: _local_search is roughly O(N^2) per step due to energy calc?
            # Energy calc is O(N). So step is O(N).
            # Let's just set a high max_iterations and let it break on time.
            rng = random.Random(42)
            improved = _local_search(sequence, dimension, best_chain, rng, max_iterations=5000)
            improved_energy = _energy_correct(sequence, improved)
            if improved_energy < best_energy:
                best_chain = improved
                best_energy = improved_energy
            
            # Second pass with different seed if time remains
            if _time_left() > 10.0:
                rng2 = random.Random(123)
                improved2 = _local_search(sequence, dimension, best_chain, rng2, max_iterations=5000)
                improved2_energy = _energy_correct(sequence, improved2)
                if improved2_energy < best_energy:
                    best_chain = improved2
                    best_energy = improved2_energy

    # Fallback: straight line if no valid chain found
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
        best_energy = _energy_correct(sequence, best_chain)

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
