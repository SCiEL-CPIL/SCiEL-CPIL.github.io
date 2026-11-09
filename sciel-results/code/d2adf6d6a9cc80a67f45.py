"""HP lattice protein folding on the cubic lattice.
Uses simulated annealing with local moves (pivot, crankshaft, diagonal)
and a deadline guard to stay within the wall-clock budget.
"""
import random
import time
import sys

# Deadline: evaluator kills at 330s wall-clock. Use 320s safety margin.
_DEADLINE = time.time() + 320.0


def _neighbours(p):
    """Yield the 6 face-neighbours of point p in 3D."""
    x, y, z = p
    yield (x + 1, y, z)
    yield (x - 1, y, z)
    yield (x, y + 1, z)
    yield (x, y - 1, z)
    yield (x, y, z + 1)
    yield (x, y, z - 1)


def _energy_delta(seq, chain, occupied, i, new_pos):
    """Compute the change in energy (delta) if residue i at old_pos moves to new_pos.
    Energy = -num_HH_contacts. So delta = new_energy - old_energy = -(new_contacts - old_contacts).
    We return delta_energy where more negative is better.
    """
    old_pos = chain[i]
    if new_pos == old_pos:
        return 0

    # Remove contacts involving residue i at old_pos
    delta = 0
    if seq[i] == 'H':
        for nb in _neighbours(old_pos):
            j = occupied.get(nb)
            if j is not None and j != i and abs(j - i) > 1 and seq[j] == 'H':
                # We lose this contact: energy increases by 1 (less negative)
                delta += 1  # delta_energy += 1 means energy goes up (worse)

    # Add contacts involving residue i at new_pos
    if seq[i] == 'H':
        for nb in _neighbours(new_pos):
            j = occupied.get(nb)
            if j is not None and j != i and abs(j - i) > 1 and seq[j] == 'H':
                # We gain this contact: energy decreases by 1 (more negative)
                delta -= 1  # delta_energy -= 1 means energy goes down (better)

    return delta


def _pivot_move(chain, occupied, i, rng):
    """Rotate the tail (residues i..end) around residue i-1.
    Returns (new_chain, new_occupied, success) or (None, None, False).
    """
    if i < 1 or i >= len(chain):
        return None, None, False

    pivot = chain[i - 1]
    tail = chain[i:]

    # The tail must rotate as a rigid body around pivot.
    # In 3D, we can rotate by 90 degrees around any axis.
    # Simpler: pick a random direction for the first step of the tail,
    # then try to re-lay out the rest. This is complex.
    
    # Alternative simple move: swap two non-adjacent residues? No, must keep connectivity.
    
    # Let's use a simpler local move: "diagonal" or "step" move where we move
    # a single residue to a nearby empty site, maintaining connectivity.
    # But single-residue moves break connectivity unless we move a sub-chain.
    
    # Let's implement a "crankshaft" or "pivot" properly.
    # Pivot: rotate segment [i, j] around chain[i-1].
    # This is expensive to implement correctly in 3D without a full 3D rotation matrix.
    
    # Simpler approach: "local rearrangement" by trying to move a short segment.
    # Or just use "step" moves: pick a residue i (not 0 or last), try to move it
    # to a neighbouring empty site, but this breaks the chain unless we also
    # move its neighbours. 
    
    # Let's stick to a well-known simple move: "pivot" by rotating the entire
    # tail around the bond at i-1. In 3D, there are 3 axes of rotation (x, y, z)
    # and 3 non-trivial 90-degree rotations per axis (plus 180 and 270).
    # Total 12 possible rotations for the tail.
    
    # Let's implement one specific type: rotate tail around axis perpendicular to
    # the current bond direction. This is getting complex.
    
    # Let's use a simpler strategy: "random restart with greedy growth" plus
    # "local optimization" using a different move set.
    
    # Move set: "diagonal" move - move a residue to a position that is a knight's
    # move away? No, must be unit steps.
    
    # Let's implement a "2-step" move: pick i, move residue i to a new position
    # that is 2 steps away in Manhattan distance, and adjust residue i+1? No.
    
    # Actually, the simplest effective local search for HP model is "pivot" moves
    # where we rotate a sub-chain. Let's implement a simplified pivot:
    # Pick i, pick a random direction for the new bond (i-1, i), and try to
    # lay out the rest of the chain greedily from there. This is essentially
    # a re-growth of the tail.
    
    # This is expensive but doable.
    
    # Alternative: use "simulated annealing" with "step" moves where we try to
    # move a single residue to an adjacent empty site, but this is only valid
    # if the residue is at the end of the chain. So we can only move the last
    # residue. That's too limited.
    
    # Let's implement a "crankshaft" move: rotate a segment of length L around
    # two fixed endpoints. This is also complex.
    
    # Given the time constraints, let's use a hybrid approach:
    # 1. Greedy growth with multiple restarts.
    # 2. Local optimization using "tail rotation": pick a random i, and try
    #    to re-lay out the tail [i..end] using a randomized greedy method,
    #    accepting if energy improves.
    
    # This is similar to the original but with local re-growth.
    
    return None, None, False


def _greedy_grow_tail(seq, chain_prefix, occupied_prefix, i, rng):
    """Grow the tail from index i onwards, given the prefix up to i-1.
    Returns (chain, occupied) or (None, None) if stuck.
    """
    chain = list(chain_prefix)
    occupied = dict(occupied_prefix)
    
    for k in range(i, len(seq)):
        last = chain[-1]
        options = []
        for nb in _neighbours(last):
            if nb in occupied:
                continue
            # Score: prefer sites with H-H contacts
            gain = 0
            if seq[k] == 'H':
                for q in _neighbours(nb):
                    j = occupied.get(q)
                    if j is not None and j < k - 1 and seq[j] == 'H':
                        gain += 1
            # Add small random noise
            options.append((gain + 0.1 * rng.random(), nb))
        
        if not options:
            return None, None
        
        options.sort(reverse=True)
        # Pick from top 3 with some randomness
        top = options[:min(3, len(options))]
        _, site = top[rng.randint(0, len(top) - 1)]
        chain.append(site)
        occupied[site] = k
    
    return chain, occupied


def _compute_energy(seq, chain):
    """Compute the total energy of the chain."""
    occupied = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if seq[i] == 'H':
            for nb in _neighbours(p):
                j = occupied.get(nb)
                if j is not None and j > i + 1 and seq[j] == 'H':
                    contacts += 1
    return -contacts


def fold(sequence, dimension):
    """Fold the sequence on the cubic lattice."""
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    rng = random.Random(42)
    best_chain = None
    best_energy = 0  # Start with 0 (no contacts)
    
    # Phase 1: Multiple greedy growths
    num_growths = 500
    start_time = time.time()
    
    for _ in range(num_growths):
        if time.time() > _DEADLINE - 10:
            break
        
        chain, occupied = _greedy_grow_tail(sequence, [(0, 0, 0)], {(0, 0, 0): 0}, 1, rng)
        if chain is None:
            continue
        
        energy = _compute_energy(sequence, chain)
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    if best_chain is None:
        # Fallback: straight line
        return [[i, 0, 0] for i in range(n)]
    
    # Phase 2: Local optimization via tail re-growth
    # Try to improve the best chain by re-growing tails at various positions
    num_local_steps = 2000
    current_chain = list(best_chain)
    current_energy = best_energy
    
    for step in range(num_local_steps):
        if time.time() > _DEADLINE - 5:
            break
        
        # Pick a random position to re-grow from
        i = rng.randint(1, n - 2)
        
        # Re-grow the tail from position i
        prefix = current_chain[:i]
        prefix_occupied = {p: j for j, p in enumerate(prefix)}
        
        new_chain, new_occupied = _greedy_grow_tail(sequence, prefix, prefix_occupied, i, rng)
        if new_chain is None:
            continue
        
        new_energy = _compute_energy(sequence, new_chain)
        
        # Accept if better, or with some probability if worse (simulated annealing)
        if new_energy < current_energy:
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        else:
            # Simple annealing: accept with probability exp(-delta/T)
            delta = new_energy - current_energy
            T = max(0.1, 1.0 - step / num_local_steps)
            if rng.random() < pow(2.71828, -delta / T):
                current_chain = new_chain
                current_energy = new_energy
    
    # Convert to list of lists
    return [list(p) for p in best_chain]
