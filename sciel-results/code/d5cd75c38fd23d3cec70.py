"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math
import time
from collections import defaultdict


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _build_scaffold(n, dimension, rng):
    """Build a simple self-avoiding walk scaffold (a snake-like path)."""
    # Start at origin, go in +x direction
    chain = [(0, 0, 0)]
    occupied = {chain[0]: 0}
    
    for i in range(1, n):
        # Try to continue in the same direction (x-axis)
        next_pos = (chain[-1][0] + 1, 0, 0)
        if next_pos not in occupied:
            chain.append(next_pos)
            occupied[next_pos] = i
        else:
            # If blocked, try other directions
            found = False
            for axis in range(dimension):
                for step in (-1, 1):
                    q = list(chain[-1])
                    q[axis] += step
                    q = tuple(q)
                    if q not in occupied:
                        chain.append(q)
                        occupied[q] = i
                        found = True
                        break
                if found:
                    break
            if not found:
                # Should not happen for reasonable n, but fallback
                # Just extend in x direction ignoring self-avoidance (shouldn't occur)
                chain.append((chain[-1][0] + 1, 0, 0))
                occupied[chain[-1]] = i
    
    return chain


def _compute_energy(sequence, chain, dimension):
    """Compute energy: -number of H-H contacts on neighboring lattice sites not adjacent in chain."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _try_move(chain, move_type, rng, dimension):
    """
    Attempt a move on the chain. Returns (new_chain, new_occupied) if successful, else None.
    move_type: 'pivot' or 'crankshaft' or 'slide'
    """
    n = len(chain)
    if n < 3:
        return None
    
    occupied = {p: i for i, p in enumerate(chain)}
    
    if move_type == 'pivot':
        # Pick a pivot point and rotate the tail
        if n <= 2:
            return None
        pivot_idx = rng.randint(0, n - 2)
        pivot = chain[pivot_idx]
        # The bond from pivot to next
        next_pos = chain[pivot_idx + 1]
        # Try all 5 possible directions for the new bond (excluding the one coming from pivot)
        # Actually, we rotate the tail around the pivot. The bond (pivot, next) can point in 5 other directions.
        # But we need to ensure self-avoidance.
        
        # Get current direction
        dx = next_pos[0] - pivot[0]
        dy = next_pos[1] - pivot[1]
        dz = next_pos[2] - pivot[2]
        
        # Generate all 6 neighbors of pivot
        candidates = []
        for axis in range(dimension):
            for step in (-1, 1):
                q = list(pivot)
                q[axis] += step
                q = tuple(q)
                candidates.append(q)
        
        # Exclude the current direction
        candidates = [c for c in candidates if c != next_pos]
        
        if not candidates:
            return None
        
        new_next = rng.choice(candidates)
        
        # Now try to rotate the rest of the chain (from pivot_idx+1 onwards)
        # This is complex; for simplicity, let's just try to place the next residue and then continue
        # Actually, pivot moves are hard to implement correctly for self-avoidance without backtracking.
        # Let's use a simpler move: crankshaft or just local swaps.
        
        # For now, let's skip pivot and use other moves
        return None
    
    elif move_type == 'crankshaft':
        # Pick two non-adjacent residues and rotate the segment between them
        # This is also complex. Let's use a simpler approach: try to move a single residue to a neighboring site.
        # But that would break the chain connectivity.
        
        # Let's implement a "local move": pick an interior residue i (1 <= i <= n-2) and try to move it to a neighboring site
        # such that the chain remains self-avoiding and connected.
        # This is a "kink" move.
        
        i = rng.randint(1, n - 2)
        p = chain[i]
        prev = chain[i - 1]
        nxt = chain[i + 1]
        
        # Find all neighbors of p that are not prev and not nxt
        candidates = []
        for q in _neighbours(p):
            if q != prev and q != nxt:
                # Check if q is occupied by any other residue
                if q not in occupied:
                    candidates.append(q)
        
        if not candidates:
            return None
        
        new_p = rng.choice(candidates)
        
        # Check if the new position is valid:
        # 1. new_p is not occupied (already checked)
        # 2. The chain remains connected: prev-new_p and new_p-nxt must be unit steps (they are, by construction)
        # 3. No self-avoidance violation: new_p is not in the chain (except at position i, which we're moving)
        
        # Since new_p is not in occupied, and we're only moving residue i, this should be fine.
        # But we need to make sure that prev and nxt are still unit steps from new_p.
        # By construction, new_p is a neighbor of p, but is it a neighbor of prev and nxt?
        # No! new_p is a neighbor of p, but prev and nxt are also neighbors of p.
        # For the chain to remain valid, new_p must be a unit step from prev AND from nxt.
        # This is not guaranteed.
        
        # Let's check:
        def is_unit_step(a, b):
            return sum(abs(a[k] - b[k]) for k in range(dimension)) == 1
        
        if not is_unit_step(prev, new_p) or not is_unit_step(new_p, nxt):
            return None
        
        # Also need to ensure that moving to new_p doesn't cause self-avoidance issues with other residues.
        # Since new_p was not in occupied, and we're only changing position i, this should be fine.
        # But we need to remove p from occupied and add new_p.
        
        new_chain = chain[:]
        new_chain[i] = new_p
        new_occupied = {p2: i2 for i2, p2 in enumerate(new_chain)}
        
        # Verify self-avoidance
        if len(new_occupied) != len(new_chain):
            return None
        
        return new_chain, new_occupied
    
    elif move_type == 'slide':
        # Try to shift a segment of the chain
        # Pick a start and end index, and try to move the segment by one step in some direction
        if n < 4:
            return None
        
        length = rng.randint(1, min(3, n - 2))  # Short segments
        start = rng.randint(1, n - 1 - length)
        end = start + length
        
        # The segment is chain[start:end+1]
        # We want to move it by one step in some direction
        # The direction must be such that the new positions are not occupied and the connections remain valid
        
        # This is complex. Let's skip for now and use only crankshaft moves.
        return None
    
    return None


def _metropolis_move(sequence, chain, dimension, temperature, rng):
    """
    Perform a single Metropolis move. Returns the new chain.
    """
    n = len(chain)
    
    # Choose a move type
    move_type = rng.choice(['crankshaft', 'crankshaft', 'crankshaft', 'slide', 'pivot'])
    
    result = _try_move(chain, move_type, rng, dimension)
    if result is None:
        return chain
    
    new_chain, _ = result
    
    # Compute energy difference
    old_energy = _compute_energy(sequence, chain, dimension)
    new_energy = _compute_energy(sequence, new_chain, dimension)
    delta_e = new_energy - old_energy
    
    # Metropolis acceptance
    if delta_e <= 0:
        return new_chain
    else:
        if temperature > 0:
            accept_prob = math.exp(-delta_e / temperature)
            if rng.random() < accept_prob:
                return new_chain
    
    return chain


def fold(sequence, dimension):
    """
    Parallel tempering Monte Carlo to find the best conformation.
    """
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    rng = random.Random(12345 + len(sequence))
    
    # Build initial scaffold
    chain = _build_scaffold(n, dimension, rng)
    
    # Compute initial energy
    best_chain = chain[:]
    best_energy = _compute_energy(sequence, chain, dimension)
    
    # Parallel tempering: maintain multiple chains at different temperatures
    num_replicas = 8
    temperatures = [0.1, 0.3, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    
    # Initialize replicas
    replicas = [chain[:] for _ in range(num_replicas)]
    replica_energies = [best_energy] * num_replicas
    
    # Run Monte Carlo
    start_time = time.time()
    time_budget = 200  # seconds, leaving buffer for the 240s total
    
    while time.time() - start_time < time_budget:
        # Perform moves in each replica
        for r in range(num_replicas):
            new_chain = _metropolis_move(sequence, replicas[r], dimension, temperatures[r], rng)
            if new_chain != replicas[r]:
                replicas[r] = new_chain
                replica_energies[r] = _compute_energy(sequence, new_chain, dimension)
        
        # Attempt swap between adjacent replicas (higher and lower temperature)
        if rng.random() < 0.5:
            r1 = rng.randint(0, num_replicas - 2)
            r2 = r1 + 1
            # Metropolis swap criterion
            e1 = replica_energies[r1]
            e2 = replica_energies[r2]
            t1 = temperatures[r1]
            t2 = temperatures[r2]
            
            # Delta = (e1/t2 - e2/t1) - (e2/t2 - e1/t1) = (e1 - e2) * (1/t2 - 1/t1)
            delta = (e1 - e2) * (1.0 / t2 - 1.0 / t1)
            
            if delta <= 0 or rng.random() < math.exp(-delta):
                # Swap
                replicas[r1], replicas[r2] = replicas[r2], replicas[r1]
                replica_energies[r1], replica_energies[r2] = replica_energies[r2], replica_energies[r1]
        
        # Track best
        for r in range(num_replicas):
            if replica_energies[r] < best_energy:
                best_energy = replica_energies[r]
                best_chain = replicas[r][:]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
