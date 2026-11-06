"""HP lattice protein folding on the cubic lattice."""
# EVOLVE-BLOCK-START
import time
import random
import heapq
from collections import deque

_DEADLINE = time.time() + 320  # safety margin before 330s kill


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


def _greedy_grow(sequence, dimension, rng, time_limit=None):
    """One randomized greedy chain growth."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        if time_limit is not None and time.time() > time_limit:
            return None
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


def _local_refine(sequence, chain, dimension, max_iters=5000, rng=None):
    """Local search with kink moves and simple swaps to escape local optima."""
    if len(chain) < 3:
        return chain
    
    if rng is None:
        rng = random.Random()
        
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    
    # Build position lookup
    pos_to_idx = {p: i for i, p in enumerate(best_chain)}
    
    # Kink move: pick two non-adjacent residues i, j (|i-j| >= 3) and try to
    # move residue j to an adjacent site of residue i, if that site is free
    # and doesn't break self-avoidance.
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 10:
            break
        
        improved = False
        n = len(best_chain)
        
        # Try kink moves: pick random i, j with |i-j| >= 3
        for _ in range(50):
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if abs(i - j) < 3:
                continue
            
            old_pos = best_chain[j]
            new_pos = best_chain[i]
            
            # Try moving j to a neighbor of i
            for neighbor in _neighbours(new_pos):
                if neighbor == old_pos:
                    continue
                if neighbor in pos_to_idx:
                    # Check if the occupant is j itself or adjacent in chain
                    occ = pos_to_idx[neighbor]
                    if occ == j:
                        continue
                    if abs(occ - j) <= 1:
                        continue
                    continue
                
                # Also check that moving j doesn't cause collision with j's neighbors
                # Actually we need to check all neighbors of j's old position and new position
                # Let's just do a full self-avoidance check for simplicity in local refine
                
                # Tentative move
                best_chain[j] = neighbor
                new_pos_to_idx = {p: idx for idx, p in enumerate(best_chain)}
                
                # Check self-avoidance
                if len(new_pos_to_idx) != n:
                    best_chain[j] = old_pos
                    continue
                
                # Check unit steps
                valid = True
                for k in range(n - 1):
                    p1, p2 = best_chain[k], best_chain[k + 1]
                    dist = sum(abs(a - b) for a, b in zip(p1, p2))
                    if dist != 1:
                        valid = False
                        break
                
                if not valid:
                    best_chain[j] = old_pos
                    continue
                
                e = _energy(sequence, best_chain)
                if e < best_energy:
                    best_energy = e
                    pos_to_idx = new_pos_to_idx
                    improved = True
                    break
                else:
                    best_chain[j] = old_pos
            if improved:
                break
        
        if not improved:
            break
    
    return best_chain


def fold(sequence, dimension):
    """Best conformation using greedy growth + local refinement."""
    rng = random.Random(12345 + len(sequence))
    best = None
    best_energy = 0  # energy is negative, so 0 means no contacts
    
    seq_len = len(sequence)
    
    # Determine number of restarts based on sequence length
    if seq_len <= 48:
        n_restarts = 800
    elif seq_len <= 64:
        n_restarts = 400
    elif seq_len <= 96:
        n_restarts = 200
    else:
        n_restarts = 100
    
    # Time budget per sequence
    seq_deadline = _DEADLINE - 5
    
    for r in range(n_restarts):
        if time.time() > seq_deadline:
            break
        
        chain = _greedy_grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        # Local refinement
        refined = _local_refine(sequence, chain, dimension, max_iters=min(2000, 50 * seq_len), rng=rng)
        value = _energy(sequence, refined)
        
        if value < best_energy:
            best = refined
            best_energy = value
    
    if best is None:
        # Fallback: straight line
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(seq_len)]
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
