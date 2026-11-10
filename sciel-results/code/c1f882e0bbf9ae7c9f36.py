"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import sys

# Explicit deadline: evaluator kills after 210s wall-clock; keep 30s safety margin.
_DEADLINE = time.time() + 170.0

def _neighbours(point):
    """Yield the 4 unit-step neighbours of a 2D point (tuple)."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _compute_energy(sequence, chain):
    """Compute the negative number of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts // 2

def _generate_initial_saw(sequence, rng):
    """
    Generate a random self-avoiding walk (SAW).
    Uses a simple rejection sampling method with a bias towards compaction 
    (though pure random SAW is often a good start for min-conflicts).
    To ensure it fits in reasonable space, we don't pre-allocate a grid, 
    but we might need to restart if it grows too large (unlikely for N<=100).
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    for _ in range(1000): # Try up to 1000 times to generate a valid SAW
        chain = [(0, 0)]
        occupied = {(0, 0)}
        
        for _ in range(n - 1):
            last = chain[-1]
            neighbors = list(_neighbours(last))
            rng.shuffle(neighbors)
            
            placed = False
            for nb in neighbors:
                if nb not in occupied:
                    chain.append(nb)
                    occupied.add(nb)
                    placed = True
                    break
            
            if not placed:
                break
        
        if len(chain) == n:
            return chain
    
    # Fallback: straight line (valid but high energy)
    return [(i, 0) for i in range(n)]

def _min_conflicts_fold(sequence, chain, rng, max_steps=50000):
    """
    Min-Conflicts style local search.
    We define 'conflicts' as H-H pairs that are NOT in contact.
    However, standard min-conflicts moves variables to minimize total conflicts.
    Here, we want to MAXIMIZE contacts.
    
    Strategy:
    1. Identify all H residues.
    2. For each H, check its neighbors. If a neighbor is H and not adjacent in chain, it's a contact (good).
    3. If a neighbor is empty, it's a potential contact.
    4. We want to move the chain to create more contacts.
    
    Instead of abstract min-conflicts, we use a guided local search:
    - Pick a random segment of the chain (length 2-5).
    - Try to rotate/move this segment to a new position that is valid (self-avoiding, connected) 
      and increases the energy (makes it more negative).
    
    To make it faster and more effective than simple random moves:
    - Focus moves on H residues or residues adjacent to H residues.
    """
    n = len(chain)
    if n < 4:
        return chain, _compute_energy(sequence, chain)
    
    current_chain = list(chain)
    current_energy = _compute_energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Precompute H positions for quick lookup
    h_indices = [i for i, c in enumerate(sequence) if c == 'H']
    if not h_indices:
        return current_chain, current_energy

    # For efficient neighbor checking, maintain an occupied set
    occupied = {p: i for i, p in enumerate(current_chain)}
    
    no_improve = 0
    
    for step in range(max_steps):
        if time.time() > _DEADLINE - 5:
            break
        
        # Select a pivot point. 
        # Bias towards H residues as they contribute to energy.
        if rng.random() < 0.7 and h_indices:
            i = rng.choice(h_indices)
        else:
            i = rng.randint(1, n - 2)
        
        # Try to move a small segment around i
        # Let's try moving the single residue i first (kink move)
        # Then try moving a segment of length 2 or 3
        
        moved = False
        # Try single residue move (i)
        if 1 <= i <= n - 2:
            p = current_chain[i]
            prev_p = current_chain[i-1]
            next_p = current_chain[i+1]
            
            # Candidate positions for p
            candidates = list(_neighbours(prev_p))
            candidates += list(_neighbours(next_p))
            # Remove current pos and invalid ones
            candidates = [c for c in candidates if c != p and c != prev_p and c != next_p]
            
            rng.shuffle(candidates)
            
            for new_p in candidates:
                if new_p in occupied and occupied[new_p] != i:
                    continue
                
                # Check connectivity: new_p must be adjacent to prev_p and next_p
                if abs(new_p[0]-prev_p[0]) + abs(new_p[1]-prev_p[1]) != 1:
                    continue
                if abs(new_p[0]-next_p[0]) + abs(new_p[1]-next_p[1]) != 1:
                    continue
                
                # Check self-avoidance: new_p is not occupied by any other residue
                # (already checked above)
                
                # Perform move
                del occupied[p]
                current_chain[i] = new_p
                occupied[new_p] = i
                
                new_energy = _compute_energy(sequence, current_chain)
                
                if new_energy < current_energy:
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(current_chain)
                    no_improve = 0
                    moved = True
                    break
                else:
                    # Revert
                    del occupied[new_p]
                    current_chain[i] = p
                    occupied[p] = i
            
            if moved:
                continue
        
        # If single move failed or wasn't tried, try a 2-residue segment move (pivot)
        # This allows for more complex rearrangements
        if 1 <= i <= n - 3:
            p1 = current_chain[i]
            p2 = current_chain[i+1]
            prev_p = current_chain[i-1]
            next_p = current_chain[i+2]
            
            # We want to move (p1, p2) to (new_p1, new_p2)
            # new_p1 must be adjacent to prev_p
            # new_p2 must be adjacent to new_p1
            # new_p2 must be adjacent to next_p
            
            candidates_p1 = list(_neighbours(prev_p))
            rng.shuffle(candidates_p1)
            
            for new_p1 in candidates_p1:
                if new_p1 == p1:
                    continue
                if new_p1 in occupied and occupied[new_p1] not in [i, i+1]:
                    continue
                
                candidates_p2 = list(_neighbours(new_p1))
                rng.shuffle(candidates_p2)
                
                for new_p2 in candidates_p2:
                    if new_p2 == p2:
                        continue
                    # new_p2 must be adjacent to next_p
                    if abs(new_p2[0]-next_p[0]) + abs(new_p2[1]-next_p[1]) != 1:
                        continue
                    # new_p2 must not be occupied by other residues
                    if new_p2 in occupied and occupied[new_p2] not in [i, i+1]:
                        continue
                    # new_p1 must not be new_p2
                    if new_p1 == new_p2:
                        continue
                    
                    # Also ensure new_p1 is not next_p or prev_p (handled by adjacency usually, but good to check)
                    # Ensure new_p2 is not prev_p or next_p
                    
                    # Perform move
                    del occupied[p1]
                    del occupied[p2]
                    current_chain[i] = new_p1
                    current_chain[i+1] = new_p2
                    occupied[new_p1] = i
                    occupied[new_p2] = i+1
                    
                    new_energy = _compute_energy(sequence, current_chain)
                    
                    if new_energy < current_energy:
                        current_energy = new_energy
                        if new_energy < best_energy:
                            best_energy = new_energy
                            best_chain = list(current_chain)
                        no_improve = 0
                        moved = True
                        break
                    else:
                        # Revert
                        del occupied[new_p1]
                        del occupied[new_p2]
                        current_chain[i] = p1
                        current_chain[i+1] = p2
                        occupied[p1] = i
                        occupied[p2] = i+1
                
                if moved:
                    break
            
            if moved:
                continue
        
        no_improve += 1
        if no_improve > 2000:
            break
            
    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Fold using Min-Conflicts / Local Search on random SAWs.
    """
    rng = random.Random(42)
    best_chain = None
    best_energy = 0
    n = len(sequence)

    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]

    # Time budget allocation
    # We will run multiple iterations of (Generate SAW -> Local Search)
    # To avoid getting stuck in local minima, we restart with new SAWs.
    
    start_time = time.time()
    
    # Number of restarts depends on sequence length
    # Shorter sequences can afford more restarts
    if n <= 20:
        max_restarts = 50
    elif n <= 50:
        max_restarts = 20
    else:
        max_restarts = 10
        
    for r in range(max_restarts):
        if time.time() > _DEADLINE - 10:
            break
            
        # Generate initial conformation
        initial_chain = _generate_initial_saw(sequence, rng)
        
        # If generation failed (returned fallback), it's a straight line.
        # We can still try to improve it, but it's less likely to be good.
        
        # Local Search
        improved_chain, improved_energy = _min_conflicts_fold(
            sequence, initial_chain, rng, max_steps=20000)
        
        if best_chain is None or improved_energy < best_energy:
            best_chain = improved_chain
            best_energy = improved_energy
            
            # If we found a very good energy (close to theoretical max), stop early?
            # Max contacts is roughly N/4 or so? 
            # Let's just let it run or stop if we hit a known good value.
            
        # If we are running out of time, break
        if time.time() > _DEADLINE - 20:
            break
            
    # Fallback
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
