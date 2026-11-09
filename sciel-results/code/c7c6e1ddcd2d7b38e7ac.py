"""HP lattice protein folding using CSP with backjumping."""
# EVOLVE-BLOCK-START
import time
import sys
import random
from collections import defaultdict

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _compute_energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _fold_csp(sequence, dimension, time_limit=10.0):
    n = len(sequence)
    start_time = time.time()
    
    best_chain = None
    best_energy = 0  # We want most negative (most contacts)
    
    # Symmetry breaking: 
    # Residue 0 at (0,0)
    # Residue 1 at (1,0) (if n > 1)
    # This fixes the orientation and reduces search space by factor of 8 (4 rotations * 2 reflections)
    
    def _search(chain, occupied, current_energy):
        nonlocal best_chain, best_energy, start_time
        
        if time.time() - start_time > time_limit:
            return
        
        i = len(chain)
        if i == n:
            # Complete chain found
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain[:]
            return
            
        last_pos = chain[-1]
        current_residue = sequence[i]
        
        # Generate candidates
        candidates = []
        for site in _neighbours(last_pos):
            if site in occupied:
                continue
                
            # Heuristic Scoring
            score = 0.0
            
            # 1. Immediate Contact Gain
            if current_residue == 'H':
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if sequence[j] == 'H':
                            score += 10.0 # Strong preference for forming contacts
            
            # 2. Free Space / Trap Avoidance
            # Count free neighbors for the next step
            free_count = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free_count += 1
            
            # If placing here leaves 0 free neighbors for the next residue (and i < n-1), it's a dead end
            if free_count == 0 and i < n - 1:
                continue # Prune immediately
            
            # Prefer sites with more free space to avoid trapping
            score += free_count * 0.5
            
            # 3. Potential Future Contacts
            # If current is H, check if any nearby (dist 2) free spots could be H?
            # Too expensive. Stick to local heuristic.
            
            # 4. Randomness for diversity in restarts
            score += random.random() * 0.1
            
            candidates.append((score, site))
            
        if not candidates:
            return
            
        # Sort by score descending
        candidates.sort(key=lambda x: -x[0])
        
        for score, site in candidates:
            # Try placing at site
            chain.append(site)
            occupied[site] = i
            
            # Calculate new energy increment
            delta_energy = 0
            if current_residue == 'H':
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if sequence[j] == 'H':
                            delta_energy -= 1 # Energy decreases (improves) by 1 per contact
            
            _search(chain, occupied, current_energy + delta_energy)
            
            # Backtrack
            chain.pop()
            del occupied[site]
            
            if time.time() - start_time > time_limit:
                return

    # Run search with multiple restarts if time allows
    # For small n, we can do exhaustive (with pruning). For large n, we rely on heuristics.
    
    # Initial state
    init_chain = [(0, 0)]
    init_occupied = {(0, 0): 0}
    
    if n > 1:
        # Fix residue 1 at (1, 0) to break symmetry
        # Check if (1,0) is valid (it is, since (0,0) is occupied)
        init_chain.append((1, 0))
        init_occupied[(1, 0)] = 1
        
        # Initial energy is 0 (no contacts between 0 and 1 as they are adjacent in chain)
        # Wait, energy is defined as -contacts between non-adjacent residues.
        # So adjacent residues do not contribute to energy.
        
        _search(init_chain, init_occupied, 0)
    else:
        best_chain = init_chain[:]
        best_energy = 0

    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = 0
    else:
        # Verify best_energy calculation (it should be correct from incremental updates)
        # But let's recompute to be safe if we modified the logic
        best_energy = _compute_energy(sequence, best_chain)

    return best_chain, best_energy

def fold(sequence, dimension):
    """Fold an HP sequence using CSP-inspired search with local refinement."""
    n = len(sequence)
    
    # Time budget per sequence
    # Total budget 120s for 10 sequences -> 12s avg.
    # Allocate based on length.
    if n <= 20:
        time_limit = 10.0
    elif n <= 40:
        time_limit = 8.0
    elif n <= 60:
        time_limit = 6.0
    else:
        time_limit = 4.0
        
    # Use a fixed seed for reproducibility in benchmarking, but allow slight randomness for restarts if we add them.
    # For now, single run with heuristic.
    random.seed(42) # Seed for heuristic noise
    
    chain, energy = _fold_csp(sequence, dimension, time_limit=time_limit)
    
    return [list(p) for p in chain]
# EVOLVE-BLOCK-END
