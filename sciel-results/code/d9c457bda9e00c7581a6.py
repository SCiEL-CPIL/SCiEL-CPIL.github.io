"""Improved HP lattice protein folding using multi-start local search."""
# EVOLVE-BLOCK-START
import random
import time
import math

# Global deadline guard
_DEADLINE = None
_BUDGET_FRACTION = 0.88  # Use 88% of the 330s budget to be safe

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 330 * _BUDGET_FRACTION
    return _DEADLINE

def _is_deadline_exceeded():
    return time.time() > _get_deadline()

def _neighbours(point):
    """Yield the 6 neighbours of a point in 3D (for dimension=3)."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    """Compute energy as -number of H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    n = len(sequence)
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _initial_grow(sequence, dimension, rng):
    """Generate an initial self-avoiding walk using randomized greedy growth."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]
    
    # Start at origin
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Score: prefer positions with more H-H contacts and more free space
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == "H":
                            gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 2.0 + free * 0.1 + rng.random()
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick best option
        options.sort(reverse=True)
        site = options[0][1]
        chain.append(site)
        occupied[site] = i
    
    return chain

def _local_search(sequence, chain, dimension, rng, max_iterations=2000):
    """Improve a chain using single-residue re-insertion local search."""
    n = len(sequence)
    if n <= 2:
        return chain
    
    # Build position map
    pos_map = {p: i for i, p in enumerate(chain)}
    
    def compute_energy():
        contacts = 0
        for i, p in enumerate(chain):
            if sequence[i] == "H":
                for q in _neighbours(p):
                    j = pos_map.get(q)
                    if j is not None and j > i + 1 and sequence[j] == "H":
                        contacts += 1
        return -contacts
    
    current_energy = compute_energy()
    best_energy = current_energy
    best_chain = list(chain)
    
    # Precompute which indices are H
    is_h = [1 if s == "H" else 0 for s in sequence]
    
    for iteration in range(max_iterations):
        if _is_deadline_exceeded():
            break
        
        # Pick a random index to try re-inserting
        i = rng.randint(1, n - 2)  # Don't move ends
        if rng.random() < 0.1:
            i = rng.randint(0, n - 1)
        
        # Remove residue i
        p_i = chain[i]
        chain.pop(i)
        del pos_map[p_i]
        
        # Try inserting at a random position
        j = rng.randint(0, len(chain))
        
        # Find a valid site near chain[j] or chain[j-1]
        neighbors_to_try = set()
        if j > 0:
            neighbors_to_try.update(_neighbours(chain[j-1]))
        if j < len(chain):
            neighbors_to_try.update(_neighbours(chain[j]))
        
        valid_sites = []
        for site in neighbors_to_try:
            if site in pos_map:
                continue
            # Check if site is adjacent to the two neighbors it would connect to
            # After insertion, site must be adjacent to chain[j-1] (if exists) and chain[j] (if exists)
            valid = True
            if j > 0:
                if abs(site[0]-chain[j-1][0]) + abs(site[1]-chain[j-1][1]) + abs(site[2]-chain[j-1][2]) != 1:
                    valid = False
            if valid and j < len(chain):
                if abs(site[0]-chain[j][0]) + abs(site[1]-chain[j][1]) + abs(site[2]-chain[j][2]) != 1:
                    valid = False
            if valid:
                valid_sites.append(site)
        
        if not valid_sites:
            # Put it back
            chain.insert(i, p_i)
            pos_map[p_i] = i
            continue
        
        site = valid_sites[rng.randint(0, len(valid_sites)-1)]
        chain.insert(j, site)
        
        # Rebuild pos_map
        pos_map = {p: idx for idx, p in enumerate(chain)}
        
        new_energy = compute_energy()
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_chain = list(chain)
        
        if new_energy < current_energy:
            current_energy = new_energy
        else:
            # Revert
            chain.pop(j)
            p_old = chain[i] if j > i else chain[i]
            # Need to restore original
            chain = list(best_chain) if False else chain
            # Simpler: just revert by putting p_i back at position i
            # But chain has changed length. Let's just accept or revert properly.
            # For simplicity, if not better, revert to previous state
            # We'll just keep going with the new state (simulated annealing style isn't needed)
            # Actually, let's do strict descent
            # Restore original chain
            # This is getting complex; let's use a simpler approach
            pass
    
    return best_chain

def _local_search_simple(sequence, chain, dimension, rng, max_iterations=500):
    """Simpler local search: try moving one residue to a neighbouring site."""
    n = len(sequence)
    if n <= 2:
        return chain
    
    def compute_energy(chain, pos_map):
        contacts = 0
        for i, p in enumerate(chain):
            if sequence[i] == "H":
                for q in _neighbours(p):
                    j = pos_map.get(q)
                    if j is not None and j > i + 1 and sequence[j] == "H":
                        contacts += 1
        return -contacts
    
    current_energy = compute_energy(chain, {p: i for i, p in enumerate(chain)})
    best_energy = current_energy
    best_chain = list(chain)
    best_pos_map = {p: i for i, p in enumerate(best_chain)}
    
    for iteration in range(max_iterations):
        if _is_deadline_exceeded():
            break
        
        # Pick a random internal residue
        i = rng.randint(1, n - 2)
        if rng.random() < 0.15:
            i = rng.randint(0, n - 1)
        
        p_i = chain[i]
        
        # Try all 6 neighbours
        for site in _neighbours(p_i):
            if site == chain[i-1] if i > 0 else False:
                continue
            if site == chain[i+1] if i < n-1 else False:
                continue
            if site in best_pos_map:
                continue
            
            # Check if moving p_i to site is valid
            # Must be adjacent to chain[i-1] and chain[i+1]
            if i > 0:
                if abs(site[0]-chain[i-1][0]) + abs(site[1]-chain[i-1][1]) + abs(site[2]-chain[i-1][2]) != 1:
                    continue
            if i < n-1:
                if abs(site[0]-chain[i+1][0]) + abs(site[1]-chain[i+1][1]) + abs(site[2]-chain[i+1][2]) != 1:
                    continue
            
            # Try the move
            old_site = chain[i]
            chain[i] = site
            del best_pos_map[old_site]
            best_pos_map[site] = i
            
            new_energy = compute_energy(chain, best_pos_map)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(chain)
                current_energy = new_energy
            else:
                # Revert
                chain[i] = old_site
                del best_pos_map[site]
                best_pos_map[old_site] = i
    
    return best_chain

def fold(sequence, dimension):
    """Best conformation using multi-start greedy growth and local search refinement."""
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 1
    
    # Multiple restarts with greedy growth + local search
    for restart in range(50):
        if _is_deadline_exceeded():
            break
        
        chain = _initial_grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        # Apply local search to improve
        chain = _local_search_simple(sequence, chain, dimension, rng, max_iterations=100)
        
        value = _energy(sequence, chain)
        if value < best_energy:
            best, best_energy = chain, value
    
    if best is None:  # a straight line is always self-avoiding
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
