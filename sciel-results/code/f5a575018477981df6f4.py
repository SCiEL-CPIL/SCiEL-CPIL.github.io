import time

def _neighbours(point):
    x, y, z = point
    yield (x+1, y, z)
    yield (x-1, y, z)
    yield (x, y+1, z)
    yield (x, y-1, z)
    yield (x, y, z+1)
    yield (x, y, z-1)

def fold(sequence, dimension):
    """
    Fold the sequence using Beam Search on the 3D cubic lattice.
    Returns a list of lists of integers representing the coordinates.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    if n == 2:
        return [[0, 0, 0], [1, 0, 0]]
    
    start_time = time.time()
    deadline = start_time + 290
    
    # Symmetry breaking:
    # R0 at (0,0,0), R1 at (1,0,0)
    initial_chain = [(0, 0, 0), (1, 0, 0)]
    initial_occupied = {(0, 0, 0), (1, 0, 0)}
    initial_pos_to_idx = {(0, 0, 0): 0, (1, 0, 0): 1}
    initial_energy = 0
    
    beam = [(initial_chain, initial_occupied, initial_pos_to_idx, initial_energy)]
    
    # Dynamic beam width
    if n < 50:
        K = 300
    else:
        K = 600
        
    for i in range(2, n):
        new_beam_candidates = []
        residue = sequence[i]
        
        for state in beam:
            chain, occupied, pos_to_idx, energy = state
            last_pos = chain[-1]
            
            for neighbor in _neighbours(last_pos):
                if neighbor in occupied:
                    continue
                
                # Calculate new energy
                new_energy = energy
                if residue == 'H':
                    for nb in _neighbours(neighbor):
                        if nb in occupied:
                            j = pos_to_idx[nb]
                            # Exclude the immediately preceding residue in the chain
                            if j != i - 1 and sequence[j] == 'H':
                                new_energy -= 1
                            
                # Calculate free space for heuristic
                free_space = 0
                for nb in _neighbours(neighbor):
                    if nb not in occupied:
                        free_space += 1
                            
                # Score: Energy is primary (minimize), free space is secondary (maximize)
                # Increased weight on energy, reduced on free space to prioritize compactness
                score = new_energy - 0.08 * free_space
                
                # Store reference to parent state to avoid copying yet
                # (score, energy, neighbor, parent_chain, parent_occupied, parent_pos_to_idx)
                new_beam_candidates.append((score, new_energy, neighbor, chain, occupied, pos_to_idx))
        
        if not new_beam_candidates:
            break
            
        # Sort by score
        new_beam_candidates.sort(key=lambda x: x[0])
        
        # Keep top K and materialize the states
        beam = []
        for item in new_beam_candidates[:K]:
            score, energy, neighbor, parent_chain, parent_occupied, parent_pos_to_idx = item
            # Now we copy for the survivors only
            new_chain = parent_chain + [neighbor]
            new_occupied = parent_occupied | {neighbor}
            new_pos_to_idx = parent_pos_to_idx | {neighbor: i}
            beam.append((new_chain, new_occupied, new_pos_to_idx, energy))
            
        if not beam:
            break
            
        # Time check
        if time.time() > deadline:
            break

    if not beam:
        # Fallback: straight line
        return [[i, 0, 0] for i in range(n)]
    
    # Return the best chain (lowest score, which corresponds to lowest energy)
    best_chain = beam[0][0]
    return [list(p) for p in best_chain]
