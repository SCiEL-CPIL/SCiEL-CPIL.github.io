"""HP lattice protein folding on the square lattice using Beam Search."""
# EVOLVE-BLOCK-START
import time
import random

_T0 = time.time()
_DEADLINE = 195.0  # 210s wall-clock limit, 15s safety margin


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _neighbours(point):
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def _energy(sequence, chain):
    """Compute exact energy: -number of H-H non-bonded contacts."""
    n = len(chain)
    if n < 3:
        return 0
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    h_set = set(i for i, c in enumerate(sequence) if c == "H")
    contacts = 0
    for i in h_set:
        x, y = chain[i]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            q = (x + dx, y + dy)
            j = pos_to_idx.get(q)
            if j is not None and j > i + 1 and sequence[j] == "H":
                contacts += 1
    return -contacts


def _grow_beam(sequence, dimension, beam_width, time_limit):
    """
    Beam search to find low-energy conformations.
    Returns a list of chains (the top ones from the final beam).
    """
    n = len(sequence)
    if n == 0:
        return []
    
    # Precompute H indices
    h_indices = set(i for i, c in enumerate(sequence) if c == "H")
    
    # Initial state: chain=[(0,0)], occupied={(0,0):0}, energy=0
    # State: (chain, occupied_dict, energy)
    beam = [([(0, 0)], {(0, 0): 0}, 0)]
    
    for i in range(1, n):
        new_beam = []
        is_h = sequence[i] == "H"
        
        for chain, occupied, energy in beam:
            last_pos = chain[-1]
            prev_idx = i - 1
            
            for npos in _neighbours(last_pos):
                if npos in occupied:
                    continue
                
                # Calculate immediate energy gain
                gain = 0
                if is_h:
                    for nq in _neighbours(npos):
                        j = occupied.get(nq)
                        if j is not None and j < prev_idx and sequence[j] == "H":
                            gain += 1
                
                new_energy = energy - gain
                
                # Heuristic: prefer states with more free space to avoid trapping
                # Count free neighbors of npos
                free_space = 0
                for nq in _neighbours(npos):
                    if nq not in occupied:
                        free_space += 1
                
                # Score: lower is better. 
                # We want low energy. We also want high free space (so subtract free_space penalty? or add?).
                # If free_space is low, it's bad. So we penalize low free_space.
                # score = energy - 0.1 * free_space ?
                # If energy is -10 and free_space is 0, score = -10.
                # If energy is -9 and free_space is 4, score = -9.4.
                # -10 < -9.4, so we prefer the trapped one? That's wrong.
                # We want to prefer high free space.
                # So score = energy - 0.5 * free_space.
                # If energy is -10, free=0 -> -10.
                # If energy is -9, free=4 -> -9 - 2 = -11.
                # -11 < -10, so we prefer the one with more free space. Correct.
                
                score = new_energy - 0.5 * free_space
                
                # Create new state
                new_chain = chain + [npos]
                new_occupied = occupied.copy()
                new_occupied[npos] = i
                
                new_beam.append((new_chain, new_occupied, new_energy, score))
        
        if not new_beam:
            break
            
        # Sort by score (ascending) and keep top beam_width
        new_beam.sort(key=lambda x: x[3])
        beam = [(c, o, e) for c, o, e, s in new_beam[:beam_width]]
        
        # Time check
        if _remaining() < 5.0:
            # If time is critical, we might want to stop early or reduce beam width
            # But we must complete the chain. Let's just break if we can't finish.
            # Actually, if we break here, we return partial chains.
            # We need to ensure we return a full chain.
            # If we don't have time, we should probably just return the best full chain found so far?
            # But we don't have full chains yet.
            # Better to just continue with a smaller beam or finish quickly.
            # Let's reduce beam width to 10 for subsequent steps to speed up.
            beam_width = min(beam_width, 10)
            
    # Final evaluation of the top states in the beam
    # The beam contains the best partials. We need to compute exact energy for them.
    # Actually, the energy stored in the state is the exact energy of the partial chain.
    # Since the chain is complete, the stored energy IS the exact energy.
    
    if not beam:
        return []
        
    # Return the chains with the lowest energy
    beam.sort(key=lambda x: x[2])
    return [c for c, o, e in beam[:5]]


def fold(sequence, dimension):
    """Best conformation using beam search."""
    if not sequence:
        return []
        
    # Adaptive beam width based on sequence length and time
    n = len(sequence)
    base_beam_width = 50
    if n > 50:
        base_beam_width = 30
    if n > 80:
        base_beam_width = 20
        
    best_chain = None
    best_energy = 0
    
    # Try a few different beam widths or just one good one?
    # Given time constraints, one well-tuned beam search is better than many small ones.
    # But randomization in the tie-breaking or slight variations can help.
    # Let's run one main beam search.
    
    candidates = _grow_beam(sequence, dimension, base_beam_width, _remaining())
    
    if candidates:
        # Find the best among candidates
        for chain in candidates:
            e = _energy(sequence, chain)
            if best_chain is None or e < best_energy:
                best_chain = chain
                best_energy = e
    else:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _energy(sequence, best_chain)

    # If time allows, try a second pass with different parameters?
    # Maybe a smaller beam but with different heuristic?
    # For now, stick to one robust pass to ensure safety.
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
