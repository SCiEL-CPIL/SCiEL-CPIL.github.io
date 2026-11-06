"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

START_TIME = time.time()
TIME_LIMIT = 300.0  # Safety margin before 330s kill

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _compute_energy(sequence, chain):
    """Compute energy of a full chain."""
    if not chain:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _beam_search(sequence, dimension, beam_width=50):
    """
    Beam search for HP folding.
    State: (chain, occupied_set, energy)
    We keep the top 'beam_width' states at each step.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    # Initial state
    start = (0,) * dimension
    initial_chain = [start]
    initial_occupied = {start: 0}
    initial_energy = 0
    
    # Beam contains list of (chain, occupied_dict, energy)
    # To save memory and time, we might store chain as list of tuples.
    # Occupied dict is {tuple: index}
    
    beam = [(initial_chain, initial_occupied, initial_energy)]
    
    for i in range(1, n):
        next_beam = []
        # For each state in current beam, try all valid moves
        for chain, occupied, energy in beam:
            last_pos = chain[-1]
            for site in _neighbours(last_pos):
                if site in occupied:
                    continue
                
                # Check if we can add site
                # Update energy if sequence[i] is H
                new_energy = energy
                if sequence[i] == "H":
                    # Check neighbors of site for H residues in occupied
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == "H":
                            new_energy -= 1
                
                new_chain = chain + [site]
                # Copying occupied dict is expensive. 
                # For beam search, we can be smarter, but for now let's try with shallow copy or just re-computing if needed?
                # Actually, copying dict of size i is O(i). With beam width 50 and n=100, this is 50*100*100 operations roughly. 
                # 500k ops per step * 100 steps = 50M ops. Might be slow in Python.
                
                # Optimization: Store occupied as a set? No, we need indices for energy check.
                # Let's try to optimize the loop.
                
                # Alternative: Don't copy the whole dict. 
                # But we need the new state for the next iteration.
                # Let's just do it and see if it fits in time.
                # To speed up, we can pre-filter.
                
                new_occupied = dict(occupied)
                new_occupied[site] = i
                
                next_beam.append((new_chain, new_occupied, new_energy))
        
        if not next_beam:
            break
            
        # Prune to beam_width
        # Sort by energy (lower is better)
        # If tie, we can break ties arbitrarily or by some heuristic
        next_beam.sort(key=lambda x: x[2])
        beam = next_beam[:beam_width]
        
        # Time check
        if time.time() - START_TIME > TIME_LIMIT * 0.8:
             # If we are running out of time, break and return best so far?
             # But we need a full chain. If we break early, we have partial chains.
             # We should complete the best partial chain greedily?
             # Or just return the best full chain found in previous complete passes?
             # Actually, beam search builds up to n. If we break, we don't have a full chain.
             # So we should only break if we can't finish.
             # Let's add a time check inside the loop to abort if too slow.
             pass

    if not beam:
        return None
        
    # Return the best chain
    best_chain = beam[0][0]
    return best_chain

def _local_optimize(sequence, chain, dimension):
    """
    Simple local optimization: try shifting each residue to a nearby empty spot if it improves energy.
    """
    if not chain:
        return chain, 0
        
    n = len(chain)
    current_energy = _compute_energy(sequence, chain)
    improved = True
    max_iters = 100
    
    while improved and max_iters > 0:
        improved = False
        max_iters -= 1
        
        # Build occupied set
        occupied = {p: i for i, p in enumerate(chain)}
        
        for i in range(n):
            # Skip ends? No, try all.
            # Get current pos
            pos = chain[i]
            
            # Try moving to neighbors of pos
            # But the new pos must be empty and connected to neighbors in chain
            # Actually, standard move: pick a residue, move it to a new position 
            # such that it remains connected to its chain neighbors.
            # This is complex. 
            # Simpler: "kink" moves or "shift" moves.
            # Let's try a simpler heuristic: 
            # For each i, try all empty sites within distance 2? No, too many.
            
            # Let's stick to a very simple local search:
            # Try swapping? No.
            # Try moving a single residue to an adjacent empty site if it maintains connectivity?
            # Connectivity is hard to maintain with single moves unless it's a "head" or "tail" or a loop.
            
            # Given time constraints, maybe skip complex local opt if beam search is good enough?
            # Beam search with width 50 is quite strong.
            # Let's try a very simple "hill climb" on the tail?
            
            # Actually, let's just return the beam search result. It's deterministic and robust.
            # If we have time, we can add a simple tail-optimization.
            
            # Let's implement a simple tail extension optimization:
            # If the last few residues are stuck, try to re-fold the tail.
            # This is effectively what beam search does, but maybe we can refine the last 10 residues?
            
            pass
            
    return chain, current_energy

def fold(sequence, dimension):
    """
    Best conformation using Beam Search.
    """
    start_time = time.time()
    
    # Determine beam width based on sequence length and time budget
    # Longer sequences need more time, so maybe smaller beam? 
    # Or just fixed beam.
    beam_width = 50
    
    # Run beam search
    best_chain = _beam_search(sequence, dimension, beam_width)
    
    if best_chain is None or len(best_chain) != len(sequence):
        # Fallback to simple greedy or straight line
        # Try a quick greedy
        rng_seed = 12345 + len(sequence)
        import random
        rng = random.Random(rng_seed)
        start = (0,) * dimension
        chain = [start]
        occupied = {start: 0}
        for i in range(1, len(sequence)):
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
                break
            site = max(options)[1]
            occupied[site] = i
            chain.append(site)
        
        if len(chain) == len(sequence):
            best_chain = chain
        else:
            # Straight line fallback
            best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
            
    # Check time, if we have plenty left, we could try multiple beams or local opt.
    # But beam search is already doing the work.
    
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
