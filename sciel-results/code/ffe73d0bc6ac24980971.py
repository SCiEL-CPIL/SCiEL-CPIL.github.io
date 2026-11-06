# EVOLVE-BLOCK-START
import time
import sys
import math
import random
import numpy as np
from heapq import heappush, heappop

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    """
    Compute the energy of a chain.
    Energy = - (number of H-H contacts).
    """
    if not chain:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    n = len(chain)
    for i in range(n):
        if sequence[i] == "H":
            p = chain[i]
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _pivot_move(sequence, chain, i, dimension, rng):
    """
    Attempt a pivot move at index i.
    Returns the new chain if successful and energy improves, else None.
    """
    n = len(chain)
    if i <= 0 or i >= n - 1:
        return None
    
    prev = chain[i]
    curr = chain[i+1]
    
    valid_moves = []
    # Check neighbors of prev that are not curr and not in head (0..i)
    # We need a set of head for fast lookup
    # Building the set every time is O(N). 
    # For N=136, this is acceptable if we don't do it too often.
    head_occupied = set(chain[:i+1])
    
    for q in _neighbours(prev):
        if q == curr:
            continue
        if q in head_occupied:
            continue
        valid_moves.append(q)
        
    if not valid_moves:
        return None
        
    new_curr = rng.choice(valid_moves)
    delta = tuple(new_curr[k] - curr[k] for k in range(dimension))
    
    # Check if the tail can be translated by delta without collision
    # Tail is i+1..n-1
    # We need to check against head_occupied and the tail itself
    tail_occupied = set()
    new_tail = []
    
    # Pre-compute tail positions for speed?
    # Let's just iterate.
    for j in range(i+1, n):
        old_p = chain[j]
        if j == i+1:
            new_p = new_curr
        else:
            new_p = tuple(old_p[k] + delta[k] for k in range(dimension))
            
        if new_p in head_occupied or new_p in tail_occupied:
            return None
            
        tail_occupied.add(new_p)
        new_tail.append(new_p)
        
    new_chain = chain[:i+1] + new_tail
    return new_chain

def _local_optimize(sequence, chain, dimension, time_limit=2.0):
    """
    Local optimization using pivot moves.
    Uses a simple acceptance criterion: accept if energy improves.
    To escape local minima, we can occasionally accept neutral or slightly worse moves?
    For now, strict improvement or equal energy (to move around flat landscapes).
    """
    n = len(chain)
    if n < 3:
        return chain
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    start_time = time.time()
    rng = random.Random(42)
    
    # Track recent moves to avoid cycling?
    # Simple approach: keep trying random pivots.
    
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() - start_time < time_limit:
        if no_improve_count > max_no_improve:
            # Random restart within the local search?
            # Or just break?
            # Let's try a random perturbation: pick a random pivot and force a move even if worse?
            # No, let's just break and rely on multiple initial seeds.
            break
            
        i = rng.randint(1, n - 2)
        new_chain = _pivot_move(sequence, current_chain, i, dimension, rng)
        
        if new_chain is None:
            continue
            
        new_energy = _energy(sequence, new_chain)
        
        if new_energy < current_energy:
            current_chain = new_chain
            current_energy = new_energy
            no_improve_count = 0
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(current_chain)
        elif new_energy == current_energy:
            # Accept with some probability to explore
            if rng.random() < 0.5:
                current_chain = new_chain
                no_improve_count += 1
        else:
            no_improve_count += 1
            
    return best_chain

def _beam_search_fold(sequence, dimension, beam_width=50, time_limit=2.0):
    """
    Beam search that builds the chain from the N-terminus.
    """
    n = len(sequence)
    if n <= 1:
        return [tuple([0]*dimension)]
    
    start = (0, 0, 0)
    initial_chain = (start,)
    initial_occupied = {start}
    initial_energy = 0
    
    beam = [(initial_energy, initial_chain, initial_occupied)]
    
    start_time = time.time()
    
    for step in range(1, n):
        if time.time() - start_time > time_limit:
            break
            
        new_beam = []
        current_residue = sequence[step]
        
        for energy, chain, occupied in beam:
            last = chain[-1]
            
            for next_pos in _neighbours(last):
                if next_pos in occupied:
                    continue
                
                new_energy = energy
                if current_residue == "H":
                    # Check for new contacts
                    for q in _neighbours(next_pos):
                        if q in occupied:
                            # Find index of q
                            # This is O(N)
                            for j, p in enumerate(chain):
                                if p == q:
                                    if sequence[j] == "H":
                                        new_energy -= 1
                                    break
                            
                new_chain = chain + (next_pos,)
                new_occupied = occupied | {next_pos}
                new_beam.append((new_energy, new_chain, new_occupied))
                
        if not new_beam:
            break
            
        new_beam.sort(key=lambda x: x[0])
        beam = new_beam[:beam_width]
        
    if beam:
        return list(beam[0][1])
    return None

def fold(sequence, dimension):
    """
    Uses multiple beam search strategies and local optimization.
    """
    n = len(sequence)
    
    if not hasattr(fold, '_global_start'):
        fold._global_start = time.time()
        
    global_deadline = fold._global_start + 320.0  # Safety margin for 330s kill
    current_time = time.time()
    
    if current_time >= global_deadline:
        # Emergency fallback
        return [list([i] + [0] * (dimension - 1)) for i in range(n)]
    
    # Allocate time based on chain length
    # Shorter chains are faster to search.
    if n < 50:
        call_time = 8.0
    elif n < 100:
        call_time = 12.0
    else:
        call_time = 18.0
        
    call_deadline = current_time + call_time
    if call_deadline > global_deadline:
        call_deadline = global_deadline
        
    best_chain = None
    best_energy = 0
    
    # Strategy 1: Forward Beam Search
    time_per_search = (call_deadline - current_time) / 3.0
    if time_per_search > 0.5:
        chain1 = _beam_search_fold(sequence, dimension, beam_width=80, time_limit=time_per_search)
        if chain1:
            e1 = _energy(sequence, chain1)
            if best_chain is None or e1 < best_energy:
                best_chain = chain1
                best_energy = e1
                
    # Strategy 2: Backward Beam Search
    if time.time() < call_deadline - 0.5:
        time_per_search = (call_deadline - time.time()) / 2.0
        if time_per_search > 0.5:
            rev_seq = sequence[::-1]
            chain2 = _beam_search_fold(rev_seq, dimension, beam_width=80, time_limit=time_per_search)
            if chain2:
                chain2 = list(reversed(chain2))
                e2 = _energy(sequence, chain2)
                if best_chain is None or e2 < best_energy:
                    best_chain = chain2
                    best_energy = e2
                    
    # Strategy 3: Local Optimization on best found
    if best_chain and time.time() < call_deadline - 0.5:
        opt_time = max(0.5, call_deadline - time.time() - 0.5)
        optimized = _local_optimize(sequence, best_chain, dimension, time_limit=opt_time)
        opt_energy = _energy(sequence, optimized)
        
        if opt_energy < best_energy:
            best_chain = optimized
            best_energy = opt_energy
            
    # Strategy 4: If we still have time, try a second local optimization from a different start?
    # Or just run local optimization again on the best chain with a different seed?
    # The current _local_optimize uses a fixed seed.
    # Let's add a simple random walk if time permits.
    
    if best_chain is None:
        # Fallback
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = 0
        
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
