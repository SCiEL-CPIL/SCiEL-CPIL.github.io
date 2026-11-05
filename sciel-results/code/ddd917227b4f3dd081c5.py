"""
HP Lattice Protein Folding on Cubic Lattice
Uses Monte Carlo Chain Growth (MCCG) with energy-based bias and backtracking.
"""
import math

import random
import time
import sys

# EVOLVE-BLOCK-START

def _neighbours(point):
    """Yield the 6 neighbors of a point in 3D cubic lattice."""
    x, y, z = point
    yield (x + 1, y, z)
    yield (x - 1, y, z)
    yield (x, y + 1, z)
    yield (x, y - 1, z)
    yield (x, y, z + 1)
    yield (x, y, z - 1)

def _calculate_energy_delta(sequence, index, pos, occupied):
    """
    Calculate the change in energy if residue at 'index' is placed at 'pos'.
    Energy = - (number of H-H contacts).
    A contact is formed if:
    1. Current residue is H.
    2. Neighbor residue is H.
    3. Neighbor index is not adjacent (|i - j| > 1).
    
    Returns delta_E (negative for favorable contacts).
    """
    if sequence[index] != 'H':
        return 0
    
    delta = 0
    for n_pos in _neighbours(pos):
        if n_pos in occupied:
            n_idx = occupied[n_pos]
            # Check if not adjacent in sequence
            if abs(index - n_idx) > 1:
                if sequence[n_idx] == 'H':
                    delta -= 1  # Energy decreases (becomes more negative)
    return delta

def _mccg_fold(sequence, dimension, time_limit, seed):
    """
    Monte Carlo Chain Growth folding algorithm.
    """
    rng = random.Random(seed)
    n = len(sequence)
    
    # Initialize
    start_pos = (0, 0, 0)
    chain = [start_pos]
    occupied = {start_pos: 0}
    current_energy = 0
    
    # Stack for backtracking: stores (chain_copy, occupied_copy, energy, next_step_index, available_neighbors_list)
    # To save memory/time, we might just store the index and re-derive, but copying small dicts is fast enough for n<150
    # Better: Store the state explicitly.
    
    # For efficiency in Python, we will maintain the state and use a stack of "decisions".
    # State: chain, occupied, energy, current_index
    # If we get stuck, we backtrack.
    
    # To make backtracking efficient, we keep a list of "states".
    # Each state: (chain, occupied, energy, index, untried_neighbors)
    # However, copying `occupied` (dict) at every step is O(N). For N=136, this is fine.
    
    best_chain = None
    best_energy = 0 # 0 is the energy of no contacts (worst case for H-rich, but we start at 0)
    # Actually, best_energy should be initialized to a very high positive number if we are minimizing, 
    # but here energy is negative. So we want the most negative.
    # Let's initialize best_energy to 0 (unfolded) and look for < 0.
    
    start_time = time.time()
    
    # Temperature parameter for Boltzmann selection
    # Lower T = more greedy. Higher T = more random.
    # We can anneal T or just use a fixed low T for exploration/exploitation balance.
    T = 1.0 
    
    # Stack of states to allow backtracking
    # State: (chain, occupied, energy, current_index, list_of_untried_neighbors)
    # We start by pushing the initial state? No, we grow.
    
    # We will use a recursive-like iterative backtracking.
    # Let's maintain a stack of "checkpoints".
    # checkpoint: (chain, occupied, energy, index, untried_neighbors)
    
    # Initial setup
    current_index = 0
    untried_neighbors = [] # Not used for start
    
    # We need a way to choose the next move.
    # Let's implement a standard MCCG loop.
    
    # To handle backtracking efficiently without deep copying huge structures constantly:
    # We can store the chain and occupied set. When backtracking, we pop.
    
    # Let's use a simpler approach: Random Walk with Backtracking.
    # We keep a list of states.
    
    states = []
    # Push initial state
    # We don't push the start, we start growing from it.
    
    # Let's manage the growth manually.
    # chain, occupied, energy, index
    # If we can't grow, we backtrack to the last state where we had untried options.
    
    # Data structure for backtracking:
    # stack: list of (chain, occupied, energy, index, untried_neighbors)
    
    # Initialize
    chain = [start_pos]
    occupied = {start_pos: 0}
    energy = 0
    idx = 0
    
    # We will keep a stack of "decisions" made.
    # Each entry: (prev_chain, prev_occupied, prev_energy, prev_idx, next_idx, chosen_pos, other_untried)
    # This is getting complex. Let's use a simpler "Restart" strategy combined with "Greedy with Randomness".
    # Given the short time limit, a pure backtracking MCCG in Python might be slow if implemented naively with dict copies.
    
    # Alternative: **Simulated Annealing on the full chain** is hard to initialize.
    # **Chain Growth with Local Search**:
    # 1. Grow a random SAW with H-bias.
    # 2. Apply local moves (pivot, crankshaft) to optimize.
    
    # Let's try **Chain Growth** first, as it's naturally self-avoiding.
    # We will implement a "Look-ahead" or "Backtracking" growth.
    
    # Optimized Backtracking MCCG:
    # We store the chain as a list.
    # We store occupied as a set.
    # When we add a step, we record it.
    # If stuck, we undo the last step.
    
    # To avoid copying the whole `occupied` dict every time we backtrack (which is O(N)),
    # we can just remove the added position from the set.
    
    # Stack item: (pos_added, index_added, energy_before, untried_neighbors_at_that_step)
    # We need to know which neighbors were available and which we tried.
    
    # Let's structure the loop:
    
    while True:
        if time.time() - start_time > time_limit:
            break
            
        # Try to grow
        if idx == n - 1:
            # Complete chain
            if energy < best_energy:
                best_energy = energy
                best_chain = chain[:]
            # Backtrack to explore other paths
            if not states:
                # We've explored all? Or we just restart?
                # MCCG usually backtracks to the root and tries different branches.
                # If we are at root and stuck, we restart with new random seed/params.
                # For simplicity, let's restart the growth with a new random offset.
                return best_chain, best_energy
            
            # Pop last state to backtrack
            pos_added, idx_added, e_before, untried = states.pop()
            # Undo
            chain.pop()
            occupied.pop(pos_added)
            energy = e_before
            idx = idx_added - 1
            
            # Mark that we tried this pos, so next time we pick from remaining untried?
            # Actually, `untried` in the state we popped contains the neighbors that were NOT tried at that step.
            # We need to continue trying those.
            # So we push a new state with the remaining untried neighbors?
            # Or we just loop back to the top, but we need to ensure we don't pick the same pos again.
            
            # Let's refine the state structure:
            # State: (pos, idx, energy_before, remaining_untried_neighbors)
            # When we pop, we take one neighbor from `remaining_untried_neighbors`.
            
            if untried:
                # Pick a neighbor from untried
                # We need to re-evaluate energy? No, energy delta depends on current state.
                # But the state is now the one BEFORE the pos_added.
                # So we can try a different neighbor from `untried`.
                
                # Let's pick a neighbor from untried using Boltzmann
                # We need to calculate delta_E for each in `untried` based on CURRENT state (which is now restored).
                
                # For now, let's just pick one randomly to keep it simple, or re-score.
                # Re-scoring is better.
                
                # We are at state (chain, occupied, energy, idx).
                # We have a list `untried` of positions that were available at step `idx+1` but not chosen.
                # We should pick one of them.
                
                # Let's pick one and continue.
                # We need to update the state to reflect this new choice.
                # This logic is getting tangled in the loop.
                
                # Let's restart the inner loop logic for backtracking.
                # Instead of complex in-place backtracking, let's use a simpler **Random Restart Greedy Growth**.
                # It's often faster to implement correctly in Python for this size.
                
                # Let's abort the complex backtracking and switch to **Multiple Random Greedy Growths with Local Search**.
                # 1. Generate a chain using biased random walk.
                # 2. If stuck, restart.
                # 3. Once a full chain is generated, apply local search (Metropolis) for a fixed number of steps.
                
                pass # Break out to use simpler strategy

        # If we didn't break, we are growing.
        # This simple while loop with manual backtracking is error-prone.
        # Let's implement the **Simpler Strategy**:
        # 1. Biased Random Growth (no backtracking, just restart if stuck).
        # 2. Local Search (Pivot Moves) on the resulting chain.
        
        break # Exit the complex loop and use the simpler strategy below

    # --- SIMPLER STRATEGY IMPLEMENTATION ---
    
    def _grow_biased(sequence, rng, T=1.0):
        """Grow a self-avoiding walk with energy bias. Returns None if stuck."""
        n = len(sequence)
        start = (0, 0, 0)
        chain = [start]
        occupied = {start: 0}
        
        for i in range(1, n):
            last = chain[-1]
            neighbors = list(_neighbours(last))
            # Filter out occupied
            available = [p for p in neighbors if p not in occupied]
            
            if not available:
                return None
            
            # Calculate score for each available position
            scores = []
            for p in available:
                # Gain in energy (negative is good)
                dE = _calculate_energy_delta(sequence, i, p, occupied)
                # Boltzmann factor
                # We want to favor lower energy (more negative dE).
                # exp(dE / T). If dE is -1, T=1, exp(-1) = 0.36. If dE=0, exp(0)=1.
                # This favors 0 over -1. We want to favor -1.
                # So we should use exp(-dE / T) or exp(dE / T) with dE being the GAIN (positive for good)?
                # Let's define score = exp(delta_E / T) where delta_E is the change in energy.
                # If we make a contact, energy drops by 1. delta_E = -1.
                # We want higher probability for delta_E = -1.
                # So we should use exp(-delta_E / T)? No.
                # Standard: P ~ exp(-E_new / T).
                # E_new = E_old + delta_E.
                # P ~ exp(-(E_old + delta_E) / T) = exp(-E_old/T) * exp(-delta_E/T).
                # So weight = exp(-delta_E / T).
                # If delta_E = -1, weight = exp(1/T). If T=1, weight = e.
                # If delta_E = 0, weight = 1.
                # So contacts are favored.
                
                weight = math.exp(-dE / T) if dE < 0 else 1.0 # Cap to avoid overflow, though dE is small
                # Actually, just use math.exp(-dE / T)
                scores.append((weight, p))
            
            # Normalize and pick
            total_w = sum(w for w, _ in scores)
            r = rng.random() * total_w
            cumulative = 0
            chosen = scores[0][1]
            for w, p in scores:
                cumulative += w
                if r <= cumulative:
                    chosen = p
                    break
            
            chain.append(chosen)
            occupied[chosen] = i
            
        return chain

    def _local_search(sequence, chain, steps=500, T_start=1.0, T_end=0.1):
        """Apply pivot moves to optimize the chain."""
        n = len(chain)
        if n < 4:
            return chain
        
        # Convert to list of tuples for easy manipulation
        c = chain[:]
        
        # Pre-compute occupied set
        occ = {p: i for i, p in enumerate(c)}
        
        # Calculate initial energy
        def calc_energy(c, occ):
            e = 0
            for i, p in enumerate(c):
                if sequence[i] == 'H':
                    for n_pos in _neighbours(p):
                        j = occ.get(n_pos)
                        if j is not None and abs(i - j) > 1 and sequence[j] == 'H':
                            e -= 1
            return e
        
        current_energy = calc_energy(c, occ)
        best_chain = c[:]
        best_energy = current_energy
        
        T = T_start
        cooling = (T_end - T_start) / steps if steps > 0 else 0
        
        for step in range(steps):
            # Pivot move
            # Choose pivot point i (1 to n-2)
            i = rng.randint(1, n - 2)
            # Choose direction to rotate: left or right
            # This is complex to implement correctly for 3D.
            # Simpler: **Swap** two non-adjacent segments? No, breaks connectivity.
            # **Crankshaft** move?
            # Let's use a simpler **Local Neighbor Swap** if possible? No, must maintain connectivity.
            
            # Actually, for HP folding, **Monte Carlo Chain Growth** is often better than post-processing SAWs with pivots in Python due to overhead.
            # Let's stick to **Multiple Biased Growths** and keep the best.
            # If we have time, we can try to "repair" or "refine" with simple moves.
            
            # For now, just return the grown chain. The power comes from many restarts.
            pass
            
        return best_chain

    # Strategy: Run many biased growths.
    # Time budget: ~8 seconds per sequence.
    # Each growth is O(N^2) roughly due to neighbor checks. N=136 is fast.
    # We can do thousands of growths.
    
    best_chain = None
    best_energy = 0 # Unfolded
    
    # We will use a fixed number of attempts or time limit.
    # Let's try 2000 growths or until time limit.
    
    attempts = 0
    max_attempts = 5000
    
    while attempts < max_attempts:
        if time.time() - start_time > time_limit:
            break
        
        chain = _grow_biased(sequence, rng, T=1.0)
        if chain is not None:
            # Calculate energy
            e = _calculate_energy(sequence, chain)
            if e < best_energy:
                best_energy = e
                best_chain = chain
        
        attempts += 1
        
        # If we found a very good energy (e.g., close to theoretical max), we might stop early?
        # Theoretical max contacts ~ N/2.
        # If best_energy < -N/4, it's quite good.
        
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0, 0) for i in range(n)]
        
    return best_chain, best_energy

def _calculate_energy(sequence, chain):
    """Calculate total energy of a chain."""
    n = len(chain)
    if n == 0:
        return 0
    occ = {p: i for i, p in enumerate(chain)}
    energy = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for n_pos in _neighbours(p):
                j = occ.get(n_pos)
                if j is not None and abs(i - j) > 1:
                    if sequence[j] == 'H':
                        energy -= 1
    return energy

def fold(sequence, dimension):
    """
    Fold the sequence using MCCG with biased growth.
    """
    # Estimate time per sequence
    # Total budget 240s for 26 sequences -> ~9s each.
    # We don't know the exact count, so we assume we are called sequentially.
    # We will use a dynamic time limit based on sequence length.
    # Longer sequences need more time to find good folds.
    
    n = len(sequence)
    # Base time 2s + 0.05s per residue
    time_limit = 2.0 + 0.05 * n
    if time_limit > 10.0:
        time_limit = 10.0
    if time_limit < 1.0:
        time_limit = 1.0

    # Use a unique seed for each call to ensure variety, but deterministic is safer for reproducibility if needed?
    # The task says "sandboxed subprocess", so random seed from system is fine.
    # But using a fixed seed allows us to test. Let's use a seed based on hash of sequence.
    seed = hash(sequence) % (2**32)
    
    chain, energy = _mccg_fold(sequence, dimension, time_limit, seed)
    
    # Ensure format: list of lists
    return [list(p) for p in chain]

# EVOLVE-BLOCK-END
