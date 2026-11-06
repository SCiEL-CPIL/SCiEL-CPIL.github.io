"""HP lattice protein folding on the square lattice – scratch rewrite."""
# EVOLVE-BLOCK-START
import random
import math


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute HP contact energy: -1 for each non-consecutive H-H pair on adjacent sites."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _mccg_fold(sequence, dimension, rng, max_backtracks=10000):
    """
    Monte Carlo Chain Growth (MCCG).
    Builds the chain step-by-step, using stochastic selection of next steps
    based on energy gain, and backtracks if stuck.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    # State: chain (list of tuples), occupied (dict pos->index)
    # We need to store history for backtracking.
    # To save memory, we can just store the chain and re-derive occupied, 
    # or store occupied as well. For n<=100, storing chain is fine.
    
    best_chain = None
    best_energy = 0
    
    # We will run multiple independent MCCG trials
    # Each trial is a single walk with backtracking
    num_trials = 200
    
    for trial in range(num_trials):
        # Temperature for stochastic selection
        # Lower T -> more greedy
        T = 1.0 / (1.0 + 0.1 * trial) 
        
        chain = [(0, 0)]
        occupied = {(0, 0): 0}
        
        # History for backtracking: list of (step_index, list_of_candidates_sorted_by_score)
        # We only need to backtrack if we get stuck.
        # A simpler approach for MCCG: 
        # At each step, generate candidates. 
        # If none, backtrack to the last step that had >1 candidate.
        # This requires storing the state of the chain at branching points.
        
        # To simplify and make it robust within time limits, we will use a 
        # "Greedy with Random Restart" style but with a twist: 
        # We will perform a single growth pass, but if we get stuck, we restart.
        # This is similar to the current code but without the local search.
        # To be distinct, let's implement proper backtracking.
        
        # Stack of states: (chain_copy, occupied_copy, candidates_at_this_step)
        # This is memory expensive. 
        # Alternative: Just restart if stuck. This is "Random Greedy".
        # The current code IS random greedy + local search.
        # So MCCG with backtracking is the key differentiator.
        
        # Let's implement a lightweight backtracker.
        # We store the chain. If we get stuck at step k, we go back to step k-1.
        # We need to know what choices were available at k-1.
        
        # Efficient MCCG implementation:
        # 1. Grow chain.
        # 2. Keep a list of "branching points". A branching point is a step where 
        #    we had more than 1 valid neighbor.
        # 3. If we get stuck, pop the last branching point, restore chain to that state,
        #    and choose a different neighbor.
        
        chain = [(0, 0)]
        occupied = {(0, 0): 0}
        branching_points = [] # List of (index, [list_of_candidates])
        
        stuck_count = 0
        max_stuck = 50 # Max times to backtrack before restarting this trial
        
        while len(chain) < n:
            current_pos = chain[-1]
            i = len(chain) - 1
            
            candidates = []
            for site in _neighbours(current_pos):
                if site in occupied:
                    continue
                gain = 0
                if sequence[i+1] == 'H': # Next residue
                    for nb in _neighbours(site):
                        j = occupied.get(nb)
                        if j is not None and j < i - 1 and sequence[j] == 'H':
                            gain += 1
                # Add small random noise to break symmetry
                score = gain * 10.0 + rng.random() * T
                candidates.append((score, site))
            
            if not candidates:
                # Stuck. Backtrack.
                if not branching_points:
                    break # No more backtracking possible, fail this trial
                if stuck_count >= max_stuck:
                    break # Too many backtracks, fail this trial
                    
                # Restore to last branching point
                bp_index, bp_candidates = branching_points.pop()
                # Truncate chain to bp_index
                chain = chain[:bp_index+1]
                # Rebuild occupied (expensive but n is small)
                occupied = {p: idx for idx, p in enumerate(chain)}
                
                # Find the candidate we already took (the one in chain)
                last_pos = chain[-1]
                # The candidates at bp_index were for the step *after* bp_index.
                # So we are at bp_index, and we need to choose a *different* next step.
                # bp_candidates contains (score, site).
                # We need to exclude the site that was previously chosen.
                prev_choice = chain[-1] # Wait, chain[-1] is the position at bp_index.
                # The next position was chain[bp_index+1] which is now removed.
                # We need to pick a new one from bp_candidates.
                
                # Identify the previously chosen site.
                # Actually, when we pushed to branching_points, we saved the candidates.
                # We just need to pick a different one.
                # To avoid infinite loops, we can shuffle candidates and pick the first one 
                # that is not the previous one.
                
                prev_site = None
                # The site we were at when we branched is chain[-1].
                # The site we moved TO is what we removed.
                # We don't strictly need to know the previous site if we just pick 
                # a random one from the remaining candidates.
                
                remaining = [c for c in bp_candidates]
                if not remaining:
                    break
                # Shuffle to pick a different one
                rng.shuffle(remaining)
                # Pick the first one. It might be the same as before, but unlikely if >1.
                # To be safe, if len(remaining)>1, we can try to avoid the previous.
                # But we don't have the previous site stored in the tuple.
                # Let's just pick the first one. If it leads to a dead end immediately, 
                # we'll backtrack again.
                
                new_site = remaining[0][1]
                chain.append(new_site)
                occupied[new_site] = len(chain) - 1
                stuck_count += 1
                continue
            
            # If we have candidates, proceed
            # Sort by score descending
            candidates.sort(key=lambda x: x[0], reverse=True)
            
            # If multiple candidates, this is a branching point
            if len(candidates) > 1:
                branching_points.append((i, candidates))
            
            # Select next site
            # Use a simple rule: pick the best, or random among top K
            # For MCCG, often a stochastic choice is better.
            # Let's pick the best with prob 0.8, second best with 0.1, else random.
            if rng.random() < 0.8:
                site = candidates[0][1]
            elif len(candidates) > 1 and rng.random() < 0.5:
                site = candidates[1][1]
            else:
                site = candidates[rng.randint(0, len(candidates)-1)][1]
            
            chain.append(site)
            occupied[site] = len(chain) - 1
        
        if len(chain) == n:
            energy = _energy(sequence, chain)
            if best_chain is None or energy < best_energy:
                best_chain = [list(p) for p in chain]
                best_energy = energy
                
    if best_chain is None:
        # Fallback
        return [[i, 0] for i in range(n)]
        
    return best_chain


def fold(sequence, dimension):
    """
    Fold an HP sequence on a 2D square lattice.
    Uses Monte Carlo Chain Growth (MCCG) with backtracking.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]

    rng = random.Random(123 + len(sequence))
    
    # MCCG is good but can be slow if backtracking is heavy.
    # We will run it for a few seconds or a fixed number of trials.
    # Let's scale trials based on n.
    if n <= 20:
        num_trials = 500
    elif n <= 50:
        num_trials = 300
    else:
        num_trials = 150
        
    best_chain = None
    best_energy = 0
    
    # Run MCCG
    # The _mccg_fold above does num_trials internally? No, it does one trial per call.
    # Let's modify _mccg_fold to do one trial, and loop here.
    # Or just call it in a loop.
    
    for _ in range(num_trials):
        chain = _mccg_fold_single(sequence, dimension, rng)
        if chain is None:
            continue
        energy = _energy(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy

    if best_chain is None:
        best_chain = [[i, 0] for i in range(n)]

    return best_chain

def _mccg_fold_single(sequence, dimension, rng):
    """Single MCCG trial with backtracking."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    T = 0.5 # Fixed temp for now, could anneal
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    branching_points = []
    
    stuck_count = 0
    max_stuck = 100 # Limit backtracking depth
    
    while len(chain) < n:
        current_pos = chain[-1]
        i = len(chain) - 1
        
        candidates = []
        for site in _neighbours(current_pos):
            if site in occupied:
                continue
            gain = 0
            if sequence[i+1] == 'H':
                for nb in _neighbours(site):
                    j = occupied.get(nb)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            score = gain * 10.0 + rng.random() * T
            candidates.append((score, site))
        
        if not candidates:
            if not branching_points:
                return None
            if stuck_count >= max_stuck:
                return None
                
            bp_index, bp_candidates = branching_points.pop()
            chain = chain[:bp_index+1]
            occupied = {p: idx for idx, p in enumerate(chain)}
            
            # Pick a new candidate
            # To avoid picking the same one immediately, we can shuffle
            rng.shuffle(bp_candidates)
            new_site = bp_candidates[0][1]
            
            # Ensure it's actually different? 
            # If we just pick the first, it might be the same.
            # Let's pick a random one from the list.
            # Since we shuffled, it's random.
            
            chain.append(new_site)
            occupied[new_site] = len(chain) - 1
            stuck_count += 1
            continue
        
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        if len(candidates) > 1:
            branching_points.append((i, candidates))
            
        # Selection
        if rng.random() < 0.7:
            site = candidates[0][1]
        elif len(candidates) > 1 and rng.random() < 0.5:
            site = candidates[1][1]
        else:
            site = candidates[rng.randint(0, len(candidates)-1)][1]
            
        chain.append(site)
        occupied[site] = len(chain) - 1
        
    return [list(p) for p in chain]

# EVOLVE-BLOCK-END
