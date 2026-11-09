"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12
import time
import random
from collections import deque

# EVOLVE-BLOCK-START

def _precompute_transitions(a_map, b_map):
    """Precompute bitmask transition tables for O(1) subset transitions."""
    # For each letter, precompute the transition of every possible bitmask
    # trans_table[letter][mask] = new_mask
    # But 2^12 = 4096 masks, so we can precompute for all masks
    # Actually, we can compute on-the-fly more efficiently with precomputed per-state mappings
    
    # For each letter, create a mapping: mask -> new_mask
    # We'll use a dict for lazy computation or precompute all
    # Precompute all 4096 masks for each letter
    trans_tables = [[None] * (1 << N) for _ in range(2)]
    
    for letter in range(2):
        # For each state, what state does it go to?
        if letter == 0:
            state_map = a_map
        else:
            state_map = b_map
        
        # Precompute for all masks
        for mask in range(1 << N):
            new_mask = 0
            for s in range(N):
                if mask & (1 << s):
                    new_mask |= (1 << state_map[s])
            trans_tables[letter][mask] = new_mask
    
    return trans_tables

def _compute_threshold_bitmask(a_map, b_map):
    """Compute shortest reset word length using bitmask BFS with precomputed tables."""
    full_mask = (1 << N) - 1
    
    # Precompute transition tables
    trans_tables = _precompute_transitions(a_map, b_map)
    
    # BFS over subsets (represented as bitmasks)
    visited = set()
    visited.add(full_mask)
    queue = deque()
    queue.append((full_mask, 0))
    
    while queue:
        mask, dist = queue.popleft()
        
        # Check if this is a singleton (reset achieved)
        if mask & (mask - 1) == 0:  # power of 2
            return dist
        
        # Generate next states for letter 'a'
        new_mask_a = trans_tables[0][mask]
        if new_mask_a not in visited:
            visited.add(new_mask_a)
            queue.append((new_mask_a, dist + 1))
        
        # Generate next states for letter 'b'
        new_mask_b = trans_tables[1][mask]
        if new_mask_b not in visited:
            visited.add(new_mask_b)
            queue.append((new_mask_b, dist + 1))
    
    return -1  # Not synchronizing

def _pawlik_automaton():
    """Generate the Pawlik automaton structure."""
    a_map = [(i + 1) % N for i in range(N)]
    b_map = [0 if i == 0 else i - 1 for i in range(N)]
    return [a_map, b_map]

def _cycle_fold_automaton(k):
    """Generate a cycle+fold automaton variant.
    
    State 0..N-1. Letter 'a' is a cycle: a(i) = (i+1) % N
    Letter 'b' folds: b(0) = 0, b(i) = i-1 for i > 0, but with some states mapped to 0
    to create a "fold" that helps synchronization but delays it.
    """
    a_map = [(i + 1) % N for i in range(N)]
    b_map = [0] * N
    # Fold: for i from 1 to N-1, b(i) = i-1, but every k-th state maps to 0
    for i in range(1, N):
        if i % k == 0:
            b_map[i] = 0
        else:
            b_map[i] = i - 1
    return [a_map, b_map]

def _double_cycle_automaton(offset):
    """Generate a double-cycle automaton with two interleaved cycles."""
    # Split states into two groups: even and odd
    # a: even states cycle among evens, odd states cycle among odds
    # b: maps each state to the corresponding state in the other group, shifted
    a_map = [0] * N
    b_map = [0] * N
    
    evens = [i for i in range(N) if i % 2 == 0]
    odds = [i for i in range(N) if i % 2 == 1]
    
    for i, s in enumerate(evens):
        a_map[s] = evens[(i + 1) % len(evens)]
    for i, s in enumerate(odds):
        a_map[s] = odds[(i + 1) % len(odds)]
    
    # b: map even to odd and vice versa with offset
    for i, s in enumerate(evens):
        b_map[s] = odds[(i + offset) % len(odds)]
    for i, s in enumerate(odds):
        b_map[s] = evens[(i + offset) % len(evens)]
    
    return [a_map, b_map]

def _perturbed_cycle_automaton(perturbations):
    """Generate a perturbed cycle automaton.
    
    Start with a cycle for 'a', and 'b' is mostly a shift but with perturbations.
    """
    a_map = [(i + 1) % N for i in range(N)]
    b_map = [(i + 1) % N for i in range(N)]  # Start with same cycle
    
    # Apply perturbations: for each (state, new_dest), change b[state]
    for state, new_dest in perturbations:
        b_map[state] = new_dest
    
    return [a_map, b_map]

def _local_search(initial_automaton, max_steps=5000, time_limit=50.0):
    """Perform local search to improve the reset threshold."""
    start_time = time.time()
    best_a = initial_automaton[0][:]
    best_b = initial_automaton[1][:]
    best_score = _compute_threshold_bitmask(best_a, best_b)
    
    if best_score < 0:
        return None
    
    current_a = best_a[:]
    current_b = best_b[:]
    current_score = best_score
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break
        
        # Random perturbation: change one state's transition for one letter
        letter = random.choice([0, 1])
        state = random.randint(0, N - 1)
        old_val = current_a[state] if letter == 0 else current_b[state]
        new_val = random.randint(0, N - 1)
        
        if letter == 0:
            current_a[state] = new_val
        else:
            current_b[state] = new_val
        
        new_score = _compute_threshold_bitmask(current_a, current_b)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_a = current_a[:]
                best_b = current_b[:]
        else:
            # Revert
            if letter == 0:
                current_a[state] = old_val
            else:
                current_b[state] = old_val
    
    return [best_a, best_b] if best_score > 0 else None

def _simulated_annealing(time_budget):
    """Simulated annealing search for high-threshold automata."""
    start_time = time.time()
    
    # Start from Pawlik
    current = _pawlik_automaton()
    current_score = _compute_threshold_bitmask(current[0], current[1])
    
    best = [current[0][:], current[1][:]]
    best_score = current_score
    
    temperature = 5.0
    cooling = 0.999
    
    while time.time() - start_time < time_budget:
        # Perturb
        letter = random.choice([0, 1])
        state = random.randint(0, N - 1)
        old_val = current[letter][state]
        new_val = random.randint(0, N - 1)
        current[letter][state] = new_val
        
        new_score = _compute_threshold_bitmask(current[0], current[1])
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best[0] = current[0][:]
                best[1] = current[1][:]
        elif new_score >= 0:
            # Accept worse with probability
            delta = new_score - current_score
            if random.random() < min(1.0, temperature / max(1, abs(delta))):
                current_score = new_score
        else:
            # Not synchronizing - revert
            current[letter][state] = old_val
        
        temperature *= cooling
        if temperature < 0.01:
            break
    
    return [best[0], best[1]] if best_score > 0 else None

def _systematic_search_structures(time_budget):
    """Search through a family of structured automata."""
    start_time = time.time()
    best = None
    best_score = -1
    
    # Try cycle+fold variants
    for k in range(2, N + 1):
        if time.time() - start_time > time_budget:
            break
        auto = _cycle_fold_automaton(k)
        score = _compute_threshold_bitmask(auto[0], auto[1])
        if score > best_score:
            best_score = score
            best = auto[:]
    
    # Try double-cycle variants
    for offset in range(N):
        if time.time() - start_time > time_budget:
            break
        auto = _double_cycle_automaton(offset)
        score = _compute_threshold_bitmask(auto[0], auto[1])
        if score > best_score:
            best_score = score
            best = auto[:]
    
    # Try perturbed cycles with random perturbations
    for _ in range(200):
        if time.time() - start_time > time_budget:
            break
        # Randomly perturb 1-3 states
        num_perts = random.randint(1, 3)
        perturbations = []
        for _ in range(num_perts):
            state = random.randint(0, N - 1)
            new_dest = random.randint(0, N - 1)
            perturbations.append((state, new_dest))
        auto = _perturbed_cycle_automaton(perturbations)
        score = _compute_threshold_bitmask(auto[0], auto[1])
        if score > best_score:
            best_score = score
            best = auto[:]
    
    return [best[0], best[1]] if best_score > 0 else None

def construct():
    """
    Constructs a synchronizing automaton with high reset threshold.
    Uses systematic structure search, simulated annealing, and local search.
    """
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    
    # Start with Pawlik as baseline
    pawlik = _pawlik_automaton()
    pawlik_score = _compute_threshold_bitmask(pawlik[0], pawlik[1])
    
    best = [pawlik[0][:], pawlik[1][:]]
    best_score = pawlik_score
    
    # Try systematic structure search
    struct_budget = 20.0
    if time.time() + struct_budget < deadline:
        struct_result = _systematic_search_structures(struct_budget)
        if struct_result:
            struct_score = _compute_threshold_bitmask(struct_result[0], struct_result[1])
            if struct_score > best_score:
                best = struct_result
                best_score = struct_score
    
    # Try simulated annealing with time budget
    sa_budget = 30.0
    if time.time() + sa_budget < deadline:
        sa_result = _simulated_annealing(sa_budget)
        if sa_result:
            sa_score = _compute_threshold_bitmask(sa_result[0], sa_result[1])
            if sa_score > best_score:
                best = sa_result
                best_score = sa_score
    
    # Try local search from best found
    ls_budget = 25.0
    if time.time() + ls_budget < deadline:
        ls_result = _local_search(best, max_steps=20000, time_limit=ls_budget)
        if ls_result:
            ls_score = _compute_threshold_bitmask(ls_result[0], ls_result[1])
            if ls_score > best_score:
                best = ls_result
                best_score = ls_score
    
    # Final check
    final_score = _compute_threshold_bitmask(best[0], best[1])
    if final_score <= 0:
        # Fallback to Pawlik if something went wrong
        return pawlik
    
    return best
# EVOLVE-BLOCK-END
