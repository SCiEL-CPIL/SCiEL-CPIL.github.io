"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np
from collections import defaultdict

def _primes_upto(n):
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n+1) if sieve[i]]

def _build_counts_np(residues, primes, max_L):
    """Build count array using numpy for speed."""
    counts = np.zeros(max_L + 1, dtype=np.int32)
    for p, r in zip(primes, residues):
        start = r if r >= 1 else r + p
        if start > max_L:
            continue
        idxs = np.arange(start, max_L + 1, p)
        counts[idxs] += 1
    return counts

def _find_prefix_length(counts, max_L):
    """Find prefix length from counts array."""
    L = 0
    for x in range(1, max_L + 1):
        if counts[x] >= 2:
            L = x
        else:
            break
    return L

def _apply_move(counts, residues, primes, idx, new_r, max_L):
    """Apply a residue change and update counts."""
    p = primes[idx]
    old_r = residues[idx]
    
    # Remove old coverage
    start_old = old_r if old_r >= 1 else old_r + p
    if start_old <= max_L:
        idxs_old = np.arange(start_old, max_L + 1, p)
        counts[idxs_old] -= 1
    
    # Add new coverage
    start_new = new_r if new_r >= 1 else new_r + p
    if start_new <= max_L:
        idxs_new = np.arange(start_new, max_L + 1, p)
        counts[idxs_new] += 1
    
    residues[idx] = new_r

def _undo_move(counts, residues, primes, idx, old_r, max_L):
    """Undo a residue change."""
    p = primes[idx]
    new_r = residues[idx]
    
    # Remove new coverage
    start_new = new_r if new_r >= 1 else new_r + p
    if start_new <= max_L:
        idxs_new = np.arange(start_new, max_L + 1, p)
        counts[idxs_new] -= 1
    
    # Add back old coverage
    start_old = old_r if old_r >= 1 else old_r + p
    if start_old <= max_L:
        idxs_old = np.arange(start_old, max_L + 1, p)
        counts[idxs_old] += 1
    
    residues[idx] = old_r

def _greedy_init(primes, max_L):
    """Greedy initialization: choose residue for each prime to maximize coverage of under-covered positions."""
    n_primes = len(primes)
    residues = [0] * n_primes
    counts = np.zeros(max_L + 1, dtype=np.int32)
    
    for idx in range(n_primes):
        p = primes[idx]
        best_r = 0
        best_metric = -1
        
        if p <= max_L:
            for r in range(p):
                start_x = r if r >= 1 else r + p
                if start_x > max_L:
                    gain = 0
                else:
                    idxs = np.arange(start_x, max_L + 1, p)
                    gain = np.sum(counts[idxs] < 2)
                if gain > best_metric:
                    best_metric = gain
                    best_r = r
        else:
            best_r = 0
            best_metric = -1
            for r in range(1, min(p, max_L + 1)):
                gain = 1 if counts[r] < 2 else 0
                if gain > best_metric:
                    best_metric = gain
                    best_r = r
            if best_metric <= 0:
                best_r = 0
        
        residues[idx] = best_r
        start_x = best_r if best_r >= 1 else best_r + p
        if start_x <= max_L:
            idxs = np.arange(start_x, max_L + 1, p)
            counts[idxs] += 1
    
    return residues, counts

def _simulated_annealing(residues, counts, primes, max_L, deadline, iterations_budget):
    """Run simulated annealing to improve the solution."""
    n_primes = len(primes)
    current_score = _find_prefix_length(counts, max_L)
    best_score = current_score
    best_residues = residues[:]
    best_counts = counts.copy()
    
    T_initial = 5.0
    T_final = 0.05
    cooling_rate = 0.9995
    T = T_initial
    
    iteration = 0
    while time.time() < deadline - 15.0 and iteration < iterations_budget:
        iteration += 1
        
        # Choose a prime to modify - bias toward small primes (more impact)
        if random.random() < 0.7:
            # Weighted toward small primes
            weights = [1.0 / primes[i] for i in range(n_primes)]
            total_w = sum(weights)
            r = random.random() * total_w
            cum = 0
            idx = 0
            for i in range(n_primes):
                cum += weights[i]
                if r <= cum:
                    idx = i
                    break
        else:
            idx = random.randint(0, n_primes - 1)
        
        p = primes[idx]
        old_r = residues[idx]
        new_r = random.randint(0, p - 1)
        
        if new_r == old_r:
            continue
        
        # Apply move
        _apply_move(counts, residues, primes, idx, new_r, max_L)
        new_score = _find_prefix_length(counts, max_L)
        
        # Acceptance
        if new_score > current_score:
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_residues = residues[:]
                best_counts = counts.copy()
        else:
            delta = new_score - current_score
            if delta >= 0 or (T > T_final and random.random() < math.exp(delta / T)):
                current_score = new_score
            else:
                _undo_move(counts, residues, primes, idx, old_r, max_L)
        
        T *= cooling_rate
        if T < T_final:
            T = T_final
    
    return best_residues, best_counts, best_score

def _local_search(residues, counts, primes, max_L, deadline):
    """Targeted local search to fix gaps."""
    n_primes = len(primes)
    current_score = _find_prefix_length(counts, max_L)
    best_score = current_score
    best_residues = residues[:]
    best_counts = counts.copy()
    
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        current_score = _find_prefix_length(counts, max_L)
        
        if current_score >= max_L:
            break
        
        gap_pos = current_score + 1
        
        # Find primes covering gap_pos
        covering = []
        for i in range(n_primes):
            p = primes[i]
            r = residues[i]
            if gap_pos % p == r % p:
                covering.append(i)
        
        if len(covering) < 2:
            # Try to change a non-covering prime to cover gap_pos
            candidates = []
            for i in range(n_primes):
                if i in covering:
                    continue
                p = primes[i]
                old_r = residues[i]
                new_r = gap_pos % p
                if new_r == old_r:
                    continue
                # Check loss
                start_old = old_r if old_r >= 1 else old_r + p
                loss = 0
                if start_old <= max_L:
                    idxs = np.arange(start_old, min(current_score, max_L) + 1, p)
                    loss = np.sum(counts[idxs] == 2)
                if loss == 0:
                    candidates.append((i, new_r))
            
            if candidates:
                # Pick the first candidate
                i, new_r = candidates[0]
                _apply_move(counts, residues, primes, i, new_r, max_L)
                improved = True
        
        if improved:
            new_score = _find_prefix_length(counts, max_L)
            if new_score > best_score:
                best_score = new_score
                best_residues = residues[:]
                best_counts = counts.copy()
    
    return best_residues, best_counts, best_score

def _pairwise_local_search(residues, counts, primes, max_L, deadline):
    """Try pairwise swaps to improve coverage."""
    n_primes = len(primes)
    current_score = _find_prefix_length(counts, max_L)
    best_score = current_score
    best_residues = residues[:]
    best_counts = counts.copy()
    
    while time.time() < deadline - 10.0:
        current_score = _find_prefix_length(counts, max_L)
        if current_score >= max_L:
            break
        
        # Try a few random pairwise changes
        found_improvement = False
        for _ in range(50):
            if time.time() > deadline - 10.0:
                break
            i = random.randint(0, n_primes - 1)
            j = random.randint(0, n_primes - 1)
            if i == j:
                continue
            
            p_i = primes[i]
            p_j = primes[j]
            old_ri = residues[i]
            old_rj = residues[j]
            
            # Try swapping residues
            new_ri = old_rj % p_i
            new_rj = old_ri % p_j
            
            if new_ri == old_ri and new_rj == old_rj:
                continue
            
            # Apply both moves
            _apply_move(counts, residues, primes, i, new_ri, max_L)
            _apply_move(counts, residues, primes, j, new_rj, max_L)
            
            new_score = _find_prefix_length(counts, max_L)
            
            if new_score > current_score:
                current_score = new_score
                found_improvement = True
                if current_score > best_score:
                    best_score = current_score
                    best_residues = residues[:]
                    best_counts = counts.copy()
            else:
                # Undo
                _undo_move(counts, residues, primes, j, new_rj, max_L)
                _undo_move(counts, residues, primes, i, new_ri, max_L)
        
        if not found_improvement:
            break
    
    return best_residues, best_counts, best_score

def _constraint_satisfaction_search(residues, counts, primes, max_L, deadline):
    """
    Structural change: Model as constraint satisfaction / matching problem.
    For each position x in 1..L, we need at least 2 primes covering it.
    We use a backtracking approach with constraint propagation to find valid assignments.
    This is a different optimization paradigm: exact search with pruning.
    """
    n_primes = len(primes)
    current_score = _find_prefix_length(counts, max_L)
    best_score = current_score
    best_residues = residues[:]
    best_counts = counts.copy()
    
    if current_score >= max_L:
        return best_residues, best_counts, best_score
    
    # Work on a smaller window for tractability
    window_start = 1
    window_end = min(current_score + 50, max_L)
    
    # For each position in the window, track which primes cover it
    pos_covered_by = defaultdict(list)  # pos -> list of prime indices covering it
    for x in range(window_start, window_end + 1):
        for i in range(n_primes):
            p = primes[i]
            r = residues[i]
            if x % p == r:
                pos_covered_by[x].append(i)
    
    # Find positions with < 2 coverage
    under_covered = [x for x in range(window_start, window_end + 1) if len(pos_covered_by[x]) < 2]
    
    if not under_covered:
        return best_residues, best_counts, best_score
    
    # Try to fix under-covered positions by changing residues
    # For each under-covered position, find a prime that could cover it
    # and check if changing that prime's residue would help
    
    improved = False
    for x in under_covered[:10]:  # Limit to first 10 under-covered positions
        if time.time() > deadline - 10.0:
            break
        
        # Find primes that could cover x (i.e., x % p is a valid residue)
        for i in range(n_primes):
            p = primes[i]
            new_r = x % p
            old_r = residues[i]
            
            if new_r == old_r:
                continue
            
            # Check if this prime already covers x
            if x % p == old_r:
                continue
            
            # Apply the change and check if it improves things
            _apply_move(counts, residues, primes, i, new_r, max_L)
            
            # Recalculate coverage for the window
            new_pos_covered_by = defaultdict(list)
            for xx in range(window_start, window_end + 1):
                for ii in range(n_primes):
                    pp = primes[ii]
                    rr = residues[ii]
                    if xx % pp == rr:
                        new_pos_covered_by[xx].append(ii)
            
            new_under_covered = [xx for xx in range(window_start, window_end + 1) if len(new_pos_covered_by[xx]) < 2]
            
            if len(new_under_covered) < len(under_covered):
                # Improvement - keep this change
                under_covered = new_under_covered
                improved = True
                new_score = _find_prefix_length(counts, max_L)
                if new_score > best_score:
                    best_score = new_score
                    best_residues = residues[:]
                    best_counts = counts.copy()
                break
            else:
                # Undo
                _undo_move(counts, residues, primes, i, old_r, max_L)
        
        if improved:
            break
    
    return best_residues, best_counts, best_score

def _genetic_algorithm(residues, counts, primes, max_L, deadline):
    """
    Structural change: Use a genetic algorithm with population-based search.
    Different representation: population of candidate solutions with crossover and mutation.
    """
    n_primes = len(primes)
    current_score = _find_prefix_length(counts, max_L)
    best_score = current_score
    best_residues = residues[:]
    best_counts = counts.copy()
    
    population_size = 20
    population = [residues[:]]
    
    # Initialize population with variations
    for _ in range(population_size - 1):
        candidate = residues[:]
        for _ in range(5):
            idx = random.randint(0, n_primes - 1)
            p = primes[idx]
            candidate[idx] = random.randint(0, p - 1)
        population.append(candidate)
    
    generation = 0
    while time.time() < deadline - 10.0 and generation < 50:
        generation += 1
        
        # Evaluate population
        scored_population = []
        for ind in population:
            counts_ind = _build_counts_np(ind, primes, max_L)
            score_ind = _find_prefix_length(counts_ind, max_L)
            scored_population.append((score_ind, ind, counts_ind))
        
        # Sort by score (descending)
        scored_population.sort(key=lambda x: x[0], reverse=True)
        
        # Update best
        if scored_population[0][0] > best_score:
            best_score = scored_population[0][0]
            best_residues = scored_population[0][1][:]
            best_counts = scored_population[0][2].copy()
        
        # Selection: keep top half
        survivors = [x[1] for x in scored_population[:population_size // 2]]
        
        # Create next generation
        next_population = survivors[:]
        
        while len(next_population) < population_size:
            if time.time() > deadline - 10.0:
                break
            
            # Crossover
            p1 = random.choice(survivors)
            p2 = random.choice(survivors)
            child = p1[:]
            for i in range(n_primes):
                if random.random() < 0.5:
                    child[i] = p2[i]
            
            # Mutation
            for _ in range(3):
                idx = random.randint(0, n_primes - 1)
                p = primes[idx]
                child[idx] = random.randint(0, p - 1)
            
            next_population.append(child)
        
        population = next_population
    
    return best_residues, best_counts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin
    
    primes = _primes_upto(1000)
    n_primes = len(primes)
    
    WORK_L = 2000
    
    best_overall_residues = None
    best_overall_score = 0
    
    # Multiple restarts with different strategies
    num_restarts = 4
    time_per_restart = (deadline - start_time - 20.0) / num_restarts
    
    strategies = [
        ('sa', _simulated_annealing),
        ('ls', _local_search),
        ('csp', _constraint_satisfaction_search),
        ('ga', _genetic_algorithm)
    ]
    
    for restart in range(num_restarts):
        if time.time() > deadline - 20.0:
            break
        
        # Greedy initialization
        residues, counts = _greedy_init(primes, WORK_L)
        score = _find_prefix_length(counts, WORK_L)
        
        # Choose strategy based on restart number
        strategy_name, strategy_func = strategies[restart % len(strategies)]
        
        if strategy_name == 'sa':
            sa_deadline = min(deadline - 25.0, start_time + time_per_restart * 0.6)
            iterations = int(200000 * (time_per_restart * 0.6) / 10.0)
            residues, counts, score = _simulated_annealing(residues, counts, primes, WORK_L, sa_deadline, iterations)
            
            # Follow up with local search
            ls_deadline = min(deadline - 20.0, start_time + time_per_restart * 0.9)
            residues, counts, score = _local_search(residues, counts, primes, WORK_L, ls_deadline)
            
        elif strategy_name == 'ls':
            ls_deadline = min(deadline - 20.0, start_time + time_per_restart * 0.7)
            residues, counts, score = _local_search(residues, counts, primes, WORK_L, ls_deadline)
            
            # Follow up with pairwise search
            pl_deadline = min(deadline - 15.0, start_time + time_per_restart * 0.95)
            residues, counts, score = _pairwise_local_search(residues, counts, primes, WORK_L, pl_deadline)
            
        elif strategy_name == 'csp':
            csp_deadline = min(deadline - 20.0, start_time + time_per_restart * 0.8)
            residues, counts, score = _constraint_satisfaction_search(residues, counts, primes, WORK_L, csp_deadline)
            
            # Follow up with local search
            ls_deadline = min(deadline - 15.0, start_time + time_per_restart * 0.95)
            residues, counts, score = _local_search(residues, counts, primes, WORK_L, ls_deadline)
            
        elif strategy_name == 'ga':
            ga_deadline = min(deadline - 20.0, start_time + time_per_restart * 0.8)
            residues, counts, score = _genetic_algorithm(residues, counts, primes, WORK_L, ga_deadline)
            
            # Follow up with local search
            ls_deadline = min(deadline - 15.0, start_time + time_per_restart * 0.95)
            residues, counts, score = _local_search(residues, counts, primes, WORK_L, ls_deadline)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_residues = residues[:]
    
    # Final local search on best solution
    if best_overall_residues is not None and time.time() < deadline - 10.0:
        counts = _build_counts_np(best_overall_residues, primes, WORK_L)
        final_deadline = deadline - 5.0
        best_overall_residues, _, _ = _local_search(best_overall_residues, counts, primes, WORK_L, final_deadline)
    
    if best_overall_residues is None:
        # Fallback
        best_overall_residues, _ = _greedy_init(primes, WORK_L)
    
    # Ensure all residues are Python ints
    return [int(r) for r in best_overall_residues]
# EVOLVE-BLOCK-END
