"""Constraint satisfaction & bottleneck repair for double coverage."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

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
        r = int(r)
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
    old_r = int(residues[idx])
    new_r = int(new_r)
    
    start_old = old_r if old_r >= 1 else old_r + p
    if start_old <= max_L:
        idxs_old = np.arange(start_old, max_L + 1, p)
        counts[idxs_old] -= 1
    
    start_new = new_r if new_r >= 1 else new_r + p
    if start_new <= max_L:
        idxs_new = np.arange(start_new, max_L + 1, p)
        counts[idxs_new] += 1
    
    residues[idx] = new_r

def _undo_move(counts, residues, primes, idx, old_r, max_L):
    """Undo a residue change."""
    p = primes[idx]
    new_r = int(residues[idx])
    old_r = int(old_r)
    
    start_new = new_r if new_r >= 1 else new_r + p
    if start_new <= max_L:
        idxs_new = np.arange(start_new, max_L + 1, p)
        counts[idxs_new] -= 1
    
    start_old = old_r if old_r >= 1 else old_r + p
    if start_old <= max_L:
        idxs_old = np.arange(start_old, max_L + 1, p)
        counts[idxs_old] += 1
    
    residues[idx] = old_r

def _greedy_density_init(primes, max_L):
    """
    Phase 1: Greedy density maximization.
    For each prime in order, pick the residue that covers the most
    currently under-covered (count < 2) positions in [1, max_L].
    """
    n_primes = len(primes)
    residues = [0] * n_primes
    counts = np.zeros(max_L + 1, dtype=np.int32)
    
    for idx in range(n_primes):
        p = primes[idx]
        best_r = 0
        best_gain = -1
        
        for r in range(p):
            start_x = r if r >= 1 else r + p
            if start_x > max_L:
                gain = 0
            else:
                idxs = np.arange(start_x, max_L + 1, p)
                gain = int(np.sum(counts[idxs] < 2))
            if gain > best_gain:
                best_gain = gain
                best_r = r
        
        residues[idx] = int(best_r)
        start_x = best_r if best_r >= 1 else best_r + p
        if start_x <= max_L:
            idxs = np.arange(start_x, max_L + 1, p)
            counts[idxs] += 1
    
    return residues, counts

def _bottleneck_repair(residues, counts, primes, max_L, deadline):
    """
    Phase 2: Iterative bottleneck repair.
    Repeatedly find the first position with count < 2, and try to
    reassign a prime to cover it without reducing coverage elsewhere.
    """
    n_primes = len(primes)
    best_score = _find_prefix_length(counts, max_L)
    best_residues = residues[:]
    best_counts = counts.copy()
    
    while time.time() < deadline:
        current_score = _find_prefix_length(counts, max_L)
        if current_score >= max_L:
            break
        
        gap_pos = current_score + 1
        if gap_pos > max_L:
            break
        
        # Find which primes currently cover gap_pos
        covering = []
        for i in range(n_primes):
            p = primes[i]
            r = int(residues[i])
            if gap_pos % p == r % p:
                covering.append(i)
        
        if len(covering) >= 2:
            # Already double-covered; shouldn't happen if count < 2
            break
        
        # Try to find a prime to reassign to cover gap_pos
        # Priority: primes that don't cover gap_pos, prefer large primes
        # (less impact on other positions)
        candidates = []
        for i in range(n_primes):
            if i in covering:
                continue
            p = primes[i]
            new_r = gap_pos % p
            old_r = int(residues[i])
            if new_r == old_r:
                continue
            
            # Compute the "cost": how many positions in [1, current_score]
            # would go from count==2 to count==1 if we remove old coverage
            start_old = old_r if old_r >= 1 else old_r + p
            loss = 0
            if start_old <= max_L:
                end_check = min(current_score, max_L)
                if end_check >= start_old:
                    idxs = np.arange(start_old, end_check + 1, p)
                    loss = int(np.sum(counts[idxs] == 2))
            
            # Also check: does the new position add coverage to any
            # currently under-covered spots in [1, current_score]?
            start_new = new_r if new_r >= 1 else new_r + p
            gain_in_prefix = 0
            if start_new <= max_L:
                end_check = min(current_score, max_L)
                if end_check >= start_new:
                    idxs = np.arange(start_new, end_check + 1, p)
                    gain_in_prefix = int(np.sum(counts[idxs] < 2))
            
            # Net effect on prefix: gain - loss
            net = gain_in_prefix - loss
            candidates.append((net, -p, i, new_r))
        
        if not candidates:
            break
        
        # Sort by net benefit (desc), then by prime size (desc)
        candidates.sort(key=lambda x: (-x[0], x[1]))
        
        # Try the best candidate
        net, neg_p, i, new_r = candidates[0]
        
        _apply_move(counts, residues, primes, i, new_r, max_L)
        
        new_score = _find_prefix_length(counts, max_L)
        if new_score > best_score:
            best_score = new_score
            best_residues = residues[:]
            best_counts = counts.copy()
        elif new_score < current_score:
            # Reverted the prefix; undo
            _undo_move(counts, residues, primes, i, new_r, max_L)
            # Try next candidate
            found = False
            for net2, neg_p2, i2, new_r2 in candidates[1:]:
                if net2 >= 0:
                    _apply_move(counts, residues, primes, i2, new_r2, max_L)
                    new_score2 = _find_prefix_length(counts, max_L)
                    if new_score2 > best_score:
                        best_score = new_score2
                        best_residues = residues[:]
                        best_counts = counts.copy()
                    elif new_score2 < current_score:
                        _undo_move(counts, residues, primes, i2, new_r2, max_L)
                    found = True
                    break
            if not found:
                break
    
    return best_residues, best_counts, best_score

def _targeted_local_search(residues, counts, primes, max_L, deadline):
    """
    Phase 3: Targeted local search on primes covering the bottleneck.
    Only modify primes that cover positions near the current prefix end.
    """
    n_primes = len(primes)
    best_score = _find_prefix_length(counts, max_L)
    best_residues = residues[:]
    best_counts = counts.copy()
    
    while time.time() < deadline:
        current_score = _find_prefix_length(counts, max_L)
        if current_score >= max_L:
            break
        
        gap_pos = current_score + 1
        if gap_pos > max_L:
            break
        
        # Find primes covering gap_pos
        covering = []
        for i in range(n_primes):
            p = primes[i]
            r = int(residues[i])
            if gap_pos % p == r % p:
                covering.append(i)
        
        if len(covering) >= 2:
            break
        
        # Also consider primes that cover positions in [gap_pos, gap_pos+50]
        extended_candidates = []
        for i in range(n_primes):
            if i in covering:
                continue
            p = primes[i]
            r = int(residues[i])
            next_pos = gap_pos + ((r - gap_pos) % p)
            if next_pos <= gap_pos + 50:
                extended_candidates.append(i)
        
        all_candidates = list(set(extended_candidates + covering))
        
        improved = False
        for i in all_candidates:
            p = primes[i]
            old_r = int(residues[i])
            new_r = gap_pos % p
            if new_r == old_r:
                continue
            
            _apply_move(counts, residues, primes, i, new_r, max_L)
            new_score = _find_prefix_length(counts, max_L)
            
            if new_score > current_score:
                current_score = new_score
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_residues = residues[:]
                    best_counts = counts.copy()
            else:
                _undo_move(counts, residues, primes, i, new_r, max_L)
        
        if not improved:
            # Try random perturbations on non-bottleneck primes
            found = False
            for _ in range(20):
                if time.time() > deadline:
                    break
                i = random.randint(0, n_primes - 1)
                p = primes[i]
                old_r = int(residues[i])
                new_r = random.randint(0, p - 1)
                if new_r == old_r:
                    continue
                
                _apply_move(counts, residues, primes, i, new_r, max_L)
                new_score = _find_prefix_length(counts, max_L)
                
                if new_score > current_score:
                    current_score = new_score
                    found = True
                    if current_score > best_score:
                        best_score = current_score
                        best_residues = residues[:]
                        best_counts = counts.copy()
                else:
                    _undo_move(counts, residues, primes, i, new_r, max_L)
            
            if not found:
                break
    
    return best_residues, best_counts, best_score

def _simulated_annealing_focused(residues, counts, primes, max_L, deadline):
    """
    Phase 4: Simulated annealing focused on small primes and bottleneck region.
    """
    n_primes = len(primes)
    current_score = _find_prefix_length(counts, max_L)
    best_score = current_score
    best_residues = residues[:]
    best_counts = counts.copy()
    
    T = 3.0
    T_final = 0.1
    cooling = 0.999
    
    while time.time() < deadline:
        # Focus on small primes (first 30) and primes near bottleneck
        if random.random() < 0.6:
            idx = random.randint(0, min(29, n_primes - 1))
        else:
            idx = random.randint(0, n_primes - 1)
        
        p = primes[idx]
        old_r = int(residues[idx])
        new_r = random.randint(0, p - 1)
        
        if new_r == old_r:
            continue
        
        _apply_move(counts, residues, primes, idx, new_r, max_L)
        new_score = _find_prefix_length(counts, max_L)
        
        delta = new_score - current_score
        if delta >= 0 or (T > T_final and random.random() < math.exp(delta / T)):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_residues = residues[:]
                best_counts = counts.copy()
        else:
            _undo_move(counts, residues, primes, idx, old_r, max_L)
        
        T *= cooling
        if T < T_final:
            T = T_final
    
    return best_residues, best_counts, best_score

def _smart_init(primes, max_L):
    """
    Smarter initialization: prioritize small primes with residues that
    maximize coverage of early positions [1, min(max_L, 200)].
    """
    n_primes = len(primes)
    residues = [0] * n_primes
    counts = np.zeros(max_L + 1, dtype=np.int32)
    
    # Focus window for early positions
    focus_end = min(max_L, 200)
    
    for idx in range(n_primes):
        p = primes[idx]
        best_r = 0
        best_score = -1
        
        for r in range(p):
            start_x = r if r >= 1 else r + p
            if start_x > max_L:
                score = 0
            else:
                # Count how many positions in [1, focus_end] get covered
                # and prioritize positions that are currently under-covered
                if start_x <= focus_end:
                    idxs_focus = np.arange(start_x, focus_end + 1, p)
                    # Weight: more weight for under-covered positions
                    under = int(np.sum(counts[idxs_focus] < 2))
                    total = len(idxs_focus)
                    score = under * 2 + total
                else:
                    score = 0
                # Also consider full range for larger primes
                if start_x <= max_L:
                    idxs_full = np.arange(start_x, max_L + 1, p)
                    score += int(np.sum(counts[idxs_full] < 2)) * 0.1
            
            if score > best_score:
                best_score = score
                best_r = r
        
        residues[idx] = int(best_r)
        start_x = best_r if best_r >= 1 else best_r + p
        if start_x <= max_L:
            idxs = np.arange(start_x, max_L + 1, p)
            counts[idxs] += 1
    
    return residues, counts

def construct():
    start_time = time.time()
    deadline = start_time + 143.0  # 7s safety margin
    
    primes = _primes_upto(1000)
    n_primes = len(primes)
    
    WORK_L = 3000
    
    best_overall_residues = None
    best_overall_score = 0
    
    # Multiple restarts with different strategies
    num_restarts = 4
    time_per_restart = (deadline - start_time - 10.0) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        phase1_deadline = start_time + time_per_restart * 0.15
        phase2_deadline = start_time + time_per_restart * 0.45
        phase3_deadline = start_time + time_per_restart * 0.75
        phase4_deadline = min(deadline - 5.0, start_time + time_per_restart * 0.95)
        
        # Alternate between greedy density and smart init
        if restart % 2 == 0:
            residues, counts = _greedy_density_init(primes, WORK_L)
        else:
            residues, counts = _smart_init(primes, WORK_L)
        score = _find_prefix_length(counts, WORK_L)
        
        # Phase 2: Bottleneck repair
        residues, counts, score = _bottleneck_repair(residues, counts, primes, WORK_L, phase2_deadline)
        
        # Phase 3: Targeted local search
        residues, counts, score = _targeted_local_search(residues, counts, primes, WORK_L, phase3_deadline)
        
        # Phase 4: Focused simulated annealing
        residues, counts, score = _simulated_annealing_focused(residues, counts, primes, WORK_L, phase4_deadline)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_residues = residues[:]
    
    # Final polish on best solution
    if best_overall_residues is not None and time.time() < deadline - 5.0:
        counts = _build_counts_np(best_overall_residues, primes, WORK_L)
        final_deadline = deadline - 3.0
        best_overall_residues, _, _ = _targeted_local_search(best_overall_residues, counts, primes, WORK_L, final_deadline)
    
    if best_overall_residues is None:
        # Fallback
        best_overall_residues, _ = _greedy_density_init(primes, WORK_L)
    
    # Ensure all residues are Python ints
    return [int(r) for r in best_overall_residues]
# EVOLVE-BLOCK-END
