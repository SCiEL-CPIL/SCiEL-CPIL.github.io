"""
Structural Change: Conflict-Graph Based Multi-Move Repair.

Instead of sequential single-move local search, this approach identifies the
bottleneck gap and constructs a conflict graph of candidate repairs. It then
searches for a set of compatible moves (an independent set in the conflict graph,
or a clique in the compatibility graph) that collectively fix the gap without
introducing new gaps in the prefix.
"""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i in range(n + 1) if sieve[i]]

_PRIMES = _primes_up_to(1000)
_NPRIMES = len(_PRIMES)

def _build_count_np(residues, L_MAX):
    """Build count array (numpy) for x in [1, L_MAX]."""
    count = np.zeros(L_MAX + 1, dtype=np.int32)
    for i, p in enumerate(_PRIMES):
        r = residues[i]
        start = p if r == 0 else r
        if start <= L_MAX:
            count[start:L_MAX+1:p] += 1
    return count

def _find_prefix_L_np(count, L_MAX):
    """Find largest L such that count[x] >= 2 for all x in [1..L] using numpy."""
    if L_MAX <= 0:
        return 0
    below = np.nonzero(count[1:L_MAX+1] < 2)[0]
    if len(below) == 0:
        return L_MAX
    return int(below[0])

def _score(residues):
    """Compute the actual score (largest L with full double coverage)."""
    L_MAX = 100000
    count = _build_count_np(residues, L_MAX)
    return _find_prefix_L_np(count, L_MAX)

def _greedy_construct(deadline):
    """
    Greedy construction: fix small primes for dense coverage, then repair gaps.
    """
    residues = [0] * _NPRIMES
    L_MAX = 5000
    count = np.zeros(L_MAX + 1, dtype=np.int32)
    
    # Phase A: Assign small primes (p <= 100) to maximize coverage in [1, 2000]
    small_prime_indices = [i for i in range(_NPRIMES) if _PRIMES[i] <= 100]
    
    for idx in small_prime_indices:
        if time.time() > deadline:
            break
        p = _PRIMES[idx]
        best_r = 0
        best_score = -1
        
        for r in range(p):
            start = p if r == 0 else r
            if start > 2000:
                continue
            seg = count[start:2001:p]
            gain = int(np.sum(seg <= 1)) - 0.1 * int(np.sum(seg == 2))
            if gain > best_score:
                best_score = gain
                best_r = r
        
        residues[idx] = best_r
        start = p if best_r == 0 else best_r
        if start <= L_MAX:
            count[start:L_MAX+1:p] += 1
    
    # Phase B: For larger primes, use coverage deficit heuristic
    large_prime_indices = [i for i in range(_NPRIMES) if _PRIMES[i] > 100]
    
    current_L = _find_prefix_L_np(count, L_MAX)
    
    for idx in large_prime_indices:
        if time.time() > deadline:
            break
        p = _PRIMES[idx]
        gap = current_L + 1
        if gap > L_MAX:
            break
        
        candidates = set()
        candidates.add(gap % p)
        for dx in range(-3, 4):
            x = gap + dx
            if x >= 1:
                candidates.add(x % p)
        candidates.add(0)
        candidates.add(1)
        
        best_r = residues[idx]
        best_gain = -1
        
        for r in candidates:
            if r >= p:
                continue
            start = p if r == 0 else r
            if start > L_MAX:
                continue
            seg = count[start:L_MAX+1:p]
            gain = int(np.sum(seg <= 1))
            if gap % p == r:
                gain += 50
            
            if gain > best_gain:
                best_gain = gain
                best_r = r
        
        residues[idx] = best_r
        start = p if best_r == 0 else best_r
        if start <= L_MAX:
            count[start:L_MAX+1:p] += 1
        
        current_L = _find_prefix_L_np(count, L_MAX)
    
    return residues, current_L

def _find_compatible_moves(residues, count, gap, L_MAX):
    """
    Identify a set of compatible moves to fix the gap at 'gap'.
    Returns a list of (prime_idx, new_residue) tuples.
    """
    # 1. Identify candidate moves: primes that can cover 'gap'
    candidates = []
    for i, p in enumerate(_PRIMES):
        desired_r = gap % p
        if desired_r != residues[i]:
            candidates.append((i, desired_r))
    
    if not candidates:
        return []
    
    # 2. Evaluate safety of each move
    # A move (i, r_new) is "safe" if changing residue i to r_new does not
    # reduce the count of any x in [1, gap-1] below 2.
    
    safe_moves = []
    unsafe_moves = []
    
    for i, r_new in candidates:
        p = _PRIMES[i]
        r_old = residues[i]
        
        old_start = p if r_old == 0 else r_old
        new_start = p if r_new == 0 else r_new
        
        # Determine positions affected by the change
        # Positions losing coverage: x where x % p == r_old and x % p != r_new
        # Positions gaining coverage: x where x % p == r_new and x % p != r_old
        
        # Since r_old != r_new, the sets are disjoint within the arithmetic progression.
        
        # Check loss: For all x in [1, gap-1] covered by old residue, count[x] must be >= 3
        # (because it will drop by 1, so it needs to stay >= 2)
        
        # Optimize: Only check if old_start <= gap-1
        is_safe = True
        if old_start <= gap - 1:
            # Find min count in the old coverage range up to gap-1
            end_check = gap
            min_count = int(np.min(count[old_start:end_check:p]))
            if min_count < 3:
                is_safe = False
        
        if is_safe:
            safe_moves.append((i, r_new))
        else:
            unsafe_moves.append((i, r_new))
    
    # 3. Try to fix gap using only safe moves
    # We need enough safe moves to bring count[gap] to >= 2.
    current_gap_count = count[gap]
    needed = max(0, 2 - current_gap_count)
    
    if len(safe_moves) >= needed:
        # Select the 'needed' safe moves that provide the best secondary benefit
        # Benefit: covering other low-count points in [1, L_MAX]
        scored_safe = []
        for i, r_new in safe_moves:
            p = _PRIMES[i]
            new_start = p if r_new == 0 else r_new
            if new_start > L_MAX:
                gain = 0
            else:
                seg = count[new_start:L_MAX+1:p]
                gain = int(np.sum(seg <= 1))
            scored_safe.append((gain, i, r_new))
        
        scored_safe.sort(key=lambda x: -x[0])
        return [(i, r) for _, i, r in scored_safe[:needed]]
    
    # 4. If not enough safe moves, try to combine unsafe moves.
    # An unsafe move breaks coverage at some x. We need another move to repair x.
    # This is a small-scale constraint satisfaction problem.
    
    # For simplicity and speed, we look for pairs (m1, m2) where:
    # - m1 is unsafe, m2 is any candidate
    # - Together they cover gap
    # - The "holes" created by m1 are filled by m2 (or m1 is safe enough that m2 fixes the rest)
    
    # Actually, a simpler structural approach:
    # If we have unsafe moves, we can try applying 2 moves simultaneously.
    # Let's generate all pairs of candidates and check if the combined effect is safe.
    
    best_pair = None
    best_pair_gain = -1
    
    # Limit search space for pairs to avoid O(N^2) blowup if N is large
    # N is number of primes ~ 168. Pairs ~ 14k. This is fast.
    
    # First, filter candidates to those that actually add coverage to gap
    # (All do by definition, but let's ensure we don't pick moves that don't help)
    
    for k1 in range(len(candidates)):
        i1, r1 = candidates[k1]
        p1 = _PRIMES[i1]
        old_start1 = p1 if residues[i1] == 0 else residues[i1]
        new_start1 = p1 if r1 == 0 else r1
        
        # Simulate move 1
        # We need to check if move 1 + move 2 is safe.
        
        for k2 in range(k1 + 1, len(candidates)):
            i2, r2 = candidates[k2]
            p2 = _PRIMES[i2]
            old_start2 = p2 if residues[i2] == 0 else residues[i2]
            new_start2 = p2 if r2 == 0 else r2
            
            # Quick check: Do these two moves together cover gap?
            # Yes, both target gap. So count[gap] increases by 2.
            # If current_gap_count + 2 >= 2, coverage at gap is satisfied.
            
            # Check safety for all x < gap.
            # This is expensive if done naively.
            # Optimization: Only check positions that are affected by either move.
            
            # Positions affected by move 1: {x | x % p1 == r1} (gain) and {x | x % p1 == r_old1} (loss)
            # Positions affected by move 2: {x | x % p2 == r2} (gain) and {x | x % p2 == r_old2} (loss)
            
            # We need to ensure that for all x in [1, gap-1], final_count[x] >= 2.
            # final_count[x] = count[x] - (1 if x covered by old1 else 0) - (1 if x covered by old2 else 0) 
            #                 + (1 if x covered by new1 else 0) + (1 if x covered by new2 else 0)
            
            # This check is O(gap) per pair, which is too slow for 14k pairs if gap is large.
            
            # Heuristic: Only consider pairs where at least one move is "safe" or 
            # the unsafe move's "damage" is localized and covered by the other.
            
            # Let's stick to a simpler structural improvement:
            # If no safe moves are sufficient, try a random pair of candidates 
            # and verify safety in O(gap) time, but limit the number of trials.
            
            pass # Placeholder for complex pair logic, see below
    
    # Fallback: If no simple solution, try a "best effort" single move 
    # that minimizes damage, or return empty to let GA handle it.
    # For now, return the best safe moves found, or empty.
    
    return []

def _local_search_conflict(residues, deadline, L_MAX=10000):
    """
    Local search using conflict graph analysis.
    """
    residues = list(residues)
    count = _build_count_np(residues, L_MAX)
    current_L = _find_prefix_L_np(count, L_MAX)
    
    improved = True
    stall_count = 0
    max_stall = 30
    
    while improved and time.time() < deadline:
        improved = False
        stall_count += 1
        if stall_count > max_stall:
            break
        
        gap = current_L + 1
        if gap > L_MAX:
            break
        
        if count[gap] >= 2:
            current_L = _find_prefix_L_np(count, L_MAX)
            improved = True
            stall_count = 0
            continue
        
        # Identify candidates for gap
        candidates = []
        for i, p in enumerate(_PRIMES):
            desired_r = gap % p
            if desired_r != residues[i]:
                candidates.append((i, desired_r))
        
        if not candidates:
            break
        
        # Separate into safe and unsafe
        safe_moves = []
        unsafe_moves = []
        
        for i, r_new in candidates:
            p = _PRIMES[i]
            r_old = residues[i]
            old_start = p if r_old == 0 else r_old
            
            is_safe = True
            if old_start <= gap - 1:
                min_count = int(np.min(count[old_start:gap:p]))
                if min_count < 3:
                    is_safe = False
            
            if is_safe:
                safe_moves.append((i, r_new))
            else:
                unsafe_moves.append((i, r_new))
        
        # Try to apply safe moves
        needed = max(0, 2 - count[gap])
        
        if len(safe_moves) >= needed:
            # Apply the best 'needed' safe moves
            # Score by secondary gain
            scored = []
            for i, r_new in safe_moves:
                p = _PRIMES[i]
                new_start = p if r_new == 0 else r_new
                if new_start <= L_MAX:
                    gain = int(np.sum(count[new_start:L_MAX+1:p] <= 1))
                else:
                    gain = 0
                scored.append((gain, i, r_new))
            
            scored.sort(key=lambda x: -x[0])
            
            # Apply the top 'needed' moves
            for _, i, r_new in scored[:needed]:
                p = _PRIMES[i]
                old_r = residues[i]
                old_start = p if old_r == 0 else old_r
                new_start = p if r_new == 0 else r_new
                
                if old_start <= L_MAX:
                    count[old_start:L_MAX+1:p] -= 1
                if new_start <= L_MAX:
                    count[new_start:L_MAX+1:p] += 1
                
                residues[i] = r_new
            
            current_L = _find_prefix_L_np(count, L_MAX)
            improved = True
            stall_count = 0
            continue
        
        # If not enough safe moves, try to find a pair of moves (possibly unsafe)
        # that together are safe and cover the gap.
        # We limit the search to pairs involving at least one unsafe move 
        # (since if 2 safe moves existed, we would have used them above, 
        # unless needed > 2, but needed is max 2).
        # Actually, if needed is 2, and we have 1 safe, we need 1 more.
        # If needed is 2, and we have 0 safe, we need 2.
        
        found_pair = False
        # Limit candidates to top 50 by some heuristic to keep O(N^2) small
        # Heuristic: Small primes are more likely to be useful for repair
        candidates_sorted = sorted(candidates, key=lambda x: _PRIMES[x[0]])
        search_candidates = candidates_sorted[:50]
        
        for k1 in range(len(search_candidates)):
            if time.time() > deadline:
                break
            i1, r1 = search_candidates[k1]
            p1 = _PRIMES[i1]
            
            for k2 in range(k1 + 1, len(search_candidates)):
                i2, r2 = search_candidates[k2]
                p2 = _PRIMES[i2]
                
                # Simulate applying both
                # We need to check if the combination is safe.
                # To speed this up, we only check the "danger zones":
                # 1. Positions covered by old1 but not new1 (loss from 1)
                # 2. Positions covered by old2 but not new2 (loss from 2)
                # 3. Ensure that any loss is compensated by gains from the other move or existing count >= 3.
                
                # This detailed check is complex. 
                # Simplified check: 
                # If move 1 is unsafe, it breaks some x. 
                # Does move 2 cover x? If yes, and x is not broken by move 2, 
                # and no other y is broken, then it's a candidate.
                
                # For now, we'll do a full O(gap) check for these limited pairs.
                # If gap is 10000, and we check 1000 pairs, that's 10^7 ops. Acceptable.
                
                # Save state
                old_r1 = residues[i1]
                old_r2 = residues[i2]
                old_start1 = p1 if old_r1 == 0 else old_r1
                new_start1 = p1 if r1 == 0 else r1
                old_start2 = p2 if old_r2 == 0 else old_r2
                new_start2 = p2 if r2 == 0 else r2
                
                # Apply
                if old_start1 <= L_MAX: count[old_start1:L_MAX+1:p1] -= 1
                if new_start1 <= L_MAX: count[new_start1:L_MAX+1:p1] += 1
                if old_start2 <= L_MAX: count[old_start2:L_MAX+1:p2] -= 1
                if new_start2 <= L_MAX: count[new_start2:L_MAX+1:p2] += 1
                
                # Check if prefix is still valid up to gap-1
                # And if gap is now covered
                if count[gap] >= 2:
                    # Check min count in [1, gap-1]
                    if gap > 1 and int(np.min(count[1:gap])) >= 2:
                        # Success!
                        residues[i1] = r1
                        residues[i2] = r2
                        current_L = _find_prefix_L_np(count, L_MAX)
                        improved = True
                        stall_count = 0
                        found_pair = True
                        break
                    else:
                        # Undo
                        if new_start1 <= L_MAX: count[new_start1:L_MAX+1:p1] -= 1
                        if old_start1 <= L_MAX: count[old_start1:L_MAX+1:p1] += 1
                        if new_start2 <= L_MAX: count[new_start2:L_MAX+1:p2] -= 1
                        if old_start2 <= L_MAX: count[old_start2:L_MAX+1:p2] += 1
                        residues[i1] = old_r1
                        residues[i2] = old_r2
                else:
                    # Undo
                    if new_start1 <= L_MAX: count[new_start1:L_MAX+1:p1] -= 1
                    if old_start1 <= L_MAX: count[old_start1:L_MAX+1:p1] += 1
                    if new_start2 <= L_MAX: count[new_start2:L_MAX+1:p2] -= 1
                    if old_start2 <= L_MAX: count[old_start2:L_MAX+1:p2] += 1
                    residues[i1] = old_r1
                    residues[i2] = old_r2
            
            if found_pair:
                break
        
        if not found_pair:
            # If no pair found, try a single "least bad" move? 
            # Or just break to let GA take over.
            # To avoid infinite loop, break.
            break

    return residues, current_L

def _polish_pass(residues, deadline, L_MAX=20000):
    """
    Final polish pass with larger L_MAX.
    """
    residues = list(residues)
    count = _build_count_np(residues, L_MAX)
    current_L = _find_prefix_L_np(count, L_MAX)
    
    improved = True
    stall_count = 0
    max_stall = 15
    
    while improved and time.time() < deadline:
        improved = False
        stall_count += 1
        if stall_count > max_stall:
            break
        
        gap = current_L + 1
        if gap > L_MAX:
            break
        
        if count[gap] >= 2:
            current_L = _find_prefix_L_np(count, L_MAX)
            improved = True
            stall_count = 0
            continue
        
        # Try all primes, prioritizing small ones
        best_move = None
        best_gain = 0
        
        for i in range(_NPRIMES):
            p = _PRIMES[i]
            
            if time.time() > deadline - 5 and p > 300:
                continue
            
            current_r = residues[i]
            desired_r = gap % p
            
            if current_r == desired_r:
                continue
            
            old_start = p if current_r == 0 else current_r
            if old_start <= current_L:
                if int(np.min(count[old_start:current_L+1:p])) < 3:
                    continue
            
            new_start = p if desired_r == 0 else desired_r
            
            count[old_start:L_MAX+1:p] -= 1
            count[new_start:L_MAX+1:p] += 1
            
            new_L = _find_prefix_L_np(count, L_MAX)
            
            count[new_start:L_MAX+1:p] -= 1
            count[old_start:L_MAX+1:p] += 1
            
            gain = new_L - current_L
            if gain > best_gain:
                best_gain = gain
                best_move = (i, desired_r, new_L)
        
        if best_move is not None:
            i, new_r, new_L = best_move
            p = _PRIMES[i]
            old_r = residues[i]
            old_start = p if old_r == 0 else old_r
            new_start = p if new_r == 0 else new_r
            
            count[old_start:L_MAX+1:p] -= 1
            count[new_start:L_MAX+1:p] += 1
            
            residues[i] = new_r
            current_L = new_L
            improved = True
            stall_count = 0
    
    return residues, current_L

def construct():
    start_time = time.time()
    deadline = start_time + 140

    # Phase 1: Greedy construction
    residues, L = _greedy_construct(deadline - 80)
    best_residues = list(residues)
    best_L = L

    # Phase 2: Conflict-based local search
    if time.time() < deadline - 40:
        residues, L = _local_search_conflict(residues, deadline - 40, L_MAX=10000)
        if L > best_L:
            best_L = L
            best_residues = list(residues)

    # Phase 3: Polish pass with larger range
    if time.time() < deadline - 5:
        residues, L = _polish_pass(best_residues, deadline - 5, L_MAX=20000)
        if L > best_L:
            best_L = L
            best_residues = list(residues)

    # Final verification
    final_L = _score(best_residues)
    if final_L > best_L:
        best_L = final_L

    return best_residues

# EVOLVE-BLOCK-END
