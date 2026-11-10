"""Optimized doubly-covered prefix: NumPy-accelerated CSP and local search."""
import time
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
    """Build count array for x in [1, L_MAX] using NumPy."""
    count = np.zeros(L_MAX + 1, dtype=np.int32)
    for i, p in enumerate(_PRIMES):
        r = residues[i]
        # Indices are 1-based in problem, 0-based in array.
        # x = r mod p.
        # If r == 0, x is p, 2p, ... -> indices p-1, 2p-1, ...
        # If r != 0, x is r, r+p, ... -> indices r-1, r+p-1, ...
        start_idx = (p - 1) if r == 0 else (r - 1)
        if start_idx < L_MAX:
            count[start_idx::p] += 1
    return count

def _find_prefix_L_np(count, L_MAX):
    """Find largest L such that count[x] >= 2 for all x in [1..L]."""
    # We look for the first index i (0-based) where count[i+1] < 2
    # i.e. count[i] < 2 in 0-based array corresponding to value i+1
    # Note: count[0] is for x=1? No, count is size L_MAX+1.
    # Let's align indices carefully.
    # Problem: x in [1, L].
    # Array `count` has indices 0..L_MAX.
    # My `_build_count_np` used `count[start_idx::p]`.
    # If r=1, start_idx=0. x=1, 1+p...
    # So `count[k]` corresponds to integer `k+1`.
    
    # We want largest L such that for all x in 1..L, count[x] >= 2.
    # In array terms: for all k in 0..L-1, count[k] >= 2.
    
    # Find first k where count[k] < 2.
    # If no such k in 0..L_MAX-1, then L = L_MAX.
    
    # Note: count[L_MAX] corresponds to x=L_MAX+1, which is out of bounds for L<=L_MAX.
    # So we only care about indices 0 to L_MAX-1.
    
    bad_indices = np.where(count[:L_MAX] < 2)[0]
    if len(bad_indices) == 0:
        return L_MAX
    else:
        first_bad = bad_indices[0]
        # If first_bad is 0, L=0.
        # If first_bad is k, then indices 0..k-1 are good.
        # Number of good integers is k.
        return first_bad

def _score(residues):
    """Compute the actual score (largest L with full double coverage)."""
    L_MAX = 100000
    count = _build_count_np(residues, L_MAX)
    return _find_prefix_L_np(count, L_MAX)

def _greedy_csp_construct(deadline):
    """
    Constraint-satisfaction-style greedy using NumPy for speed.
    """
    residues = [0] * _NPRIMES
    L_MAX = 5000 # Smaller window for greedy to be fast, verify later
    
    # We will maintain a count array for the greedy phase
    count = np.zeros(L_MAX + 1, dtype=np.int32)
    
    # Phase 1: Small primes (p <= 50)
    # Objective: Maximize coverage in [1, L_MAX]
    # We iterate primes in increasing order.
    
    for i, p in enumerate(_PRIMES):
        if time.time() > deadline:
            break
            
        if p > 50:
            continue # Handle large primes later or in same loop
            
        best_r = 0
        best_score = -1
        
        # Try all residues for small primes?
        # p is small, so this is cheap.
        for r in range(p):
            start_idx = (p - 1) if r == 0 else (r - 1)
            if start_idx >= L_MAX:
                continue
                
            # Calculate gain
            # We want to increase counts where count < 2
            # Gain = sum(1 for x in covered if count[x] < 2)
            # This is roughly (number of covered) - (number of covered where count >= 2)
            
            # Vectorized gain calculation
            # covered = count[start_idx::p]
            # But we need to be careful not to modify count yet.
            
            # Simulate:
            # new_count = count.copy() is too slow.
            # Just compute the gain directly.
            
            # Indices covered: start_idx, start_idx+p, ...
            # We can slice count.
            covered_counts = count[start_idx::p]
            
            # Gain:
            # +2 for each 0
            # +1 for each 1
            # 0 for >= 2
            gain = np.sum(covered_counts == 0) * 2 + np.sum(covered_counts == 1) * 1
            
            if gain > best_score:
                best_score = gain
                best_r = r
        
        residues[i] = best_r
        start_idx = (p - 1) if best_r == 0 else (best_r - 1)
        if start_idx < L_MAX:
            count[start_idx::p] += 1

    # Phase 2: Large primes
    # Use deficit heuristic
    for i, p in enumerate(_PRIMES):
        if time.time() > deadline:
            break
        if p <= 50:
            continue
            
        # Find first gap in [1, L_MAX]
        # We need to check where count < 2
        # But we only have count for [1, L_MAX].
        # If L_MAX is small, this is fine.
        
        # Find first x (1-based) where count[x-1] < 2
        # Note: count array index k corresponds to x=k+1
        bad_indices = np.where(count[:L_MAX] < 2)[0]
        
        if len(bad_indices) == 0:
            # All covered in this window
            residues[i] = 0
            start_idx = p - 1
            if start_idx < L_MAX:
                count[start_idx::p] += 1
            continue
            
        gap_idx = bad_indices[0] # 0-based index in count, so x = gap_idx + 1
        gap_x = gap_idx + 1
        
        # We want to cover gap_x.
        # Residue r must satisfy gap_x % p == r (with 0 if divisible)
        # Note: Python % returns 0 for divisible.
        # Our start_idx logic:
        # If r == 0, start_idx = p-1. x=p, 2p...
        # If r != 0, start_idx = r-1. x=r, r+p...
        
        # Candidate residues:
        # 1. The one that covers gap_x
        r_gap = gap_x % p
        
        # 2. Maybe others that cover nearby gaps?
        # For speed, just try r_gap and a few neighbors.
        
        candidates = set([r_gap])
        for dx in range(-5, 6):
            x = gap_x + dx
            if 1 <= x <= L_MAX:
                candidates.add(x % p)
        
        best_r = r_gap
        best_score = -1
        
        for r in candidates:
            start_idx = (p - 1) if r == 0 else (r - 1)
            if start_idx >= L_MAX:
                continue
                
            covered_counts = count[start_idx::p]
            
            # Score:
            # High weight for covering the specific gap
            # Medium weight for other under-covered spots
            # Penalty for over-covered?
            
            gain = np.sum(covered_counts == 0) * 3 + np.sum(covered_counts == 1) * 2
            
            # Bonus if it covers the gap
            # Check if gap_idx is in start_idx::p
            # (gap_idx - start_idx) % p == 0
            if (gap_idx - start_idx) % p == 0:
                gain += 100
            
            if gain > best_score:
                best_score = gain
                best_r = r
        
        residues[i] = best_r
        start_idx = (p - 1) if best_r == 0 else (best_r - 1)
        if start_idx < L_MAX:
            count[start_idx::p] += 1

    # Get current L using this local count
    current_L = _find_prefix_L_np(count, L_MAX)
    return residues, current_L

def _local_search_targeted(residues, deadline, L_MAX=5000):
    """
    Targeted local search with NumPy.
    """
    residues = list(residues)
    count = _build_count_np(residues, L_MAX)
    current_L = _find_prefix_L_np(count, L_MAX)
    
    stall_count = 0
    max_stall = 20
    
    while time.time() < deadline:
        if stall_count > max_stall:
            break
        
        if current_L >= L_MAX:
            break
            
        gap_x = current_L + 1
        gap_idx = gap_x - 1
        
        if count[gap_idx] >= 2:
            # Extend
            while current_L < L_MAX and count[current_L] >= 2: # count[current_L] is x=current_L+1
                current_L += 1
            stall_count = 0
            continue
        
        # Find primes that can cover gap_x
        # And are "safe" to move (don't break prefix)
        
        best_move = None
        best_gain = 0
        
        for i, p in enumerate(_PRIMES):
            current_r = residues[i]
            
            # Desired residue to cover gap
            desired_r = gap_x % p
            
            if current_r == desired_r:
                continue
                
            # Check safety:
            # Old positions covered: old_start_idx :: p
            # New positions covered: new_start_idx :: p
            # Lost positions: old_start_idx :: p but not in new_start_idx :: p
            # Gained positions: new_start_idx :: p but not in old_start_idx :: p
            
            old_start_idx = (p - 1) if current_r == 0 else (current_r - 1)
            new_start_idx = (p - 1) if desired_r == 0 else (desired_r - 1)
            
            # Safety: All positions in [1, current_L] that lose coverage must have count >= 3.
            # Losing positions are those in old_set \ new_set.
            # This is complex to compute directly.
            # Approximation: Check if any position in [1, current_L] covered by old_r is NOT covered by new_r.
            # If count[x] < 3 for any such x, unsafe.
            
            # Efficient check:
            # Iterate through positions covered by old_r in [1, current_L].
            # There are at most current_L / p positions.
            # For p=2, ~2500 checks. For p=1000, ~5 checks.
            # Total checks per prime ~ L/p. Sum over primes ~ L * sum(1/p) ~ L ln ln P.
            # 5000 * 3 = 15000 checks per iteration.
            # This is fast enough in Python if done simply, but let's try to vectorize or optimize.
            
            safe = True
            # We only need to check positions x in [1, current_L] such that x % p == current_r (with 0 mod p)
            # AND x % p != desired_r (with 0 mod p)
            
            # Actually, if current_r != desired_r, then for any x, it's either in old or new or neither.
            # If x is in old, is it in new?
            # x % p == current_r.
            # x % p == desired_r.
            # Since current_r != desired_r, x cannot be in both.
            # So ALL positions covered by old_r are LOST from new_r?
            # Wait.
            # If I change residue from r1 to r2, the set of covered integers changes completely.
            # Old set: {x | x % p == r1}
            # New set: {x | x % p == r2}
            # Intersection is empty if r1 != r2.
            # So we LOSE all coverage from old_r and GAIN all coverage from new_r.
            
            # Therefore, safety check is:
            # For all x in [1, current_L] such that x % p == current_r:
            #   count[x] must be >= 3 (so that after losing 1, it becomes >= 2).
            
            # Let's check this.
            # Positions:
            start_check = old_start_idx
            # We need x <= current_L => idx <= current_L - 1
            # So indices in [old_start_idx, current_L - 1] with step p.
            
            if old_start_idx < current_L: # If old_start_idx >= current_L, no positions in prefix are lost
                indices = np.arange(old_start_idx, current_L, p)
                if len(indices) > 0:
                    min_count = np.min(count[indices])
                    if min_count < 3:
                        safe = False
            
            if not safe:
                continue
            
            # If safe, calculate gain.
            # Gain = extension of L.
            # We need to simulate the change to see how far L extends.
            # But simulation is expensive.
            # Heuristic: Gain is roughly (number of new positions covered in [current_L+1, current_L+K] with count < 2)
            
            # Let's just simulate if the prime is small or if the gap is likely to be fixed.
            # Actually, let's just do a quick simulation for the best few candidates.
            
            # For now, let's just pick the one that covers the gap and is safe.
            # If multiple, pick the one with smallest p (more frequent future help).
            
            # We can just take the first safe one that covers the gap.
            # Or score them.
            
            # Score: Number of positions in [current_L+1, current_L+100] that become covered (and had count < 2)
            # We can approximate this.
            
            # Let's just accept the first safe move that covers the gap.
            # This is a greedy local search.
            
            # To be slightly better, we can look for the move that maximizes the count at gap_x after move.
            # Since we only change one prime, count[gap_idx] increases by 1 if desired_r covers it.
            # All candidates cover gap_x.
            # So they all increase count[gap_idx] by 1.
            # The difference is in the surrounding area.
            
            # Let's just take the first safe one.
            best_move = (i, desired_r)
            break # Take first safe one for speed
            
        if best_move is None:
            # No single prime flip can fix the gap safely?
            # Maybe we need to move a prime that doesn't cover the gap but frees up space?
            # Or maybe we are stuck.
            stall_count += 1
            # Try random moves?
            # For now, break or try harder.
            if stall_count > 10:
                break
            continue
        
        i, new_r = best_move
        p = _PRIMES[i]
        old_r = residues[i]
        
        old_start_idx = (p - 1) if old_r == 0 else (old_r - 1)
        new_start_idx = (p - 1) if new_r == 0 else (new_r - 1)
        
        # Update count
        # Subtract old
        if old_start_idx < L_MAX:
            count[old_start_idx::p] -= 1
        # Add new
        if new_start_idx < L_MAX:
            count[new_start_idx::p] += 1
            
        residues[i] = new_r
        
        # Update L
        new_L = current_L
        while new_L < L_MAX and count[new_L] >= 2:
            new_L += 1
            
        if new_L > current_L:
            current_L = new_L
            stall_count = 0
        else:
            # No improvement, undo?
            # If we made a move that didn't extend L, it might still be useful for future.
            # But if it didn't extend L, it might have broken something else?
            # We checked safety for [1, current_L].
            # So [1, current_L] is still valid.
            # But L didn't extend.
            # We can keep the change or revert.
            # Keeping might help future steps.
            # Reverting might be safer if we are stuck.
            # Let's keep it for now.
            stall_count += 1
            
    return residues, current_L

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Phase 1: Greedy
    residues, L = _greedy_csp_construct(deadline - 100)
    best_residues = list(residues)
    best_L = L
    
    # Phase 2: Local Search
    if time.time() < deadline - 20:
        residues, L = _local_search_targeted(residues, deadline - 20, L_MAX=5000)
        if L > best_L:
            best_L = L
            best_residues = list(residues)
            
    # Phase 3: Final Verification and Polish
    # We should verify the score with full range 100000
    # But if L from local search (on 5000) is high, it's likely good.
    # Let's do a quick score check.
    # If time permits, run local search on larger window?
    # 5000 is a good balance.
    
    # One more local search pass on the best result found
    if time.time() < deadline - 10:
        residues, L = _local_search_targeted(best_residues, deadline - 10, L_MAX=5000)
        if L > best_L:
            best_L = L
            best_residues = list(residues)
            
    # Return best residues
    return best_residues

# EVOLVE-BLOCK-END
