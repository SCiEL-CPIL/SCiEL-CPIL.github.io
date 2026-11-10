"""Optimized doubly-covered prefix: informed greedy + 2-flip local search."""
import time
import random

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

def _build_count(residues, L_MAX):
    """Build count array for x in [1, L_MAX]."""
    count = [0] * (L_MAX + 1)
    for i, p in enumerate(_PRIMES):
        r = residues[i]
        start = p if r == 0 else r
        for x in range(start, L_MAX + 1, p):
            count[x] += 1
    return count

def _find_prefix_L(count, L_MAX):
    """Find largest L such that count[x] >= 2 for all x in [1..L]."""
    L = 0
    for x in range(1, L_MAX + 1):
        if count[x] >= 2:
            L = x
        else:
            break
    return L

def _score(residues):
    """Compute the actual score (largest L with full double coverage)."""
    L_MAX = 100000
    count = _build_count(residues, L_MAX)
    return _find_prefix_L(count, L_MAX)

def _greedy_construct(deadline):
    """
    Informed Greedy Construction.
    Strategy: Process primes in increasing order. For each prime, choose the residue
    that maximizes the number of positions in [1, L_TARGET] that reach count >= 2 
    or are already >= 2 (stability).
    """
    residues = [0] * _NPRIMES
    L_TARGET = 2000  # Focus on extending the prefix to this range initially
    count = [0] * (L_TARGET + 1)
    
    # We want to maximize double coverage.
    # A position x is "good" if count[x] >= 2.
    # When we add a prime p with residue r, we increment count[x] for x in {r, r+p, ...}.
    # Gain = number of x where count[x] becomes 2 (from 1) + number of x where count[x] stays >= 2.
    # Penalty = number of x where count[x] was 0 and becomes 1 (neutral) or if we break something? 
    # Actually, we just want to maximize the prefix length.
    
    # A better heuristic: 
    # For each prime p, try all residues r.
    # Simulate the addition.
    # Score the residue by how far it extends the current prefix L.
    
    current_L = 0
    prime_order = list(range(_NPRIMES))
    
    # Small primes are more important for dense coverage.
    # We process them in order.
    
    for idx in prime_order:
        if time.time() > deadline:
            break
        p = _PRIMES[idx]
        best_r = 0
        best_L = -1
        
        # Candidate residues: 
        # 1. Residues that cover the current gap (current_L + 1)
        # 2. Residues that cover many positions in [current_L, current_L + p]
        
        # To be efficient, we only test a subset of residues if p is large.
        # But for p <= 1000, testing all p residues is feasible if we do it fast.
        # However, updating the count array for each r is O(L_TARGET/p).
        # Total cost per prime: sum(p * L_TARGET/p) = L_TARGET * 168. ~336k ops. 168 primes -> 56M ops.
        # This might be too slow in Python.
        
        # Optimization: 
        # Instead of full simulation, use a heuristic score.
        # Score(r) = number of x in [1, L_TARGET] such that (x % p == r) AND count[x] == 1.
        # Plus a bonus if it covers current_L + 1.
        
        gap = current_L + 1
        best_score = -1
        
        # We can precompute the count array state.
        # For a given r, the positions covered are r, r+p, r+2p...
        # We want to maximize the number of these positions that currently have count == 1.
        
        # Let's try a few promising residues:
        # 1. gap % p
        # 2. 0
        # 3. Residues that cover positions with count == 1 in the range [gap, gap + 10*p]
        
        candidates = set()
        candidates.add(gap % p)
        candidates.add(0)
        
        # Add residues that cover positions with count == 1
        # We sample a few positions in the gap region
        for dx in range(0, min(10, L_TARGET - gap)):
            x = gap + dx
            if x <= L_TARGET and count[x] == 1:
                candidates.add(x % p)
        
        # Also add random candidates if p is small
        if p < 50:
            for _ in range(5):
                candidates.add(random.randint(0, p-1))
        
        for r in candidates:
            if r >= p: continue
            # Calculate score
            score = 0
            start = p if r == 0 else r
            for x in range(start, L_TARGET + 1, p):
                if count[x] == 1:
                    score += 10  # High reward for completing double coverage
                elif count[x] >= 2:
                    score += 1   # Small reward for maintaining coverage
                # count[x] == 0: no reward
        
        # Tie-breaker: prefer covering the gap
        if gap % p in candidates:
            # We already added it. Let's boost it if it's the best.
            pass
            
        # Actually, the above loop calculates score for each candidate.
        # We need to pick the best one.
        
        # Re-iterate to find best
        best_r = 0
        max_score = -1
        for r in candidates:
             if r >= p: continue
             score = 0
             start = p if r == 0 else r
             for x in range(start, L_TARGET + 1, p):
                 if count[x] == 1:
                     score += 10
                 elif count[x] >= 2:
                     score += 1
             if score > max_score:
                 max_score = score
                 best_r = r
        
        residues[idx] = best_r
        
        # Update count
        start = p if best_r == 0 else best_r
        for x in range(start, L_TARGET + 1, p):
            count[x] += 1
            
        # Update current_L
        while current_L + 1 <= L_TARGET and count[current_L + 1] >= 2:
            current_L += 1
            
    return residues, current_L

def _local_search_1flip(residues, deadline, L_MAX=5000):
    """
    Local search with single flips.
    Allows moves that extend the prefix, even if they break previous safety constraints,
    as long as the net prefix length increases.
    """
    residues = list(residues)
    count = _build_count(residues, L_MAX)
    current_L = _find_prefix_L(count, L_MAX)
    
    improved = True
    stall_count = 0
    max_stall = 20
    
    while improved and time.time() < deadline:
        improved = False
        stall_count += 1
        if stall_count > max_stall:
            break
            
        if current_L >= L_MAX:
            break
            
        gap = current_L + 1
        
        # Try flipping each prime
        best_move = None
        best_new_L = current_L
        
        # Prioritize primes that cover the gap or are small
        # To speed up, we only try primes where p <= 200 or p divides gap
        # Actually, any prime can be flipped. But large primes have sparse coverage.
        # Let's try all primes, but limit the work.
        
        for i in range(_NPRIMES):
            p = _PRIMES[i]
            current_r = residues[i]
            
            # Try a few candidate residues for this prime
            # 1. gap % p (to cover the gap)
            # 2. 0
            # 3. Current residue (skip)
            
            candidates = [gap % p, 0]
            # Add some random ones if p is small
            if p < 50:
                candidates.append(random.randint(0, p-1))
            
            for new_r in candidates:
                if new_r == current_r or new_r >= p:
                    continue
                
                # Simulate flip
                old_start = p if current_r == 0 else current_r
                new_start = p if new_r == 0 else new_r
                
                # Apply
                for x in range(old_start, L_MAX + 1, p):
                    count[x] -= 1
                for x in range(new_start, L_MAX + 1, p):
                    count[x] += 1
                
                # Find new L
                # Optimization: we know the change is local? No, it affects all multiples.
                # But we can scan from current_L downwards if it broke, or upwards if it extended.
                # Actually, if it broke, we need to find the new L which might be much smaller.
                # Let's just scan from 1 to L_MAX? Too slow.
                
                # Better: 
                # If count[gap] >= 2, try extending.
                # If count[some x < current_L] < 2, we broke it.
                
                # Let's check if we broke the prefix.
                # We can check the minimum count in [1, current_L] quickly?
                # No, we have to scan.
                
                # Heuristic: Only accept if count[gap] >= 2.
                # This ensures we don't break the prefix if we were careful before?
                # No, flipping might break an earlier x.
                
                # Let's do a full scan for new L, but limit it.
                new_L = 0
                # If we broke it, new_L will be smaller.
                # If we extended it, new_L will be larger.
                
                # To save time, we can check if any x in [1, current_L] has count < 2.
                # If yes, new_L is at most that x - 1.
                # If no, new_L is at least current_L.
                
                broke = False
                for x in range(1, current_L + 1):
                    if count[x] < 2:
                        broke = True
                        new_L = x - 1
                        break
                
                if not broke:
                    # Try to extend
                    new_L = current_L
                    while new_L + 1 <= L_MAX and count[new_L + 1] >= 2:
                        new_L += 1
                
                # Undo
                for x in range(new_start, L_MAX + 1, p):
                    count[x] -= 1
                for x in range(old_start, L_MAX + 1, p):
                    count[x] += 1
                
                if new_L > best_new_L:
                    best_new_L = new_L
                    best_move = (i, new_r, new_L)
        
        if best_move is not None:
            i, new_r, new_L = best_move
            p = _PRIMES[i]
            old_r = residues[i]
            old_start = p if old_r == 0 else old_r
            new_start = p if new_r == 0 else new_r
            
            for x in range(old_start, L_MAX + 1, p):
                count[x] -= 1
            for x in range(new_start, L_MAX + 1, p):
                count[x] += 1
            
            residues[i] = new_r
            current_L = new_L
            improved = True
            stall_count = 0
    
    return residues, current_L

def _local_search_2flip(residues, deadline, L_MAX=3000):
    """
    Local search with 2-flips to escape local minima.
    """
    residues = list(residues)
    count = _build_count(residues, L_MAX)
    current_L = _find_prefix_L(count, L_MAX)
    
    improved = True
    stall_count = 0
    max_stall = 5
    
    while improved and time.time() < deadline:
        improved = False
        stall_count += 1
        if stall_count > max_stall:
            break
            
        if current_L >= L_MAX:
            break
            
        gap = current_L + 1
        
        # Try pairs of primes
        # To limit complexity, only try pairs of small primes
        small_primes_idx = [i for i in range(_NPRIMES) if _PRIMES[i] < 50]
        
        best_move = None
        best_new_L = current_L
        
        for i1 in range(len(small_primes_idx)):
            idx1 = small_primes_idx[i1]
            p1 = _PRIMES[idx1]
            r1_old = residues[idx1]
            
            # Candidates for r1
            cand_r1 = [gap % p1, 0]
            
            for r1_new in cand_r1:
                if r1_new == r1_old or r1_new >= p1:
                    continue
                
                old_start1 = p1 if r1_old == 0 else r1_old
                new_start1 = p1 if r1_new == 0 else r1_new
                
                # Apply flip 1
                for x in range(old_start1, L_MAX + 1, p1):
                    count[x] -= 1
                for x in range(new_start1, L_MAX + 1, p1):
                    count[x] += 1
                
                for i2 in range(i1 + 1, len(small_primes_idx)):
                    idx2 = small_primes_idx[i2]
                    p2 = _PRIMES[idx2]
                    r2_old = residues[idx2]
                    
                    cand_r2 = [gap % p2, 0]
                    
                    for r2_new in cand_r2:
                        if r2_new == r2_old or r2_new >= p2:
                            continue
                        
                        old_start2 = p2 if r2_old == 0 else r2_old
                        new_start2 = p2 if r2_new == 0 else r2_new
                        
                        # Apply flip 2
                        for x in range(old_start2, L_MAX + 1, p2):
                            count[x] -= 1
                        for x in range(new_start2, L_MAX + 1, p2):
                            count[x] += 1
                        
                        # Evaluate
                        new_L = 0
                        broke = False
                        for x in range(1, current_L + 1):
                            if count[x] < 2:
                                broke = True
                                new_L = x - 1
                                break
                        
                        if not broke:
                            new_L = current_L
                            while new_L + 1 <= L_MAX and count[new_L + 1] >= 2:
                                new_L += 1
                        
                        # Undo flip 2
                        for x in range(new_start2, L_MAX + 1, p2):
                            count[x] -= 1
                        for x in range(old_start2, L_MAX + 1, p2):
                            count[x] += 1
                        
                        if new_L > best_new_L:
                            best_new_L = new_L
                            best_move = (idx1, r1_new, idx2, r2_new, new_L)
                
                # Undo flip 1
                for x in range(new_start1, L_MAX + 1, p1):
                    count[x] -= 1
                for x in range(old_start1, L_MAX + 1, p1):
                    count[x] += 1
        
        if best_move is not None:
            idx1, r1_new, idx2, r2_new, new_L = best_move
            p1 = _PRIMES[idx1]
            r1_old = residues[idx1]
            p2 = _PRIMES[idx2]
            r2_old = residues[idx2]
            
            # Apply both
            old_start1 = p1 if r1_old == 0 else r1_old
            new_start1 = p1 if r1_new == 0 else r1_new
            old_start2 = p2 if r2_old == 0 else r2_old
            new_start2 = p2 if r2_new == 0 else r2_new
            
            for x in range(old_start1, L_MAX + 1, p1):
                count[x] -= 1
            for x in range(new_start1, L_MAX + 1, p1):
                count[x] += 1
            for x in range(old_start2, L_MAX + 1, p2):
                count[x] -= 1
            for x in range(new_start2, L_MAX + 1, p2):
                count[x] += 1
            
            residues[idx1] = r1_new
            residues[idx2] = r2_new
            current_L = new_L
            improved = True
            stall_count = 0
    
    return residues, current_L

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Phase 1: Greedy Construction
    residues, L = _greedy_construct(deadline - 60)
    best_residues = list(residues)
    best_L = L
    
    # Phase 2: 1-flip Local Search
    if time.time() < deadline - 30:
        residues, L = _local_search_1flip(residues, deadline - 30, L_MAX=5000)
        if L > best_L:
            best_L = L
            best_residues = list(residues)
    
    # Phase 3: 2-flip Local Search
    if time.time() < deadline - 10:
        residues, L = _local_search_2flip(residues, deadline - 10, L_MAX=3000)
        if L > best_L:
            best_L = L
            best_residues = list(residues)
    
    # Phase 4: Final 1-flip polish
    if time.time() < deadline - 5:
        residues, L = _local_search_1flip(residues, deadline - 5, L_MAX=5000)
        if L > best_L:
            best_L = L
            best_residues = list(residues)
    
    # Final verification
    final_L = _score(best_residues)
    if final_L > best_L:
        best_L = final_L
    
    return best_residues

# EVOLVE-BLOCK-END
