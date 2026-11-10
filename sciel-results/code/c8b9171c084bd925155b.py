import random
import math
import time

def construct():
    # Generate primes up to 1000
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n_primes = len(primes)
    
    # We will search for a solution that covers up to M
    M = 5000  # Increased M to allow for higher scores
    
    # Initialize residues: 
    # A good initial heuristic is to set r_p = 0 for all p? 
    # Or random? Let's try random but seeded for reproducibility in dev, 
    # but for the task, random is fine. 
    # Actually, setting r_p = 0 means all multiples of p are covered. 
    # This covers many small numbers.
    # Let's start with r_p = 0 for all p.
    residues = [0] * n_primes
    
    def compute_counts(res):
        counts = [0] * (M + 1)
        for i in range(n_primes):
            p = primes[i]
            r = res[i]
            if r == 0:
                start = p
            else:
                start = r
            for x in range(start, M + 1, p):
                counts[x] += 1
        return counts

    def get_score(counts):
        for x in range(1, M + 1):
            if counts[x] < 2:
                return x - 1
        return M

    counts = compute_counts(residues)
    current_score = get_score(counts)
    best_score = current_score
    best_residues = residues[:]
    
    start_time = time.time()
    time_limit = 110.0 # Leave buffer
    
    # Precompute for each prime p, the list of x in [1, M] that p covers for a given residue?
    # That's too much memory.
    # Instead, we will maintain counts.
    
    # To speed up impact analysis, we can precompute for each prime p and each possible residue r,
    # the set of indices x in 1..M covered? No, M is 5000, p is up to 1000.
    # Number of covered points is M/p.
    # We can precompute `covers[p_idx][r]` as a list of x's?
    # Total memory: sum_{p in primes} p * (M/p) = n_primes * M = 168 * 5000 = 840,000 integers.
    # This is feasible.
    
    covers = {}
    for i, p in enumerate(primes):
        covers[i] = {}
        for r in range(p):
            lst = []
            start = r if r > 0 else p
            for x in range(start, M + 1, p):
                lst.append(x)
            covers[i][r] = lst
            
    # We need to quickly find which points are deficient (count < 2).
    # We can maintain a list of deficient points?
    # Or just scan counts? Scanning 5000 is fast.
    
    while (time.time() - start_time) < time_limit:
        # Find the first deficient point
        fail_idx = -1
        for x in range(1, M + 1):
            if counts[x] < 2:
                fail_idx = x
                break
        
        if fail_idx == -1:
            # Covered up to M
            current_score = M
            if current_score > best_score:
                best_score = current_score
                best_residues = residues[:]
            # Try to extend? We are capped at M. 
            # We can try to randomize to find a "more robust" solution or just break.
            # Since M is fixed, we can't get higher score.
            # But maybe we can increase M? 
            # The task says "capped at 10^5". 
            # If we reach M=5000, we should try to go higher.
            # But resizing counts and covers is expensive.
            # Let's stick to M=5000 for now. If we hit M, we break.
            break
            
        # We have a failure at fail_idx.
        # Find primes that do NOT cover fail_idx.
        # For each such prime i, if we change residue to fail_idx % p, 
        # how many deficient points do we fix?
        
        best_gain = -1
        best_i = -1
        best_new_r = -1
        
        # To make this faster, we only consider a subset of primes?
        # Or all? 168 primes is small.
        
        # Identify deficient points to check impact?
        # Checking all M points for each of 168 primes is 168 * 5000 = 840k ops.
        # This is fast in Python? Maybe. 110s / 840k ~ 130 iterations/s? 
        # That might be too slow if we do full scan every time.
        
        # Optimization: 
        # Instead of checking all deficient points, we can just check the specific fail_idx 
        # and maybe a few others?
        # Actually, the "impact" is: 
        # 1. fail_idx becomes covered (gain +1 if it was 0, +1 if it was 1? 
        #    Wait, if count[fail_idx] was 0, it becomes 1. Still deficient. 
        #    If count[fail_idx] was 1, it becomes 2. Fixed.
        # So we only "fix" fail_idx if count[fail_idx] == 1.
        # If count[fail_idx] == 0, we still need another prime to cover it.
        
        # So, if count[fail_idx] == 0:
        #   We need to pick a prime to cover it. 
        #   This move alone doesn't fix it.
        #   But it might fix other points.
        
        # Let's refine the heuristic:
        # We want to minimize the number of deficient points.
        
        # Let's calculate the delta in the number of deficient points.
        
        # For a candidate move (i, new_r):
        # Old residue: old_r = residues[i]
        # New residue: new_r = fail_idx % primes[i]
        
        # Points affected:
        # Lost: x in covers[i][old_r]
        # Gained: x in covers[i][new_r]
        
        # For each x in Lost:
        #   counts[x] decreases by 1.
        #   If counts[x] was 2, it becomes 1 -> becomes deficient (loss)
        #   If counts[x] was 3, it becomes 2 -> stays non-deficient
        #   If counts[x] was 1, it becomes 0 -> stays deficient (no change in count of deficient, but worse)
        
        # For each x in Gained:
        #   counts[x] increases by 1.
        #   If counts[x] was 1, it becomes 2 -> fixed (gain)
        #   If counts[x] was 0, it becomes 1 -> stays deficient
        #   If counts[x] was >= 2, stays non-deficient
        
        # So, Gain = number of x in Gained where counts[x] == 1
        # Loss = number of x in Lost where counts[x] == 2
        
        # Net change in deficient count = Loss - Gain.
        # We want to minimize this (i.e., maximize Gain - Loss).
        
        # We also care about the *position* of the first failure.
        # If we fix fail_idx, the first failure moves to the right.
        
        # Let's compute this for all candidates.
        
        candidates = []
        for i in range(n_primes):
            p = primes[i]
            old_r = residues[i]
            new_r = fail_idx % p
            if new_r == old_r:
                continue
            
            lost = covers[i][old_r]
            gained = covers[i][new_r]
            
            loss_count = 0
            for x in lost:
                if counts[x] == 2:
                    loss_count += 1
            
            gain_count = 0
            for x in gained:
                if counts[x] == 1:
                    gain_count += 1
            
            # Bonus if we actually fix fail_idx?
            # If counts[fail_idx] was 1, and fail_idx is in gained (which it is, by definition),
            # then it is counted in gain_count.
            # If counts[fail_idx] was 0, it is not counted.
            
            # We prefer moves that fix the current first failure.
            # If counts[fail_idx] == 1, this move fixes it.
            # If counts[fail_idx] == 0, this move doesn't fix it, but might fix others.
            
            # Heuristic score:
            # Primary: Fix fail_idx if possible (counts[fail_idx]==1)
            # Secondary: Maximize (gain_count - loss_count)
            
            score = 0
            if counts[fail_idx] == 1:
                score += 1000 # Huge bonus for fixing the first failure
            score += (gain_count - loss_count)
            
            candidates.append((score, i, new_r, loss_count, gain_count))
        
        if not candidates:
            # All primes already cover fail_idx?
            # Then counts[fail_idx] >= n_primes? No, we checked counts[fail_idx] < 2.
            # So counts[fail_idx] is 0 or 1.
            # If counts[fail_idx] == 0, no prime covers it.
            # So there are n_primes candidates.
            # If counts[fail_idx] == 1, exactly 1 prime covers it.
            # So there are n_primes - 1 candidates.
            # So candidates should not be empty.
            # Unless n_primes == 1? No.
            break
            
        # Sort by score descending
        candidates.sort(reverse=True, key=lambda x: x[0])
        
        # Pick the best one
        # To add some randomness to escape local minima, pick from top 3?
        # Let's pick the best one.
        best_score_move, best_i, best_new_r, _, _ = candidates[0]
        
        # Apply move
        old_r = residues[best_i]
        residues[best_i] = best_new_r
        
        lost = covers[best_i][old_r]
        gained = covers[best_i][best_new_r]
        
        # Update counts
        for x in lost:
            counts[x] -= 1
        for x in gained:
            counts[x] += 1
            
        new_score = get_score(counts)
        
        if new_score > best_score:
            best_score = new_score
            best_residues = residues[:]
            
    return best_residues
