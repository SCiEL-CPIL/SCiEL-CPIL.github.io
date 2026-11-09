import numpy as np
import time

N = 5000

# EVOLVE-BLOCK-START
def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

# Precompute primes up to a reasonable limit, e.g., 600
PRIMES = _primes(600)
PRIME_SET = set(PRIMES)

def construct():
    start_time = time.time()
    deadline = start_time + 290  # 10s margin
    
    # X is the array of integers 0..5000
    X = np.arange(N)
    
    # We will iterate k downwards
    # Max possible k is bounded by N, but realistically ~500
    # Let's start from 500 and go down
    for k in range(500, 0, -1):
        if time.time() > deadline:
            break
            
        # Get primes <= k
        primes_k = [p for p in PRIMES if p <= k]
        if not primes_k:
            continue
            
        # Local search for residues
        # Initialize residues to 0
        residues = {p: 0 for p in primes_k}
        
        # Compute initial valid mask
        valid = np.ones(N, dtype=bool)
        for p in primes_k:
            valid &= (X % p != residues[p])
            
        count = np.sum(valid)
        
        # If we already have enough, return
        if count >= k:
            # Extract k distinct integers
            valid_indices = np.where(valid)[0]
            return valid_indices[:k].tolist()
            
        # Local search
        # We want to maximize count
        # Limit iterations to avoid getting stuck
        max_iters = 50
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            best_gain = 0
            best_p = None
            best_r = None
            
            for p in primes_k:
                current_r = residues[p]
                # Try all other residues
                for r in range(p):
                    if r == current_r:
                        continue
                    
                    # Calculate change in count
                    # Affected x: those where X%p is current_r or r
                    mask_affected = (X % p == current_r) | (X % p == r)
                    affected_indices = np.where(mask_affected)[0]
                    
                    if len(affected_indices) == 0:
                        continue
                    
                    # For each affected x, check if it becomes valid or invalid
                    # x is valid if for all q in primes_k, X[x] % q != residues[q]
                    # We only change p, so we only need to check p and the others for the affected x
                    
                    new_count_delta = 0
                    # Current validity of affected x
                    # We know they are currently valid or not
                    # Let's just re-evaluate validity for affected x under new residue r
                    
                    # To speed up, we can check the condition for p first
                    # If X[x] % p == r, it fails the new constraint for p.
                    # If X[x] % p != r, it passes the new constraint for p.
                    
                    # Let's create a temporary validity array for affected indices
                    # valid_new[x] = (X[x] % p != r) AND (for all q != p, X[x] % q != residues[q])
                    
                    # We can use the existing 'valid' array?
                    # No, 'valid' reflects the current residues.
                    # We need to check the other primes.
                    
                    # Optimization: Precompute "other_valid" for affected x?
                    # other_valid[x] = (for all q != p, X[x] % q != residues[q])
                    
                    # Let's just loop over affected indices if count is small?
                    # No, use vectorization.
                    
                    # Check other primes
                    # This is expensive if done for every r.
                    # However, affected_indices size is ~ 2*N/p.
                    # For small p, this is large.
                    
                    # Alternative:
                    # Count how many affected x are currently valid.
                    # Among those, how many will become invalid?
                    # And how many currently invalid will become valid?
                    
                    # Let V be the set of currently valid x.
                    # Let I be the set of currently invalid x.
                    
                    # For x in affected:
                    # If x in V:
                    #   It is currently valid.
                    #   It will remain valid if X[x] % p != r.
                    #   But x is in affected, so X[x] % p is current_r or r.
                    #   If X[x] % p == current_r:
                    #       It was passing p (since current_r is the residue we are NOT allowed to be? No.)
                    #       Wait. The constraint is x NOT congruent to residues[p].
                    #       So if x % p == current_r, it was FAILING the constraint for p.
                    #       So if x was valid, it must have x % p != current_r.
                    #       So if x is in V, x % p != current_r.
                    #       Since x is in affected, x % p must be r.
                    #       So if x in V and x % p == r, it will become INVALID (fails new constraint).
                    #       If x in V and x % p != r, it remains valid. (But x % p is r, so this case is impossible).
                    #       So: x in V and x % p == r -> becomes invalid.
                    
                    # If x in I:
                    #   It is currently invalid.
                    #   It might become valid if it was failing ONLY p, and now passes p.
                    #   It fails p if x % p == current_r.
                    #   So if x in I and x % p == current_r:
                    #       It was failing p.
                    #       Now it has x % p = current_r or r.
                    #       If x % p == current_r, it still fails p.
                    #       If x % p == r, it passes p.
                    #       So if x in I, x % p == r, AND it passes all other primes, it becomes valid.
                    
                    # So:
                    # Loss: count of x in V such that x % p == r
                    # Gain: count of x in I such that x % p == r AND passes all other primes
                    
                    # We can compute this efficiently.
                    
                    # 1. Count loss
                    # V_mask = valid
                    # loss_mask = V_mask & (X % p == r)
                    # loss = sum(loss_mask)
                    
                    # 2. Count gain
                    # I_mask = ~valid
                    # gain_candidate_mask = I_mask & (X % p == r)
                    # For these x, check if they pass all other primes.
                    # If yes, they become valid.
                    
                    # To check "passes all other primes", we can use a precomputed array?
                    # Or just check them.
                    # Number of candidates is small?
                    # If p is small, N/p is large.
                    
                    # Let's just do the vectorized check for the candidates.
                    candidates = np.where(gain_candidate_mask)[0]
                    if len(candidates) > 0:
                        # Check other primes for candidates
                        # valid_other = True
                        # for q in primes_k:
                        #   if q != p:
                        #     valid_other &= (X[candidates] % q != residues[q])
                        # gain = sum(valid_other)
                        
                        # To speed up, we can precompute a "validity score" or similar?
                        # No, just loop over primes for the candidates.
                        # Candidates size is ~ N/p.
                        # Primes count ~ 80.
                        # Ops ~ 80 * N/p.
                        # For p=2, 2000 * 80 = 160k ops.
                        # For all p, sum is ~ N * sum(1/p) * 80 ~ N * log(log k) * 80.
                        # 5000 * 2 * 80 = 800k ops per iteration of local search.
                        # This is acceptable.
                        
                        gain = 0
                        # We can vectorize the check for candidates
                        # Create a boolean array for candidates
                        cand_valid = np.ones(len(candidates), dtype=bool)
                        for q in primes_k:
                            if q == p:
                                continue
                            cand_valid &= (X[candidates] % q != residues[q])
                        gain = np.sum(cand_valid)
                    else:
                        gain = 0
                    
                    loss = np.sum(valid & (X % p == r))
                    
                    delta = gain - loss
                    if delta > best_gain:
                        best_gain = delta
                        best_p = p
                        best_r = r
            
            if best_gain > 0:
                # Apply change
                p = best_p
                r = best_r
                old_r = residues[p]
                residues[p] = r
                
                # Update valid array
                # Affected: X%p == old_r or X%p == r
                mask_affected = (X % p == old_r) | (X % p == r)
                affected_indices = np.where(mask_affected)[0]
                
                # Recalculate validity for affected
                # new_valid[x] = (X[x] % p != r) AND (for all q != p, X[x] % q != residues[q])
                
                # We can reuse the logic from above
                # For x in affected:
                # Check p: X[x] % p != r
                # Check others: 
                
                # Let's just recompute for affected_indices
                new_validity = np.ones(len(affected_indices), dtype=bool)
                for q in primes_k:
                    new_validity &= (X[affected_indices] % q != residues[q])
                
                valid[affected_indices] = new_validity
                count += best_gain
            else:
                # No improvement, break or random restart?
                # For simplicity, break and try next k or random restart
                # Let's do a few random restarts if time allows
                # But for now, break
                break
        
        # If we found a solution in the loop above, it would have returned.
        # If we exited the loop without returning, it means count < k for this k.
        # Continue to next k.
        
    # Fallback: return a small valid tuple
    # Just return [0] or something simple?
    # The task requires admissible tuple.
    # [0] is admissible?
    # Primes <= 1: none. So yes.
    # But we want max size.
    # If we fail, return empty or [0]?
    # The seed returns a list.
    return [0]
# EVOLVE-BLOCK-END
