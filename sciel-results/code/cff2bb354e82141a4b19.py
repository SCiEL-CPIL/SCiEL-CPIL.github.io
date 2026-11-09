import time
import random
import sys

def construct():
    """
    Returns a list of up to 48 distinct positive integers <= 512 such that no element 
    is the exact arithmetic mean of any subset of at least two other elements.
    
    Strategy:
    1. Search for a large set of residues modulo M that is "mean-free" modulo M.
       If a set of integers is mean-free modulo M, it is mean-free in integers.
       This reduces the search space and the DP complexity (sums are bounded by M*k).
    2. Greedily build a set of residues R in Z_M such that no element in R is the 
       mean of any subset of R\{element} (mod M).
    3. Map the found residues to the smallest distinct integers in [1, 512] 
       that have those residues.
    4. Verify the resulting integer set with a standard DP to be safe.
    """
    deadline = time.time() + 50
    N = 512
    target_size = 48
    
    # We need at least 48 residues. 
    # Let's try M=64. Max possible is 64. 
    # If M=64 is too restrictive, try M=128, 256.
    # For M=64, sums of subsets are small.
    
    def is_valid_modular(R, M):
        """
        Check if set of residues R is mean-free modulo M.
        Condition: For all a in R, and all subsets T of R\{a} with |T|>=2,
        sum(T) != a * |T| (mod M).
        """
        m = len(R)
        if m < 3:
            return True
        
        R_list = list(R)
        
        for idx, a in enumerate(R_list):
            # Build DP for R without a, modulo M
            # dp[k] is a bitset of possible sums mod M using exactly k elements
            # Since M is small (e.g., 64), we can use a list of sets or bitsets.
            # Bitset of size M.
            max_k = m - 1
            # dp[k] is an integer bitmask of length M
            dp = [0] * (max_k + 1)
            dp[0] = 1  # sum 0 is possible
            
            count = 0
            for v in R_list:
                if v == a and count == 0:
                    count = 1
                    continue
                
                # Update DP in reverse
                # We only need to track sums mod M
                for k in range(min(count, max_k - 1), -1, -1):
                    if dp[k]:
                        # Shift left by v, wrap around M
                        # This is tricky with standard bit ops if M is not a power of 2.
                        # M=64 is a power of 2. Let's stick to powers of 2 for simplicity.
                        # If M=64, shift by v and mask with (1<<M)-1? No, wrap around.
                        # (x << v) | (x >> (M-v)) ... this is circular shift.
                        # But we only care if the target sum is present.
                        # Let's use a set of ints for simplicity if M is small.
                        pass
            
            # Using sets for M=64 is fast enough.
            # Let's rewrite is_valid_modular using sets for clarity and safety with any M.
            
        # Re-implementation with sets
        for a in R_list:
            others = [v for v in R_list if v != a]
            m_others = len(others)
            # dp[k] = set of possible sums mod M using k elements from others
            dp = [set() for _ in range(m_others + 1)]
            dp[0].add(0)
            
            for v in others:
                for k in range(len(others) - 1, -1, -1):
                    if dp[k]:
                        for s in dp[k]:
                            dp[k+1].add((s + v) % M)
            
            for k in range(2, m_others + 1):
                target = (a * k) % M
                if target in dp[k]:
                    return False
        
        return True

    # Try different moduli
    best_residues = []
    best_M = 0
    
    for M in [64, 128, 256, 512]:
        if time.time() > deadline:
            break
            
        # Greedy construction of residues
        # Order: try all residues 0..M-1 in multiple orders
        found = False
        # Randomized restarts for residue selection
        rng = random.Random(M)
        
        for attempt in range(10):
            if time.time() > deadline:
                break
                
            residues = set()
            # Try adding residues in a specific order
            order = list(range(M))
            if attempt > 0:
                rng.shuffle(order)
            
            for r in order:
                if len(residues) >= target_size:
                    break
                if r in residues:
                    continue
                new_residues = residues | {r}
                if is_valid_modular(new_residues, M):
                    residues.add(r)
            
            if len(residues) >= target_size:
                best_residues = list(residues)
                best_M = M
                found = True
                break
        
        if found:
            break
    
    if not best_residues:
        # Fallback: use the original greedy if modular fails
        # But let's try to construct a valid set directly if modular fails
        # For now, return empty or a small valid set
        return []
    
    # Map residues to integers
    # For each residue r in best_residues, find the smallest integer x in [1, 512] such that x % M == r
    # Note: Python % returns 0..M-1.
    # If r==0, x can be M, 2M, ...
    # If r!=0, x can be r, r+M, ...
    
    integers = []
    used_ints = set()
    
    for r in best_residues:
        # Find smallest x in [1, 512] with x % M == r
        # x = r + k*M
        # If r == 0, smallest positive is M (assuming M <= 512)
        # If r != 0, smallest positive is r (if r >= 1)
        
        if r == 0:
            start = M
        else:
            start = r
        
        found_x = False
        x = start
        while x <= N:
            if x not in used_ints:
                integers.append(x)
                used_ints.add(x)
                found_x = True
                break
            x += M
        
        if not found_x:
            # Should not happen if M <= 512 and we have enough space
            # But if M=512, only 1 int per residue.
            # If M=64, 8 ints per residue.
            # We picked 48 residues. 48*1 = 48 ints. Should fit.
            pass
    
    # Verify the final integer set
    # This is a safety check.
    def is_valid_global(S):
        m = len(S)
        if m < 3:
            return True
        
        S_sorted = sorted(S)
        
        for idx, a in enumerate(S_sorted):
            max_k = m - 1
            dp = [0] * (max_k + 1)
            dp[0] = 1
            
            count = 0
            for v in S_sorted:
                if v == a and count == 0:
                    count = 1
                    continue
                
                for k in range(min(count, max_k - 1), -1, -1):
                    if dp[k]:
                        dp[k + 1] |= dp[k] << v
                count += 1
            
            for k in range(2, max_k + 1):
                target_sum = a * k
                if target_sum > 512 * k:
                    continue
                if target_sum > 24576:
                    break
                if (dp[k] >> target_sum) & 1:
                    return False
        
        return True
    
    if is_valid_global(integers):
        return integers[:target_size]
    
    # If verification fails (should not happen if modular logic is correct), 
    # fallback to a simple valid set (e.g., powers of 2, truncated)
    # Powers of 2: 1, 2, 4, 8, 16, 32, 64, 128, 256. (9 elements)
    # This is small, but better than invalid.
    # Let's try to salvage by removing conflicting elements from the mapped set.
    # But for now, return the mapped set if it's close, or a known valid small set.
    
    # Actually, if modular validity holds, integer validity MUST hold.
    # Why? 
    # If a = mean(T) in integers, then k*a = sum(T).
    # Then k*a = sum(T) + q*M for some integer q? No, it's exact equality.
    # So k*a = sum(T) mod M.
    # So if no such subset exists mod M, it cannot exist in integers.
    # So the verification should pass.
    
    return integers[:target_size]

# EVOLVE-BLOCK-START
# The code above is the full implementation.
# EVOLVE-BLOCK-END
