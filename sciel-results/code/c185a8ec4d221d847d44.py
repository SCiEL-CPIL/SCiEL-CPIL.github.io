"""Improved program using branch-and-bound backtracking with suffix-sum pruning."""
import time

_DEADLINE = 85.0  # Safety margin before 90s kill

def construct():
    start_time = time.time()
    
    # Upper bound on length:
    # The smallest sum is 1, largest is 256.
    # For length L, there are L(L+1)/2 consecutive blocks.
    # All sums must be distinct integers.
    # The sums must be within some range.
    # Actually, the sums are not necessarily in [1, 256].
    # But the elements are in [1, 256].
    # Let's just use a safe upper bound.
    # If we pick 1, 2, 4, 8, ..., 2^k, the sums are distinct.
    # Max power of 2 <= 256 is 128 (2^7). So we can get at least 8 elements (1, 2, 4, 8, 16, 32, 64, 128).
    # Wait, 1, 2, 4, 8, 16, 32, 64, 128 are 8 elements.
    # Next is 256, but 128+256 > 256? No, elements must be <= 256.
    # So 1, 2, 4, 8, 16, 32, 64, 128, 256 is 9 elements.
    # Let's verify if this is valid.
    # Sums of consecutive blocks of powers of 2 are distinct because binary representation is unique.
    # So length 9 is achievable.
    # Can we do better?
    # L(L+1)/2 <= MaxSum - MinSum.
    # MinSum is at least 1. MaxSum is at most 9*256 = 2304.
    # 9*10/2 = 45. 2304 is plenty.
    # The constraint is strict increase and distinct sums.
    
    best_seq = []
    
    def backtrack(current_seq, current_sums, max_len):
        nonlocal best_seq, start_time
        if time.time() > _DEADLINE:
            return
        
        if len(current_seq) > len(best_seq):
            best_seq = current_seq[:]
        
        # If we can't beat the best length, prune
        # This is a weak bound, but helps.
        # A better bound: 
        # The remaining elements must be > last element.
        # Let last = current_seq[-1] if current_seq else 0.
        # Next element >= last + 1.
        # Max possible length if we pick consecutive integers: 256 - last.
        # But we need distinct sums, which is harder.
        
        if not current_seq:
            start_val = 1
        else:
            start_val = current_seq[-1] + 1
            
        # If start_val > 256, we can't add more
        if start_val > 256:
            return
            
        # Try to extend
        # Heuristic: try smaller values first to allow more room?
        # Or try values that create new sums efficiently?
        
        # Suffix sums of current_seq
        # If current_seq is [a1, ..., ak], suffix sums are:
        # a_k
        # a_{k-1} + a_k
        # ...
        # a_1 + ... + a_k
        
        # When we add x, new suffix sums are:
        # x
        # x + a_k
        # x + a_{k-1} + a_k
        # ...
        # x + sum(current_seq)
        
        # We need these to not be in current_sums.
        # Also x must not be in current_sums (since it's a block of length 1).
        
        # Optimization:
        # If x > max(current_sums), then x is definitely new.
        # Also x + suffix_sum > max(current_sums) if x is large enough.
        
        # Let's iterate x from start_val to 256.
        # To find longer sequences faster, maybe we should prioritize?
        # Actually, for backtracking, the order matters for finding a good solution early.
        # Smaller x allows more elements to fit in [1, 256].
        # So iterate x from start_val upwards.
        
        for x in range(start_val, 257):
            if time.time() > _DEADLINE:
                return
            
            # Check if x is valid
            if x in current_sums:
                continue
                
            # Compute new suffix sums
            # We can maintain a list of suffix sums for efficiency
            # But for now, let's just compute them.
            
            # Actually, maintaining suffix sums is O(N) per step.
            # Let's maintain a list `suffixes` where suffixes[i] is sum of current_seq[i:].
            # Wait, I don't have suffixes in the closure. Let's pass it or recompute.
            # Recomputing is O(N^2) total for the path. N is small (~10-15).
            
            # Let's just compute the set of new sums to add.
            new_sums_to_add = set()
            new_sums_to_add.add(x)
            
            s = x
            for a in reversed(current_seq):
                s += a
                new_sums_to_add.add(s)
                
            # Check intersection
            if current_sums.intersection(new_sums_to_add):
                continue
                
            # Valid move
            current_seq.append(x)
            current_sums.update(new_sums_to_add)
            
            backtrack(current_seq, current_sums, max_len)
            
            current_seq.pop()
            current_sums.difference_update(new_sums_to_add)
            
            # If we found a sequence of length, say, 12, and we are at depth 10,
            # and we know max possible is 12, we might prune.
            # But we don't know max possible easily.
            
            # Pruning: if we have already found a sequence of length L_best,
            # and the maximum possible length from here is <= L_best, return.
            # Max possible length: if we pick start_val, start_val+1, ..., 
            # count = 256 - start_val + 1.
            # current_len + count <= len(best_seq) -> prune
            if len(current_seq) + (256 - start_val + 1) <= len(best_seq):
                return

    backtrack([], set(), 0)
    
    # Fallback if backtracking didn't find anything (shouldn't happen)
    if not best_seq:
        # Simple greedy fallback
        values, sums = [], set()
        for x in range(1, 257):
            tails = [x]
            total = x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        best_seq = values
        
    return best_seq
