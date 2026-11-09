import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 55  # Safety margin
    
    best_set = [31, 32, 34]
    best_size = 3
    
    # Precompute divisibility relationships for all possible subset sums
    # Max subset sum for 30 numbers from 1..256: at most 30*256 = 7680
    # But realistically with distinct numbers, max sum is around 256+255+...+227 ≈ 7440
    # Let's use a reasonable bound
    MAX_SUM = 8000
    
    # divisors[i] = set of j < i such that i % j == 0
    # This helps us check divisibility quickly
    divisors = [set() for _ in range(MAX_SUM + 1)]
    multiples = [set() for _ in range(MAX_SUM + 1)]
    for i in range(1, MAX_SUM + 1):
        for j in range(i * 2, MAX_SUM + 1, i):
            divisors[j].add(i)
            multiples[i].add(j)
    
    def is_valid_full(S):
        """Check if set S is valid using full subset sum enumeration."""
        if not S:
            return True
        sums = {0}
        for x in S:
            new_sums = set()
            for s in sums:
                new_sums.add(s + x)
            sums = sums | new_sums
        
        non_empty = [s for s in sums if s != 0]
        # Check distinctness (already guaranteed by set)
        # Check divisibility
        non_empty_set = set(non_empty)
        for u in non_empty:
            # Check if any multiple of u is in the set
            for m in multiples[u]:
                if m in non_empty_set:
                    return False
            # Check if any divisor of u is in the set
            for d in divisors[u]:
                if d in non_empty_set:
                    return False
        return True
    
    def try_add(S, x, sums_set):
        """
        Try to add x to S. S is assumed valid.
        sums_set is the set of all subset sums of S (including 0).
        Returns (True, new_sums_set) if valid, (False, None) otherwise.
        """
        # New sums are x + s for s in sums_set
        # Check distinctness: (sums_set) & (sums_set + x) should be empty
        # i.e., no s1, s2 in sums_set with s1 = s2 + x
        # Equivalently, sums_set and (sums_set + x) are disjoint
        
        new_sums = set()
        for s in sums_set:
            ns = s + x
            if ns in sums_set:
                return (False, None)
            new_sums.add(ns)
        
        # Now check divisibility among new_sums and between new_sums and sums_set
        all_sums = sums_set | new_sums
        non_empty_all = [s for s in all_sums if s != 0]
        non_empty_all_set = set(non_empty_all)
        
        for u in non_empty_all:
            for m in multiples[u]:
                if m in non_empty_all_set:
                    return (False, None)
            for d in divisors[u]:
                if d in non_empty_all_set:
                    return (False, None)
        
        return (True, all_sums)
    
    def greedy_build(candidates, start_S=None, start_sums=None):
        """Greedy build: iterate through candidates, add if valid."""
        if start_S is None:
            S = []
            sums_set = {0}
        else:
            S = list(start_S)
            sums_set = set(start_sums)
        
        for x in candidates:
            if x in S:
                continue
            ok, new_sums = try_add(S, x, sums_set)
            if ok:
                S.append(x)
                sums_set = new_sums
        
        return S, sums_set
    
    # Strategy: Use binary signature approach to ensure distinct subset sums
    # Then filter for divisibility
    def construct_binary_signature():
        """
        Assign each integer a unique power of 2 as its "signature".
        Numbers with unique highest bits ensure subset sum distinctness.
        """
        by_highest_bit = {}
        for n in range(1, 257):
            hb = n.bit_length() - 1
            if hb not in by_highest_bit:
                by_highest_bit[hb] = []
            by_highest_bit[hb].append(n)
        
        best_local = []
        
        for attempt in range(50):
            if time.time() - start_time > time_limit:
                break
            
            bit_positions = sorted(by_highest_bit.keys(), reverse=True)
            random.shuffle(bit_positions)
            
            S = []
            sums_set = {0}
            
            for bp in bit_positions:
                if time.time() - start_time > time_limit:
                    break
                
                candidates = by_highest_bit[bp][:]
                random.shuffle(candidates)
                
                for x in candidates:
                    ok, new_sums = try_add(S, x, sums_set)
                    if ok:
                        S.append(x)
                        sums_set = new_sums
                        break
            
            if len(S) > len(best_local):
                best_local = S[:]
        
        return best_local
    
    # Run binary signature approach
    binary_result = construct_binary_signature()
    if len(binary_result) > best_size:
        best_size = len(binary_result)
        best_set = binary_result[:]
    
    # Now do randomized greedy with larger numbers first
    # Larger numbers are less likely to divide smaller sums
    
    max_restarts = 5000
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Create candidate list biased towards larger numbers
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        # Sort by descending value with some randomness
        # Use a weighted random selection approach
        # For now, just shuffle and try greedy
        
        S, sums_set = greedy_build(candidates)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            
            # If we found a good set, try to extend it further
            # by trying all remaining numbers
            remaining = [x for x in range(1, 257) if x not in S]
            random.shuffle(remaining)
            S2, sums_set2 = greedy_build(remaining, S, sums_set)
            if len(S2) > best_size:
                best_size = len(S2)
                best_set = S2[:]
    
    # Try local search: swap one element for another to enable growth
    if best_size < 30 and time.time() - start_time < time_limit - 5:
        current_S = list(best_set)
        # Compute sums for current set
        sums_set = {0}
        for x in current_S:
            new_sums = set()
            for s in sums_set:
                new_sums.add(s + x)
            sums_set = sums_set | new_sums
        
        # Try removing each element and adding a different one
        for i in range(len(current_S)):
            if time.time() - start_time > time_limit - 2:
                break
            
            removed = current_S[i]
            S_reduced = current_S[:i] + current_S[i+1:]
            
            # Recompute sums without removed
            sums_reduced = {0}
            for x in S_reduced:
                new_sums = set()
                for s in sums_reduced:
                    new_sums.add(s + x)
                sums_reduced = sums_reduced | new_sums
            
            # Try adding each candidate
            for x in range(1, 257):
                if x in S_reduced:
                    continue
                ok, new_sums = try_add(S_reduced, x, sums_reduced)
                if ok:
                    new_S = S_reduced + [x]
                    if len(new_S) > best_size:
                        best_size = len(new_S)
                        best_set = new_S[:]
    
    # Final verification
    if len(best_set) > 0:
        if not is_valid_full(best_set):
            best_set = [31, 32, 34]
            best_size = 3
    
    return list(best_set)

if __name__ == "__main__":
    res = construct()
    print(res)
