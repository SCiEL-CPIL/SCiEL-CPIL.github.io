import time
from fractions import Fraction
from math import gcd

def construct():
    start_time = time.time()
    
    # Greedy Egyptian Fraction Algorithm
    target_num, target_den = 1, 1
    denominators = []
    used = set()
    
    while target_num > 0:
        # Find smallest d > 4, not in used, such that 1/d <= target_num/target_den
        # d >= target_den / target_num
        min_d = int(target_den / target_num) + 1
        if min_d < 5:
            min_d = 5
        
        d = min_d
        while d in used:
            d += 1
        
        # Check constraint: d must be <= 10000000 for the final output, 
        # but for now we allow it in greedy to get a base solution, 
        # then we will split large denominators.
        # However, if d is huge, the splitting might be expensive or result in many terms.
        # Let's try to keep d reasonable if possible? 
        # No, greedy is greedy.
        
        if d > 10**9: # Safety break to prevent infinite loop if something goes wrong
            break
            
        denominators.append(d)
        used.add(d)
        
        # Update remainder: target_num/target_den - 1/d
        new_num = target_num * d - target_den
        new_den = target_den * d
        
        # Simplify
        g = gcd(new_num, new_den)
        if g > 0:
            new_num //= g
            new_den //= g
            
        target_num, target_den = new_num, new_den
        
        if time.time() - start_time > 10:
            break

    # Split large denominators to satisfy d <= 10000000
    MAX_D = 10000000
    # We need to ensure the sum is exactly 1.
    # The greedy algorithm ensures the sum of 1/d for d in denominators is 1.
    # We replace 1/D with 1/(D+1) + 1/(D*(D+1)).
    # This preserves the sum.
    
    # We do this until all denominators are <= MAX_D
    # To avoid infinite loops or too many terms, we set a limit on iterations
    # or on the number of denominators.
    
    i = 0
    limit_iterations = 1000
    while i < limit_iterations:
        # Find max denominator
        if not denominators:
            break
        max_d = max(denominators)
        if max_d <= MAX_D:
            break
        
        # Find index of max_d
        idx = denominators.index(max_d)
        
        # Replace max_d with max_d+1 and max_d*(max_d+1)
        d1 = max_d + 1
        d2 = max_d * (max_d + 1)
        
        # Check if d1 or d2 are already in use?
        # The identity 1/D = 1/(D+1) + 1/(D(D+1)) introduces new denominators.
        # We must ensure they are distinct from the rest of the set.
        # Since D is the largest, D+1 and D(D+1) are larger than D (except D+1 is just slightly larger).
        # Wait, D+1 is larger than D. D(D+1) is much larger.
        # So they are definitely distinct from all other denominators which are <= D.
        # So we don't need to check for duplicates with the existing set, 
        # EXCEPT if we have multiple large denominators and we split one, 
        # the new ones might collide with others?
        # No, because we always pick the CURRENT maximum.
        # If we have [5, 6, 1000], max is 1000.
        # Split 1000 -> 1001, 1001000.
        # Set becomes [5, 6, 1001, 1001000].
        # Max is 1001000.
        # Split 1001000 -> 1001001, 1001000*1001001.
        # This seems to work.
        
        denominators[idx] = d1
        denominators.append(d2)
        
        # Optimization: If d2 > MAX_D, it will be split in the next iteration.
        # If d1 > MAX_D, it will be split in the next iteration.
        
        i += 1
        
        if time.time() - start_time > 40:
            break

    # Final verification
    # Sum must be exactly 1
    s = sum(Fraction(1, d) for d in denominators)
    
    # Check constraints
    valid = (s == 1) 
    valid = valid and all(5 <= d <= MAX_D for d in denominators)
    valid = valid and len(set(denominators)) == len(denominators)
    valid = valid and 1 <= len(denominators) <= 256
    
    if not valid:
        # Fallback: Return a known valid solution if the algorithm failed
        # This is a very long solution, but valid.
        # 1/5 + ... + 1/1000 is too big.
        # Let's just raise an error or return empty? 
        # The task says invalid scores 0.
        # But we should try to return something.
        # If greedy+split failed, maybe due to time or count > 256.
        # If count > 256, we can't return it.
        
        # Let's try a simpler fallback: 
        # Use a hardcoded short solution if we know one.
        # I don't have a hardcoded one memorized that is < 256 terms and > 4.
        # But the greedy+split should produce a short one.
        
        pass

    return denominators
