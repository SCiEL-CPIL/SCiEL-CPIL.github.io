import math
import time

def construct():
    deadline = time.time() + 50.0
    max_mod = 128
    
    # Initial set: powers of 2
    # [residue, modulus]
    current_set = [[(1 << k) - 1, 1 << (k + 1)] for k in range(7)]
    
    # Candidates: even numbers 2..128 not in current_set
    # We must add multiples of 4 to be compatible with m=2 (residue 1, odd)
    # because gcd(2, m) must be 2, requiring residue parity flip.
    # If m is 2*odd, gcd(2, m)=2. r_m must be even.
    # If we add two 2*odd numbers, their gcd is 2, requiring parity flip, but both are even. Fail.
    # So we can add at most one 2*odd number.
    # But 4 is in the set. gcd(4, 2*odd) = 2. r_{2*odd} must be even. r_4 is odd. OK.
    # However, we established we can't have two 2*odd.
    # Multiples of 4 are safer.
    
    candidates = [m for m in range(2, max_mod + 1) if m % 2 == 0]
    # Remove those already in current_set
    current_mods = {m for _, m in current_set}
    candidates = [m for m in candidates if m not in current_mods]
    
    # Sort candidates by something? Maybe smaller gcds are easier?
    # Or just try all.
    
    def is_compatible(res, mod, existing_set):
        for r_ex, m_ex in existing_set:
            g = math.gcd(mod, m_ex)
            if (res - r_ex) % g == 0:
                return False
        return True

    def find_residue(mod, existing_set):
        for r in range(mod):
            if is_compatible(r, mod, existing_set):
                return r
        return None

    # Try to add candidates
    # We should probably prioritize moduli that are "harder" or "easier"?
    # Let's just iterate.
    
    # To maximize, we might want to try adding in a specific order.
    # Multiples of high powers of 2 might be easier?
    
    added_count = 0
    # We can try to add all possible.
    # Since we have a deadline, we can do a simple greedy.
    
    # Re-sort candidates to try multiples of 4 first?
    # Actually, let's just try to add any valid one.
    
    # To ensure we get a good result, we can try to add in order of modulus.
    
    for m in candidates:
        if time.time() > deadline:
            break
        r = find_residue(m, current_set)
        if r is not None:
            current_set.append([r, m])
            added_count += 1
            
    return current_set
