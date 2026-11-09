"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # Precompute LCM of 1..30
    def lcm(a, b):
        return abs(a * b) // gcd(a, b)
    
    L = reduce(lcm, range(1, 31))
    
    n = 30
    half = 15
    
    # Weights for first half (indices 0..14) correspond to denominators 1..15
    w1 = [L // (i + 1) for i in range(half)]
    # Weights for second half (indices 15..29) correspond to denominators 16..30
    w2 = [L // (i + 1) for i in range(half, n)]
    
    # Function to encode coefficients in {-1, 0, 1} into a base-3 integer
    # c[0] is the least significant digit
    def encode_coeffs(coeffs):
        val = 0
        power = 1
        for c in coeffs:
            # Map -1 -> 0, 0 -> 1, 1 -> 2
            # Actually, let's just use a simple mapping to an integer.
            # We can use base 3 with digits 0, 1, 2 representing -1, 0, 1.
            digit = c + 1
            val += digit * power
            power *= 3
        return val

    def decode_coeffs(encoded, length):
        coeffs = [0] * length
        power = 1
        for i in range(length):
            digit = (encoded // power) % 3
            coeffs[i] = digit - 1
            power *= 3
        return coeffs

    # Generate all sums and encoded coeffs for the first half
    # Start with [(0, 0)] where 0 is sum and 0 is encoded coeffs
    states1 = [(0, 0)]
    
    for i in range(half):
        w = w1[i]
        new_states = []
        for s, enc in states1:
            # Option 0: coeff is 0
            new_states.append((s, enc))
            # Option +1: coeff is 1
            # Digit at position i is 2 (since 1+1=2)
            enc_plus = enc + 2 * (3 ** i)
            new_states.append((s + w, enc_plus))
            # Option -1: coeff is -1
            # Digit at position i is 0 (since -1+1=0)
            enc_minus = enc + 0 * (3 ** i) # Wait, if digit is 0, it adds 0.
            # But we need to be careful. The encoding is sum(digit * 3^i).
            # If we start with 0, and set digit i to 0, it stays same? No.
            # The initial state is all zeros (coeffs 0). Enc is 0.
            # If coeff is 0, digit is 1. Enc contribution is 1 * 3^i.
            # My encode_coeffs function maps c=0 to digit 1.
            # So initial state (all 0 coeffs) should have enc = sum(1 * 3^i) = (3^15 - 1)/2.
            # Let's restart the encoding logic to be simpler.
            
            pass # The above logic was getting complicated. Let's use a simpler encoding.
        
        # Simpler encoding: Just store the tuple of coeffs? 
        # 14M tuples of 15 ints is too much memory.
        # Let's stick to base-3 but initialize correctly.
        # Digit 0: -1, Digit 1: 0, Digit 2: 1.
        # Initial coeffs all 0 -> Digits all 1.
        # Enc = sum(1 * 3^i for i in 0..14) = (3^15 - 1) // 2.
        
        # Let's re-implement generation with this understanding.
        pass

    # Corrected Generation Strategy
    # State: (sum, encoded_coeffs)
    # Init: sum=0, coeffs all 0 -> digits all 1.
    init_enc = sum(1 * (3 ** i) for i in range(half))
    
    states1 = [(0, init_enc)]
    
    for i in range(half):
        w = w1[i]
        pow3 = 3 ** i
        new_states = []
        for s, enc in states1:
            # Current digit is 1 (represents 0).
            # To change to -1 (digit 0): subtract 1 * pow3
            enc_minus = enc - pow3
            new_states.append((s - w, enc_minus))
            
            # Keep 0 (digit 1): no change
            new_states.append((s, enc))
            
            # Change to +1 (digit 2): add 1 * pow3
            enc_plus = enc + pow3
            new_states.append((s + w, enc_plus))
        
        states1 = new_states
        
        if time.time() > deadline - 15:
             # Fallback if too slow
             return [0] * 28 + [1, -1]

    # Build dictionary for first half: sum -> encoded_coeffs
    # If multiple coeffs give same sum, we keep one (the first one found)
    sum_to_enc1 = {}
    for s, enc in states1:
        if s not in sum_to_enc1:
            sum_to_enc1[s] = enc

    # Generate second half and check for matches
    # We want total sum S such that S/L is minimized non-zero.
    # So we look for S = 1, -1, 2, -2, ...
    
    states2 = [(0, init_enc)]
    
    for i in range(half):
        w = w2[i]
        pow3 = 3 ** i
        new_states = []
        for s, enc in states2:
            enc_minus = enc - pow3
            new_states.append((s - w, enc_minus))
            
            new_states.append((s, enc))
            
            enc_plus = enc + pow3
            new_states.append((s + w, enc_plus))
        
        states2 = new_states
        
        if time.time() > deadline - 5:
             # If we run out of time generating, we can't check.
             # But states2 generation is fast. The check is fast too.
             pass

    # Check for targets
    # We iterate through states2 and check if (target - s2) is in sum_to_enc1
    # To be efficient, we can just iterate through states2.
    
    targets = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5]
    
    best_result = None
    best_abs_val = float('inf')
    
    # Optimization: Instead of checking all targets for all s2, 
    # we can just check the most likely ones first.
    # Actually, just checking 1 and -1 is usually enough.
    
    for target in targets:
        found = False
        for s2, enc2 in states2:
            s1_needed = target - s2
            if s1_needed in sum_to_enc1:
                enc1 = sum_to_enc1[s1_needed]
                
                # Decode coeffs
                c1 = decode_coeffs(enc1, half)
                c2 = decode_coeffs(enc2, half)
                
                result = c1 + c2
                
                # Verify
                total_sum = sum(c * (L // (i+1)) for i, c in enumerate(result))
                if total_sum == target:
                    best_result = result
                    best_abs_val = abs(total_sum)
                    found = True
                    break
        
        if found:
            # Since we check targets in increasing abs order, the first found is optimal
            return best_result

    # If no small target found (very unlikely), fallback
    return [0] * 28 + [1, -1]

# EVOLVE-BLOCK-END
