import numpy as np
import time
import random

def construct():
    N = 18
    MIN_DIST = 6
    
    # Construct the extended QR(17) code
    # p = 17
    p = 17
    # Quadratic residues modulo 17
    # x^2 mod 17 for x in 1..8
    residues = set()
    for x in range(1, p):
        residues.add((x * x) % p)
    # residues = {1, 2, 4, 8, 9, 13, 15, 16}
    
    # Create the base vector v of length 17
    # v[i] = 1 if i in residues else 0
    v = 0
    for i in range(p):
        if i in residues:
            v |= (1 << i)
    
    # Generate the 8 cyclic shifts of v
    basis = []
    current = v
    for _ in range((p - 1) // 2): # 8 shifts
        basis.append(current)
        # Rotate left by 1 bit within p bits
        # (current << 1) | (current >> (p-1))
        # Mask to p bits
        mask = (1 << p) - 1
        current = ((current << 1) | (current >> (p - 1))) & mask
        
    # Add the all-ones vector
    basis.append((1 << p) - 1)
    
    # Generate all 2^9 = 512 codewords
    code = []
    num_basis = len(basis)
    for mask in range(1 << num_basis):
        word = 0
        for i in range(num_basis):
            if mask & (1 << i):
                word ^= basis[i]
        
        # Compute weight of the 17-bit word
        weight = word.bit_count()
        
        # Determine parity bit
        # If weight is odd, add 1 to make it even (total weight even)
        # If weight is even, add 0
        parity = weight % 2
        
        # Form 18-bit integer
        # The 17 bits are the original, the 18th bit is the parity
        # The task says "18-bit words". Standard representation:
        # Usually, the parity bit is appended.
        # Integer value: (word << 1) | parity
        full_word = (word << 1) | parity
        
        code.append(full_word)
        
    # Local search to attempt to add more codewords
    # Start with the 512 codewords
    best_code = list(code)
    start_time = time.time()
    deadline = start_time + 55.0 # 5s margin
    
    # Pre-compute distances or use a fast check
    # We try to add random words or swap words
    # Since 512 is likely optimal, this will likely add nothing.
    
    # Helper to check if a word can be added
    def can_add(word, current_code):
        for c in current_code:
            if (word ^ c).bit_count() < MIN_DIST:
                return False
        return True

    # Try to add random words
    # We need to be careful not to spend too much time.
    # 512 words * 2^18 candidates is too many.
    # We can try to find words at distance >= 6 from all existing.
    
    # A smarter approach: Try to find a word that is at distance >= 6 from all 512.
    # This is equivalent to finding a codeword in the "union of spheres" complement.
    
    # Given the tightness of the bound, random search is unlikely to succeed quickly.
    # We will perform a limited random search.
    
    random.seed(42)
    attempts = 0
    max_attempts = 50000
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        # Generate a random 18-bit word
        w = random.randint(0, (1 << N) - 1)
        
        # Quick check: weight must be between 6 and 12?
        # If weight < 6, dist to 0 is < 6. 0 is in code?
        # 0 is in code (mask=0 -> word=0 -> weight=0 -> parity=0 -> full=0).
        # So dist(w, 0) = weight(w). Must be >= 6.
        # Also dist(w, all_ones) = 18 - weight(w). Must be >= 6.
        # So 6 <= weight(w) <= 12.
        
        wt = w.bit_count()
        if wt < 6 or wt > 12:
            continue
            
        # Check against all codewords
        valid = True
        for c in best_code:
            if (w ^ c).bit_count() < MIN_DIST:
                valid = False
                break
        
        if valid:
            best_code.append(w)
            # Restart attempts? Or just continue?
            # If we add one, we might be able to add more.
            # But we are likely at the limit.
            
    return best_code
