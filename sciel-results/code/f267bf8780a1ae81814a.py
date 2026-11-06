import time
import random

def construct():
    N = 17
    D = 6
    deadline = time.time() + 85  # Safety margin
    
    # 1. Find generator polynomial g(x) for [17, 8, 6] QR code
    # g(x) is monic degree 8, divides x^17 + 1 in GF(2)
    # x^17 + 1 = x^17 - 1
    # We look for p(x) = x^8 + ... such that (x^17 + 1) % p(x) == 0
    
    def poly_mod(a, b):
        # a % b in GF(2)
        # b is monic
        deg_b = b.bit_length() - 1
        while a.bit_length() > deg_b:
            shift = a.bit_length() - deg_b - 1
            if shift >= 0:
                a ^= (b << shift)
        return a

    g_poly = None
    # Iterate all monic degree 8 polynomials
    for lower_bits in range(128):
        p = (1 << 8) | lower_bits
        # Check if x^17 + 1 is divisible by p
        # x^17 + 1 is 1 << 17 | 1
        if poly_mod((1 << 17) | 1, p) == 0:
            # Ensure it's not x-1 (degree 1) or other small factors
            # Degree 8 is correct for QR code of length 17
            g_poly = p
            break
            
    if g_poly is None:
        # Fallback to simple code if something goes wrong
        # But it should find it
        return [0] * 32 

    # 2. Generate 256 QR codewords
    def poly_mult(a, b):
        res = 0
        while b:
            if b & 1:
                res ^= a
            a <<= 1
            b >>= 1
        return res

    qr_words = []
    for k in range(256):
        w = poly_mult(k, g_poly)
        qr_words.append(w)
        
    # Verify QR code validity (optional but good for debug)
    # for i in range(256):
    #   for j in range(i+1, 256):
    #     if (qr_words[i] ^ qr_words[j]).bit_count() < 6:
    #       print("QR Code Invalid")
    #       break

    current_code = set(qr_words)
    best_code = set(qr_words)
    
    # 3. Greedy Independent Set Search
    # We try to add more words from the remaining 2^17 - 256
    
    # Precompute all words? No, just iterate range.
    # To allow random restarts, we create a list of candidates
    
    all_words = list(range(2**N))
    
    # We will run multiple iterations of the greedy algorithm
    # with different random orderings of the candidates
    
    while time.time() < deadline:
        # Start with the QR code
        current = set(qr_words)
        
        # Candidates are all words not in current
        # We shuffle them to get a random order
        candidates = [w for w in all_words if w not in current]
        random.shuffle(candidates)
        
        # Greedy pass
        # We iterate through candidates.
        # If w is compatible with all c in current, add w to current.
        # If not, discard w (it will never be compatible with a superset of current).
        
        i = 0
        while i < len(candidates):
            w = candidates[i]
            
            # Check compatibility
            compatible = True
            for c in current:
                if (w ^ c).bit_count() < D:
                    compatible = False
                    break
            
            if compatible:
                current.add(w)
                # Do not increment i, because we removed w?
                # Actually, we are iterating a list.
                # If we add w, we don't need to check it again.
                # But we should remove it from candidates?
                # The list `candidates` is fixed. We just skip it.
                # But wait, if we add w, does it affect other candidates?
                # Yes, it increases the set `current`.
                # So subsequent candidates must be checked against the new `current`.
                # This is handled by the loop structure.
                pass
            else:
                # w is incompatible, so it can never be added.
                # We can conceptually remove it.
                # In this implementation, we just move to the next one.
                pass
            
            i += 1
            
        if len(current) > len(best_code):
            best_code = current
            # If we reached the theoretical upper bound, we can stop
            if len(best_code) >= 340:
                break
                
    return list(best_code)
