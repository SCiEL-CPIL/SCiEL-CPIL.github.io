import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 18
    MIN_DIST = 6
    
    # Step 1: Try to construct the Extended QR Code [18, 9, 6]
    # Generator polynomial for QR(17): x^8 + x^5 + x^4 + x^3 + 1
    # Bits: 1 0 0 1 1 1 0 0 1 (MSB to LSB) -> 313
    g_val = 313 
    # We need 9 basis vectors (codewords) of the QR(17) code
    # They are x^i * g(x) mod (x^17 - 1) for i=0..8
    
    basis = []
    modulus = (1 << 17) - 1 # x^17 - 1
    
    for i in range(9):
        # Calculate x^i * g(x)
        # g(x) is represented as an integer
        val = g_val << i
        # Reduce mod x^17 - 1
        # Since val < x^17 * x^9 = x^26, we might need multiple reductions
        # But x^17 = 1, so x^k = x^(k%17)
        # We can just take the lower 17 bits and the upper bits shifted?
        # No, (a*x^17 + b) mod (x^17-1) = a + b.
        # So we fold the bits.
        res = val
        while res >= (1 << 17):
            # Shift right by 17 and XOR with lower 17 bits
            high = res >> 17
            low = res & ((1 << 17) - 1)
            res = low ^ high
            # Note: res might still be >= 1<<17 if high was large?
            # Max val is 313 * 2^8 ~ 80k. 2^17 is 131k.
            # So one reduction is enough for i=0..8?
            # 313 * 256 = 80128 < 131072.
            # So for i=0..8, val < 2^17. No reduction needed?
            # Wait, g_val is degree 8. x^8 * g_val is degree 16.
            # So x^i * g_val for i=0..8 has max degree 16.
            # So no reduction is needed for the basis vectors!
        basis.append(res)
        
    # Verify basis vectors are distinct and valid
    # Generate all 512 codewords of length 17
    qr_words_17 = []
    for m in range(512):
        word = 0
        for j in range(9):
            if (m >> j) & 1:
                word ^= basis[j]
        qr_words_17.append(word)
        
    # Extend to length 18 with parity bit
    qr_words_18 = []
    for w in qr_words_17:
        parity = w.bit_count() % 2
        w18 = w | (parity << 17)
        qr_words_18.append(w18)
        
    # Validate the QR code
    # Check if all pairwise distances are >= 6
    # To be fast, we can sample or check against 0 first
    # If 0 is in the code (it is, m=0), all words must have weight >= 6
    valid_qr = True
    for w in qr_words_18:
        if w.bit_count() < 6:
            valid_qr = False
            break
            
    if valid_qr:
        # Full pairwise check is expensive for 512 words (512*512/2 = 130k checks)
        # But it's fast enough in Python? 130k * bit_count is ~0.1s.
        # Let's do it to be sure.
        for i in range(len(qr_words_18)):
            if time.time() > deadline:
                break
            for j in range(i+1, len(qr_words_18)):
                if (qr_words_18[i] ^ qr_words_18[j]).bit_count() < 6:
                    valid_qr = False
                    break
            if not valid_qr:
                break
                
    if valid_qr and time.time() < deadline:
        return qr_words_18
        
    # Fallback: Greedy Search
    # Start with the valid subset of QR words, or just {0}
    code = []
    for w in qr_words_18:
        if w.bit_count() >= 6:
            # Check against existing code
            ok = True
            for c in code:
                if (w ^ c).bit_count() < 6:
                    ok = False
                    break
            if ok:
                code.append(w)
                
    # If we have a good start, try to add more from the full space
    # But full space is 262k. Greedy might be slow if not vectorized.
    # Given time, we might just return the QR code if it's valid.
    # If QR code is invalid, the fallback greedy might be weak.
    # But the QR code *should* be valid.
    
    return code
