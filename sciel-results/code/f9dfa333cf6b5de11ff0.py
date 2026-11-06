"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    N = 18
    MIN_DIST = 6
    deadline = time.time() + 55  # 55s safety margin
    
    def verify_code(codes):
        """Verify all pairs have Hamming distance >= MIN_DIST using vectorized approach."""
        if len(codes) <= 1:
            return True
        arr = np.array(codes, dtype=np.uint32)
        n = len(arr)
        # Check all pairs using broadcasting (memory efficient for n<=512)
        for i in range(n):
            diffs = arr[i] ^ arr[i+1:]
            d = np.bitwise_count(diffs)
            if np.any(d < MIN_DIST):
                return False
        return True
    
    # Construct Extended Quadratic Residue Code [18, 9, 6]
    # Use the standard generator polynomial for QR(17)
    # g(x) = x^8 + x^5 + x^4 + x^3 + 1
    g_poly = 0b100111001
    
    def generate_qr_code(g_poly):
        codes = []
        for msg in range(512):
            cw = 0
            for i in range(9):
                if (msg >> i) & 1:
                    cw ^= (g_poly << i)
            # Extend with parity bit
            parity = bin(cw).count('1') % 2
            cw_ext = cw | (parity << 17)
            codes.append(cw_ext)
        return codes
    
    # Try the QR code
    qr_codes = generate_qr_code(g_poly)
    
    if verify_code(qr_codes):
        return qr_codes
    
    # Fallback: Reed-Muller RM(1,4) padded to 18 bits
    # RM(1,4) is [16, 5, 8] code with 32 codewords
    basis = [
        0b1111111111111111,
        0b1010101010101010,
        0b0110011001100110,
        0b0011001100110011,
        0b0001000100010001
    ]
    
    rm_codes = []
    for m in range(32):
        cw = 0
        for i in range(5):
            if (m >> i) & 1:
                cw ^= basis[i]
        rm_codes.append(cw)
    
    if verify_code(rm_codes):
        return rm_codes
    
    return [0]

# EVOLVE-BLOCK-END
