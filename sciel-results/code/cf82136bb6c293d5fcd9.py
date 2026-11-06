"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    # Construct the extended quadratic residue code of length 18
    # QR code of length 17 over F_19, extended by parity bit
    # Parameters: [18, 9, 6] with 2^9 = 512 codewords
    
    p = 19
    # Quadratic residues mod 19
    qr = set((x * x) % p for x in range(1, p))
    qr = sorted(qr)
    # qr = [1, 4, 5, 6, 7, 9, 11, 16, 17]
    
    # The QR code generator: use the standard construction
    # For p ≡ 3 (mod 4), the extended QR code has dimension (p-1)/2 = 9
    # We construct the generator matrix G (9 x 17) then extend to 18
    
    # The generator matrix rows correspond to shifts of the QR indicator
    # Position i (0-indexed) corresponds to element (i+1) mod p, but we use 0..16
    # Let's use the standard: generator matrix where row k has 1s at positions
    # where (pos + k) mod p is a quadratic residue (for pos = 0..16)
    
    # Actually, for the QR code, the generator matrix is circulant with first row
    # being the indicator of QR elements at positions 0, 1, ..., 16
    # Position j corresponds to field element j (0 to 16), but 0 is not in QR set
    # Standard: use positions 1..17 mapped to 0..16, or 0..16 directly
    
    # Let me use the standard construction:
    # G[i][j] = 1 if (j - i) mod p is in QR_set (for j=0..16, i=0..8)
    # But 0 is not a QR, so we need to handle this carefully
    
    # Alternative standard construction:
    # The QR code has generator matrix where row i (i=0..8) has 1s at positions j
    # such that (j+i) mod 19 is a quadratic residue mod 19
    
    # Let's build it properly
    # Positions 0..16 correspond to field elements 0..16
    # QR set mod 19: {1, 4, 5, 6, 7, 9, 11, 16, 17}
    
    qr_set = set((x * x) % p for x in range(1, p))
    
    # Generator matrix: 9 rows, 17 columns
    # Row i: 1 at position j if (j + i + 1) % p is in qr_set
    # Using offset to avoid 0 issues
    G = [[0] * 17 for _ in range(9)]
    for i in range(9):
        for j in range(17):
            # Map position j to field element (j+1) % p, check if (j+1-i) % p is QR
            # Standard: G[i][j] = 1 iff (j - i) mod p in QR
            val = (j - i) % p
            if val in qr_set:
                G[i][j] = 1
    
    # Now generate all 2^9 = 512 codewords
    codewords = []
    for mask in range(512):
        word = 0
        for i in range(9):
            if (mask >> i) & 1:
                for j in range(17):
                    if G[i][j]:
                        word ^= (1 << j)
        # Extend with parity bit (bit 17)
        parity = word.bit_count() % 2
        word |= (parity << 17)
        codewords.append(word)
    
    # Remove duplicates (shouldn't be any, but just in case)
    codewords = list(set(codewords))
    
    # Verify minimum distance is at least 6
    # For efficiency, spot check or full check if time permits
    if len(codewords) <= 512:
        min_dist = 18
        n = len(codewords)
        for i in range(n):
            for j in range(i + 1, n):
                d = (codewords[i] ^ codewords[j]).bit_count()
                if d < min_dist:
                    min_dist = d
                if min_dist < 6:
                    break
            if min_dist < 6:
                break
        # If distance check fails, fall back to Reed-Muller
        if min_dist < 6:
            words = []
            for slope in range(16):
                for intercept in range(2):
                    word = 0
                    for point in range(16):
                        bit = ((slope & point).bit_count() + intercept) % 2
                        word |= bit << point
                    words.append(word)
            return words
    
    return codewords
# EVOLVE-BLOCK-END
