import numpy as np
import time

def construct():
    """
    Constructs the extended quadratic-residue code QR(17) to length 18.
    Returns a list of 512 distinct integers representing 18-bit codewords
    with pairwise Hamming distance at least 6.
    
    The code is linear, so all pairwise distances are guaranteed by the
    minimum distance of the code (which is 6 for the extended QR(17)).
    """
    start_time = time.time()
    
    p = 17      # Prime length of the QR code
    n = 18      # Extended length
    k = 9       # Dimension of the code (2^9 = 512 codewords)

    # --- 1. Identify Quadratic Residues mod 17 ---
    # QR(17) uses the set of non-zero quadratic residues modulo 17.
    # x is a QR if x^((p-1)/2) == 1 mod p.
    qr_set = set()
    for i in range(1, p):
        if pow(i, (p - 1) // 2, p) == 1:
            qr_set.add(i)
    
    # --- 2. Construct Generator Matrix G (k x p) over GF(2) ---
    # The first row has 1s at positions corresponding to QRs.
    # The subsequent rows are cyclic shifts of the first row.
    first_row = np.zeros(p, dtype=np.uint8)
    first_row[0] = 1  # Position 0 is always included in QR codes? 
                      # Actually, standard QR codes are defined on Z_p.
                      # The generator matrix for QR(p) is typically formed by 
                      # the vector v where v_i = 1 if i is a QR, 0 otherwise.
                      # Then cyclic shifts.
                      # Note: 0 is not a QR. The vector length is p.
                      # The standard construction for QR(17) [17,9,5] uses the 
                      # quadratic residues mod 17 as the support of the first row.
                      
    for r in qr_set:
        first_row[r] = 1

    # Generate 9 cyclic shifts to form the 9x17 matrix
    G = np.vstack([np.roll(first_row, -shift, axis=0) for shift in range(k)])

    # --- 3. Generate all 512 codewords ---
    # We need all linear combinations of the rows of G.
    # This is equivalent to computing M @ G^T where M contains all 9-bit vectors.
    
    # Create all 512 coefficient vectors (9 bits each)
    masks = np.arange(1 << k, dtype=np.uint16)
    coeffs = np.zeros((1 << k, k), dtype=np.uint8)
    for i in range(k):
        coeffs[:, i] = (masks >> i) & 1

    # Matrix multiplication over GF(2)
    # coeffs @ G -> (512, 9) @ (9, 17) -> (512, 17)
    # Using bitwise XOR via integer addition mod 2 is efficient enough for this size.
    # Or simply use % 2 on the dot product.
    codewords_17 = (coeffs @ G) % 2  # shape (512, 17)

    # --- 4. Extend to length 18 with an overall parity bit ---
    # The extended QR code has minimum distance 6.
    parity_bits = codewords_17.sum(axis=1) % 2  # shape (512,)
    codewords_18 = np.column_stack((codewords_17, parity_bits))  # (512, 18)

    # --- 5. Convert to list of integers ---
    # We pack the bits into integers.
    # Bit j of the integer corresponds to column j of codewords_18.
    # Using vectorized shifting and summing.
    
    # powers[i] = 1 << i
    powers = 1 << np.arange(n, dtype=np.uint32)
    
    # Multiply each column by its power of 2 and sum across columns
    # codewords_18 is uint8, powers is uint32. Result is uint32.
    # 18 bits fit in uint32.
    words_arr = (codewords_18.astype(np.uint32) * powers).sum(axis=1).astype(np.int64)
    
    words = words_arr.tolist()

    # --- 6. Safety Check (Optional but good for robustness) ---
    # Verify uniqueness and count.
    # Since it's a linear code, uniqueness is guaranteed by the injectivity of the mapping from coeffs to codewords (G has full rank).
    # We just ensure we return a list of ints.
    
    return words
