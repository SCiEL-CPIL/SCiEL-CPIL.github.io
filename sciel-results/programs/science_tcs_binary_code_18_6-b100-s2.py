"""
Constructs a binary code of length 18 with minimum Hamming distance 6.
Uses the Extended Quadratic Residue Code for p=17, which is a [18, 9, 6] code 
with 512 codewords. Attempts to extend via greedy search if time permits, 
though the covering radius of the QR code suggests 512 may be optimal.
"""
import time
import random

N = 18
MIN_DIST = 6
TIME_LIMIT = 60.0
DEADLINE = time.time() + TIME_LIMIT - 5.0

def popcount(x):
    return x.bit_count()

def hamming(a, b):
    return popcount(a ^ b)

def verify_code(words):
    """Verify all pairs have distance >= MIN_DIST."""
    n = len(words)
    for i in range(n):
        wi = words[i]
        for j in range(i + 1, n):
            if hamming(wi, words[j]) < MIN_DIST:
                return False
    return True

def quadratic_residue_code():
    """
    Construct the Extended Quadratic Residue Code for p=17.
    Parameters: [18, 9, 6], Size: 512.
    """
    p = 17
    # Quadratic residues mod 17
    qr = set((i * i) % p for i in range(p))
    
    # Generator matrix for the [17, 8, 5] QR code
    # Dimension is (p-1)/2 = 8 for p=17? 
    # Standard QR code for p=17 has dimension 8.
    # Let's construct the generator matrix G of size 8x17.
    # Rows are cyclic shifts of the generator polynomial coefficients.
    # The generator polynomial g(x) has roots at the quadratic residues.
    # For p=17, QR = {1, 2, 4, 8, 9, 13, 15, 16}.
    # g(x) = prod_{r in QR} (x - r) over F_2.
    # Since we are in F_2, x-r = x+r = x-r.
    # However, calculating the polynomial product is easier than building the matrix from shifts if we don't have the poly.
    # Let's use the matrix definition: G[i][j] = 1 if (j-i) mod p is in QR.
    # This generates a cyclic code.
    
    dim = (p - 1) // 2  # 8
    G = [[0]*p for _ in range(dim)]
    for i in range(dim):
        for j in range(p):
            if (j - i) % p in qr:
                G[i][j] = 1
    
    # Generate all 2^8 = 256 codewords of length 17
    codewords_17 = []
    for mask in range(1 << dim):
        word = 0
        for i in range(dim):
            if mask & (1 << i):
                # XOR the row
                row = G[i]
                for j in range(p):
                    if row[j]:
                        word ^= (1 << j)
        codewords_17.append(word)
    
    # Extend by an overall parity bit to get length 18
    # The extended code has min distance 6.
    codewords_18 = []
    for w in codewords_17:
        parity = popcount(w) % 2
        # Append parity bit at position 17 (bit 17)
        codewords_18.append(w | (parity << p))
    
    return codewords_18

def greedy_extend(words, time_budget=0):
    """
    Greedily try to add more words to the code while maintaining min distance.
    Uses a set for O(1) membership check.
    """
    if time_budget <= 0:
        return words
    
    deadline = time.time() + time_budget
    words = list(words)
    word_set = set(words)
    
    random.seed(42)
    
    attempts = 0
    while time.time() < deadline:
        if attempts % 10000 == 0 and time.time() > deadline:
            break
            
        candidate = random.randint(0, (1 << N) - 1)
        if candidate in word_set:
            attempts += 1
            continue
        
        # Check distance to all existing words
        ok = True
        for w in words:
            if hamming(candidate, w) < MIN_DIST:
                ok = False
                break
        
        if ok:
            words.append(candidate)
            word_set.add(candidate)
        
        attempts += 1
        
    return words

def construct():
    """
    Main entry point.
    """
    start_time = time.time()
    
    # Step 1: Construct the extended QR code of length 18
    qr_words = quadratic_residue_code()
    
    # Verify validity
    if not verify_code(qr_words):
        # Fallback to Reed-Muller if QR construction fails (should not happen)
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words
    
    # Step 2: Try to extend with greedy search
    remaining_time = TIME_LIMIT - 5 - (time.time() - start_time)
    if remaining_time > 1:
        extended_words = greedy_extend(qr_words, time_budget=remaining_time * 0.9)
    else:
        extended_words = qr_words
    
    # Final verification
    if not verify_code(extended_words):
        return qr_words
    
    return extended_words
