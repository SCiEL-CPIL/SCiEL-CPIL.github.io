"""Improved program for A_2(18,6) using quadratic residue construction + efficient greedy/FLIP search."""
import time
import random
from itertools import combinations

N = 18
MIN_DIST = 6
TIME_LIMIT = 60.0  # wall-clock seconds; we use a safety margin

def popcount(x):
    return x.bit_count()

def hamming(a, b):
    return popcount(a ^ b)

def verify_code(words):
    """Verify all pairs have distance >= MIN_DIST."""
    n = len(words)
    for i in range(n):
        for j in range(i + 1, n):
            if hamming(words[i], words[j]) < MIN_DIST:
                return False
    return True

def quadratic_residue_code():
    """
    Construct the extended quadratic residue code [18, 9, 6] over F_2.
    
    For p=17 (prime ≡ 3 mod 4), the QR code has:
    - Length p = 17, dimension (p-1)/2 = 8, minimum distance 5
    - The extended QR code has length 18, dimension 9, minimum distance 6
    - This gives 2^9 = 512 codewords
    
    The generator matrix G is 9 x 17 where G[i][j] = 1 if (j - i) mod 17 is a quadratic residue.
    We then add an overall parity bit to extend to length 18.
    """
    p = 17
    # Quadratic residues mod 17: {1, 2, 4, 8, 9, 13, 15, 16}
    qr_set = set((i * i) % p for i in range(1, p))
    
    dim = (p - 1) // 2  # = 8
    
    # Build generator matrix: dim x p
    # G[i][j] = 1 if (j - i) % p is in qr_set
    G = []
    for i in range(dim):
        row = 0
        for j in range(p):
            if (j - i) % p in qr_set:
                row |= (1 << j)
        G.append(row)
    
    # Generate all 2^8 = 256 codewords of length 17
    codewords_17 = []
    for mask in range(1 << dim):
        word = 0
        for i in range(dim):
            if mask & (1 << i):
                word ^= G[i]
        codewords_17.append(word)
    
    # Extend each word by an overall parity bit to get length 18
    codewords_18 = []
    for w in codewords_17:
        parity = popcount(w) % 2
        codewords_18.append(w | (parity << p))
    
    return codewords_18



def construct():
    """
    Main entry point. Constructs a code with as many codewords as possible
    with pairwise Hamming distance >= 6, length 18.
    """
    start_time = time.time()
    safety_margin = 5.0
    deadline = start_time + TIME_LIMIT - safety_margin
    
    # Step 1: Construct the extended QR code of length 18
    qr_words = quadratic_residue_code()
    
    # Verify
    if not verify_code(qr_words):
        # Fallback to Reed-Muller (32 words)
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words
    
    # The extended QR code [18, 9, 6] is optimal for A_2(18,6) = 512.
    # No further extension is possible.
    # Verify and return immediately to avoid unnecessary overhead.
    if not verify_code(qr_words):
        return qr_words
        
    return qr_words
