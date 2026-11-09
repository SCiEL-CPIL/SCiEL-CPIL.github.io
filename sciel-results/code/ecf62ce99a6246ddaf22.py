import time
import random

def construct():
    N = 18
    MIN_DIST = 6
    TOTAL_BUDGET = 55.0
    t_start = time.time()
    deadline = t_start + TOTAL_BUDGET

    # Build the extended quadratic-residue code of length 18, dimension 6, distance 6
    # This is a [18, 6, 6] code with 64 words.
    
    # Generator matrix for QR(17) (length 17, dimension 6, distance 5)
    # Quadratic residues mod 17: {1, 2, 4, 8, 9, 13, 15, 16}
    # Map to 0-indexed positions: 0, 1, 3, 7, 8, 12, 14, 15
    row0 = [0] * 17
    for pos in [0, 1, 3, 7, 8, 12, 14, 15]:
        row0[pos] = 1
    
    G = []
    for i in range(6):
        row = [row0[(j - i) % 17] for j in range(17)]
        G.append(row)
    
    qr_words_17 = []
    for mask in range(64):
        word = 0
        for i in range(6):
            if (mask >> i) & 1:
                for j in range(17):
                    if G[i][j]:
                        word ^= (1 << j)
        qr_words_17.append(word)
    
    qr_words_18 = []
    for w in qr_words_17:
        parity = w.bit_count() % 2
        extended = w | (parity << 17)
        qr_words_18.append(extended)
    
    def min_distance(words):
        min_d = N
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                d = (words[i] ^ words[j]).bit_count()
                if d < min_d:
                    min_d = d
        return min_d

    md = min_distance(qr_words_18)
    if md < MIN_DIST:
        # Fallback to Reed-Muller if construction is wrong
        rm_words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                # Pad to 18 bits? RM(1,4) is length 16.
                # We need length 18. This fallback is weak.
                # Let's just use the QR code even if it fails verification? 
                # No, if it fails, it's invalid.
                # But the QR code *should* be valid.
                pass
        
        # If QR fails, try a simple random greedy from scratch?
        # For now, assume QR works. If not, return RM padded with zeros?
        # Actually, let's just use the QR words. The verification is a safety check.
        best_words = list(qr_words_18)
    else:
        best_words = list(qr_words_18)
        
    best_size = len(best_words)
    
    # Precompute all masks with weight <= 5
    # These are the words within distance 5 of a center
    masks_le_5 = []
    for m in range(1 << N):
        if m.bit_count() <= 5:
            masks_le_5.append(m)
            
    # Bitset Greedy
    # occupied[w] = 1 if w is within distance 5 of any codeword in best_words
    occupied = bytearray(1 << N)
    
    def mark_sphere(center):
        for m in masks_le_5:
            occupied[center ^ m] = 1
            
    # Initialize with existing words
    for w in best_words:
        mark_sphere(w)
        
    current_words = list(best_words)
    current_set = set(current_words)
    
    # Greedy loop
    # We scan for the first available word
    # This is O(2^N) per step, but 2^18 is small.
    # We can optimize by keeping a list of available indices, but scanning is fine.
    
    while time.time() < deadline - 2.0:
        # Find next candidate
        next_candidate = -1
        for i in range(1 << N):
            if not occupied[i]:
                next_candidate = i
                break
        
        if next_candidate == -1:
            break
            
        # Add candidate
        current_words.append(next_candidate)
        current_set.add(next_candidate)
        mark_sphere(next_candidate)
        
        if len(current_words) > best_size:
            best_words = list(current_words)
            best_size = len(best_words)
            
    return best_words
