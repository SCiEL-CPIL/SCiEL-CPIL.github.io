"""Construct a large binary code of length 17 with minimum distance 6."""
import time
import random
from collections import deque

N = 17
MIN_DIST = 6
DEADLINE = 55.0  # seconds, safety margin before 60s kill

def _hamming(a, b):
    return (a ^ b).bit_count()

def _min_dist_to_set(word, codewords):
    """Return minimum Hamming distance from word to any codeword."""
    best = N + 1
    for c in codewords:
        d = (word ^ c).bit_count()
        if d < best:
            best = d
            if best < MIN_DIST:
                return best
    return best

def _build_seed_code():
    """
    Construct the 256-word [18,9,6] shortened quadratic-residue code.
    The [18,9,6] QR code over F_2: generator polynomial for QR(17) extended.
    We use the known structure: the [17,9,5] QR code extended to [18,9,6],
    then shortened by one coordinate to get [17,9,5]... 
    Actually, let's use the RM(1,4) code on 16 coords + 1 parity to get [17,5,8] (32 words),
    but that's only 32. For 256 words we need a [17,9,6] code.
    
    The extended QR code [18,9,6] has 256 codewords. Shortening one coordinate
    gives a [17,9,5] code with 256 words? No - shortening reduces length but 
    keeps dimension if we fix that coordinate to 0, giving [17,9,5] with 256 words.
    But we need distance 6, not 5.
    
    Alternative: Use the [18,9,6] code directly but we need length 17.
    Let me use a different approach: construct a [17,9,6] code if it exists,
    or use the 256 words from the extended QR by taking all words with bit 17 = 0
    in the [18,9,6] code, which gives [17,9,5]... 
    
    Actually, the task says "the 256-word linear reference is the [18,9,6] extended 
    quadratic-residue code shortened by one coordinate". So we get 256 words of 
    length 17 with distance at least 6 (shortening preserves or increases distance).
    
    Let me construct the [18,9,6] QR code. The QR code of length 17 over F_2 has 
    parameters [17,9,5]. Its extension is [18,9,6]. The generator polynomial for 
    QR(17) is the product of minimal polynomials of alpha^1 where alpha is a 
    primitive 17th root of unity. For p=17, the QR code has generator polynomial:
    g(x) = M_1(x) * M_4(x) where M_i are minimal polynomials.
    
    For simplicity, let me use a known [17,9,5] QR code generator matrix and 
    extend it, or just use a systematic construction.
    
    Actually, let me use a simpler approach: use the first-order Reed-Muller code 
    RM(1,4) which gives 32 words of length 16 with distance 8, extend to 17 bits.
    That's only 32 words though.
    
    For 256 words, I need a [17,9,6] code. Let me use the BCH code approach.
    The binary BCH code of length 17 with designed distance 6 has parameters [17,9,6].
    This is the QR code! Let me construct it.
    
    The generator polynomial for the [17,9,6] BCH/QR code:
    alpha = primitive element of GF(2^8) or use the cyclotomic cosets mod 17.
    Cyclotomic cosets mod 17 (order 2):
    C0 = {0}
    C1 = {1, 2, 4, 8, 16, 15, 13, 9}  (size 8)
    C3 = {3, 6, 12, 7, 14, 11, 5, 10}  (size 8)
    C5 = {5, 10, 3, 6, 12, 7, 14, 11}  = C3
    So we need minimal polynomials for alpha^1 and alpha^3.
    
    Let me just use a precomputed generator matrix for the [17,9,6] QR code.
    """
    # Generator matrix for [17,9,6] binary QR code (systematic form)
    # This is a known code. Let me use the parity-check based construction.
    # 
    # Alternative: use the fact that the [17,9,6] code is the dual of the [17,8,7] code.
    # 
    # For practical purposes, let me construct it using BCH theory.
    # alpha is a primitive 17th root of unity in GF(2^8).
    # We need alpha to be a root of x^17 - 1 over GF(2).
    # 
    # Let me use a simpler method: enumerate all 2^9 = 512 candidate words from 
    # a [17,9] linear code and keep those with min distance >= 6.
    # But that's circular.
    #
    # Let me just use the known generator matrix. The [17,9,6] QR code generator:
    
    # I'll use a different approach: start with RM(1,4) extended, and use 
    # a greedy algorithm to build up to ~256 words.
    
    # Actually, let me just use the 32-word RM(1,4) code as seed and 
    # greedily add words. The key insight is that we can potentially get 
    # well above 32 with a good greedy algorithm.
    
    # Start with the 32-word RM(1,4) code on 16 coordinates, with 17th bit = 0
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            # 17th bit (index 16) = 0
            words.append(word)
    return words

def _greedy_extend(codewords, deadline, rng):
    """
    Greedily extend the code by adding words that maintain min distance >= 6.
    Use a priority queue based on number of conflicts.
    """
    codewords = list(codewords)
    start_time = time.time()
    
    # Precompute which words conflict with current code
    # A word conflicts if its distance to any codeword is < 6
    
    # For efficiency, we'll iterate through candidate words in a smart order
    # and check if they can be added.
    
    # Strategy: iterate through all 2^17 words, but use early termination
    # in the distance check.
    
    added = 0
    # We'll do multiple passes, each time trying to add one more word
    # until we can't add any more or hit the deadline.
    
    # Use a set for O(1) lookup
    code_set = set(codewords)
    
    # For the greedy step, we want to find words that are "furthest" from 
    # the current code. We'll sample candidates.
    
    # Precompute: for each word in 0..2^17-1, we could compute min distance 
    # to code, but that's 2^17 * |code| which is too slow for large |code|.
    
    # Instead, use a smarter approach: 
    # 1. Generate random candidates and check
    # 2. Use a structured search
    
    # Let's do a structured approach: iterate through all words in Gray code 
    # order or just sequential, checking each one.
    
    # For |code| = 32, checking all 2^17 words takes 2^17 * 32 = 4M ops, fine.
    # For |code| = 256, it's 2^17 * 256 = 33M ops, still feasible in < 1s.
    
    max_code_size = 400  # safety cap
    iteration = 0
    
    while len(codewords) < max_code_size:
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Find a word to add: iterate through all possible words
        # and find one with min distance >= 6 to all codewords
        # To make this efficient, we'll use a random sampling approach 
        # combined with structured search.
        
        # First, try a structured scan
        found = False
        # Use a random starting point for diversity
        offset = rng.randint(0, 2**N - 1)
        
        # Check words in a shuffled order for diversity
        # But full shuffle of 2^17 is expensive. Use block-based randomization.
        
        # Simpler: just scan sequentially with early termination
        for i in range(2**N):
            if time.time() - start_time > DEADLINE - 5:
                return codewords
            
            word = (offset + i) % (2**N)
            if word in code_set:
                continue
            
            # Check if word can be added
            can_add = True
            for c in codewords:
                if (word ^ c).bit_count() < MIN_DIST:
                    can_add = False
                    break
            
            if can_add:
                codewords.append(word)
                code_set.add(word)
                added += 1
                found = True
                break
        
        if not found:
            # No more words can be added
            break
    
    return codewords

def construct():
    """
    Construct a large set of 17-bit words with pairwise Hamming distance >= 6.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    # Start with the 32-word RM(1,4) code
    codewords = _build_seed_code()
    
    # Greedily extend
    codewords = _greedy_extend(codewords, DEADLINE, rng)
    
    # Try a few random restarts with different seeds
    best = codewords
    
    while time.time() - start_time < DEADLINE - 10:
        rng2 = random.Random()
        codewords2 = _build_seed_code()
        # Shuffle the initial code to get different greedy paths
        rng2.shuffle(codewords2)
        codewords2 = _greedy_extend(codewords2, DEADLINE, rng2)
        
        if len(codewords2) > len(best):
            best = codewords2
        
        if time.time() - start_time > DEADLINE - 5:
            break
    
    return best
