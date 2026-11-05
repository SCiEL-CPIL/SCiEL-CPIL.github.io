import time
import random
import numpy as np

# EVOLVE-BLOCK-START
N = 18
MIN_DIST = 6

def hamming_distance(a, b):
    return (a ^ b).bit_count()

def reed_muller_construction():
    """First-order Reed-Muller code on 16 coordinates, extended by two zero bits.
    Every affine Boolean function on F_2^4 is a word: 32 words, distance 8."""
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    return words

def is_valid_word(candidate, words):
    """Check if candidate has distance >= MIN_DIST from all words in words."""
    for w in words:
        if (candidate ^ w).bit_count() < MIN_DIST:
            return False
    return True

def greedy_add(words, word_set, candidate):
    """Try to add candidate to words if valid. Returns True if added."""
    if candidate in word_set:
        return False
    for w in words:
        if (candidate ^ w).bit_count() < MIN_DIST:
            return False
    words.append(candidate)
    word_set.add(candidate)
    return True

def greedy_random_construction(seed=42, max_iter=1000000, deadline=None, base_words=None):
    """Greedy random construction to add words to a base code."""
    rng = random.Random(seed)
    if base_words is None:
        words = reed_muller_construction()
    else:
        words = list(base_words)
    word_set = set(words)
    
    for i in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        # Generate random word with various weight distributions
        r = rng.random()
        if r < 0.1:
            weight = 0
        elif r < 0.25:
            weight = rng.choice([1, 1, 2, 2, 2])
        elif r < 0.5:
            weight = rng.choice([3, 3, 3, 4, 4, 4, 5])
        elif r < 0.75:
            weight = rng.choice([5, 6, 6, 7, 7, 8])
        elif r < 0.9:
            weight = rng.choice([8, 9, 9, 10, 10])
        else:
            weight = rng.randint(10, N)
        
        positions = rng.sample(range(N), min(weight, N))
        word = 0
        for p in positions:
            word |= 1 << p
        
        if word in word_set:
            continue
        
        if is_valid_word(word, words):
            words.append(word)
            word_set.add(word)
    
    return words

def greedy_with_local_search(seed=42, max_iter=500000, deadline=None, base_words=None):
    """Greedy construction with occasional local search perturbations."""
    rng = random.Random(seed)
    if base_words is None:
        words = reed_muller_construction()
    else:
        words = list(base_words)
    word_set = set(words)
    
    for i in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        # Occasionally do a local search: try flipping bits of an existing word
        if rng.random() < 0.1 and len(words) > 0:
            base_idx = rng.randint(0, len(words) - 1)
            base = words[base_idx]
            # Try flipping 1-3 bits
            n_flips = rng.randint(1, 3)
            flip_positions = rng.sample(range(N), n_flips)
            candidate = base
            for p in flip_positions:
                candidate ^= (1 << p)
            if candidate not in word_set and is_valid_word(candidate, words):
                words.append(candidate)
                word_set.add(candidate)
        else:
            # Standard random generation
            weight = rng.choice([0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 7, 8, 9])
            positions = rng.sample(range(N), min(weight, N))
            word = 0
            for p in positions:
                word |= 1 << p
            
            if word in word_set:
                continue
            
            if is_valid_word(word, words):
                words.append(word)
                word_set.add(word)
    
    return words

def systematic_search(seed=42, deadline=None):
    """Systematic search over low-weight words, then medium-weight."""
    rng = random.Random(seed)
    words = []
    word_set = set()
    
    # First, add the all-zero word
    words.append(0)
    word_set.add(0)
    
    # Generate all words of weight 1 through 5
    # This is manageable: C(18,1)+C(18,2)+C(18,3)+C(18,4)+C(18,5) = 18+153+816+3060+8568 = 12615
    from itertools import combinations
    
    for weight in range(1, 6):
        if deadline and time.time() > deadline:
            break
        for positions in combinations(range(N), weight):
            if deadline and time.time() > deadline:
                break
            word = 0
            for p in positions:
                word |= 1 << p
            if word in word_set:
                continue
            if is_valid_word(word, words):
                words.append(word)
                word_set.add(word)
    
    # Then try random higher-weight words
    for i in range(200000):
        if deadline and time.time() > deadline:
            break
        weight = rng.randint(6, 12)
        positions = rng.sample(range(N), weight)
        word = 0
        for p in positions:
            word |= 1 << p
        if word in word_set:
            continue
        if is_valid_word(word, words):
            words.append(word)
            word_set.add(word)
    
    return words

def construct_quadratic_residue_code():
    """
    Constructs the extended quadratic residue code of length 18.
    This is the [18, 9, 6] code, containing 512 codewords.
    """
    # GF(2^8) arithmetic
    # Primitive polynomial: x^8 + x^4 + x^3 + x^2 + 1
    # Let's use a standard one, e.g., 0x11D (x^8+x^4+x^3+x+1) or 0x187 (x^8+x^6+x^5+x^4+1)
    # We need to find a primitive 17th root of unity.
    # 17 divides 2^8 - 1 = 255.
    # We need an element alpha in GF(2^8) such that alpha^17 = 1 and alpha != 1.
    # Order of alpha must be 17.
    
    # Build log/exp tables for GF(2^8)
    # Using primitive polynomial x^8 + x^4 + x^3 + x^2 + 1 (0x119)
    # Let's pick one that works.
    # Actually, any primitive polynomial will do, as long as we find the right alpha.
    
    # Let's implement GF(2^8) multiplication
    def gf_mult(a, b, poly=0x119):
        # poly is the reduction polynomial, excluding the x^8 term
        # e.g., x^8 + x^4 + x^3 + x^2 + 1 -> 0x19 (x^4+x^3+x^2+1)
        # Wait, 0x119 is x^8+x^4+x^3+x^2+1. The lower 8 bits are 0x19.
        res = 0
        while b:
            if b & 1:
                res ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= poly
        return res

    # Find primitive element
    # We need to check if an element is primitive.
    # For GF(2^8), primitive elements have order 255.
    # But we just need an element of order 17.
    # Any element alpha = g^k where g is primitive and k is multiple of 255/17 = 15.
    # So if we find a primitive g, then alpha = g^15 will have order 17.
    
    # Let's find a primitive polynomial and element.
    # Using x^8 + x^4 + x^3 + x^2 + 1 (0x119)
    # Let g = x (which is 0x02 in field representation if x is the generator)
    # Actually, it's easier to just iterate through all 255 non-zero elements and find one with order 17.
    
    # To find order, we check if a^17 == 1 and a^d != 1 for d < 17.
    # Since 17 is prime, we just need a^17 == 1 and a != 1.
    
    alpha = None
    for a in range(1, 256):
        # Check if a^17 == 1
        # Compute a^17
        val = 1
        for _ in range(17):
            val = gf_mult(val, a, 0x19) # 0x19 is lower 8 bits of 0x119
        if val == 1 and a != 1:
            alpha = a
            break
            
    if alpha is None:
        return [] # Fallback
        
    # Now we have alpha, a primitive 17th root of unity.
    # We need the minimal polynomial of alpha over GF(2).
    # The roots are alpha, alpha^2, alpha^4, ..., alpha^(2^7).
    # These are the conjugates.
    # The polynomial is M(x) = (x - alpha) (x - alpha^2) ... (x - alpha^128)
    # In GF(2), subtraction is addition.
    # M(x) = x^8 + c7 x^7 + ... + c0
    
    # We can compute the coefficients by multiplying the linear factors.
    # Start with P(x) = x
    # Multiply by (x + root) for each root.
    
    def poly_mul(p, q):
        # p, q are lists of coefficients [c0, c1, ...]
        # Returns list of coefficients of product
        res = [0] * (len(p) + len(q) - 1)
        for i in range(len(p)):
            for j in range(len(q)):
                res[i+j] ^= p[i] & q[j]
        return res
        
    # Roots are alpha^(2^i) for i=0..7
    # We need to compute alpha^(2^i)
    roots = []
    curr = alpha
    for i in range(8):
        roots.append(curr)
        curr = gf_mult(curr, curr, 0x19)
        
    # Build polynomial
    # M(x) = prod (x + r_i)
    # Note: In GF(2), x - r = x + r
    # So factors are (x + r_i) -> coefficients [r_i, 1]
    
    M = [1] # 1
    for r in roots:
        factor = [r, 1]
        M = poly_mul(M, factor)
        
    # M is now [c0, c1, ..., c8]
    # The generator polynomial g(x) is M(x).
    # The code is cyclic of length 17.
    # Codewords are c(x) = m(x) g(x) where deg(m) < 9.
    # c(x) will have degree < 17.
    
    codewords_17 = []
    for m in range(256): # 2^8 messages? No, dimension is 9.
        # Wait, dimension is (17+1)/2 = 9.
        # So message space is 2^9 = 512.
        # g(x) has degree 8.
        # m(x) has degree < 9.
        pass
        
    # Correction:
    # The generator polynomial has degree 8.
    # The message polynomial m(x) has degree < 9.
    # So there are 2^9 = 512 messages.
    
    codewords_17 = []
    for m_val in range(512):
        # Construct m(x) coefficients
        m_coeffs = []
        for i in range(9):
            m_coeffs.append((m_val >> i) & 1)
            
        # Multiply m(x) by g(x)
        # g_coeffs = M (reversed? M is [c0, c1...c8])
        # Standard poly mult
        c_coeffs = poly_mul(m_coeffs, M)
        
        # c_coeffs has length 9+8-1 = 16?
        # deg(m) < 9, deg(g) < 9 (actually deg is 8).
        # Max degree 8+8=16.
        # So c_coeffs has 17 coefficients (indices 0 to 16).
        # If length is less than 17, pad with zeros.
        while len(c_coeffs) < 17:
            c_coeffs.append(0)
            
        # Convert to integer
        word = 0
        for i in range(17):
            if c_coeffs[i]:
                word |= (1 << i)
                
        codewords_17.append(word)
        
    # Extend to length 18 by adding parity bit
    codewords_18 = []
    for w in codewords_17:
        parity = w.bit_count() % 2
        w_ext = (w << 1) | parity
        codewords_18.append(w_ext)
        
    return codewords_18

def construct():
    """
    Constructs the largest possible collection of 18-bit words with pairwise
    Hamming distance at least 6.
    """
    start_time = time.time()
    deadline = start_time + 80.0
    
    # Try the optimal QR code construction first
    try:
        qr_words = construct_quadratic_residue_code()
        if len(qr_words) == 512:
            # Verify validity quickly
            # Sample check or full check? 512 words is small enough for full check in Python?
            # 512 * 511 / 2 = 130k pairs.
            # Each pair: XOR and bit_count.
            # 130k * 100ns = 13ms. Very fast.
            valid = True
            for i in range(len(qr_words)):
                for j in range(i+1, len(qr_words)):
                    if (qr_words[i] ^ qr_words[j]).bit_count() < MIN_DIST:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                return qr_words
    except Exception:
        pass
        
    # Fallback to greedy search
    best_words = []
    
    if time.time() < deadline:
        words = systematic_search(seed=42, deadline=deadline)
        if len(words) > len(best_words):
            best_words = words
            
    seeds = [42, 123, 999, 2023, 777, 555, 111, 888, 333, 666, 1234, 5678, 9012, 3456, 7890]
    for seed in seeds:
        if time.time() > deadline:
            break
        current_words = greedy_random_construction(seed=seed, max_iter=500000, deadline=deadline)
        if len(current_words) > len(best_words):
            best_words = current_words
            
    if not best_words:
        best_words = reed_muller_construction()
        
    return best_words

# EVOLVE-BLOCK-END
