import time
import random
import numpy as np

def construct():
    N = 17
    MIN_DIST = 6
    start_time = time.time()
    time_limit = 80.0  # Safety margin for 90s limit

    def hamming_dist(a, b):
        return bin(a ^ b).count('1')

    # Build the 256-word shortened quadratic residue code as a base set.
    # The [18,9,6] extended quadratic-residue code over GF(2) has a generator matrix
    # based on the quadratic residues of 17: {1,2,4,8,9,13,15,16}.
    # We construct the [17,9,6] code by shortening one coordinate.
    
    # The QR code of length 17 has generator polynomial g(x) = 1 + x + x^4 + x^6 + x^7 + x^10 + x^11 + x^14 + x^16
    # Actually, let's construct the [17,9,6] code properly.
    # The quadratic residues mod 17 are {1,2,4,8,9,13,15,16}
    # The generator polynomial for the [17,9,6] QR code is:
    # g(x) = (x^2 + x + 1)(x^3 + x + 1)(x^4 + x^3 + x^2 + x + 1) ... no
    # Let me use a known generator matrix approach.
    
    # For the [18,9,6] extended QR code, we can use the fact that it's a cyclic code.
    # The generator polynomial for the [17,9,6] QR code over GF(2) is:
    # g(x) = x^8 + x^7 + x^5 + x^4 + x^3 + x + 1
    # Let's verify: degree 8, so k = 17-8 = 9. Good.
    
    # Actually, the standard QR code [17,9,6] has generator polynomial:
    # g(x) = 1 + x + x^4 + x^6 + x^7 + x^10 + x^11 + x^14 + x^16
    # Wait, that's degree 16, which would give k=1. That's wrong.
    
    # Let me look up the correct generator polynomial.
    # The [17,9,6] quadratic residue code has generator polynomial of degree 8.
    # g(x) = x^8 + x^7 + x^5 + x^4 + x^3 + x + 1
    
    # Let's construct all 2^9 = 512... wait, that's too many. k=9 means 2^9=512 codewords?
    # No, [17,9,6] means length 17, dimension 9, distance 6. So 2^9 = 512 codewords.
    # But the task says the linear reference is 256 words. So it must be [17,8,6] or similar.
    # Actually, re-reading: "the 256-word linear reference is the [18,9,6] extended quadratic-residue code shortened by one coordinate"
    # [18,9,6] has 2^9 = 512 codewords. Shortening by one coordinate: we take all codewords with 0 in that coordinate.
    # That gives 2^8 = 256 codewords, forming a [17,8,6] code (or possibly [17,8,7]).
    
    # So I need to construct the [18,9,6] extended QR code, then shorten it.
    # The extended QR code of length 18 is obtained by adding a parity bit to the [17,9,6] QR code.
    # So first construct [17,9,6], extend to [18,9,6], then shorten.
    
    # Generator polynomial for [17,9,6] QR code:
    # g(x) = x^8 + x^7 + x^5 + x^4 + x^3 + x + 1
    # In binary: 1 1 0 1 1 1 0 1 1 (from x^8 down to x^0)
    # That's 255 in binary? Let me check: 110111011 = 255? 
    # 128+64+0+16+8+4+0+2+1 = 223. So g = 223.
    
    g_poly = 0b110111011  # 223
    
    # Generate all codewords of the [17,9,6] cyclic code
    # A codeword is c(x) = m(x) * g(x) mod (x^17 - 1)
    # Since we're in GF(2), multiplication is XOR-based.
    
    def multiply_poly(a, b, modulus):
        """Multiply two polynomials over GF(2), reducing mod modulus."""
        result = 0
        while a:
            if a & 1:
                result ^= b
            a >>= 1
            b <<= 1
        # Reduce mod modulus
        # modulus is x^17 - 1, which means x^17 = 1
        # So we need to reduce terms of degree >= 17
        deg_mod = 17
        while b != 0:
            deg_b = b.bit_length() - 1
            if deg_b >= deg_mod:
                # x^deg_b = x^(deg_b - 17) * x^17 = x^(deg_b-17)
                # So we XOR with modulus shifted
                shift = deg_b - deg_mod
                result ^= (modulus << shift)
                # Actually, we need to reduce properly
                # Let me use a different approach
        return result

    # Simpler approach: just enumerate all 2^9 message polynomials and compute codewords
    # by polynomial multiplication mod (x^17 - 1)
    
    def poly_mult_gf2(a, b):
        """Multiply two polynomials over GF(2)."""
        result = 0
        while a:
            if a & 1:
                result ^= b
            a >>= 1
            b <<= 1
        return result
    
    def poly_mod_gf2(a, modulus):
        """Reduce polynomial a mod modulus over GF(2).
        modulus = x^17 - 1, so x^17 = 1."""
        # x^17 = 1 means we can replace x^17 with 1, x^18 with x, etc.
        # So x^(17+k) = x^k
        result = a
        # Handle high degrees
        while result.bit_length() > 17:
            deg = result.bit_length() - 1
            # x^deg = x^(deg-17) * x^17 = x^(deg-17)
            # So we XOR with x^(deg-17)
            result ^= (1 << (deg - 17))
            # Clear the high bit
            result &= ~(1 << deg)
        return result
    
    # Generate all 512 codewords of the [17,9,6] code
    qr_codewords_17 = []
    for msg in range(512):
        # msg is the message polynomial (degree < 9)
        cw = poly_mult_gf2(msg, g_poly)
        cw = poly_mod_gf2(cw, None)  # mod x^17 - 1
        # Ensure it's 17 bits
        cw &= (1 << 17) - 1
        qr_codewords_17.append(cw)
    
    # Extend to [18,9,6] by adding a parity bit
    # The parity bit is the sum of all bits mod 2
    extended_codewords = []
    for cw in qr_codewords_17:
        parity = bin(cw).count('1') % 2
        ext_cw = (cw << 1) | parity  # bit 17 is parity
        extended_codewords.append(ext_cw)
    
    # Shorten by one coordinate: take all codewords with 0 in coordinate 0 (LSB)
    # Wait, shortening means we fix one coordinate to 0 and remove it.
    # Let's fix coordinate 17 (the parity bit) to 0, then remove it.
    # That gives us the original [17,9,6] code words with even parity... 
    # Actually, let's fix coordinate 0 to 0.
    
    # Shorten by fixing the last bit (bit 17) to 0
    shortened = []
    for cw in extended_codewords:
        if (cw >> 17) & 1 == 0:
            # Remove bit 17, keep bits 0-16
            shortened.append(cw & ((1 << 17) - 1))
    
    # Remove duplicates
    shortened = list(set(shortened))
    
    # Verify minimum distance
    if len(shortened) > 1:
        min_d = min(hamming_dist(a, b) for i, a in enumerate(shortened) for b in shortened[i+1:])
        # Should be at least 6
    
    base_words = shortened
    if len(base_words) < 2:
        # Fallback: use Reed-Muller-like base
        base_words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    dot = bin(slope & point).count('1') % 2
                    bit = (dot + intercept) % 2
                    word |= bit << point
                base_words.append(word)
    
    codewords = list(base_words)
    best_codewords = list(codewords)
    best_size = len(codewords)
    
    rng = random.Random(42)
    
    def is_compatible_with_all(w, word_list):
        for c in word_list:
            if hamming_dist(w, c) < MIN_DIST:
                return False
        return True
    
    # Phase 1: Greedy addition of random compatible words
    attempts = 0
    max_attempts_greedy = 300000
    while attempts < max_attempts_greedy:
        if time.time() - start_time > time_limit * 0.3:
            break
        
        w = rng.getrandbits(N)
        if w not in codewords and is_compatible_with_all(w, codewords):
            codewords.append(w)
            if len(codewords) > best_size:
                best_size = len(codewords)
                best_codewords = list(codewords)
        attempts += 1
    
    # Phase 2: Structured candidate generation and clique finding
    # Generate candidates from weight-7,8,9 vectors and mutations of existing codewords
    candidate_set = set()
    
    # Add weight-7,8,9 vectors
    for weight in [7, 8, 9]:
        # Sample some vectors of this weight
        for _ in range(2000):
            positions = rng.sample(range(N), weight)
            w = 0
            for p in positions:
                w |= (1 << p)
            candidate_set.add(w)
    
    # Add mutations of existing codewords (flip 1-3 bits)
    for c in codewords:
        # Flip 1 bit
        for bit in range(N):
            w = c ^ (1 << bit)
            candidate_set.add(w)
        # Flip 2 bits
        for i in range(N):
            for j in range(i+1, N):
                w = c ^ (1 << i) ^ (1 << j)
                candidate_set.add(w)
    
    # Also add random candidates
    for _ in range(10000):
        candidate_set.add(rng.getrandbits(N))
    
    # Filter candidates that are compatible with all current codewords
    valid_candidates = []
    for w in candidate_set:
        if w not in codewords and is_compatible_with_all(w, codewords):
            valid_candidates.append(w)
    
    # Find a large clique among valid_candidates using greedy heuristic
    def greedy_clique(candidates):
        """Find a large clique using greedy heuristic."""
        if not candidates:
            return []
        
        clique = []
        remaining = list(candidates)
        
        while remaining:
            if time.time() - start_time > time_limit * 0.75:
                break
            
            # Find vertex with highest degree in remaining
            best_idx = 0
            best_deg = -1
            for i, c in enumerate(remaining):
                deg = 0
                for other in remaining:
                    if other != c and hamming_dist(c, other) >= MIN_DIST:
                        deg += 1
                if deg > best_deg:
                    best_deg = deg
                    best_idx = i
            
            c = remaining.pop(best_idx)
            clique.append(c)
            
            # Remove all candidates incompatible with c
            remaining = [other for other in remaining 
                        if hamming_dist(c, other) >= MIN_DIST]
        
        return clique
    
    if valid_candidates and time.time() - start_time < time_limit * 0.7:
        # Limit the number of candidates
        if len(valid_candidates) > 8000:
            valid_candidates = rng.sample(valid_candidates, 8000)
        
        clique = greedy_clique(valid_candidates)
        
        if clique:
            # Verify all clique members are compatible with existing codewords
            valid_clique = []
            for w in clique:
                if is_compatible_with_all(w, codewords):
                    valid_clique.append(w)
            
            if valid_clique:
                codewords.extend(valid_clique)
                if len(codewords) > best_size:
                    best_size = len(codewords)
                    best_codewords = list(codewords)
    
    # Phase 3: Final greedy polish
    attempts = 0
    max_attempts_final = 200000
    while attempts < max_attempts_final:
        if time.time() - start_time > time_limit:
            break
        
        w = rng.getrandbits(N)
        if w not in codewords and is_compatible_with_all(w, codewords):
            codewords.append(w)
            if len(codewords) > best_size:
                best_size = len(codewords)
                best_codewords = list(codewords)
        attempts += 1
    
    return best_codewords
