import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 50.0  # Safety margin

    # 1. GF(2^8) setup
    # Irreducible polynomial: x^8 + x^4 + x^3 + x^2 + 1 = 0x11D
    def gf_mul(a, b):
        res = 0
        while b:
            if b & 1:
                res ^= a
            a <<= 1
            if a & 0x100:
                a ^= 0x11D
            b >>= 1
        return res

    def gf_pow(a, n):
        res = 1
        while n:
            if n & 1:
                res = gf_mul(res, a)
            a = gf_mul(a, a)
            n >>= 1
        return res

    # 2. Find alpha
    # For the [17,9,6] extended QR code shortened to length 17,
    # we need the narrow-sense BCH code with designed distance 6.
    # The generator polynomial has roots alpha, alpha^2, alpha^3, alpha^4
    # where alpha is a primitive 17th root of unity in GF(2^8).
    # alpha = beta^15 where beta = x (element 2 in GF(2^8))
    # Since 2^8 - 1 = 255 = 15 * 17, alpha = beta^(255/17) = beta^15
    
    beta = 2
    alpha = gf_pow(beta, 15)
    
    # 3. Compute generator polynomial g(x)
    # For BCH code with designed distance 6, roots are alpha, alpha^2, alpha^3, alpha^4
    # g(x) = lcm(M(x,alpha), M(x,alpha^2), M(x,alpha^3), M(x,alpha^4))
    # In GF(2), alpha^2 is conjugate of alpha, so M(x,alpha) = M(x,alpha^2)
    # alpha^4 = (alpha^2)^2, so M(x,alpha^4) = M(x,alpha)
    # alpha^3 has its own minimal polynomial
    # So g(x) = M(x,alpha) * M(x,alpha^3)
    
    # Find minimal polynomial of alpha: degree 8
    def find_minpoly(a):
        # Find minimal polynomial of element a in GF(2^8)
        # Try all monic polynomials of degree 8
        for lower_bits in range(256):
            candidate = (1 << 8) | lower_bits
            # Evaluate at a
            res = 0
            for i in range(9):
                if (candidate >> i) & 1:
                    res ^= gf_pow(a, i)
            if res == 0:
                return candidate
        return None
    
    # Find minimal polynomial of alpha
    m_alpha = find_minpoly(alpha)
    
    # Find minimal polynomial of alpha^3
    alpha3 = gf_pow(alpha, 3)
    m_alpha3 = find_minpoly(alpha3)
    
    # g(x) = m_alpha * m_alpha3 (polynomial multiplication over GF(2))
    def poly_mul(a, b):
        res = 0
        for i in range(32):
            if (a >> i) & 1:
                res ^= (b << i)
        return res
    
    g_poly = poly_mul(m_alpha, m_alpha3)
    
    # 4. Generate codewords: c(x) = m(x) * g(x) mod (x^17 - 1)
    # m(x) has degree < 9 (since g has degree 8, total degree < 17)
    # Actually for [17,9,6], we need 9 message bits, so m has degree < 9
    # But g has degree 8, so m*g has degree < 17, no reduction needed
    
    words = []
    for m in range(512):
        c = poly_mul(m, g_poly)
        # Keep only words with even weight (shortened code)
        if c.bit_count() % 2 == 0:
            words.append(c)
    
    # Should have 256 words
    # Now try to extend with additional words
    
    # Convert to numpy for vectorized operations
    words_arr = np.array(words, dtype=np.uint8)
    words_bits = np.zeros((len(words), 17), dtype=np.uint8)
    for i in range(17):
        words_bits[:, i] = (words_arr >> i) & 1
    
    # Precompute pairwise distances
    # dist[i,j] = Hamming distance between words[i] and words[j]
    # Use broadcasting: dist = sum(bits[i,:] != bits[j,:])
    # This is O(n^2 * 17) which for n=256 is manageable
    
    # Try to add new words greedily
    # Candidates: all 17-bit words at distance exactly 6 from word 0 (zero word)
    # There are C(17,6) = 12376 such words
    
    # Generate all candidates at distance 6 from 0
    candidates = []
    for i in range(17):
        for j in range(i+1, 17):
            for k in range(j+1, 17):
                for l in range(k+1, 17):
                    for m_idx in range(l+1, 17):
                        for n_idx in range(m_idx+1, 17):
                            word = (1 << i) | (1 << j) | (1 << k) | (1 << l) | (1 << m_idx) | (1 << n_idx)
                            candidates.append(word)
    
    candidates_arr = np.array(candidates, dtype=np.uint8)
    candidates_bits = np.zeros((len(candidates), 17), dtype=np.uint8)
    for i in range(17):
        candidates_bits[:, i] = (candidates_arr >> i) & 1
    
    # For each candidate, check if it's at distance >= 6 from all existing words
    # Use vectorized operations
    # dist_matrix = |candidates_bits[:, None, :] - words_bits[None, :, :]|.sum(axis=2)
    # But this creates a 12376 x 256 x 17 array which is ~53M elements, too large
    
    # Instead, process candidates in batches
    batch_size = 1000
    added_count = 0
    
    for batch_start in range(0, len(candidates), batch_size):
        if time.time() - start_time > deadline - 5:
            break
        
        batch_end = min(batch_start + batch_size, len(candidates))
        batch_bits = candidates_bits[batch_start:batch_end]
        batch_words = candidates_arr[batch_start:batch_end]
        
        # Check distance to all existing words
        # For each candidate in batch, compute min distance to any word in words_bits
        # dist[c, w] = popcount(batch_bits[c] ^ words_bits[w])
        
        # Use XOR and bit_count via numpy
        # XOR: batch_bits[:, None, :] ^ words_bits[None, :, :]
        # Sum along last axis
        
        xor_result = batch_bits[:, None, :] ^ words_bits[None, :, :]
        dists = xor_result.sum(axis=2)
        min_dists = dists.min(axis=1)
        
        # Find candidates with min_dist >= 6
        valid_mask = min_dists >= 6
        
        for idx in np.where(valid_mask)[0]:
            if time.time() - start_time > deadline - 5:
                break
            word_val = batch_words[idx]
            # Add to words
            words.append(word_val)
            words_bits = np.vstack([words_bits, ((word_val >> np.arange(17)) & 1).astype(np.uint8)])
            added_count += 1
    
    return words
