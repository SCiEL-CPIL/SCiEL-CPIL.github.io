import numpy as np

def construct():
    # 1. GF(2^8) setup
    PRIM = 0b100111001
    
    def gf_mul(a, b):
        res = 0
        while b:
            if b & 1:
                res ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= PRIM
        return res

    def gf_pow(base, exp):
        res = 1
        while exp:
            if exp & 1:
                res = gf_mul(res, base)
            base = gf_mul(base, base)
            exp >>= 1
        return res

    alpha = gf_pow(2, 15)
    QR = [1, 2, 4, 8, 9, 13, 15, 16]
    
    g_coeffs = [1]
    for i in QR:
        root = gf_pow(alpha, i)
        new_g = [0] * (len(g_coeffs) + 1)
        for k in range(len(g_coeffs)):
            new_g[k+1] ^= g_coeffs[k]
            new_g[k] ^= gf_mul(root, g_coeffs[k])
        g_coeffs = new_g

    G17 = np.zeros((9, 17), dtype=np.uint8)
    for i in range(9):
        for k in range(9):
            if g_coeffs[k] == 1:
                G17[i, k+i] = 1

    G18 = np.zeros((9, 18), dtype=np.uint8)
    G18[:, :17] = G17
    for i in range(9):
        parity = np.sum(G17[i]) % 2
        G18[i, 17] = parity

    codewords_18 = []
    for m in range(512):
        c_int = 0
        for i in range(9):
            if (m >> i) & 1:
                row_int = 0
                for j in range(18):
                    if G18[i, j]:
                        row_int |= (1 << j)
                c_int ^= row_int
        codewords_18.append(c_int)

    current_words = []
    seen = set()
    for c in codewords_18:
        if (c >> 17) & 1 == 0:
            short_c = c & ((1 << 17) - 1)
            if short_c not in seen:
                seen.add(short_c)
                current_words.append(short_c)
    
    # Greedy swap improvement
    # Try to find a codeword c that can be replaced by two words w1, w2
    # such that d(w1, w2) >= 6 and d(wi, c') >= 6 for all c' in C \ {c}
    
    max_swaps = 2
    for _ in range(max_swaps):
        found_swap = False
        C_list = list(current_words)
        C_set = set(C_list)
        
        for c in C_list:
            C_prime = [x for x in C_list if x != c]
            
            # Generate candidates within distance 5 of c
            # We can iterate all 2^17 and filter, or generate neighbors.
            # Generating neighbors of c up to dist 5 is faster.
            candidates = [c]
            # Add neighbors of dist 1
            for i in range(17):
                candidates.append(c ^ (1 << i))
            # Add neighbors of dist 2
            for i in range(17):
                for j in range(i+1, 17):
                    candidates.append(c ^ (1 << i) ^ (1 << j))
            # Add neighbors of dist 3
            for i in range(17):
                for j in range(i+1, 17):
                    for k in range(j+1, 17):
                        candidates.append(c ^ (1 << i) ^ (1 << j) ^ (1 << k))
            # Add neighbors of dist 4
            for i in range(17):
                for j in range(i+1, 17):
                    for k in range(j+1, 17):
                        for l in range(k+1, 17):
                            candidates.append(c ^ (1 << i) ^ (1 << j) ^ (1 << k) ^ (1 << l))
            # Add neighbors of dist 5
            for i in range(17):
                for j in range(i+1, 17):
                    for k in range(j+1, 17):
                        for l in range(k+1, 17):
                            for m in range(l+1, 17):
                                candidates.append(c ^ (1 << i) ^ (1 << j) ^ (1 << k) ^ (1 << l) ^ (1 << m))
            
            # Filter candidates
            valid_cands = []
            for w in candidates:
                if w in C_set and w != c:
                    continue
                # Check distance to C_prime
                ok = True
                for cp in C_prime:
                    if (w ^ cp).bit_count() < 6:
                        ok = False
                        break
                if ok:
                    valid_cands.append(w)
            
            if len(valid_cands) >= 2:
                # Find a pair in valid_cands with dist >= 6
                pair_found = False
                for i in range(len(valid_cands)):
                    w1 = valid_cands[i]
                    for j in range(i+1, len(valid_cands)):
                        w2 = valid_cands[j]
                        if (w1 ^ w2).bit_count() >= 6:
                            # Replace c with w1, w2
                            C_list.remove(c)
                            C_list.append(w1)
                            C_list.append(w2)
                            current_words = C_list
                            found_swap = True
                            pair_found = True
                            break
                    if pair_found:
                        break
                if found_swap:
                    break
            if found_swap:
                break
        
        if not found_swap:
            break
            
    return current_words
