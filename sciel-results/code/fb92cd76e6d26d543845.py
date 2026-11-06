import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 17
    MIN_DIST = 6
    
    # --- Step 1: Construct [17,8,6] QR Code ---
    
    def poly_mul(a, b):
        res = 0
        while b:
            if b & 1:
                res ^= a
            a <<= 1
            b >>= 1
        return res

    def poly_div(a, b):
        deg_a = a.bit_length() - 1
        deg_b = b.bit_length() - 1
        if deg_a < deg_b:
            return 0, a
        q = 0
        r = a
        shift = deg_a - deg_b
        while deg_a >= deg_b:
            if (r >> deg_a) & 1:
                q |= (1 << shift)
                r ^= (b << shift)
                deg_a = r.bit_length() - 1
                shift = deg_a - deg_b
        return q, r

    target = (1 << 17) | 1
    factors = []
    for mask in range(128):
        p = (1 << 8) | mask
        if p == 3: continue
        _, rem = poly_div(target, p)
        if rem == 0:
            factors.append(p)
            
    best_seed = []
    if len(factors) >= 2:
        for f in factors:
            g = poly_mul(3, f)
            if g.bit_length() - 1 == 9:
                basis = [g << i for i in range(8)]
                words = []
                for mask in range(256):
                    w = 0
                    for i in range(8):
                        if (mask >> i) & 1:
                            w ^= basis[i]
                    words.append(w)
                # Check min distance
                valid = True
                for i in range(256):
                    if not valid: break
                    for j in range(i+1, 256):
                        if (words[i] ^ words[j]).bit_count() < MIN_DIST:
                            valid = False
                            break
                if valid:
                    best_seed = words
                    break

    if best_seed:
        codewords = best_seed
    else:
        # Fallback: Greedy from scratch
        codewords = [0]
        while len(codewords) < 100 and time.time() < deadline - 10:
            c = random.getrandbits(N)
            if all((c ^ w).bit_count() >= MIN_DIST for w in codewords):
                codewords.append(c)

    # --- Step 2: Greedy Extension ---
    no_add_count = 0
    max_no_add = 10000
    
    while time.time() < deadline - 1:
        candidate = random.getrandbits(N)
        
        if candidate in codewords:
            continue
            
        valid = True
        for w in codewords:
            if (candidate ^ w).bit_count() < MIN_DIST:
                valid = False
                break
        
        if valid:
            codewords.append(candidate)
            no_add_count = 0
        else:
            no_add_count += 1
            if no_add_count > max_no_add:
                if len(codewords) > 1:
                    idx = random.randint(0, len(codewords) - 1)
                    old_w = codewords[idx]
                    codewords.pop(idx)
                    
                    new_w = random.getrandbits(N)
                    valid_new = True
                    for w in codewords:
                        if (new_w ^ w).bit_count() < MIN_DIST:
                            valid_new = False
                            break
                    if valid_new and new_w != old_w:
                        codewords.append(new_w)
                    else:
                        codewords.append(old_w)
                    no_add_count = 0

    return codewords
