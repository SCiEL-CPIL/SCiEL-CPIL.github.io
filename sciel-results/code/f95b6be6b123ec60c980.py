import time
import random
from itertools import product

DEADLINE = 90.0
SAFETY_MARGIN = 5.0

def _square_nonzero_count_fast(exponents, signs):
    """
    Calculate the number of nonzero terms in the square of a polynomial 
    defined by exponents and signs.
    exponents: list of 12 ints
    signs: list of 12 ints (+1 or -1)
    """
    n = len(exponents)
    coeffs = {}
    for i in range(n):
        e_i = exponents[i]
        s_i = signs[i]
        for j in range(i, n):
            e_sum = e_i + exponents[j]
            if i == j:
                coeffs[e_sum] = coeffs.get(e_sum, 0) + 1
            else:
                coeffs[e_sum] = coeffs.get(e_sum, 0) + 2 * s_i * signs[j]
    return sum(1 for v in coeffs.values() if v != 0)

def _best_signs_for_exponents(exponents):
    """
    For a fixed set of 12 exponents, find the sign assignment (+1/-1)
    that minimizes the nonzero count of the square.
    Brute forces 2^12 = 4096 possibilities.
    """
    best_count = 10**9
    best_signs = None
    n = len(exponents)
    
    for mask in range(1 << n):
        signs = [1 if (mask >> i) & 1 == 0 else -1 for i in range(n)]
        
        coeffs = {}
        for i in range(n):
            e_i = exponents[i]
            s_i = signs[i]
            for j in range(i, n):
                e_sum = e_i + exponents[j]
                if i == j:
                    coeffs[e_sum] = coeffs.get(e_sum, 0) + 1
                else:
                    coeffs[e_sum] = coeffs.get(e_sum, 0) + 2 * s_i * signs[j]
        
        count = sum(1 for v in coeffs.values() if v != 0)
        if count < best_count:
            best_count = count
            best_signs = signs
    
    return best_count, best_signs

def construct():
    start = time.time()
    deadline = start + DEADLINE - SAFETY_MARGIN
    rng = random.Random(42)
    
    best_coefs = None
    best_sq_nz = 10**9

    def try_candidate(exponents, signs):
        nonlocal best_coefs, best_sq_nz
        # Build coefficient list
        max_exp = max(exponents)
        if max_exp >= 41:
            return
        coefs = [0] * (max_exp + 1)
        for e, s in zip(exponents, signs):
            coefs[e] = s
        if len(coefs) > 41:
            return
        if coefs[-1] == 0:
            return
        if sum(1 for c in coefs if c != 0) != 12:
            return
        if any(c not in range(-64, 65) for c in coefs):
            return
        
        # We already know the count from the search, but let's be safe
        # Actually, we pass the count in.
        pass

    def update_global(exponents, signs, count):
        nonlocal best_coefs, best_sq_nz
        if count < best_sq_nz:
            best_sq_nz = count
            max_exp = max(exponents)
            coefs = [0] * (max_exp + 1)
            for e, s in zip(exponents, signs):
                coefs[e] = s
            best_coefs = coefs

    # Strategy 1: Structured Decomposition
    # We model the polynomial as P = A + B + C, where A, B, C are 4-term polynomials.
    # To minimize cross-terms, we place A, B, C in widely separated exponent ranges.
    # However, if they are too far apart, the total degree might exceed 40.
    # Let's try a "modular" approach or just wide gaps that fit in 41.
    
    # If we use gaps of size G, the exponents for block k are {k*G + offsets}.
    # Max exponent approx 2*G + max_offset.
    # If G=10, max exp ~ 20 + 3 = 23. Fits.
    # If G=12, max exp ~ 24 + 3 = 27. Fits.
    # If G=15, max exp ~ 30 + 3 = 33. Fits.
    
    # For each block, we choose 4 offsets from [0, G-1].
    # We also choose signs for all 12 terms.
    # Total search space:
    # For G=10: offsets are 4-combinations from 10. C(10,4)=210.
    # 3 blocks: 210^3 = 9.26M combinations. Too many to brute force all with 4096 signs each.
    # But we can optimize signs per block? No, cross-block terms couple signs.
    # However, if gaps are large enough that no sum from block i equals sum from block j (i!=j),
    # then the problem decouples?
    # Sum from block i: 2*i*G + o1 + o2.
    # Sum from block j: 2*j*G + o3 + o4.
    # Equality: 2(i-j)G = o3+o4 - o1-o2.
    # RHS is in [-2G, 2G]. LHS is multiple of 2G.
    # If i-j=1, LHS=2G. RHS can be 2G if o3+o4=2G-1 and o1+o2=-1 (impossible, offsets>=0).
    # Wait, offsets are in [0, G-1].
    # Max RHS: (G-1)+(G-1) - 0 - 0 = 2G-2.
    # Min RHS: 0+0 - (G-1)-(G-1) = -2G+2.
    # So 2G is never reached.
    # Thus, if we use gaps G, sums from different blocks NEVER collide.
    # This means the total nonzero count is the sum of nonzero counts of each block's square
    # PLUS the cross terms between blocks.
    # Cross terms: A*B, A*C, B*C.
    # Since exponents don't collide, each cross term (2 * s_i * s_j * x^(e_i+e_j)) is nonzero
    # unless the coefficient is 0 (impossible for +/-1) or it cancels with another term.
    # But we established that sums from different blocks don't collide with each other?
    # Wait.
    # Term from A*B: exponent is a + b.
    # Term from A*C: exponent is a + c.
    # Can a+b = a'+c'?
    # (i*G + oi) + (j*G + oj) = (i'*G + oi') + (j'*G + oj')
    # (i+j - i'-j')G = oi'+oj' - oi-oj.
    # RHS is in [-2G+2, 2G-2].
    # LHS is multiple of G.
    # If i+j != i'+j', LHS is at least G.
    # If G > 2G-2 (i.e., G < 2), impossible. But G>=1.
    # If G=10, LHS=10, 20, etc. RHS max 18. So no collision if |i+j - i'-j'| >= 1?
    # Wait. If i+j = i'+j', then LHS=0, so oi+oj = oi'+oj'.
    # So collisions can happen between A*B and A'*B' if the block indices sum to the same value.
    # Example: A*B (indices 0,1) and A*C (indices 0,2)? No, 0+1=1, 0+2=2.
    # A*B (0,1) and B*A (1,0) are the same term (2 s0 s1 x^...).
    # What about A*B (0,1) and C*B (2,1)? Sum indices 1 vs 3. No.
    # What about A*B (0,1) and A*B (0,1)? Same.
    # What about A*B (0,1) and C*C? No, C*C is internal.
    
    # Let's check A*B vs A'*B' where {i,j} != {i',j'}.
    # If i+j = i'+j', then we need oi+oj = oi'+oj'.
    # Since offsets are small, this is possible.
    # Example: Block 0 (A) and Block 1 (B). Sum index 1.
    # Block 0 (A) and Block 2 (C)? Sum index 2.
    # Block 1 (B) and Block 1 (B)? Internal.
    # So cross-block collisions only happen if the block indices sum to the same value.
    # Pairs of blocks:
    # (0,0) -> Internal A
    # (1,1) -> Internal B
    # (2,2) -> Internal C
    # (0,1) -> Cross A*B
    # (1,0) -> Cross B*A (same as A*B)
    # (0,2) -> Cross A*C
    # (2,0) -> Cross C*A
    # (1,2) -> Cross B*C
    # (2,1) -> Cross C*B
    
    # Are there collisions between (0,1) and (0,2)?
    # Indices 1 vs 2. No.
    # Between (0,1) and (1,1)? No.
    # So, if we use 3 blocks with indices 0, 1, 2, the cross terms are in distinct "index-sum" buckets: 0, 1, 2, 3, 4?
    # (0,0)->0, (1,1)->2, (2,2)->4.
    # (0,1)->1, (1,0)->1.
    # (0,2)->2, (2,0)->2. ** COLLISION with (1,1) **
    # (1,2)->3, (2,1)->3.
    
    # So A*C (index sum 2) collides with B*B (index sum 2).
    # This coupling makes it hard to decouple completely.
    
    # However, we can still use this structure to reduce the search space.
    # Instead of random exponents, we search over offsets within blocks.
    
    def search_structured():
        # Try different gap sizes
        for G in [10, 12, 15, 20]:
            if time.time() > deadline:
                return
            
            # Generate random sets of offsets for 3 blocks
            # Each block has 4 offsets in [0, G-1]
            
            # To speed up, we can precompute the best signs for a specific block configuration?
            # No, signs are global.
            
            # Let's just do a limited random search over offsets.
            # Number of offset sets per block: C(G, 4).
            # For G=10, C(10,4)=210.
            # We can't do 210^3.
            # We will sample random offset sets.
            
            for _ in range(2000):
                if time.time() > deadline:
                    return
                
                offsets_A = sorted(rng.sample(range(G), 4))
                offsets_B = sorted(rng.sample(range(G), 4))
                offsets_C = sorted(rng.sample(range(G), 4))
                
                exps = []
                exps.extend(offsets_A) # Block 0
                exps.extend([G + o for o in offsets_B]) # Block 1
                exps.extend([2*G + o for o in offsets_C]) # Block 2
                
                if max(exps) >= 41:
                    continue
                    
                # Find best signs
                count, signs = _best_signs_for_exponents(exps)
                if count < best_sq_nz:
                    update_global(exps, signs, count)
                    
    search_structured()
    
    # Strategy 2: Fallback to AP and Local Search if structured didn't find something great
    # The AP (0..11) with optimal signs is a strong baseline.
    if best_sq_nz > 20: # If we haven't beaten 20, ensure we have the AP baseline
        exps = list(range(12))
        count, signs = _best_signs_for_exponents(exps)
        if count < best_sq_nz:
            update_global(exps, signs, count)
            
    # If we have a candidate, do local search on exponents
    if best_coefs is not None and time.time() < deadline - 2.0:
        current_exps = [i for i, c in enumerate(best_coefs) if c != 0]
        current_signs = [best_coefs[i] for i in current_exps]
        
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            # Try moving one exponent
            for i in range(12):
                old_e = current_exps[i]
                # Try nearby exponents first
                for delta in [1, -1, 2, -2, 3, -3]:
                    new_e = old_e + delta
                    if new_e < 0 or new_e >= 41:
                        continue
                    if new_e in current_exps:
                        continue
                    new_exps = list(current_exps)
                    new_exps[i] = new_e
                    new_exps.sort()
                    
                    # Quick filter
                    us = len(set(a+b for a in new_exps for b in new_exps if new_exps.index(a) <= new_exps.index(b)))
                    # This is slow. Just check count.
                    
                    count, signs = _best_signs_for_exponents(new_exps)
                    if count < best_sq_nz:
                        best_sq_nz = count
                        current_exps = new_exps
                        current_signs = signs
                        improved = True
                        max_exp = max(current_exps)
                        best_coefs = [0] * (max_exp + 1)
                        for e, s in zip(current_exps, current_signs):
                            best_coefs[e] = s
                        break
                if improved:
                    break
    
    # Final check
    if best_coefs is None or not best_coefs or best_coefs[-1] == 0:
        return [1] * 12
        
    return best_coefs

if __name__ == "__main__":
    res = construct()
    print(res)
