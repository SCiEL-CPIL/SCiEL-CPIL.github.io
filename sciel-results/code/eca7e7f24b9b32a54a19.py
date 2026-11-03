import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    def fast_score(U):
        """
        Compute the score for a set U.
        Score = 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
        """
        if not U:
            return 0.0
        U_set = set(U)
        max_U = max(U_set)
        if max_U == 0:
            return 0.0
        
        n = len(U_set)
        
        # Use numpy for speed on n up to ~2000
        U_arr = np.array(sorted(U_set), dtype=np.int64)
        
        # Compute sumset and diffset
        # n=2000 -> 4M elements. ~32MB. Acceptable.
        # n=5000 -> 25M elements. ~200MB. Risky.
        if n > 2000:
             # Fallback for very large sets if needed, but we target n < 1000
             # For now, just let numpy handle it if memory allows, 
             # but our algebraic constructions are typically small n.
             pass

        sums = U_arr[:, None] + U_arr[None, :]
        diffs = U_arr[:, None] - U_arr[None, :]
        
        # Count unique elements
        sumset_size = len(np.unique(sums))
        diffset_size = len(np.unique(diffs))
        
        ratio = diffset_size / sumset_size
        if ratio <= 0:
            return 0.0
        
        try:
            exponent = 1 + math.log(ratio) / math.log(2 * max_U + 1)
        except (ValueError, OverflowError, ZeroDivisionError):
            return 0.0
        return exponent

    def generate_finite_field_set(k):
        """
        Construct a set U from the multiplicative group of F_{2^k} or a subspace.
        
        Strategy: 
        Represent elements of F_{2^k} as integers 0..2^k-1.
        We want a set of integers (representing field elements) such that 
        the map to integers preserves the "Sidon" or "Golomb" properties well.
        
        Actually, a better algebraic approach for integers:
        Use a basis of F_{2^k} over F_2. Let alpha be a primitive element.
        Consider the set U = { alpha^i for i=0..m-1 } mapped to integers.
        However, the standard integer representation (binary) doesn't directly
        give additive independence.
        
        Alternative Algebraic Construction:
        Let U be a set of integers where the binary representations are linearly
        independent over F_2 in a way that minimizes carries in addition.
        
        A known construction for large sumsets/diffsets ratios involves
        "Sidon sets" or "Golomb rulers". 
        Finite field construction:
        Let p be a prime. Consider U = { x^2 mod p : x in 1..n }? No, that's mod p.
        
        Let's use the specific "Finite Field Vector Space" hint:
        Select elements based on binary representation to enforce linear independence.
        
        Consider U = { sum_{j in S} 2^{k_j} } where S ranges over subsets of {0..k-1}.
        This is just all numbers 0..2^k-1. Sumset is 0..2^(k+1)-1. Diffset is -(2^k-1)..2^k-1.
        |U| = 2^k. |U+U| = 2^(k+1). |U-U| = 2^(k+1)-1.
        Ratio ~ 1. Score ~ 1. Not good.
        
        We need a *subset* of the vector space that is "sparse" enough to avoid sum collisions
        but "dense" enough to have large differences.
        
        Let's try a Sidon set construction from F_{2^k}.
        Let V be a 2-dimensional subspace of F_{2^k} over F_2? No, that's size 4.
        
        Let's use the "Mian-Chowla" type greedy but initialized with algebraic structure.
        Or, better: Use the set of powers of a primitive element in F_{2^k}, mapped to integers.
        Wait, the integer value of alpha^i is just the bit pattern.
        
        Let's try: U = { alpha^i : i=0..m-1 } where alpha is a primitive element of F_{2^k}.
        The integers are the bit patterns of these powers.
        Are these a Sidon set in integers?
        If a+b = c+d, does it imply something in F_2? No, integer addition != field addition.
        However, if we choose k large enough and m small, collisions are unlikely.
        
        Let's implement a generic "Algebraic Seed" generator:
        1. Pick k (field size 2^k).
        2. Find a primitive polynomial of degree k.
        3. Generate powers of alpha.
        4. Use the first m powers as U.
        
        We need to find primitive polynomials. We can use a simple test.
        """
        if k < 3:
            return [0, 1]
            
        # Find a primitive polynomial of degree k
        # A polynomial x^k + ... + 1 is primitive if it is irreducible and 
        # the order of x mod P(x) is 2^k - 1.
        
        def is_primitive_poly(coeffs, k):
            # coeffs is list of coefficients for x^0 ... x^k
            # We check if x^((2^k-1)/q) != 1 mod P(x) for all prime factors q of 2^k-1
            # And x^(2^k-1) == 1 mod P(x)
            
            def poly_mul(a, b, mod_poly):
                # Multiply two polynomials mod mod_poly
                res = [0] * (len(a) + len(b) - 1)
                for i, ai in enumerate(a):
                    if ai == 0: continue
                    for j, bj in enumerate(b):
                        if bj == 0: continue
                        res[i+j] ^= (ai & bj) # F_2
                # Reduce mod mod_poly
                # mod_poly is x^k + ...
                # We represent polynomials as int bits for speed? 
                # For small k (< 20), list is fine.
                
                # Reduce
                deg_res = len(res) - 1
                while deg_res >= k:
                    if res[deg_res] == 1:
                        # Subtract x^(deg_res-k) * mod_poly
                        for i in range(k+1):
                            if mod_poly[i] == 1:
                                res[deg_res - k + i] ^= 1
                    deg_res -= 1
                return res[:k] # Truncate to degree < k
            
            def poly_pow(base, exp, mod_poly):
                result = [1] + [0]*(k-1) # 1
                base = base[:]
                while exp > 0:
                    if exp & 1:
                        result = poly_mul(result, base, mod_poly)
                    base = poly_mul(base, base, mod_poly)
                    exp >>= 1
                return result
            
            n_order = (1 << k) - 1
            # Find prime factors of n_order
            factors = []
            temp = n_order
            d = 3
            while d * d <= temp:
                if temp % d == 0:
                    factors.append(d)
                    while temp % d == 0:
                        temp //= d
                d += 2
            if temp > 1:
                factors.append(temp)
            
            # Check x^(n_order) == 1
            x_poly = [0, 1] + [0]*(k-2) if k > 1 else [1]
            # Actually x is [0, 1] for k=1? No, x is degree 1.
            # Representation: index i is coeff of x^i.
            x_poly = [0]*k
            if k > 1:
                x_poly[1] = 1
            else:
                x_poly[0] = 1 # x mod (x+1) = 1? No, x = 1 in F2.
                
            # Simpler: just iterate multiplication
            current = [0]*k
            current[0] = 1 # 1
            # We need x^exp.
            # Let's just use the poly_pow function.
            
            # Base is x.
            x_base = [0]*k
            if k == 1:
                x_base[0] = 1 # x = 1 in F2[x]/(x+1)
            else:
                x_base[1] = 1
                
            # Check x^(n_order) == 1
            res_final = poly_pow(x_base, n_order, coeffs)
            if res_final != [1] + [0]*(k-1):
                return False
                
            # Check x^(n_order/q) != 1 for all prime factors q
            for q in factors:
                res_q = poly_pow(x_base, n_order // q, coeffs)
                if res_q == [1] + [0]*(k-1):
                    return False
            return True

        # Search for primitive polynomial
        # Try random polynomials of form x^k + ... + 1
        found = False
        poly_coeffs = None
        for _ in range(1000):
            # Random middle coefficients
            c = [1] + [random.randint(0,1) for _ in range(k-1)] + [1]
            if is_primitive_poly(c, k):
                poly_coeffs = c
                found = True
                break
        
        if not found:
            # Fallback: try specific known ones or just use greedy
            return None

        # Generate powers of alpha
        # alpha is x.
        # We want the integer values of alpha^0, alpha^1, ..., alpha^(m-1)
        # alpha^i is represented as a polynomial of degree < k.
        # The integer value is sum_{j=0}^{k-1} coeff_j * 2^j.
        
        # We need to compute powers.
        # Start with 1 (coeff [1,0,0..])
        # Next is x (coeff [0,1,0..])
        
        powers = []
        current = [0]*k
        current[0] = 1 # 1
        powers.append(current)
        
        x_poly = [0]*k
        if k == 1:
            x_poly[0] = 1
        else:
            x_poly[1] = 1
            
        for i in range(1, 100): # Generate enough
            current = [0]*k
            # Multiply current (which is alpha^(i-1)) by x
            # Shift left
            if k > 1:
                new_cur = [0]*k
                for j in range(k-1):
                    new_cur[j+1] = current[j]
                current = new_cur
            else:
                current = x_poly[:]
            
            # Reduce mod poly_coeffs
            # If current[k] was 1, we need to subtract x^k * 1?
            # Actually, the multiplication by x might have produced a term x^k.
            # In the shift above, we dropped the bit that went out?
            # No, we shifted into index k? No, size is k.
            # If current[k-1] was 1, shifting left puts 1 at index k (out of bounds).
            # So we must handle reduction.
            
            # Let's do it properly:
            # current represents alpha^(i-1).
            # alpha^i = x * alpha^(i-1).
            # Let c_j be coeffs of alpha^(i-1).
            # x * sum c_j x^j = sum c_j x^(j+1).
            # Term c_{k-1} x^k must be reduced.
            # x^k = sum_{j=0}^{k-1} poly_coeffs[j] x^j (since x^k + sum ... = 0 => x^k = sum ... in F2)
            
            new_cur = [0]*k
            for j in range(k):
                if current[j] == 1:
                    if j + 1 < k:
                        new_cur[j+1] ^= 1
                    else:
                        # Reduce x^k
                        for t in range(k):
                            if poly_coeffs[t] == 1:
                                new_cur[t] ^= 1
            current = new_cur
            powers.append(current)
            
            if len(powers) >= 100:
                break

        # Convert to integers
        U = []
        for p in powers:
            val = 0
            for j in range(k):
                if p[j] == 1:
                    val |= (1 << j)
            U.append(val)
        
        # Ensure 0 is in U. alpha^0 = 1. So U contains 1.
        # We need 0.
        if 0 not in U:
            U.append(0)
            
        return sorted(set(U))

    def local_search_refinement(U, time_limit):
        """
        Refine the set using local search.
        """
        current_set = sorted(set(U))
        current_score = fast_score(current_set)
        best_set = current_set[:]
        best_score = current_score
        
        while time.time() < time_limit:
            improved = False
            n = len(current_set)
            
            if n <= 1:
                break
                
            # Pick a random element (not 0) to modify
            idx = random.randint(1, n - 1)
            old_val = current_set[idx]
            
            # Try small perturbations
            for _ in range(50):
                # Shift by small amount
                shift = random.randint(-10, 10)
                new_val = old_val + shift
                
                if new_val <= 0:
                    continue
                
                # Check if new_val is already in set
                if new_val in current_set:
                    continue
                
                candidate_set = current_set[:]
                candidate_set[idx] = new_val
                candidate_set = sorted(set(candidate_set))
                
                new_score = fast_score(candidate_set)
                
                if new_score > current_score:
                    current_set = candidate_set
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_set = current_set[:]
                    improved = True
                    break
            
            if not improved:
                # Try larger jump
                shift = random.randint(-50, 50)
                new_val = old_val + shift
                if new_val > 0 and new_val not in current_set:
                    candidate_set = current_set[:]
                    candidate_set[idx] = new_val
                    candidate_set = sorted(set(candidate_set))
                    new_score = fast_score(candidate_set)
                    if new_score > current_score:
                        current_set = candidate_set
                        current_score = new_score
                        if new_score > best_score:
                            best_score = new_score
                            best_set = current_set[:]
                        improved = True

        return best_set, best_score

    best_overall_score = 0.0
    best_overall_set = [0, 1]
    
    # Strategy:
    # 1. Try algebraic constructions for various k.
    # 2. Try small Golomb rulers as baseline.
    # 3. Refine the best candidate.
    
    # Baseline: Small Golomb rulers
    for n in range(3, 20):
        if time.time() > deadline - 30:
            break
        # Simple greedy Golomb
        U = [0, 1]
        existing_diffs = {1}
        for _ in range(n - 2):
            current_max = max(U)
            v = current_max + 1
            while True:
                valid = True
                for u in U:
                    d = v - u
                    if d in existing_diffs:
                        valid = False
                        break
                if valid:
                    U.append(v)
                    for u in U[:-1]:
                        existing_diffs.add(v - u)
                    break
                v += 1
                if v > 10000:
                    break
            if v > 10000:
                break
        
        if len(U) == n:
            score = fast_score(U)
            if score > best_overall_score:
                best_overall_score = score
                best_overall_set = U[:]

    # Algebraic constructions
    for k in range(3, 12):
        if time.time() > deadline - 30:
            break
            
        U_alg = generate_finite_field_set(k)
        if U_alg:
            # We might want to take a subset of the generated powers to optimize n/max ratio
            # The full set has size 2^k (or close).
            # Let's try the full set first.
            score = fast_score(U_alg)
            if score > best_overall_score:
                best_overall_score = score
                best_overall_set = U_alg[:]
                
            # Also try subsets: first m elements
            for m in [5, 10, 20, 50]:
                if len(U_alg) >= m:
                    U_sub = U_alg[:m]
                    if 0 not in U_sub:
                        U_sub.append(0)
                    U_sub = sorted(set(U_sub))
                    score = fast_score(U_sub)
                    if score > best_overall_score:
                        best_overall_score = score
                        best_overall_set = U_sub[:]

    # Refine the best set found
    remaining_time = deadline - time.time() - 5
    if remaining_time > 5:
        improved_set, improved_score = local_search_refinement(best_overall_set, time.time() + remaining_time)
        if improved_score > best_overall_score:
            best_overall_score = improved_score
            best_overall_set = improved_set[:]

    return best_overall_set

if __name__ == "__main__":
    U = construct()
    print(f"Result: {U}")
    if U:
        max_U = max(U)
        sums = set()
        diffs = set()
        for a in U:
            for b in U:
                sums.add(a+b)
                diffs.add(a-b)
        if sums:
            ratio = len(diffs)/len(sums)
            score = 1 + math.log(ratio)/math.log(2*max_U+1)
            print(f"Score: {score}")
