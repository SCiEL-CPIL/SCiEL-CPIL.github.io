import time
import math
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # Leave 5s safety margin
    
    max_val = 10210
    
    # 1. Sieve of Eratosthenes for primes up to max_val
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if is_prime[i]:
            for j in range(i*i, max_val + 1, i):
                is_prime[j] = False
                
    primes = [i for i, prime in enumerate(is_prime) if prime]
    prime_to_index = {p: i for i, p in enumerate(primes)}
    d = len(primes)
    
    # 2. Compute parity vector for 210
    # 210 = 2 * 3 * 5 * 7
    t = 0
    for p in [2, 3, 5, 7]:
        t |= (1 << prime_to_index[p])
        
    # 3. Compute parity vectors for integers 211 to max_val
    vectors = [0] * (max_val - 210)
    for i in range(max_val - 210):
        x = 211 + i
        v = 0
        n = x
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                count = 0
                while n % p == 0:
                    n //= p
                    count += 1
                if count % 2 == 1:
                    v |= (1 << prime_to_index[p])
        if n > 1:
            if n in prime_to_index:
                v |= (1 << prime_to_index[n])
        vectors[i] = v

    # 4. Find minimal K where t is in span of first K vectors using incremental basis
    # Build basis incrementally and track when t becomes reducible
    
    basis = [0] * (d + 1)  # basis[msb] = vector with that msb
    rank = 0
    min_K = None
    
    # We also track the representation: when we add vector i to basis,
    # we store which previous vectors it was combined with.
    # rep[i] = bitmask of which vectors 0..i-1 are needed to express vectors[i] in terms of basis
    # Actually, we need: for each basis vector, which original vectors compose it.
    # Let's use a different approach: maintain the basis with back-references.
    
    # Simpler: just find min_K first, then do full Gaussian elimination for the solution.
    for i in range(max_val - 210):
        v = vectors[i]
        if v == 0:
            continue
        # Reduce v
        reduced = v
        while reduced != 0:
            msb = reduced.bit_length() - 1
            if basis[msb] != 0:
                reduced ^= basis[msb]
            else:
                basis[msb] = reduced
                rank += 1
                break
        
        # Check if t is in span
        check = t
        while check != 0:
            msb = check.bit_length() - 1
            if basis[msb] != 0:
                check ^= basis[msb]
            else:
                break
        if check == 0:
            min_K = i + 1
            break
    
    if min_K is None:
        # Should not happen, but fallback
        min_K = max_val - 210
    
    # 5. Find a sparse solution using Gaussian elimination on the first min_K vectors
    # Build the transposed system: d equations, min_K unknowns
    # Row j (prime j): sum of c_i * v_i[j] = t[j]
    
    # For efficiency, build rows using bit operations
    K = min_K
    rows = []
    for j in range(d):
        row = 0
        for i in range(K):
            if (vectors[i] >> j) & 1:
                row |= (1 << i)
        if (t >> j) & 1:
            row |= (1 << K)
        rows.append(row)
    
    # Gaussian elimination to RREF
    pivot_row = 0
    pivot_cols = []
    for c in range(K):
        found = False
        for r in range(pivot_row, d):
            if (rows[r] >> c) & 1:
                rows[r], rows[pivot_row] = rows[pivot_row], rows[r]
                found = True
                break
        if not found:
            continue
            
        for r in range(d):
            if r != pivot_row and ((rows[r] >> c) & 1):
                rows[r] ^= rows[pivot_row]
                
        pivot_cols.append(c)
        pivot_row += 1
        if pivot_row >= d:
            break
    
    # Extract particular solution (free vars = 0)
    c_p = 0
    for r in range(len(pivot_cols)):
        c = pivot_cols[r]
        if (rows[r] >> K) & 1:
            c_p |= (1 << c)
    
    # Build null space basis
    pivot_set = set(pivot_cols)
    free_cols = [c for c in range(K) if c not in pivot_set]
    
    null_basis = []
    for f in free_cols:
        n = 1 << f
        for r in range(len(pivot_cols)):
            c = pivot_cols[r]
            if (rows[r] >> f) & 1:
                n |= (1 << c)
        null_basis.append(n)
    
    # 6. Sparse solution via local search
    current = c_p
    cur_weight = bin(current).count('1')
    
    # Greedy single-swap
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for b in null_basis:
            new_c = current ^ b
            new_w = bin(new_c).count('1')
            if new_w < cur_weight:
                current = new_c
                cur_weight = new_w
                improved = True
    
    if cur_weight <= 100:
        selected = []
        for i in range(K):
            if (current >> i) & 1:
                selected.append(211 + i)
        return selected
    
    # 7. If still > 100, try 2-swap local search (escape local minima)
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            n_basis = len(null_basis)
            for i in range(n_basis):
                if time.time() > deadline - 3:
                    break
                for j in range(i + 1, n_basis):
                    new_c = current ^ null_basis[i] ^ null_basis[j]
                    new_w = bin(new_c).count('1')
                    if new_w < cur_weight:
                        current = new_c
                        cur_weight = new_w
                        improved = True
                        break
                if improved:
                    break
    
    if cur_weight <= 100:
        selected = []
        for i in range(K):
            if (current >> i) & 1:
                selected.append(211 + i)
        return selected
    
    # 8. Randomized restarts with random null space combinations
    if time.time() < deadline - 3:
        best_c = current
        best_w = cur_weight
        rng = random.Random(42)
        
        # Try random combinations of up to 4 null space vectors
        for _ in range(5000):
            if time.time() > deadline - 3:
                break
            k = rng.randint(1, 4)
            idxs = rng.sample(range(len(null_basis)), k) if len(null_basis) >= k else []
            if not idxs:
                break
            delta = 0
            for idx in idxs:
                delta ^= null_basis[idx]
            new_c = c_p ^ delta
            new_w = bin(new_c).count('1')
            if new_w < best_w:
                best_c = new_c
                best_w = new_w
        
        if best_w <= 100:
            current = best_c
            cur_weight = best_w
    
    if cur_weight <= 100:
        selected = []
        for i in range(K):
            if (current >> i) & 1:
                selected.append(211 + i)
        return selected
    
    # 9. Fallback: try increasing K to get more null space dimension
    # Try K+1, K+2, ... up to a limit
    for extra in range(1, 50):
        if time.time() > deadline - 3:
            break
        K2 = K + extra
        if K2 > max_val - 210:
            break
        
        # Quick check: is t still solvable? (it should be, since adding vectors only increases span)
        # Rebuild rows for K2
        rows2 = []
        for j in range(d):
            row = 0
            for i in range(K2):
                if (vectors[i] >> j) & 1:
                    row |= (1 << i)
            if (t >> j) & 1:
                row |= (1 << K2)
            rows2.append(row)
        
        # Gaussian elimination
        pr = 0
        pc = []
        for c in range(K2):
            found = False
            for r in range(pr, d):
                if (rows2[r] >> c) & 1:
                    rows2[r], rows2[pr] = rows2[pr], rows2[r]
                    found = True
                    break
            if not found:
                continue
            for r in range(d):
                if r != pr and ((rows2[r] >> c) & 1):
                    rows2[r] ^= rows2[pr]
            pc.append(c)
            pr += 1
            if pr >= d:
                break
        
        # Particular solution
        c2 = 0
        for r in range(len(pc)):
            if (rows2[r] >> K2) & 1:
                c2 |= (1 << pc[r])
        
        w2 = bin(c2).count('1')
        if w2 <= 100:
            selected = []
            for i in range(K2):
                if (c2 >> i) & 1:
                    selected.append(211 + i)
            return selected
        
        # Null space for K2
        pc_set = set(pc)
        fc = [c for c in range(K2) if c not in pc_set]
        nb = []
        for f in fc:
            n = 1 << f
            for r in range(len(pc)):
                if (rows2[r] >> f) & 1:
                    n |= (1 << pc[r])
            nb.append(n)
        
        # Greedy on this
        cur2 = c2
        cw2 = w2
        imp = True
        while imp and time.time() < deadline - 2:
            imp = False
            for b in nb:
                nc = cur2 ^ b
                nw = bin(nc).count('1')
                if nw < cw2:
                    cur2 = nc
                    cw2 = nw
                    imp = True
        
        if cw2 <= 100:
            selected = []
            for i in range(K2):
                if (cur2 >> i) & 1:
                    selected.append(211 + i)
            return selected
    
    # Final fallback: return best found
    selected = []
    for i in range(K):
        if (current >> i) & 1:
            selected.append(211 + i)
    return selected

# Test function (optional, for local verification)
if __name__ == "__main__":
    sol = construct()
    print(f"Solution size: {len(sol)}")
    print(f"Max element: {max(sol) if sol else 0}")
    if sol:
        prod = 210
        for x in sol:
            prod *= x
        sqrt_prod = math.isqrt(prod)
        is_square = (sqrt_prod * sqrt_prod == prod)
        print(f"Is square: {is_square}")
        print(f"Raw value: {max(sol) - 210}")
