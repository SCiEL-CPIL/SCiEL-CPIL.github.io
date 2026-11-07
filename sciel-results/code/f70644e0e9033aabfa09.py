"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

P = {'n': 210, 'maximum_window': 10000, 'maximum_size': 100}

def _is_perfect_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _verify_solution(selected):
    """Verify that product of selected * 210 is a perfect square."""
    prod = 210
    for x in selected:
        prod *= x
    return _is_perfect_square(prod)

def _prime_parity_mask(n, prime_to_bit):
    """Return bitmask of primes with odd exponent in n."""
    mask = 0
    d = 2
    while d * d <= n:
        if n % d == 0:
            exp = 0
            while n % d == 0:
                n //= d
                exp += 1
            if exp % 2 == 1 and d in prime_to_bit:
                mask ^= (1 << prime_to_bit[d])
        d += 1
    if n > 1 and n in prime_to_bit:
        mask ^= (1 << prime_to_bit[n])
    return mask

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # Generate primes up to a reasonable bound
    def sieve(limit):
        is_prime = [True] * (limit + 1)
        if limit >= 0:
            is_prime[0] = False
        if limit >= 1:
            is_prime[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit+1, i):
                    is_prime[j] = False
        return [i for i in range(2, limit+1) if is_prime[i]]
    
    max_search = 2000
    primes = sieve(max_search)
    prime_to_bit = {p: i for i, p in enumerate(primes)}
    
    # Target: sf(210) = 2*3*5*7
    target_mask = 0
    for p in [2, 3, 5, 7]:
        if p in prime_to_bit:
            target_mask |= (1 << prime_to_bit[p])
    
    # Precompute parity masks for numbers 211..max_search
    masks = {}
    for i in range(211, max_search + 1):
        masks[i] = _prime_parity_mask(i, prime_to_bit)
    
    best_raw = 10000
    best_solution = [224, 240]  # Known good: raw=30
    
    # Build mask_to_numbers for full range
    mask_to_numbers = defaultdict(list)
    for i in range(211, max_search + 1):
        mask_to_numbers[masks[i]].append(i)
    
    # Phase 1: Search for 2-element solutions (O(n) with hash lookup)
    for a in range(211, max_search + 1):
        if time.time() > deadline - 30:
            break
        needed_mask = masks[a] ^ target_mask
        if needed_mask in mask_to_numbers:
            for b in mask_to_numbers[needed_mask]:
                if b > a:
                    raw = b - 210
                    if raw < best_raw:
                        if _verify_solution([a, b]):
                            best_raw = raw
                            best_solution = [a, b]
    
    # Phase 2: Search for 3-element solutions with max < 210 + best_raw
    upper_bound = min(210 + best_raw - 1, max_search)
    if upper_bound > 211:
        mask_to_numbers2 = defaultdict(list)
        for i in range(211, upper_bound + 1):
            mask_to_numbers2[masks[i]].append(i)
        
        for a in range(211, upper_bound + 1):
            if time.time() > deadline - 20:
                break
            for b in range(a + 1, upper_bound + 1):
                if time.time() > deadline - 20:
                    break
                needed_mask = masks[a] ^ masks[b] ^ target_mask
                if needed_mask in mask_to_numbers2:
                    for c in mask_to_numbers2[needed_mask]:
                        if c > b:
                            raw = c - 210
                            if raw < best_raw:
                                if _verify_solution([a, b, c]):
                                    best_raw = raw
                                    best_solution = [a, b, c]
    
    # Phase 3: Search for 4-element solutions with max < 210 + best_raw
    upper_bound4 = min(210 + best_raw - 1, max_search)
    if upper_bound4 > 211 and (upper_bound4 - 211) <= 80:
        mask_to_numbers3 = defaultdict(list)
        for i in range(211, upper_bound4 + 1):
            mask_to_numbers3[masks[i]].append(i)
        
        for a in range(211, upper_bound4 + 1):
            if time.time() > deadline - 10:
                break
            for b in range(a + 1, upper_bound4 + 1):
                if time.time() > deadline - 10:
                    break
                for c in range(b + 1, upper_bound4 + 1):
                    if time.time() > deadline - 10:
                        break
                    needed_mask = masks[a] ^ masks[b] ^ masks[c] ^ target_mask
                    if needed_mask in mask_to_numbers3:
                        for d in mask_to_numbers3[needed_mask]:
                            if d > c:
                                raw = d - 210
                                if raw < best_raw:
                                    if _verify_solution([a, b, c, d]):
                                        best_raw = raw
                                        best_solution = [a, b, c, d]
    
    # Phase 4: Iterative improvement
    for iteration in range(20):
        if time.time() > deadline - 5:
            break
        upper_bound_iter = min(210 + best_raw - 1, max_search)
        if upper_bound_iter <= 211:
            break
        
        # Try 2-element again with tighter bound
        mask_to_numbers_iter = defaultdict(list)
        for i in range(211, upper_bound_iter + 1):
            mask_to_numbers_iter[masks[i]].append(i)
        
        found_improvement = False
        for a in range(211, upper_bound_iter + 1):
            if time.time() > deadline - 5:
                break
            needed_mask = masks[a] ^ target_mask
            if needed_mask in mask_to_numbers_iter:
                for b in mask_to_numbers_iter[needed_mask]:
                    if b > a:
                        raw = b - 210
                        if raw < best_raw:
                            if _verify_solution([a, b]):
                                best_raw = raw
                                best_solution = [a, b]
                                found_improvement = True
                                break
            if found_improvement:
                break
        if found_improvement:
            continue
        
        # Try 3-element
        if (upper_bound_iter - 211) <= 100:
            for a in range(211, upper_bound_iter + 1):
                if time.time() > deadline - 5:
                    break
                for b in range(a + 1, upper_bound_iter + 1):
                    if time.time() > deadline - 5:
                        break
                    needed_mask = masks[a] ^ masks[b] ^ target_mask
                    if needed_mask in mask_to_numbers_iter:
                        for c in mask_to_numbers_iter[needed_mask]:
                            if c > b:
                                raw = c - 210
                                if raw < best_raw:
                                    if _verify_solution([a, b, c]):
                                        best_raw = raw
                                        best_solution = [a, b, c]
                                        found_improvement = True
                                        break
                    if found_improvement:
                        break
                if found_improvement:
                    break
            if found_improvement:
                continue
        
        # Try 4-element
        if (upper_bound_iter - 211) <= 60:
            for a in range(211, upper_bound_iter + 1):
                if time.time() > deadline - 5:
                    break
                for b in range(a + 1, upper_bound_iter + 1):
                    if time.time() > deadline - 5:
                        break
                    for c in range(b + 1, upper_bound_iter + 1):
                        if time.time() > deadline - 5:
                            break
                        needed_mask = masks[a] ^ masks[b] ^ masks[c] ^ target_mask
                        if needed_mask in mask_to_numbers_iter:
                            for d in mask_to_numbers_iter[needed_mask]:
                                if d > c:
                                    raw = d - 210
                                    if raw < best_raw:
                                        if _verify_solution([a, b, c, d]):
                                            best_raw = raw
                                            best_solution = [a, b, c, d]
                                            found_improvement = True
                                            break
                        if found_improvement:
                            break
                    if found_improvement:
                        break
                if found_improvement:
                    break
            if found_improvement:
                continue
        
        # Try 5-element
        if (upper_bound_iter - 211) <= 40:
            for a in range(211, upper_bound_iter + 1):
                if time.time() > deadline - 5:
                    break
                for b in range(a + 1, upper_bound_iter + 1):
                    if time.time() > deadline - 5:
                        break
                    for c in range(b + 1, upper_bound_iter + 1):
                        if time.time() > deadline - 5:
                            break
                        for d in range(c + 1, upper_bound_iter + 1):
                            if time.time() > deadline - 5:
                                break
                            needed_mask = masks[a] ^ masks[b] ^ masks[c] ^ masks[d] ^ target_mask
                            if needed_mask in mask_to_numbers_iter:
                                for e in mask_to_numbers_iter[needed_mask]:
                                    if e > d:
                                        raw = e - 210
                                        if raw < best_raw:
                                            if _verify_solution([a, b, c, d, e]):
                                                best_raw = raw
                                                best_solution = [a, b, c, d, e]
                                                found_improvement = True
                                                break
                            if found_improvement:
                                break
                        if found_improvement:
                            break
                    if found_improvement:
                        break
                if found_improvement:
                    break
            if found_improvement:
                continue
        
        # No improvement found, stop iterating
        break
    
    # Phase 5: GF(2) Gaussian elimination to find minimal M
    # Binary search on M: find smallest M such that there exists a subset of (210, M]
    # with XOR of parity masks = target_mask and size <= 100
    def find_minimal_m(lo, hi):
        """Binary search for minimal M where a valid subset exists in (210, M]."""
        def exists_subset(m):
            """Check if there's a subset of (210, m] with XOR = target_mask, size <= 100."""
            if m <= 210:
                return False, None
            
            candidates = list(range(211, m + 1))
            if not candidates:
                return False, None
            
            # Compute parity vectors as bitsets
            # Collect all primes that appear in any candidate
            all_primes_set = set()
            for num in candidates:
                mask = masks[num]
                # Extract which primes are set
                for p, idx in prime_to_bit.items():
                    if mask & (1 << idx):
                        all_primes_set.add(p)
            all_primes_set.update([2, 3, 5, 7])
            
            prime_list = sorted(all_primes_set)
            prime_to_idx = {p: i for i, p in enumerate(prime_list)}
            num_primes = len(prime_list)
            
            def to_bitset(mask_val):
                bits = 0
                for p, idx in prime_to_bit.items():
                    if mask_val & (1 << idx):
                        if p in prime_to_idx:
                            bits |= (1 << prime_to_idx[p])
                return bits
            
            target_bits = to_bitset(target_mask)
            
            # Build matrix: rows are parity vectors of candidates
            # We want to find a subset whose XOR = target_bits
            # Gaussian elimination with tracking
            
            rows = [to_bitset(masks[num]) for num in candidates]
            rows.append(target_bits)  # Last row is target
            
            n_rows = len(rows)
            
            # Track combination: comb[r] is a bitset of original indices
            # For rows 0..n-2, comb[i] = 1 << i
            # For row n-1 (target), comb starts as 0
            comb = []
            for i in range(len(candidates)):
                comb.append(1 << i)
            comb.append(0)
            
            mat = rows[:]
            rank = 0
            for col in range(num_primes):
                # Find pivot row
                pivot = -1
                for r in range(rank, n_rows):
                    if (mat[r] >> col) & 1:
                        pivot = r
                        break
                if pivot == -1:
                    continue
                # Swap
                mat[rank], mat[pivot] = mat[pivot], mat[rank]
                comb[rank], comb[pivot] = comb[pivot], comb[rank]
                # Eliminate
                for r in range(n_rows):
                    if r != rank and ((mat[r] >> col) & 1):
                        mat[r] ^= mat[rank]
                        comb[r] ^= comb[rank]
                rank += 1
            
            # Check if target is in span: last row should be 0
            if mat[-1] != 0:
                return False, None
            
            # Extract the subset from comb[-1]
            subset_mask = comb[-1]
            subset_indices = []
            for i in range(len(candidates)):
                if (subset_mask >> i) & 1:
                    subset_indices.append(i)
            
            # Check size constraint
            if len(subset_indices) > 100:
                return False, None
            
            subset = [candidates[i] for i in subset_indices]
            # Verify
            if _verify_solution(subset):
                return True, subset
            else:
                return False, None
        
        best_m = hi
        best_subset = None
        
        while lo <= hi and time.time() < deadline - 10:
            mid = (lo + hi) // 2
            exists, subset = exists_subset(mid)
            if exists:
                best_m = mid
                best_subset = subset
                hi = mid - 1
            else:
                lo = mid + 1
        
        return best_m, best_subset
    
    # Try GF(2) approach to find minimal M
    # Start with current best as upper bound
    upper_for_gf2 = min(210 + best_raw, max_search)
    if upper_for_gf2 > 211:
        # Binary search between 211 and upper_for_gf2
        found_m, found_subset = find_minimal_m(211, upper_for_gf2)
        if found_subset is not None:
            raw = found_m - 210
            if raw < best_raw:
                best_raw = raw
                best_solution = found_subset
    
    # Final verification
    if not _verify_solution(best_solution):
        best_solution = [224, 240]
        best_raw = 30
    
    # Ensure types
    result = [int(x) for x in best_solution]
    return result

# EVOLVE-BLOCK-END
