import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin (150s kill, 5s buffer)

    # Sieve primes up to 10^6 for factorization
    limit = 1000000
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]

    def compute_phi_segment(start, end):
        """Compute phi(n) for n in [start, end] using segmented approach."""
        size = end - start + 1
        phi = np.arange(start, end + 1, dtype=np.int64)
        rem = phi.copy()
        
        for p in primes:
            if p * p > end:
                break
            # Find indices where rem is divisible by p
            first = ((start + p - 1) // p) * p
            if first > end:
                continue
            idxs = np.arange(first - start, size, p)
            if len(idxs) == 0:
                continue
            mask = rem[idxs] % p == 0
            if not mask.any():
                continue
            sel = idxs[mask]
            phi[sel] -= phi[sel] // p
            rem[sel] //= p
            # Remove all factors of p
            while True:
                still = rem[sel] % p == 0
                if not still.any():
                    break
                sel = sel[still]
                rem[sel] //= p
        
        # Handle remaining prime factors > 1
        mask = rem > 1
        if mask.any():
            phi[mask] -= phi[mask] // rem[mask]
        
        return phi

    def find_longest_run_sliding(phi_vals):
        """Find longest consecutive run of distinct phi values using sliding window."""
        n = len(phi_vals)
        if n == 0:
            return 1, 0
        best_k = 1
        best_s = 0
        seen = {}
        left = 0
        for right in range(n):
            val = phi_vals[right]
            if val in seen and seen[val] >= left:
                left = seen[val] + 1
            seen[val] = right
            cur_k = right - left + 1
            if cur_k > best_k:
                best_k = cur_k
                best_s = left
            if best_k >= 200:
                break
        return best_k, best_s

    best_k = 1
    best_s = 1

    # Phase 1: Small range to get a baseline
    if time.time() - start_time < deadline - 30:
        phi_vals = compute_phi_segment(1, 100000)
        k, s = find_longest_run_sliding(phi_vals)
        if k > best_k:
            best_k = k
            best_s = 1 + s

    # Phase 2: Search at various scales - prioritize high ranges where collisions are rare
    # Use larger segments now: 1,000,000
    seg_size = 1000000
    # Scan a contiguous high range, e.g., 9e11 to 9e11 + 1e9
    # And some random spots
    bases = []
    # Contiguous scan in high range
    for i in range(10):
        bases.append(9 * 10**11 + i * seg_size)
    # Random high spots
    import random
    random.seed(42)
    for _ in range(20):
        b = random.randint(10**11, 10**12 - seg_size)
        bases.append(b)
    # Lower ranges
    bases.extend([10**9, 10**8, 10**7])

    for base in bases:
        if time.time() - start_time > deadline - 10:
            break
        if base < 1 or base > 10**12 - seg_size:
            continue
        seg_start = base
        seg_end = min(base + seg_size - 1, 10**12)
        try:
            phi_vals = compute_phi_segment(seg_start, seg_end)
            k, s = find_longest_run_sliding(phi_vals)
            actual_s = seg_start + s
            if k > best_k:
                best_k = k
                best_s = actual_s
                print(f"Found k={best_k} at s={best_s}")
        except Exception as e:
            pass

    # Phase 3: If time remains and best_k < 200, try finer-grained search in high ranges
    if time.time() - start_time < deadline - 20 and best_k < 200:
        small_seg = 500000
        high_bases = [
            95*10**10 + i*small_seg for i in range(5)
        ] + [
            90*10**10 + i*small_seg for i in range(5)
        ] + [
            80*10**10 + i*small_seg for i in range(5)
        ]
        for base in high_bases:
            if time.time() - start_time > deadline - 5:
                break
            if base > 10**12 - small_seg:
                continue
            seg_start = base
            seg_end = min(base + small_seg - 1, 10**12)
            try:
                phi_vals = compute_phi_segment(seg_start, seg_end)
                k, s = find_longest_run_sliding(phi_vals)
                actual_s = seg_start + s
                if k > best_k:
                    best_k = k
                    best_s = actual_s
                if best_k >= 200:
                    break
            except Exception:
                pass

    # Phase 4: Try to extend the best run
    if time.time() - start_time < deadline - 15 and best_k > 1:
        # Try to find a longer run starting near best_s
        for offset in [-2000, -1000, -500, -100, 0, 100, 500, 1000, 2000]:
            if time.time() - start_time > deadline - 5:
                break
            test_s = best_s + offset
            if test_s < 1:
                continue
            seg_end = min(test_s + 400, 10**12)
            try:
                phi_vals = compute_phi_segment(test_s, seg_end)
                k, s = find_longest_run_sliding(phi_vals)
                actual_s = test_s + s
                if k > best_k:
                    best_k = k
                    best_s = actual_s
            except Exception:
                pass

    # Final validation and clamping
    if best_s + best_k - 1 > 10**12:
        best_k = min(best_k, 10**12 - best_s + 1)
    if best_k > 200:
        best_k = 200
    if best_k < 1:
        best_k = 1
    if best_s < 1:
        best_s = 1

    return {'start': int(best_s), 'length': int(best_k)}
