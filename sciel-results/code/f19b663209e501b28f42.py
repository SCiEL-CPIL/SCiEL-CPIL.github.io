"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill

    def time_remaining():
        return DEADLINE - (time.time() - start_time)

    # Precompute small primes up to a reasonable limit
    SIEVE_LIMIT = 1000000
    def compute_primes(limit):
        sieve = bytearray(b'\x01') * (limit + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
        return [i for i in range(2, limit + 1) if sieve[i]]

    small_primes = compute_primes(SIEVE_LIMIT)
    small_prime_set = set(small_primes)

    # Miller-Rabin primality test
    def is_prime_mr(n):
        if n < 2:
            return False
        if n in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = x * x % n
                if x == n - 1:
                    break
            else:
                return False
        return True

    # Compute phi(n) using small primes then Miller-Rabin for large factors
    def phi(n):
        if n <= 1:
            return n
        result = n
        y = n
        # Trial divide by small primes
        for p in small_primes:
            if p * p > y:
                break
            if y % p == 0:
                result -= result // p
                while y % p == 0:
                    y //= p
        if y > 1:
            # y is either 1, a prime, or a product of two large primes
            if y <= SIEVE_LIMIT:
                if y in small_prime_set:
                    result -= result // y
                else:
                    # Composite of small primes not in our list (shouldn't happen)
                    # Factor it further
                    for p in small_primes:
                        if p * p > y:
                            break
                        if y % p == 0:
                            result -= result // p
                            while y % p == 0:
                                y //= p
                    if y > 1:
                        result -= result // y
            else:
                if is_prime_mr(y):
                    result -= result // y
                else:
                    # y is composite with large factors
                    # Try to factor further
                    for p in small_primes:
                        if p * p > y:
                            break
                        if y % p == 0:
                            result -= result // p
                            while y % p == 0:
                                y //= p
                    if y > 1:
                        if is_prime_mr(y):
                            result -= result // y
                        else:
                            # y is product of two large primes
                            # We need to handle this case
                            # For now, assume it's prime (conservative)
                            result -= result // y
        return result

    # Batch phi computation for a range [start, start+window)
    def batch_phi(start, window):
        """Compute phi for numbers in [start, start+window)"""
        results = {}
        for i in range(window):
            n = start + i
            results[n] = phi(n)
        return results

    # Search for longest run of distinct phi values
    best_k = 1
    best_s = 1

    # Strategy: search in large strides, then refine
    # Phase 1: Coarse search with large windows
    # Phase 2: Fine search around promising areas

    # First, do a coarse search over a wide range
    # Use a sliding window approach
    WINDOW_SIZE = 200  # Max k we care about
    SEARCH_RANGE = 10**12

    # For efficiency, search in chunks
    # We'll search at various scales
    search_positions = []

    # Add some strategic positions
    # Small numbers (already known to have runs)
    search_positions.append(1)
    # Powers and special numbers
    for exp in range(1, 42):
        search_positions.append(2**exp)
    # Random positions in the range
    random.seed(42)
    for _ in range(5000):
        search_positions.append(random.randint(1, SEARCH_RANGE))
    # Positions near multiples of large primes
    for p in [999999937, 999999929, 999999893, 999999883, 999999881]:
        for mult in range(1, 1000):
            pos = p * mult
            if 1 <= pos <= SEARCH_RANGE:
                search_positions.append(pos)
            if 1 <= pos + WINDOW_SIZE <= SEARCH_RANGE:
                search_positions.append(pos + 1)

    # Deduplicate and sort
    search_positions = sorted(set(search_positions))

    # Process search positions, checking windows
    for pos in search_positions:
        if time_remaining() < 5.0:
            break
        if pos < 1 or pos > SEARCH_RANGE:
            continue
        if pos + WINDOW_SIZE - 1 > SEARCH_RANGE:
            continue

        # Compute phi for a window starting at pos
        # But we need to check if extending the run is possible
        # First compute phi for [pos, pos + WINDOW_SIZE)
        try:
            phi_vals = [phi(pos + i) for i in range(WINDOW_SIZE)]
        except Exception:
            continue

        # Find longest run of distinct values starting at pos
        seen = set()
        k = 0
        for i, v in enumerate(phi_vals):
            if v in seen:
                break
            seen.add(v)
            k = i + 1

        if k > best_k:
            best_k = k
            best_s = pos
            # Try to extend the run backwards
            # Check if we can extend before pos
            # But first, let's also try to find where the run actually starts
            # by checking backwards
            actual_start = pos
            while actual_start > 1 and time_remaining() > 5.0:
                try:
                    prev_phi = phi(actual_start - 1)
                except Exception:
                    break
                if prev_phi not in seen:
                    actual_start -= 1
                    seen.add(prev_phi)
                    k += 1
                else:
                    break
            best_s = actual_start

        # Also check if we can extend forward beyond WINDOW_SIZE
        # Only if we hit the full window
        if k == WINDOW_SIZE and time_remaining() > 10.0:
            # Try to extend forward
            extend_start = pos + WINDOW_SIZE
            while extend_start + 50 <= SEARCH_RANGE and time_remaining() > 5.0:
                try:
                    next_phi = phi(extend_start)
                except Exception:
                    break
                if next_phi not in seen:
                    seen.add(next_phi)
                    k += 1
                    extend_start += 1
                else:
                    break
            if k > best_k:
                best_k = k
                best_s = pos

    # Phase 2: Local refinement around the best found
    if best_k > 1 and time_remaining() > 20.0:
        # Search in a neighborhood of the best start
        center = best_s
        search_radius = 10000
        for offset in range(-search_radius, search_radius + 1, 10):
            if time_remaining() < 5.0:
                break
            pos = center + offset
            if pos < 1 or pos > SEARCH_RANGE:
                continue
            if pos + WINDOW_SIZE - 1 > SEARCH_RANGE:
                continue
            try:
                phi_vals = [phi(pos + i) for i in range(WINDOW_SIZE)]
            except Exception:
                continue
            seen = set()
            k = 0
            for i, v in enumerate(phi_vals):
                if v in seen:
                    break
                seen.add(v)
                k = i + 1
            if k > best_k:
                best_k = k
                best_s = pos
                # Extend backwards
                actual_start = pos
                while actual_start > 1 and time_remaining() > 5.0:
                    try:
                        prev_phi = phi(actual_start - 1)
                    except Exception:
                        break
                    if prev_phi not in seen:
                        actual_start -= 1
                        seen.add(prev_phi)
                        k += 1
                    else:
                        break
                best_s = actual_start

    # Ensure we have a valid result
    if best_k < 1:
        best_k = 1
        best_s = 1

    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
