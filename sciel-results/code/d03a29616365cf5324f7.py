import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin out of 150s

    # We want M with max number of distinct k such that k*sigma(k) = M,
    # k <= sqrt(M) (always true for k>=1), M <= 10^18.
    #
    # Key insight: k*sigma(k) grows roughly as k^2 * (sigma(k)/k).
    # For k with many small prime factors, sigma(k)/k is maximized.
    # So "smooth" numbers (small prime factors) are the best candidates
    # for producing collisions at the same M.
    #
    # Strategy: Generate all k composed of small primes (smooth numbers),
    # compute k*sigma(k) for each, group by M, find M with most preimages.

    # Primes to use - small primes give the smoothest numbers
    # With primes up to 47, we can generate many numbers up to 10^18
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    num_primes = len(primes)

    # Generate all smooth numbers (products of powers of these primes)
    # that are <= some bound. We need k*sigma(k) <= 10^18.
    # sigma(k) >= k+1, so k^2 + k <= M <= 10^18, roughly k <= 10^9.
    # But for smooth numbers, sigma(k)/k can be large, so k can be smaller.
    # Let's generate smooth numbers up to 10^9 to be safe.

    # Actually, let's be smarter. Generate smooth numbers and compute
    # k*sigma(k), keeping only those with M <= 10^18.

    # Use recursive generation
    candidates = []  # list of (k, sigma_k)

    def generate(idx, current_k, current_sigma):
        if time.time() > deadline - 5:
            return
        # current_k is the product of primes[0..idx-1] raised to some powers
        # current_sigma is sigma of that product
        if current_k > 10**10:  # safety bound
            return
        # Add this candidate
        M = current_k * current_sigma
        if M <= 10**18:
            candidates.append((current_k, current_sigma, M))

        if idx >= num_primes:
            return

        p = primes[idx]
        # Try powers of p
        pk = 1
        sigma_pk = 1  # sigma of p^0 = 1
        while current_k * pk <= 10**10:
            # sigma(p^a) = 1 + p + p^2 + ... + p^a
            # When we multiply current_k by p, sigma multiplies by (1 + p)
            # But we need to track sigma properly
            # sigma(current_k * p) = sigma(current_k) * (1 + p) only if p doesn't divide current_k
            # Since we're building from scratch, p is new, so this works
            new_k = current_k * pk
            new_sigma = current_sigma * (1 + p) if pk == 1 else None
            # Actually, let me redo this more carefully
            break
        # Let me use a different approach - track factorization

    # Better approach: iterative generation with proper sigma tracking
    candidates = []

    def generate_smooth(idx, current_k, current_sigma):
        if time.time() > deadline - 5:
            return
        if current_k > 10**10:
            return
        M = current_k * current_sigma
        if M <= 10**18:
            candidates.append((current_k, current_sigma))

        if idx >= num_primes:
            return

        p = primes[idx]
        # Multiply by p^a for a = 0, 1, 2, ...
        pk = 1
        # sigma(p^0) = 1, sigma(p^1) = 1+p, sigma(p^a) = (p^(a+1)-1)/(p-1)
        # When we go from p^(a-1) to p^a, sigma multiplies by p
        # Actually: sigma(p^a) = sigma(p^(a-1)) + p^a
        # So new_sigma = old_sigma + p^a where old_sigma was sigma(p^(a-1))
        # But current_sigma already includes all previous primes.
        # We need: sigma(current_k * p^a) = sigma(current_k) * sigma(p^a)
        # since current_k and p^a are coprime.

        # So for each power a of p:
        # new_k = current_k * p^a
        # new_sigma = current_sigma * sigma(p^a)
        # where sigma(p^a) = (p^(a+1) - 1) / (p - 1)

        pk = 1
        sigma_pk = 1  # sigma(p^0) = 1
        while current_k * pk <= 10**10:
            new_k = current_k * pk
            new_sigma = current_sigma * sigma_pk
            if new_k <= 10**10 and new_k * new_sigma <= 10**18:
                generate_smooth(idx + 1, new_k, new_sigma)
            # Next power
            pk *= p
            if pk > 10**10:
                break
            # sigma(p^a) = sigma(p^(a-1)) + p^a
            # But we need to track p^a separately
            # Let me just compute sigma_pk iteratively
            # Actually this is getting complicated. Let me restart with cleaner code.
            break

    # Let me use a cleaner recursive approach
    candidates = []

    def gen(idx, k, sigma_k):
        if time.time() > deadline - 3:
            return
        if k > 10**10:
            return
        M = k * sigma_k
        if M <= 10**18:
            candidates.append((k, sigma_k, M))
        if idx >= num_primes:
            return
        p = primes[idx]
        # Try p^0, p^1, p^2, ...
        pk = 1
        # For p^a, sigma(p^a) = (p^(a+1)-1)/(p-1)
        # We'll compute iteratively: sigma(p^0)=1, and when we multiply by p,
        # the new sigma contribution is sigma(p^(a+1)) = sigma(p^a) + p^(a+1)
        # But since k and p^a are coprime: sigma(k*p^a) = sigma(k)*sigma(p^a)
        # So we track sigma_p_a separately.
        sigma_p_a = 1  # sigma(p^0)
        p_power = 1    # p^0
        while k * p_power <= 10**10:
            new_k = k * p_power
            new_sigma = sigma_k * sigma_p_a
            if new_k * new_sigma <= 10**18:
                gen(idx + 1, new_k, new_sigma)
            # Advance to next power
            p_power *= p
            if p_power > 10**10:
                break
            # sigma(p^(a+1)) = sigma(p^a) + p^(a+1)
            # But p^(a+1) = p_power (after the multiply above)
            sigma_p_a += p_power

    # Start generation
    gen(0, 1, 1)

    # Now group by M
    if time.time() > deadline - 10:
        # Not enough time, just return best found so far (empty)
        # Fallback to known solution
        return {'M': int(333312), 'k': [336, 372, 434]}

    m_to_ks = defaultdict(list)
    for k, sigma_k, M in candidates:
        m_to_ks[M].append(k)

    # Find M with most k's
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3

    for M, ks in m_to_ks.items():
        if len(ks) > best_count:
            best_count = len(ks)
            best_M = M
            best_k = ks

    # Verify and ensure k <= sqrt(M)
    # For k >= 1, k <= sqrt(k*sigma(k)) iff k <= sigma(k) iff true for all k >= 1
    # since sigma(k) >= k+1 > k for k >= 1
    # So all candidates automatically satisfy k <= sqrt(M)

    # Ensure types are correct
    best_M = int(best_M)
    best_k = [int(x) for x in best_k]

    # Deduplicate (shouldn't be needed but safety)
    best_k = list(set(best_k))

    return {'M': best_M, 'k': best_k}
