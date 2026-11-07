import time
import math
import sys
from collections import defaultdict

def construct():
    deadline = time.time() + 55.0  # safety margin before 60s
    n = 210
    max_val = 10210

    def is_square(m):
        if m < 0:
            return False
        s = math.isqrt(m)
        return s * s == m

    def square_free_kernel(x):
        """Compute the square-free kernel of x."""
        result = 1
        d = 2
        while d * d <= x:
            if x % d == 0:
                count = 0
                while x % d == 0:
                    x //= d
                    count += 1
                if count % 2 == 1:
                    result *= d
            d += 1 if d == 2 else 2  # after 2, only check odd numbers
        if x > 1:
            result *= x
        return result

    # Precompute square-free kernels for all numbers in (210, 10210]
    sf = {}
    for i in range(211, max_val + 1):
        sf[i] = square_free_kernel(i)

    # For pairs: a*b*210 must be a square
    # Let sf(a) = A, sf(b) = B. Then A*B*210 must be a square.
    # This means for every prime p, the exponent of p in A*B*210 must be even.
    # Equivalently, A*B must have the same "square-free signature" as 210.
    # 
    # More precisely: if we write 210 = product of primes with odd exponents,
    # then A*B must have exactly those primes with odd exponents (and no others).
    #
    # Key insight: A*B*210 is a square iff sf(A*B*210) = 1.
    # Since A and B are square-free, A*B*210 being a square means:
    # For each prime p: (exp_p(A) + exp_p(B) + exp_p(210)) % 2 == 0
    # where exp_p(A) is 1 if p divides A, else 0.
    #
    # So: exp_p(B) = (exp_p(A) + exp_p(210)) % 2
    # This uniquely determines B's prime factors from A!
    #
    # Let's compute what B must be given A:
    # B = product over primes p of p^((exp_p(A) + exp_p(210)) % 2)
    # But this might not be square-free if we're not careful... actually it is,
    # since we're taking mod 2.
    #
    # Wait, that's not quite right. Let me think again.
    # We need A * B * 210 to be a perfect square.
    # Let C = A * 210. Then we need B such that B * C is a square.
    # Since B is square-free, B * C is a square iff B = sf(C) (the square-free part of C).
    # Because: C = k^2 * sf(C) for some k. Then B * C = B * k^2 * sf(C).
    # For this to be a square, we need B * sf(C) to be a square.
    # Since B and sf(C) are both square-free, B * sf(C) is a square iff B = sf(C).
    #
    # So the algorithm is:
    # For each a in (211, 10210]:
    #   A = sf(a)
    #   C = A * 210
    #   B_required = sf(C)  # the square-free kernel of C
    #   Find b in (211, 10210] with b > a and sf(b) == B_required
    #   To minimize max(a,b) = b (since b > a), we want the smallest such b > a.
    #
    # Precompute: for each possible square-free value v, store all numbers x with sf(x) = v.
    # Then for each a, compute B_required, look up all b with sf(b) = B_required,
    # and find the smallest b > a.

    # Build a dictionary: square_free_value -> sorted list of numbers with that kernel
    kernel_to_numbers = defaultdict(list)
    for i in range(211, max_val + 1):
        kernel_to_numbers[sf[i]].append(i)

    # Sort each list (they're already in order since we iterate in order, but let's be safe)
    for k in kernel_to_numbers:
        kernel_to_numbers[k].sort()

    best_max = max_val + 1
    best_result = None

    # For each a from 211 upward, find the best b
    for a in range(211, max_val + 1):
        if time.time() >= deadline:
            break
        A = sf[a]
        C = A * 210
        B_required = square_free_kernel(C)

        # Look up numbers with this kernel
        if B_required in kernel_to_numbers:
            candidates = kernel_to_numbers[B_required]
            # Find the smallest b > a in candidates
            # Binary search for the first element > a
            lo, hi = 0, len(candidates)
            while lo < hi:
                mid = (lo + hi) // 2
                if candidates[mid] <= a:
                    lo = mid + 1
                else:
                    hi = mid
            if lo < len(candidates):
                b = candidates[lo]
                if b <= max_val:
                    candidate_max = b  # since b > a
                    if candidate_max < best_max:
                        best_max = candidate_max
                        best_result = (a, b)
                    # Since we're iterating a in increasing order, and we want to minimize b,
                    # once best_max is small, we can potentially break early.
                    if best_max - 210 <= 1:
                        break

    # Verify and return
    if best_result is None:
        # Fallback: single element 840 (840*210 = 176400 = 420^2)
        return [840]

    a, b = best_result
    prod = a * b * 210
    if is_square(prod):
        return [int(a), int(b)]
    else:
        return [840]
