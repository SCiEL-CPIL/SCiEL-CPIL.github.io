"""Rewritten program: greedy squarefree-kernel matching with deadline guard."""
import time

def construct():
    deadline = time.time() + 55  # safety margin before 60s limit
    N = 210
    MAX_VAL = 10210
    MAX_SIZE = 100

    def is_squarefree(n):
        """Check if n is squarefree (no prime squared divides n)."""
        if n < 2:
            return True
        d = 2
        while d * d <= n:
            if n % (d * d) == 0:
                return False
            d += 1
        return True

    def squarefree_kernel(n):
        """Compute the squarefree part of n."""
        result = 1
        d = 2
        while d * d <= n:
            if n % (d * d) == 0:
                result *= d
                while n % (d * d) == 0:
                    n //= (d * d)
            d += 1
        return result * n

    def is_perfect_square(n):
        if n < 0:
            return False
        r = int(n ** 0.5)
        # Adjust for floating point
        while r * r > n:
            r -= 1
        while (r + 1) * (r + 1) <= n:
            r += 1
        return r * r == n

    # Target: product * 210 must be a perfect square
    # 210 = 2*3*5*7 (squarefree)
    # So product must have squarefree kernel = 210
    # i.e., product = 210 * k^2 for some integer k
    # Equivalently: squarefree_kernel(product) == 210

    target_kernel = 210

    # Greedy: repeatedly pick the smallest available number that reduces
    # the "distance" from current kernel to target_kernel
    # We track: current_product_kernel (squarefree part of running product)
    # We want current_product_kernel * 210 to be a perfect square
    # i.e., current_product_kernel == 210 (since 210 is squarefree)

    used = set()
    selected = []
    current_kernel = 1  # squarefree kernel of the running product

    # Precompute squarefree kernels for all candidates
    # For efficiency, precompute for range 211..MAX_VAL
    kernels = {}
    for v in range(211, MAX_VAL + 1):
        kernels[v] = squarefree_kernel(v)

    # Greedy selection: at each step, find the smallest unused v such that
    # the new kernel (current_kernel * kernel(v)) is "closer" to 210
    # Distance metric: we want current_kernel * kernel(v) to eventually equal 210
    # Better: pick v that makes the gcd with 210 larger, or reduces the ratio

    # Actually, let's think differently:
    # We need product = 210 * k^2
    # Start with product = 1, kernel = 1
    # Each step: pick smallest unused v, update kernel = sf(kernel * v)
    # Stop when kernel == 210

    # Greedy strategy: always pick the smallest unused v that moves us closer
    # "Closer" = the new kernel shares more factors with 210, or the quotient
    # kernel/210 (in terms of prime factors) gets simpler

    # Simplest effective greedy: pick smallest unused v each time, check if
    # we've reached kernel 210. If not, continue. This might not be optimal
    # but let's try a smarter approach:

    # Better greedy: at each step, among all unused v, pick the one that
    # minimizes the "remaining work" = number of prime factors in
    # lcm(kernel, 210) / kernel that aren't yet matched with 210.

    # Even simpler: pick v such that kernel(v) has maximum overlap with
    # the "deficit" = 210 / gcd(210, kernel) ... this is getting complex.

    # Let's try: always pick the smallest unused number. Track kernel.
    # If kernel becomes 210, we're done.
    # This is O(n) per step in worst case but should work for small n.

    # Actually, the key insight: if we just pick numbers in order 211, 212, ...
    # the kernel will evolve. We stop when it hits 210.
    # But this might take many numbers. Let's be smarter.

    # Smart greedy: at each step, for each unused candidate v (in order of v),
    # compute new_kernel = sf(current_kernel * v). If new_kernel == 210, pick v.
    # Otherwise, prefer v that makes new_kernel "closer" to 210.
    # Closeness: count how many prime factors of 210 are "satisfied".

    # Let me define: deficit = 210 // gcd(210, current_kernel)  (not quite right)
    # Better: factor 210 = {2:1, 3:1, 5:1, 7:1}
    # For current_kernel, track which of these primes have odd exponent.
    # We want all four to have odd exponent (matching 210).

    # Simpler approach: just try all subsets greedily.
    # Pick smallest v, add it, check. If kernel == 210, done.
    # If we exceed 100 numbers without success, backtrack or try different order.

    # Given the constraint of 100 numbers max, and we need kernel to be exactly 210,
    # let's try: iterate through numbers in ascending order, add each one,
    # check if kernel becomes 210. The first time it does, we're done.

    # This is O(MAX_VAL) in the worst case but should find a solution quickly.

    current_kernel = 1
    selected = []
    for v in range(211, MAX_VAL + 1):
        if len(selected) >= MAX_SIZE:
            break
        new_kernel = squarefree_kernel(current_kernel * v)
        selected.append(v)
        current_kernel = new_kernel
        if current_kernel == target_kernel:
            break
        if time.time() > deadline:
            break

    # Verify
    if current_kernel != target_kernel:
        # Fallback: use 4*210 = 840 (since 840*210 = 840^2/210... let me check)
        # 840 = 4*210 = 2^2 * 210, so sf(840) = 210. 
        # So [840] alone works: 840 * 210 = 210^2 * 4 = (2*210)^2 = 420^2. Yes!
        selected = [840]
        current_kernel = 210

    # Verify the solution
    product = 1
    for x in selected:
        product *= x
    total = product * 210
    if not is_perfect_square(total):
        # Fallback
        selected = [840]

    # Ensure types are correct
    result = [int(x) for x in selected]
    return result
