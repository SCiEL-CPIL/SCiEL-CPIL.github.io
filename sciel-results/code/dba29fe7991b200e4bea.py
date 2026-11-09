import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    n = 210
    max_val = 10210

    # Sieve for primes up to max_val
    is_prime = np.ones(max_val + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(max_val ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i * i::i] = False
    primes = np.array([i for i in range(2, max_val + 1) if is_prime[i]])
    prime_index = {int(p): i for i, p in enumerate(primes)}

    # Precompute parity masks for all numbers up to max_val
    masks = [0] * (max_val + 1)
    for x in range(2, max_val + 1):
        mask = 0
        temp = x
        for idx, p in enumerate(primes):
            p_int = int(p)
            if p_int * p_int > temp:
                break
            if temp % p_int == 0:
                count = 0
                while temp % p_int == 0:
                    temp //= p_int
                    count ^= 1
                if count:
                    mask |= (1 << idx)
        if temp > 1:
            if temp in prime_index:
                mask |= (1 << prime_index[temp])
        masks[x] = mask

    # Parity mask of 210
    target = masks[210]

    # We'll add candidates from 211 upward, maintaining a linear basis.
    # basis: dict bit_pos -> (vector, history_mask)
    # kernel_deps: list of history masks that reduce to zero (span the kernel)
    basis = {}
    basis_hist = {}
    kernel_deps = []

    best_solution = None
    best_raw = float('inf')

    for M in range(211, max_val + 1):
        if time.time() > deadline:
            break

        v = masks[M]
        h = 1 << (M - 211)  # history bit for this candidate

        # Insert v into basis with history h
        tv = v
        th = h
        reduced_to_zero = False
        while tv:
            p = tv.bit_length() - 1
            if p in basis:
                tv ^= basis[p]
                th ^= basis_hist[p]
            else:
                basis[p] = tv
                basis_hist[p] = th
                break
        else:
            reduced_to_zero = True
            if th != 0:
                kernel_deps.append(th)

        # Check if target is in span of current basis
        tv = target
        th = 0
        in_span = True
        while tv:
            p = tv.bit_length() - 1
            if p in basis:
                tv ^= basis[p]
                th ^= basis_hist[p]
            else:
                in_span = False
                break

        if in_span and tv == 0:
            # We have a particular solution th
            # Need to find minimal-weight (min max index) solution in th + Kernel

            # Build kernel basis from kernel_deps
            k_basis = []
            for dep in kernel_deps:
                td = dep
                for kb in k_basis:
                    if td:
                        p = td.bit_length() - 1
                        if kb.bit_length() - 1 == p:
                            td ^= kb
                if td:
                    k_basis.append(td)

            n_kernel = len(k_basis)

            # Try to minimize max index. Strategy: greedy elimination.
            # We want to minimize the highest bit set in the solution.
            # Start with th, and try to eliminate high bits using kernel vectors.

            cand = th
            # Sort k_basis by highest bit descending
            k_sorted = sorted(k_basis, key=lambda x: x.bit_length(), reverse=True)

            # Greedy: for each high bit in cand, try to flip it off using a kernel vector
            # that has that bit set.
            improved = True
            while improved:
                improved = False
                if cand == 0:
                    break
                # Find highest bit in cand
                hi_bit = cand.bit_length() - 1
                # Find a kernel vector with this bit set
                for kb in k_sorted:
                    if kb & (1 << hi_bit):
                        new_cand = cand ^ kb
                        # Only accept if it reduces the "max index" or doesn't increase it
                        if new_cand.bit_length() <= cand.bit_length():
                            if new_cand.bit_length() < cand.bit_length() or \
                               (new_cand.bit_length() == cand.bit_length() and new_cand < cand):
                                cand = new_cand
                                improved = True
                                break

            # Also try a local search: for each kernel vector, see if XOR improves
            for _ in range(3):  # a few passes
                for kb in k_sorted:
                    new_cand = cand ^ kb
                    if new_cand.bit_length() < cand.bit_length() or \
                       (new_cand.bit_length() == cand.bit_length() and new_cand < cand):
                        cand = new_cand
                # Check if any single flip helps
                improved = False
                for kb in k_sorted:
                    new_cand = cand ^ kb
                    if new_cand.bit_length() < cand.bit_length() or \
                       (new_cand.bit_length() == cand.bit_length() and new_cand < cand):
                        cand = new_cand
                        improved = True
                if not improved:
                    break

            # If kernel dimension is small, also try full enumeration for min-max
            if n_kernel <= 12:
                best_cand = cand
                best_max_bit = cand.bit_length()
                for mask_idx in range(1 << n_kernel):
                    z = 0
                    mm = mask_idx
                    kb_idx = 0
                    while mm:
                        if mm & 1:
                            z ^= k_sorted[kb_idx]
                        mm >>= 1
                        kb_idx += 1
                    t = th ^ z
                    if t.bit_length() < best_max_bit or \
                       (t.bit_length() == best_max_bit and t < best_cand):
                        best_max_bit = t.bit_length()
                        best_cand = t
                cand = best_cand

            # Check size constraint
            size = bin(cand).count('1')
            if size <= 100:
                # Extract the solution
                sol_set = set()
                hh = cand
                while hh:
                    lsb = hh & -hh
                    idx = lsb.bit_length() - 1
                    sol_set.add(211 + idx)
                    hh ^= lsb

                if sol_set:
                    sol = list(sol_set)
                    max_sel = max(sol)
                    raw = max_sel - 210

                    # Verify the square condition
                    prod = 210
                    for x in sol:
                        prod *= x
                    s = math.isqrt(prod)
                    if s * s == prod:
                        if raw < best_raw:
                            best_raw = raw
                            best_solution = sol

                        # Since we're going M in order, the first valid solution
                        # has the minimum max value. Return immediately.
                        return [int(x) for x in sol]

    # Fallback
    if best_solution is not None:
        return [int(x) for x in best_solution]
    return [840]
