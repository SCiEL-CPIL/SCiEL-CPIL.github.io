import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    n = 210
    max_val = 10210

    # Sieve for primes up to max_val
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(max_val ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, max_val + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, max_val + 1) if is_prime[i]]
    prime_index = {p: i for i, p in enumerate(primes)}

    def get_parity_mask(x):
        """Return bitmask where bit i is set if prime primes[i] appears to odd power in x."""
        mask = 0
        temp = x
        for idx, p in enumerate(primes):
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count ^= 1
                if count:
                    mask |= (1 << idx)
        if temp > 1:
            if temp in prime_index:
                mask |= (1 << prime_index[temp])
        return mask

    # Precompute masks for all numbers up to max_val
    masks = [0] * (max_val + 1)
    for i in range(2, max_val + 1):
        masks[i] = get_parity_mask(i)

    # Parity mask of 210
    target = masks[210]

    # Basis: dict mapping highest-bit position -> (vector, history_mask)
    basis = {}
    basis_hist = {}
    
    # Store all dependency masks found so far (kernel candidates)
    # Actually, we need the kernel basis for the current set of vectors.
    # We will recompute or maintain it.
    # Since we add one vector at a time, we can maintain a list of kernel vectors.
    # But the kernel vectors change as we add more vectors?
    # No, the kernel of the first K vectors is a subspace of the kernel of the first K+1 vectors?
    # No. The map changes.
    # It's safer to compute the kernel basis from scratch when we need it, 
    # or maintain the "history" of reductions.
    
    # Let's just collect "zero-reduction" history masks.
    # Each time a new vector reduces to 0, we get a dependency.
    # The set of all such dependencies spans the kernel.
    kernel_deps = []

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
            # tv became 0
            reduced_to_zero = True
            if th != 0:
                kernel_deps.append(th)

        # Check if target is in span
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
            # We need to find the minimal weight solution in th + Kernel
            # Get Kernel Basis from kernel_deps
            
            # If kernel_deps is empty, solution is unique
            # Otherwise, find basis for kernel_deps
            
            k_basis = []
            # Simple Gaussian elimination on kernel_deps to find independent vectors
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
            
            # If kernel dimension is small, enumerate
            if n_kernel <= 15:
                min_sol = None
                min_size = float('inf')
                
                # Iterate all 2^n_kernel combinations
                # Use bit manipulation for speed
                for mask_idx in range(1 << n_kernel):
                    # Compute XOR of selected kernel vectors
                    z = 0
                    mm = mask_idx
                    kb_idx = 0
                    while mm:
                        if mm & 1:
                            z ^= k_basis[kb_idx]
                        mm >>= 1
                        kb_idx += 1
                    
                    # Candidate solution = th ^ z
                    cand = th ^ z
                    
                    # Calculate size (popcount)
                    size = bin(cand).count('1')
                    
                    if size < min_size:
                        min_size = size
                        min_sol = cand
                        
                # Check if min_sol is valid (size <= 100)
                if min_size <= 100:
                    sol_set = set()
                    hh = min_sol
                    while hh:
                        lsb = hh & -hh
                        idx = lsb.bit_length() - 1
                        sol_set.add(211 + idx)
                        hh ^= lsb
                    
                    sol = list(sol_set)
                    # Verify
                    prod = 210
                    for x in sol:
                        prod *= x
                    s = math.isqrt(prod)
                    if s * s == prod:
                        return [int(x) for x in sol]
            else:
                # Fallback: try to reduce weight of th using kernel basis
                s = th
                # Local search: greedily XOR kernel vectors that reduce weight
                improved = True
                while improved:
                    improved = False
                    current_weight = bin(s).count('1')
                    for k in k_basis:
                        new_s = s ^ k
                        new_weight = bin(new_s).count('1')
                        if new_weight < current_weight:
                            s = new_s
                            current_weight = new_weight
                            improved = True
                            
                size = bin(s).count('1')
                if size <= 100:
                    sol_set = set()
                    hh = s
                    while hh:
                        lsb = hh & -hh
                        idx = lsb.bit_length() - 1
                        sol_set.add(211 + idx)
                        hh ^= lsb
                    sol = list(sol_set)
                    prod = 210
                    for x in sol:
                        prod *= x
                    sq_root = math.isqrt(prod)
                    if sq_root * sq_root == prod:
                        return [int(x) for x in sol]
                # If still too big, continue to next M

    # Fallback (should not be reached)
    return [840]
