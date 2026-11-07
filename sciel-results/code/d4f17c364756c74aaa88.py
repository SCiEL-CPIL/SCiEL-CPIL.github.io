"""Rewritten: linear-algebra view + deterministic greedy with forward checking + local search."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit

    # Factorization setup: primes up to 40
    all_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    prime_to_idx = {p: i for i, p in enumerate(all_primes)}
    D = len(all_primes)

    # Exponent vector for each number 2..40
    def factorize(n):
        vec = [0] * D
        for i, p in enumerate(all_primes):
            while n % p == 0:
                vec[i] += 1
                n //= p
        return tuple(vec)

    candidates = list(range(2, 41))
    vecs = {n: factorize(n) for n in candidates}

    # A set S is valid iff its exponent vectors are linearly independent over Q.
    # We maintain a basis (row-echelon form) of the vectors in S.
    # Adding v is valid iff v is not in the span of the current basis.

    def basis_reduce(vec, basis):
        """Reduce vec against basis (list of (pivot_col, vector)). Return reduced vector or None if dependent."""
        v = list(vec)
        for pivot_col, bv in basis:
            if v[pivot_col] != 0:
                # Eliminate using bv
                factor = v[pivot_col]
                for j in range(pivot_col, D):
                    v[j] -= factor * bv[j]
        # Check if all remaining are zero
        for j in range(D):
            if v[j] != 0:
                return tuple(v)
        return None  # dependent

    def basis_add(basis, vec):
        """Add vec to basis. Returns new basis. Assumes vec is independent of current basis."""
        v = list(vec)
        new_basis = list(basis)
        # Reduce against existing basis
        for pivot_col, bv in new_basis:
            if v[pivot_col] != 0:
                factor = v[pivot_col]
                for j in range(pivot_col, D):
                    v[j] -= factor * bv[j]
        # Find new pivot
        for j in range(D):
            if v[j] != 0:
                new_basis.append((j, tuple(v)))
                break
        # Sort by pivot column for consistency
        new_basis.sort(key=lambda x: x[0])
        return new_basis

    def is_independent(vec, basis):
        """Check if vec is independent of basis."""
        return basis_reduce(vec, basis) is not None

    def try_add(vec, basis):
        """Try to add vec to basis. Returns new basis or None."""
        reduced = basis_reduce(vec, basis)
        if reduced is None:
            return None
        return basis_add(basis, reduced)

    # Deterministic greedy with forward checking:
    # At each step, pick the candidate that leaves the most options (fewest conflicts with remaining).
    def greedy_forward_check():
        basis = []
        S = []
        remaining = list(candidates)
        
        while remaining and len(S) < 18:
            # For each candidate, check if it can be added
            viable = []
            for x in remaining:
                if is_independent(vecs[x], basis):
                    viable.append(x)
            
            if not viable:
                break
            
            # Score each viable candidate by how many other candidates it would block
            best_x = None
            best_score = -1
            for x in viable:
                new_basis = try_add(vecs[x], basis)
                # Count how many remaining (other than x) are still viable after adding x
                count = 0
                for y in remaining:
                    if y == x:
                        continue
                    if is_independent(vecs[y], new_basis):
                        count += 1
                # We want to maximize remaining options, so pick x that blocks fewest
                # Actually, we want to pick x that leaves the most options
                # So score = count (higher is better)
                if count > best_score:
                    best_score = count
                    best_x = x
            
            if best_x is None:
                break
            
            basis = try_add(vecs[best_x], basis)
            S.append(best_x)
            remaining = [x for x in remaining if x != best_x]
        
        return S, basis

    # Local search: 1-for-1 swaps with forward checking
    def local_search(S, basis, max_iters=100):
        S = list(S)
        basis = basis
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            for i, x in enumerate(S):
                # Remove x
                S_no_x = S[:i] + S[i+1:]
                # Rebuild basis without x
                basis_no_x = []
                for y in S_no_x:
                    new_basis = try_add(vecs[y], basis_no_x)
                    if new_basis is None:
                        break
                    basis_no_x = new_basis
                else:
                    # Try adding each candidate not in S
                    for y in candidates:
                        if y in S:
                            continue
                        if is_independent(vecs[y], basis_no_x):
                            # Check if this swap improves (it doesn't in size, but might enable future additions)
                            # For now, just accept the swap to explore different configurations
                            new_basis = try_add(vecs[y], basis_no_x)
                            S[i] = y
                            basis = new_basis
                            improved = True
                            break
                    if improved:
                        break
        return S, basis

    # Run deterministic greedy first
    best_S, best_basis = greedy_forward_check()
    best_size = len(best_S)
    
    # Try multiple randomized greedy variants
    for restart in range(500):
        if time.time() > deadline - 5:
            break
        
        r = restart % 5
        if r == 0:
            # Pure deterministic
            S, basis = greedy_forward_check()
        elif r == 1:
            # Randomized greedy: shuffle candidates before scoring
            basis = []
            S = []
            remaining = list(candidates)
            random.shuffle(remaining)
            while remaining and len(S) < 18:
                viable = [x for x in remaining if is_independent(vecs[x], basis)]
                if not viable:
                    break
                # Pick randomly from top-scoring
                scored = []
                for x in viable:
                    new_basis = try_add(vecs[x], basis)
                    count = sum(1 for y in remaining if y != x and is_independent(vecs[y], new_basis))
                    scored.append((count, x))
                scored.sort(reverse=True)
                # Pick from top 3
                top = scored[:min(3, len(scored))]
                _, best_x = random.choice(top)
                basis = try_add(vecs[best_x], basis)
                S.append(best_x)
                remaining = [x for x in remaining if x != best_x]
        elif r == 2:
            # Start from best, perturb
            S = list(best_S)
            for _ in range(random.randint(1, 3)):
                if S:
                    S.pop(random.randint(0, len(S) - 1))
            # Rebuild basis
            basis = []
            valid_S = []
            for x in S:
                nb = try_add(vecs[x], basis)
                if nb is not None:
                    basis = nb
                    valid_S.append(x)
            S = valid_S
            # Complete greedily
            remaining = [x for x in candidates if x not in S]
            while remaining and len(S) < 18:
                viable = [x for x in remaining if is_independent(vecs[x], basis)]
                if not viable:
                    break
                best_x = random.choice(viable)
                basis = try_add(vecs[best_x], basis)
                S.append(best_x)
                remaining = [x for x in remaining if x != best_x]
        elif r == 3:
            # Start with a random small independent set, then greedy complete
            basis = []
            S = []
            # Build small random independent set
            pool = list(candidates)
            random.shuffle(pool)
            for x in pool:
                if len(S) >= random.randint(2, 5):
                    break
                if is_independent(vecs[x], basis):
                    basis = try_add(vecs[x], basis)
                    S.append(x)
            # Greedy complete
            remaining = [x for x in candidates if x not in S]
            while remaining and len(S) < 18:
                viable = [x for x in remaining if is_independent(vecs[x], basis)]
                if not viable:
                    break
                best_x = random.choice(viable)
                basis = try_add(vecs[best_x], basis)
                S.append(best_x)
                remaining = [x for x in remaining if x != best_x]
        else:
            # Start from primes, then try to add composites
            S = [p for p in range(2, 41) if all(p % d for d in range(2, isqrt(p) + 1))]
            basis = []
            for x in S:
                nb = try_add(vecs[x], basis)
                if nb is not None:
                    basis = nb
            # Try adding composites
            composites = [x for x in candidates if x not in S]
            random.shuffle(composites)
            for x in composites:
                if len(S) >= 18:
                    break
                if is_independent(vecs[x], basis):
                    basis = try_add(vecs[x], basis)
                    S.append(x)
        
        if len(S) > best_size:
            best_S = list(S)
            best_size = len(S)
            best_basis = basis
            if best_size == 18:
                break
    
    # Final verification using direct product check
    def verify_direct(S):
        products = {1}
        for n in S:
            new_products = set()
            for p in products:
                np = p * n
                if np in products or np in new_products:
                    return False
                new_products.add(np)
            products = products | new_products
        return True
    
    if not verify_direct(best_S):
        # Fallback to primes
        return [int(p) for p in range(2, 41) if all(p % d for d in range(2, isqrt(p) + 1))]
    
    return [int(x) for x in best_S]

# EVOLVE-BLOCK-END
