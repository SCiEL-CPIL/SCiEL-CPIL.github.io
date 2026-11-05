import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    best_score = float('inf')
    best_poly = None
    
    def update_best(poly_dict, score):
        nonlocal best_score, best_poly
        if score < best_score:
            best_score = score
            max_exp = max(poly_dict.keys())
            lst = [0] * (max_exp + 1)
            for e, c in poly_dict.items():
                lst[e] = c
            best_poly = lst

    def get_score(d):
        """Exact count of nonzero terms in square."""
        square = {}
        items = list(d.items())
        for i in range(len(items)):
            ei, ci = items[i]
            for j in range(i, len(items)):
                ej, cj = items[j]
                key = ei + ej
                if i == j:
                    val = ci * ci
                else:
                    val = 2 * ci * cj
                square[key] = square.get(key, 0) + val
        return sum(1 for v in square.values() if v != 0)

    # ====================================================================
    # NEW STRATEGY: Structured Factor Decomposition with Constraint Propagation
    # 
    # We model P(x) = A(x) * B(x) * C(x).
    # Key Insight: If we choose exponents such that the support of A^2, B^2, C^2
    # are disjoint or have specific overlap patterns, we can force cancellations
    # in P^2 = A^2 * B^2 * C^2.
    # 
    # Specifically, we will search for factor structures where:
    # 1. The exponents of A, B, C are chosen to create a "grid" of sums.
    # 2. The coefficients are chosen to satisfy linear equations that force
    #    specific convolution sums to zero.
    # ====================================================================

    def try_structured_factor_search():
        """
        Search for P = A*B*C with specific exponent patterns that allow
        for systematic cancellation in P^2.
        """
        
        # We will try several distinct exponent structures for A, B, C.
        # Each structure defines the possible exponents in the product.
        
        structures = [
            # (exp_A, exp_B, exp_C)
            # Structure 1: Arithmetic progressions
            ([0, 1], [0, 3, 6], [0, 2, 4]),
            ([0, 2], [0, 1, 3], [0, 4, 5]),
            # Structure 2: Geometric-like spacing
            ([0, 1], [0, 4, 8], [0, 2, 3]),
            ([0, 3], [0, 2, 5], [0, 1, 4]),
            # Structure 3: Mixed
            ([0, 2, 4], [0, 1, 3], [0, 5, 6]),
            ([0, 1, 2], [0, 3, 6], [0, 4, 5]),
        ]

        for exp_A, exp_B, exp_C in structures:
            if time.time() > deadline:
                break
            
            # Determine the size of factors
            nA, nB, nC = len(exp_A), len(exp_B), len(exp_C)
            # Total max terms in P is nA*nB*nC. We need exactly 12.
            # If nA*nB*nC > 12, we need cancellations in P itself.
            # If nA*nB*nC == 12, no cancellations in P, but we optimize P^2.
            
            # We focus on cases where nA*nB*nC >= 12.
            # (2,3,3) -> 18 terms. Need 6 cancellations in P.
            # (3,3,3) -> 27 terms. Need 15 cancellations in P.
            # (2,2,3) -> 12 terms. No cancellations in P.
            
            # For simplicity and robustness, we will use local search on the
            # coefficients of A, B, C to minimize the number of nonzeros in P^2,
            # subject to P having exactly 12 nonzeros.
            
            def compute_P(coeffs):
                """Compute P = A*B*C from flat coefficient list."""
                A = {exp_A[i]: coeffs[i] for i in range(nA)}
                B = {exp_B[i]: coeffs[nA+i] for i in range(nB)}
                C = {exp_C[i]: coeffs[nA+nB+i] for i in range(nC)}
                
                # Multiply A * B
                AB = {}
                for ea, ca in A.items():
                    for eb, cb in B.items():
                        key = ea + eb
                        AB[key] = AB.get(key, 0) + ca * cb
                
                # Multiply AB * C
                ABC = {}
                for ea, ca in AB.items():
                    for ec, cc in C.items():
                        key = ea + ec
                        ABC[key] = ABC.get(key, 0) + ca * cc
                
                return ABC

            def evaluate(coeffs):
                P = compute_P(coeffs)
                # Remove zeros
                P_clean = {e: c for e, c in P.items() if c != 0}
                if len(P_clean) != 12:
                    return float('inf')
                return get_score(P_clean)

            # Local Search
            def local_search(structure_id):
                n_coeffs = nA + nB + nC
                # Initialize with small integers
                coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(n_coeffs)]
                
                curr_score = evaluate(coeffs)
                if curr_score < best_score:
                    P = compute_P(coeffs)
                    P_clean = {e: c for e, c in P.items() if c != 0}
                    if len(P_clean) == 12:
                        update_best(P_clean, curr_score)
                
                T = 10.0
                for it in range(20000):
                    if time.time() > deadline:
                        return
                    
                    # Mutate 1 or 2 coefficients
                    n_mut = random.choice([1, 2])
                    indices = random.sample(range(n_coeffs), n_mut)
                    old_coeffs = list(coeffs)
                    
                    for idx in indices:
                        # Sample from a wider range to allow larger cancellations
                        coeffs[idx] = random.randint(-10, 10)
                        if coeffs[idx] == 0:
                            coeffs[idx] = 1
                    
                    new_score = evaluate(coeffs)
                    
                    if new_score < curr_score:
                        curr_score = new_score
                        if curr_score < best_score:
                            P = compute_P(coeffs)
                            P_clean = {e: c for e, c in P.items() if c != 0}
                            if len(P_clean) == 12:
                                update_best(P_clean, curr_score)
                    else:
                        delta = new_score - curr_score
                        if random.random() < pow(2.718, -delta / max(T, 0.1)):
                            curr_score = new_score
                        else:
                            coeffs = old_coeffs
                    
                    T *= 0.999

            # Run a few restarts for this structure
            for restart in range(3):
                if time.time() > deadline:
                    break
                random.seed(5000 + structure_id * 100 + restart)
                local_search(structure_id)

    # ====================================================================
    # ALTERNATIVE STRATEGY: Direct Search with "Cancellation Clusters"
    #
    # Instead of factors, we directly construct P but initialize it with
    # known cancellation patterns.
    #
    # Example: If c_i = c_j and c_k = -c_l, then terms involving these
    # may cancel.
    # ====================================================================

    def try_direct_cluster_search():
        """
        Direct search on 12 coefficients, but initialized with symmetric
        patterns that encourage cancellation in the square.
        """
        
        # Pattern 1: Symmetric coefficients
        # c_0 = c_11, c_1 = c_10, etc.
        # This makes P palindromic, which has properties for the square.
        
        init_patterns = [
            # Palindromic
            [1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1],
            # Anti-palindromic
            [1, 2, 3, 4, 5, 6, -6, -5, -4, -3, -2, -1],
            # Random symmetric
            [1, -1, 2, -2, 3, -3, 3, -3, 2, -2, 1, -1],
            # Sparse-like (many 1s and -1s)
            [1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1],
        ]
        
        for init in init_patterns:
            if time.time() > deadline:
                break
            
            curr = {}
            for i, c in enumerate(init):
                c = max(-64, min(64, c))
                if c == 0:
                    c = 1
                curr[i] = c
            
            curr_score = get_score(curr)
            if curr_score < best_score:
                update_best(curr, curr_score)
            
            # Local search
            T = 10.0
            for it in range(20000):
                if time.time() > deadline:
                    break
                
                i1 = random.randint(0, 11)
                i2 = random.randint(0, 11)
                if i1 == i2:
                    i2 = (i2 + 1) % 12
                
                old_v1 = curr[i1]
                old_v2 = curr[i2]
                
                # Sample from a range that includes small and medium integers
                new_v1 = random.randint(-5, 5)
                new_v2 = random.randint(-5, 5)
                if new_v1 == 0: new_v1 = 1
                if new_v2 == 0: new_v2 = 1
                
                if new_v1 == old_v1 and new_v2 == old_v2:
                    continue
                
                curr[i1] = new_v1
                curr[i2] = new_v2
                
                new_score = get_score(curr)
                
                if new_score < curr_score:
                    curr_score = new_score
                    if curr_score < best_score:
                        update_best(curr, curr_score)
                else:
                    delta = new_score - curr_score
                    if random.random() < pow(2.718, -delta / max(T, 0.1)):
                        curr_score = new_score
                    else:
                        curr[i1] = old_v1
                        curr[i2] = old_v2
                
                T *= 0.999

    # Run the new strategies
    try_structured_factor_search()
    
    if time.time() < deadline:
        try_direct_cluster_search()

    # Fallback
    if best_poly is None:
        best_poly = [1] * 12
    
    # Final validation
    final_dict = {i: best_poly[i] for i in range(len(best_poly)) if best_poly[i] != 0}
    if len(final_dict) != 12:
        best_poly = [1] * 12
    else:
        for c in final_dict.values():
            if not (-64 <= c <= 64):
                best_poly = [1] * 12
                break
        max_exp = max(final_dict.keys())
        if max_exp > 40:
            best_poly = [1] * 12
    
    return best_poly
