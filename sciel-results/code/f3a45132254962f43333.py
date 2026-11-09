import itertools
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    best_poly = None
    best_nz = 100  # Max possible nonzero terms in square is ~144, but we expect lower
    
    # We need exactly 12 nonzero terms in P.
    # We split the 12 terms into two sets A and B with disjoint supports.
    # Let size(A) = k, size(B) = 12-k.
    # Coefficients in A and B are in {-1, 1}.
    # This guarantees all 12 terms in P are nonzero and in [-64, 64].
    
    # The square is P^2 = A^2 + B^2 + 2AB.
    # We want to minimize the number of nonzero coefficients in P^2.
    
    # To make the search tractable, we restrict the supports to be contiguous blocks
    # or simple arithmetic progressions, and try small k values.
    # Given 60s limit, we can't do full 2^12 * support search.
    # We will try specific support structures and exhaustively search signs for small k.
    
    # Strategy:
    # 1. Try k=1 to 11.
    # 2. For each k, try a few simple support structures (e.g., interleaved, contiguous).
    # 3. For a fixed support, if k is small (e.g., <= 6), we can exhaustively try all 2^k * 2^(12-k) sign combinations.
    #    2^12 = 4096. This is very fast.
    #    So for ANY fixed support, we can exhaustively check all 4096 sign assignments.
    #    The bottleneck is choosing the support.
    
    # Let's generate a set of candidate supports for the 12 exponents.
    # The exponents must be distinct integers.
    # To allow for cancellations, we want the pairwise sums (from A^2, B^2, AB) to collide.
    # Arithmetic progressions are good for collisions.
    # Let's try supports that are subsets of an arithmetic progression.
    
    # We will iterate over a range of "base" arithmetic progressions and choose 12 distinct positions from them.
    # Actually, simpler: Just pick 12 distinct exponents from a small range, say 0..20.
    # But there are C(21, 12) ~ 293930 supports. Too many to test all with 4096 sign checks each (1.2 billion ops).
    # We need to be smarter.
    
    # Observation: If the support is an arithmetic progression $0, d, 2d, ..., 11d$, 
    # the square's support is $0, d, ..., 22d$.
    # We can test this specific support structure with all 4096 sign assignments.
    # Then try other structures.
    
    # Let's define a function to evaluate a specific support (list of 12 exponents)
    # by trying all 2^12 sign assignments.
    
    def evaluate_support(exponents):
        nonlocal best_poly, best_nz
        # exponents: list of 12 ints
        max_exp = max(exponents)
        min_exp = min(exponents)
        # Square degree is 2*max_exp. We only care about relative positions for sparsity count,
        # but we need to return the actual polynomial.
        
        # To speed up, we can precompute the index mapping.
        # But 4096 * 144 ops is small. Let's just do it.
        
        # We will iterate over all sign combinations.
        # Since 12 is small, we can use bitmasks.
        for mask in range(4096):
            # Construct signs
            signs = []
            for i in range(12):
                if (mask >> i) & 1:
                    signs.append(1)
                else:
                    signs.append(-1)
            
            # Compute P^2
            # P(x) = sum signs[i] * x^(exponents[i])
            # P(x)^2 = sum_i sum_j signs[i]signs[j] x^(e_i+e_j)
            
            # Use a dictionary or array for coefficients
            # Max degree is 2 * max_exp. Min degree 2 * min_exp.
            # Let's shift to start at 0 for calculation, then shift back.
            # Actually, just use a dict to count nonzero terms easily.
            coeffs = {}
            for i in range(12):
                for j in range(12):
                    exp = exponents[i] + exponents[j]
                    val = signs[i] * signs[j]
                    coeffs[exp] = coeffs.get(exp, 0) + val
            
            # Count nonzero
            nz = sum(1 for v in coeffs.values() if v != 0)
            
            if nz < best_nz:
                best_nz = nz
                # Reconstruct polynomial
                # Find min and max exponent in coeffs
                min_e = min(coeffs.keys())
                max_e = max(coeffs.keys())
                # Wait, we need to return the ORIGINAL polynomial P, not the square.
                # The task says: "Return coefficient list... of the candidate [polynomial]".
                # "Minimize nonzero terms in the EXACT integer square."
                # So we return P, but score is based on sparsity of P^2.
                
                # Construct P coefficients
                # P has terms at exponents[i] with coefficients signs[i].
                # We need a list in ascending degree.
                # Find min and max exponent in exponents.
                p_min_e = min(exponents)
                p_max_e = max(exponents)
                length = p_max_e - p_min_e + 1
                p_coeffs = [0] * length
                for i in range(12):
                    p_coeffs[exponents[i] - p_min_e] = signs[i]
                
                best_poly = p_coeffs
                # Early exit if we find a very good one? 
                # Theoretical minimum for 12 terms?
                # If P^2 has 1 term, P is monomial (1 term).
                # If P^2 has 2 terms, P is binomial? (a+b)^2 = a^2+2ab+b^2. 3 terms.
                # It is known that sparse squares are related to Golay sequences.
                # For length 12, a perfect autocorrelation is not possible.
                # Let's aim for < 20 nonzero terms in square.
                if best_nz <= 14:
                    return True
        return False

    # Candidate supports:
    # 1. Arithmetic progressions
    for d in [1, 2, 3, 4, 5]:
        for start in range(0, 5):
            exponents = [start + i*d for i in range(12)]
            if evaluate_support(exponents):
                if time.time() > deadline:
                    break
        if time.time() > deadline:
            break
    
    # 2. Interleaved structures (e.g., even/odd)
    # Try a few random supports with some structure
    # To save time, let's try supports derived from small sets.
    
    # Let's try a specific known good structure if we can guess one.
    # But for now, let's add a few random structured supports.
    
    # Support: 0, 1, 3, 7, ... (Golay-like?)
    # Let's just try a few random subsets of {0..20} that are "dense" enough to allow collisions.
    
    # Actually, let's try supports where the 12 points are chosen from a smaller range to force collisions.
    # Range 0..15 (16 points). Choose 12.
    # C(16, 12) = 1820.
    # 1820 * 4096 = 7.4 million evaluations.
    # Each eval is 144 additions. ~1 billion ops. Might be too slow in Python (10-30s).
    # Let's optimize the inner loop or reduce candidates.
    
    # We can prune supports that are "Sidon-like" (few collisions) as they won't cancel well.
    # But computing collisions is also work.
    
    # Let's stick to arithmetic progressions and a few specific patterns.
    
    # Pattern: Two interleaved APs
    # e.g., 0, 2, 4, ..., 10 and 1, 3, 5, ..., 11
    exponents = list(range(0, 24, 2))[:12] # 0,2,...22
    if time.time() < deadline:
        evaluate_support(exponents)
        
    exponents = [i for i in range(0, 12)] # 0..11
    if time.time() < deadline:
        evaluate_support(exponents)

    # Try a "Golay" inspired support
    # Golay sequence of length 8: 11111111 or 1111-1111 etc.
    # For 12, maybe combine two 6-term sequences?
    
    # Let's try one more specific structure: 
    # Exponents: 0, 1, 2, 4, 8, 16... no, too sparse.
    
    # If we haven't found a good one, we might need to search more.
    # But let's check the time.
    
    if best_poly is None:
        # Fallback: Return a valid polynomial with 12 terms.
        # Simple AP 0..11 with all 1s.
        best_poly = [1]*12
        # Calculate its square sparsity just in case we need to report, 
        # but the function just returns the list.
        
    return best_poly

# EVOLVE-BLOCK-END
