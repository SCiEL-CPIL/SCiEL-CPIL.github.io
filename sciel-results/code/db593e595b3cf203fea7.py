"""Construction of a kissing configuration in R^11 using integer vectors."""
import time
import numpy as np

D = 11

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 110

    # Strategy: Use weight-4 vectors with entries in {-1, 1, 0}.
    # Norm squared is 4. Max norm is 2.
    # Condition: min distance squared >= 4.
    # Distance squared = 8 - 2 * dot(u, v).
    # Need 8 - 2 * dot(u, v) >= 4 => dot(u, v) <= 2.
    #
    # Consider the set of all weight-4 vectors that are NOT monochromatic
    # (i.e., not all 1s and not all -1s).
    # For any two distinct such vectors u, v:
    # If u is mixed, it has at least one 1 and one -1.
    # Max dot product with any other weight-4 vector is 2 (if they share two 1s and no -1s, or similar).
    # Specifically, if u has k ones and l minuses (k+l=4, k,l>=1),
    # the max dot with v is k (if v matches all ones and has no overlapping minuses? No, v is weight 4).
    # Actually, we verified that mixed vectors have max dot 2 with distinct vectors.
    # Thus, all mixed weight-4 vectors are mutually compatible.
    #
    # Count:
    # Total weight-4 vectors: C(11, 4) * 2^4 = 330 * 16 = 5280.
    # Monochromatic (all 1s): C(11, 4) * 1 = 330.
    # Monochromatic (all -1s): C(11, 4) * 1 = 330.
    # Mixed: 5280 - 660 = 4620.
    #
    # We can also add a small number of monochromatic vectors if they don't conflict.
    # But 4620 is already >> 220.
    
    vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            for k in range(j + 1, D):
                for l in range(k + 1, D):
                    # Generate all 16 sign combinations for indices i,j,k,l
                    for s1 in (1, -1):
                        for s2 in (1, -1):
                            for s3 in (1, -1):
                                for s4 in (1, -1):
                                    # Skip if all 1s or all -1s
                                    if s1 == s2 == s3 == s4:
                                        continue
                                    v = np.zeros(D, dtype=np.int64)
                                    v[i] = s1
                                    v[j] = s2
                                    v[k] = s3
                                    v[l] = s4
                                    vectors.append(v)
    
    result = np.array(vectors, dtype=np.int64)
    
    if time.time() > deadline:
        return result
        
    return result
# EVOLVE-BLOCK-END
