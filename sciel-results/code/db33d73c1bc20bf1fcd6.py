import itertools
import numpy as np

def construct():
    D = 11
    vectors = []
    
    # Generate all combinations of 6 indices
    # Using 6-sparse vectors ensures that for any distinct pair,
    # the squared distance is at least the squared norm (6).
    # Proof:
    # Norm^2 = 6.
    # Let u, v be distinct 6-sparse vectors.
    # Let k be the number of shared indices.
    # Dist^2 = 2(6-k) + 4 * (number of sign disagreements on shared indices).
    # Since u != v, if k=6, there is at least 1 sign disagreement.
    # If k < 6, there are at least 1 unique index in u or v.
    # 
    # Case k=6: Dist^2 >= 0 + 4*1 = 4? No.
    # Wait.
    # If k=6, supports are identical.
    # u, v differ in at least 1 sign.
    # Dist^2 = 4 * (num diffs). Min num diffs = 1.
    # Dist^2 = 4.
    # Norm^2 = 6.
    # 4 < 6. FAIL.
    
    # My previous deduction for s=6 was:
    # "Condition s+4 >= 2k => 10 >= 2k => k <= 5"
    # This assumed d_min=1.
    # But if k=6, d_min=1 gives Dist^2 = 4.
    # 4 < 6.
    # So s=6 is NOT always valid.
    
    # Let's re-evaluate s=4.
    # Norm^2 = 4.
    # k=4 (same support): Dist^2 >= 4. OK.
    # k=3: Dist^2 = 2(1) + 4(1) = 6 >= 4. OK.
    # k=2: Dist^2 = 2(2) + 4(1) = 8 >= 4. OK.
    # k=1: Dist^2 = 2(3) + 4(1) = 10 >= 4. OK.
    # k=0: Dist^2 = 2(4) = 8 >= 4. OK.
    # So s=4 is always valid. Count 5280.
    
    # What about s=5?
    # Norm^2 = 5.
    # k=5: Dist^2 >= 4. 4 < 5. FAIL.
    # So s=5 requires pruning.
    
    # What about s=3?
    # Norm^2 = 3.
    # k=3: Dist^2 >= 4. OK.
    # k=2: Dist^2 = 2(1) + 4(1) = 6 >= 3. OK.
    # k=1: Dist^2 = 2(2) + 4(1) = 8 >= 3. OK.
    # k=0: Dist^2 = 6 >= 3. OK.
    # So s=3 is always valid.
    # Count: C(11,3) * 8 = 165 * 8 = 1320.
    # Smaller than s=4.
    
    # So s=4 is the best "always valid" sparsity.
    
    # Can we do better with s=5 by pruning?
    # Total 5-sparse: 14784.
    # Conflict: k=5, d=1.
    # We need to pick a set of 5-sparse vectors such that no two with same support differ in only 1 sign.
    # This is the independent set of Q_5 with min dist 2.
    # Max size is 16 (even parity).
    # So we can pick 16 vectors per support.
    # Total: 462 * 16 = 7392.
    # This is better than 5280.
    
    # Can we do s=6 with pruning?
    # Total 29568.
    # Conflict: k=6, d=1.
    # Need independent set in Q_6 with min dist 2.
    # Max size 32 (even parity).
    # Total: 462 * 32 = 14784.
    # This is better than 7392.
    
    # Can we do s=7 with pruning?
    # Total 42240.
    # Conflict: k=7, d=1 (Dist^2=4 < 7).
    # Also k=6, d=1 (Dist^2=6 < 7).
    # So we need min dist 2 on Q_7 (for same support) AND handle k=6 conflicts.
    # Even parity on Q_7 gives min dist 2 on same support.
    # But k=6 conflicts remain.
    # As analyzed, even parity on Q_7 still has k=6, d=1 conflicts.
    # So we need to prune further.
    # Likely reduces significantly.
    
    # s=6 with even parity seems a strong candidate.
    # Count: 14784.
    
    # Let's implement s=6 with even parity.
    
    vectors = []
    for support in itertools.combinations(range(D), 6):
        # Generate all 64 sign patterns
        for mask in range(64):
            # Check parity: number of 1s in mask should be even
            if bin(mask).count('1') % 2 == 0:
                v = [0] * D
                for i, idx in enumerate(support):
                    # 0 -> 1, 1 -> -1
                    if (mask >> i) & 1:
                        v[idx] = -1
                    else:
                        v[idx] = 1
                vectors.append(v)
                
    return np.array(vectors, dtype=np.int64)
