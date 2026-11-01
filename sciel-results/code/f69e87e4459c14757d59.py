"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115

    # 1. Generate Type A vectors (Weight 1, value +/-2, Norm^2=4)
    # All mutually compatible. Compatible with all Type B and C.
    type_a_vectors = []
    for i in range(D):
        for sign in [1, -1]:
            v = np.zeros(D, dtype=np.int64)
            v[i] = 2 * sign
            type_a_vectors.append(v)
    
    type_a_array = np.array(type_a_vectors)
    n_a = len(type_a_array) # 22

    # 2. Generate Type B vectors (Weight 2, values +/-1, Norm^2=2) -> Wait, Norm^2 = 1^2+1^2=2.
    # The task requires "minimum pairwise distance is at least the maximum vector norm".
    # If we mix Norm 2 and Norm 4 vectors, the max norm is 4 (squared).
    # Let's stick to Norm^2 = 4 for all vectors to simplify the distance constraint:
    # Dist^2 >= Max_Norm^2 = 4.
    # Type B (Weight 2, values +/-1) has Norm^2 = 2.
    # If we include Norm^2=2 vectors, the max norm in the set might be 2 (squared) if we only use those?
    # No, we have Type A (Norm^2=4) and Type C (Norm^2=4).
    # So Max Norm^2 is 4.
    # Condition: ||v_i - v_j||^2 >= 4.
    # For v (Norm^2=4) and w (Norm^2=2):
    # ||v-w||^2 = 4 + 2 - 2 v.dot(w) = 6 - 2 v.dot(w).
    # We need 6 - 2 v.dot(w) >= 4 => 2 >= 2 v.dot(w) => v.dot(w) <= 1.
    # Let's check compatibility of Type A (Norm^2=4) and Type B (Norm^2=2).
    # Type A: 2s at i. Type B: t1 at j, t2 at k.
    # If i not in {j,k}, dot=0 <= 1. OK.
    # If i in {j,k}, dot = 2s * t = +/- 2.
    # If dot = 2, condition 2 <= 1 is FALSE. Conflict!
    # So Type A and Type B are NOT fully compatible.
    # Specifically, if Type A is at index i, and Type B has a non-zero at i with same sign, they conflict.
    
    # Let's re-evaluate Type B (Weight 2, Norm^2=2).
    # Maybe we should scale them? No, "integer vectors".
    # Can we use Weight 2 vectors with values +/-1? Norm^2=2.
    # Can we use Weight 4 vectors with values +/-1? Norm^2=4.
    # Can we use Weight 1 vectors with values +/-2? Norm^2=4.
    
    # If we use a mix of Norm^2=2 and Norm^2=4, the max norm is 2 (value 4 squared? No, norm is sqrt(4)=2).
    # Max Norm = 2.
    # Min Dist >= 2.
    # Dist^2 >= 4.
    
    # Let's check Type B (Norm^2=2) vs Type B (Norm^2=2).
    # ||u-v||^2 = 2 + 2 - 2 u.dot(v) = 4 - 2 u.dot(v).
    # Need 4 - 2 u.dot(v) >= 4 => u.dot(v) <= 0.
    # Type B vectors: 2 non-zeros +/-1.
    # u=(1,1,0..), v=(1,1,0..) -> dot=2. Conflict.
    # u=(1,1,0..), v=(1,-1,0..) -> dot=0. OK.
    # u=(1,1,0..), v=(0,1,1..) -> dot=1. Conflict (1 > 0).
    # So Type B vectors have significant conflicts.
    
    # Let's check Type B (Norm^2=2) vs Type C (Norm^2=4).
    # ||u-v||^2 = 2 + 4 - 2 u.dot(v) = 6 - 2 u.dot(v).
    # Need 6 - 2 u.dot(v) >= 4 => u.dot(v) <= 1.
    # u (wt 2), v (wt 4).
    # Intersection size k.
    # If k=0, dot=0. OK.
    # If k=1, dot=+/-1. OK.
    # If k=2, dot = s1*t1 + s2*t2. Values -2, 0, 2.
    # If dot=2, 2 <= 1 is FALSE. Conflict.
    # So Type B and C are compatible UNLESS they share both support positions with same signs.
    
    # This looks complicated. The previous strategy of using ONLY Norm^2=4 vectors was cleaner.
    # But I missed that Type 2 vectors (Weight 2, Norm^2=2) are NOT Norm^2=4.
    # To get Norm^2=4 with Weight 2, we need values +/-? 1^2+1^2=2. 2^2+0=4 (Weight 1).
    # There are no integer vectors of Weight 2 with Norm^2=4. (sqrt(2) is not integer).
    # So my previous "Type 2" analysis was flawed because I assumed Norm^2=4 for Weight 2 vectors, which is impossible for integers.
    # Weight 2 integer vectors have Norm^2 = a^2+b^2.
    # If a,b are integers, min non-zero Norm^2 is 1^2+1^2=2.
    # Next is 1^2+2^2=5.
    # So there are NO integer vectors of Weight 2 with Norm^2=4.
    
    # Therefore, the set of integer vectors with Norm^2=4 is exactly:
    # 1. Weight 1: +/-2. (22 vectors)
    # 2. Weight 4: +/-1. (5280 vectors)
    # 3. Weight 2? No.
    # 4. Weight 3? 1^2+1^2+1^2=3. 1^2+1^2+sqrt(2)^2 No.
    # 5. Weight 5? 1^2*5=5.
    
    # So the current program's candidate set (Type A + Type C) is EXHAUSTIVE for Norm^2=4.
    # Can we use Norm^2=2 vectors?
    # If we include Norm^2=2 vectors, Max Norm becomes 2 (from Norm^2=4 vectors).
    # So Min Dist must be >= 2.
    # This allows us to add Norm^2=2 vectors if they satisfy the distance constraint to ALL other vectors in the set.
    
    # Let's try adding Type B (Weight 2, Norm^2=2) vectors.
    # We need to check compatibility with the selected Type A and Type C vectors.
    # This turns into a different problem: Select a subset of Type A, Type C, and Type B.
    # Given the complexity, and that the current program already scores 2.4,
    # and the candidate set for Norm^2=4 is exhausted,
    # the improvement must come from better optimization of the MIS for Type C,
    # OR adding Type B vectors if they are compatible.
    
    # However, adding Type B vectors requires checking them against the specific Type A/C chosen.
    # Since Type A is small (22), we can check compatibility easily.
    # Type C is large.
    
    # Alternative: Use a different Norm.
    # If we use only Norm^2=2 vectors (Weight 2, values 1,1), Max Norm = sqrt(2) ~ 1.41.
    # Min Dist >= 1.41.
    # Dist^2 >= 2.
    # For two Type B vectors: 4 - 2 u.dot(v) >= 2 => u.dot(v) <= 1.
    # This is less restrictive than u.dot(v) <= 0?
    # Wait, for Norm^2=2 vectors:
    # ||u-v||^2 = 2+2 - 2u.dot(v) = 4 - 2u.dot(v).
    # Max Norm^2 = 2.
    # Need 4 - 2u.dot(v) >= 2 => 2 >= 2u.dot(v) => u.dot(v) <= 1.
    # So we can pack more Type B vectors!
    # Number of Type B vectors: 220.
    # Conflict if dot > 1.
    # Dot > 1 means Dot = 2.
    # Dot=2 only if they are identical.
    # So **all** Type B vectors are mutually compatible if we only use Type B?
    # Wait, if u=v, dot=2. Dist=0.
    # We need distinct vectors.
    # If u != v, can dot be 2?
    # u=(1,1,0..), v=(1,1,0..) is same vector.
    # u=(1,1,0..), v=(1,-1,0..) dot=0.
    # u=(1,1,0..), v=(0,1,1..) dot=1.
    # Max dot for distinct vectors is 1.
    # So **all 220** Type B vectors are mutually compatible!
    
    # Score: 220.
    # Current score: ~52.
    # 220 / 52 = 4.2.
    # This is a huge improvement.
    
    # We can just return all Type B vectors.
    # Let's verify the constraint:
    # "minimum pairwise distance is at least the maximum vector norm"
    # Vectors: all Weight 2, values +/-1.
    # Norm^2 = 2. Max Norm = sqrt(2).
    # Min Dist:
    # For any distinct u, v in this set:
    # u.dot(v) <= 1.
    # ||u-v||^2 = 4 - 2 u.dot(v) >= 4 - 2(1) = 2.
    # ||u-v|| >= sqrt(2).
    # So Min Dist >= Max Norm.
    # Condition satisfied.
    
    # We can also mix in Type A?
    # Type A Norm^2=4. Max Norm becomes 2.
    # Min Dist must be >= 2.
    # Type A vs Type B:
    # ||a-b||^2 = 4 + 2 - 2 a.dot(b) = 6 - 2 a.dot(b).
    # Need 6 - 2 a.dot(b) >= 4 => a.dot(b) <= 1.
    # a=(2s, 0..), b=(t1, t2, 0..).
    # If a's index is in b's support: dot = 2s*t = +/-2.
    # If dot=2, 2 <= 1 is False.
    # So we can only include Type A vectors if they don't conflict with any included Type B.
    # Or we can just drop Type A and use only Type B.
    # 220 is better than 22 + MIS.
    
    # Can we add Type C?
    # Type C Norm^2=4. Max Norm 2.
    # Type B vs Type C:
    # ||b-c||^2 = 2 + 4 - 2 b.dot(c) = 6 - 2 b.dot(c).
    # Need >= 4 => b.dot(c) <= 1.
    # b (wt 2), c (wt 4).
    # If they share 2 indices with same signs, dot=2. Conflict.
    # So we can add Type C vectors that don't conflict with Type B.
    # But adding Type C is hard. 220 is already a big jump.
    
    # Let's return all Type B vectors.
    
    type_b_vectors = []
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = np.zeros(D, dtype=np.int64)
            for pos, s in zip(positions, signs):
                v[pos] = s
            type_b_vectors.append(v)
            
    return np.array(type_b_vectors)

# EVOLVE-BLOCK-END
