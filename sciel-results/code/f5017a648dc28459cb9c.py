import numpy as np
import itertools
import time
import random

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Strategy:
    # 1. Generate a structured pool of candidate vectors.
    #    The optimal kissing number in d=11 is known to be 224 (Leech lattice).
    #    The Leech lattice Lambda can be constructed from E8 x A2 x A1.
    #    Minimal vectors in Lambda have norm squared 4.
    #    There are 196,560 such vectors. We will generate a subset of these
    #    or a similar structured set that fits in R^11.
    #    
    #    However, constructing the full Leech lattice in Python within 150s 
    #    might be tight if done naively. A simpler structured approach:
    #    Use the known fact that the kissing configuration in d=11 is related 
    #    to the Leech lattice. 
    #    
    #    Alternative: Use the pool of all weight-4 vectors (5280 vectors) 
    #    and apply a highly optimized Minimum Degree Greedy (MDG) algorithm 
    #    for Maximum Independent Set (MIS).
    #    The MDG heuristic is known to perform significantly better than 
    #    random greedy for dense conflict graphs.
    #    
    #    Let's combine:
    #    1. Pool: All weight-4 vectors (norm^2 = 4).
    #       Why weight 4? 
    #       If we use weight 4 vectors, R^2 = 4.
    #       Condition: ||v_i - v_j||^2 >= 4.
    #       ||v_i - v_j||^2 = 4 + 4 - 2 v_i.v_j = 8 - 2 v_i.v_j.
    #       We need 8 - 2 v_i.v_j >= 4 => 2 v_i.v_j <= 4 => v_i.v_j <= 2.
    #       For weight 4 vectors with entries +/-1, 0, the dot product is an integer.
    #       Possible dot products: -4, -2, 0, 2, 4.
    #       Conflict if dot product > 2. i.e., dot product = 4.
    #       Dot product 4 means v_i = v_j (since norms are equal).
    #       Wait, if v_i != v_j, can dot product be 4?
    #       v_i = (1,1,1,1,0...), v_j = (1,1,1,1,0...). Same vector.
    #       If they are distinct weight 4 vectors, max dot product is 2 (sharing 3 indices with same sign, 1 different? No.)
    #       Let's check: v1 has support S1, v2 has support S2. |S1|=|S2|=4.
    #       dot = sum_{i in S1 cap S2} s1_i s2_i.
    #       If S1 = S2, dot = 4 if signs match, -4 if opposite, etc.
       #       If S1 != S2, |S1 cap S2| <= 3. Max dot = 3.
       #       So distinct weight-4 vectors always have dot <= 3.
       #       Condition dot <= 2.
       #       So we conflict if dot = 3 or 4.
       #       Dot = 4 only if identical.
       #       Dot = 3 if they share 3 indices with matching signs.
       #       
       #    So the conflict graph for distinct weight-4 vectors has an edge if they share 3 indices with the same sign.
       #    We want the largest set of weight-4 vectors such that no two share 3 indices with the same sign.
       #    
       #    This is a specific combinatorial problem. 
       #    The number of weight-4 vectors is 5280.
       #    We can build the adjacency matrix or use a greedy MIS on the fly.
       #    
    #    Let's try to generate a larger pool including weight 3 and 5 to see if mixing helps?
    #    If we mix weights, R is determined by the max norm.
    #    If we include weight 5 (norm^2=5), R^2=5.
    #    Then weight 4 vectors (norm^2=4) must be distance >= sqrt(5) from each other and from weight 5 vectors.
    #    dist^2(v4, v4) = 8 - 2 dot >= 5 => 2 dot <= 3 => dot <= 1.
    #    dist^2(v5, v5) = 10 - 2 dot >= 5 => 2 dot <= 5 => dot <= 2.
    #    dist^2(v4, v5) = 4 + 5 - 2 dot >= 5 => 9 - 2 dot >= 5 => 2 dot <= 4 => dot <= 2.
    #    
    #    Mixing constraints are complex. The pure weight-4 case is cleaner.
    #    Known kissing number for d=11 is 224.
    #    Can we find 224 weight-4 vectors?
    #    Total weight-4 vectors: 5280.
    #    Each vector conflicts with others sharing 3 indices.
    #    Number of neighbors for a given v:
    #    Choose 3 of 4 indices (4 ways).
    #    Fix signs on these 3.
    #    Choose 1 new index from remaining 7 (7 ways).
    #    Fix sign on new index (2 ways).
    #    Total neighbors = 4 * 7 * 2 = 56.
    #    Plus itself? No, distinct.
    #    So degree ~ 56.
    #    Graph is sparse. Greedy MIS should work well.
    #    Expected size of MIS ~ N / (d+1) = 5280 / 57 ~ 92.
    #    This is far from 224.
    #    
    #    Why? Because the optimal configuration uses vectors that are NOT all weight 4?
    #    Or is my conflict analysis wrong?
    #    Leech lattice minimal vectors have norm 4.
    #    In d=24, there are 196560.
    #    In d=11, we are looking for a kissing configuration.
    #    The bound is 224.
    #    Maybe the vectors in the optimal config have different norms?
    #    If we allow norms 1, 2, 3, 4...
    #    If we use weight 1 vectors (norm 1), R=1.
    #    dist >= 1.
    #    v=(1,0...), w=(0,1...). dist^2=2 >= 1. OK.
    #    We can pack many weight 1 vectors.
    #    Max weight 1 vectors: 2D = 22.
    #    Score 22.
    #    
    #    If we use weight 2 (norm 2), R=2.
    #    dist >= 2.
    #    v=(1,1,0...), w=(1,0,1,0...). dist^2 = 1+1+1+1 - 2(1) = 4 >= 4. OK.
    #    v=(1,1,0...), w=(-1,1,0...). dist^2 = 1+1+1+1 - 2(-1) = 6 >= 4. OK.
    #    v=(1,1,0...), w=(1,1,0...). Same, dist 0. Fail.
    #    v=(1,1,0...), w=(0,0,1,1). dist^2 = 1+1+1+1 - 0 = 4 >= 4. OK.
    #    So weight 2 vectors can be packed.
    #    How many?
    #    Total weight 2: C(11,2)*4 = 220.
    #    Conflict if dist < 2.
    #    dist^2 = 4 + 4 - 2 dot = 8 - 2 dot < 4 => 2 dot > 4 => dot > 2.
    #    Max dot for distinct weight 2 is 2 (if they share 1 index with same sign).
    #    If they share 1 index with same sign, dot=1. 8-2=6 >= 4. OK.
    #    If they share 0 indices, dot=0. 8 >= 4. OK.
    #    If they are identical, dot=2. 8-4=4 >= 4. OK?
    #    Wait, if v=w, dist=0. 0 < 2. Fail.
    #    So distinct weight 2 vectors never conflict?
    #    Let's re-verify.
    #    v=(1,1,0...), w=(1,1,0...). Identical. Dist 0.
    #    v=(1,1,0...), w=(-1,1,0...). Dot = -1+1=0. Dist^2=8. OK.
    #    v=(1,1,0...), w=(1,0,1,0...). Dot = 1. Dist^2=6. OK.
    #    v=(1,1,0...), w=(0,1,1,0...). Dot = 1. Dist^2=6. OK.
    #    v=(1,1,0...), w=(1,1,0...). Identical.
    #    
    #    It seems ANY set of DISTINCT weight-2 vectors is a valid kissing configuration with R=2?
    #    If so, k = 220.
    #    Score 220.
    #    Current best is 1.168 * seed. What is seed?
    #    If seed is ~100, score 1.16 is ~116.
    #    If I can get 220, that's a big jump.
    #    
    #    Let's verify the condition for weight 2 vectors again.
    #    Set S of distinct weight-2 vectors.
    #    Max norm R = 2.
    #    Min dist between distinct v, w in S.
    #    ||v-w||^2 = ||v||^2 + ||w||^2 - 2 v.w = 4 + 4 - 2 v.w = 8 - 2 v.w.
    #    We need ||v-w|| >= 2 => ||v-w||^2 >= 4.
    #    8 - 2 v.w >= 4 => 2 v.w <= 4 => v.w <= 2.
    #    For distinct weight-2 vectors, what is the max dot product?
    #    v has two 1s (or -1s). w has two 1s (or -1s).
    #    Case 1: v and w share 2 indices.
    #       If signs match: v=(1,1,0..), w=(1,1,0..). Identical. Excluded.
    #       If signs differ: v=(1,1,0..), w=(-1,1,0..). Dot = -1+1=0.
    #       v=(1,1,0..), w=(1,-1,0..). Dot = 1-1=0.
    #       v=(1,1,0..), w=(-1,-1,0..). Dot = -1-1=-2.
    #    Case 2: v and w share 1 index.
    #       v=(1,1,0..), w=(1,1,0..)? No, share 1 index means supports intersect in 1.
    #       v=(1,1,0..), w=(1,0,1,0..). Dot = 1*1 + 1*0 + 0*1 = 1.
    #       v=(1,1,0..), w=(-1,0,1,0..). Dot = -1.
    #       Max dot = 1.
    #    Case 3: v and w share 0 indices.
    #       Dot = 0.
    #    
    #    So for DISTINCT weight-2 vectors, max dot product is 1.
    #    1 <= 2 is always true.
    #    Therefore, ALL 220 distinct weight-2 vectors form a valid kissing configuration!
    #    
    #    Wait, is it that simple?
    #    "minimum pairwise distance is at least the maximum vector norm"
    #    Max norm is 2.
    #    Min dist is sqrt(6) (approx 2.45) for vectors sharing 1 index?
    #    Let's check min dist.
    #    We want MIN dist.
    #    Dist^2 = 8 - 2 dot.
    #    To minimize dist, we maximize dot.
    #    Max dot for distinct is 1.
    #    Min dist^2 = 8 - 2(1) = 6.
    #    Min dist = sqrt(6) ≈ 2.449.
    #    Max norm = 2.
    #    2.449 >= 2. True.
    #    
    #    So the answer is simply the set of all 220 weight-2 vectors?
    #    Why did the previous program score 1.16?
    #    Maybe the seed is high? Or the previous program failed to find this?
    #    The previous program used weights 2-6 and a complex SA.
    #    If I just return all weight-2 vectors, k=220.
    #    
    #    Can we do better?
    #    What if we add weight-1 vectors?
    #    If we add weight-1 vectors, max norm becomes 2 (since weight-2 are present).
    #    Weight-1 vectors have norm 1.
    #    Dist between weight-1 (v) and weight-2 (w):
    #    ||v||=1, ||w||=2.
    #    ||v-w||^2 = 1 + 4 - 2 v.w = 5 - 2 v.w.
    #    Need >= 4 (since R=2, R^2=4).
    #    5 - 2 v.w >= 4 => 2 v.w <= 1 => v.w <= 0.5.
    #    Since v, w are integer vectors, v.w is integer.
    #    So v.w <= 0.
    #    
    #    Let v be a weight-1 vector, e.g., (1,0,0,0,0,0,0,0,0,0,0).
    #    Let w be a weight-2 vector.
    #    v.w is the entry of w at index 0.
    #    w has two non-zero entries.
    #    If w has a non-zero entry at index 0, v.w is 1 or -1.
    #    We need v.w <= 0.
    #    So w must NOT have a positive entry at index 0?
    #    If w[0] = 1, v.w = 1 > 0. Conflict.
    #    If w[0] = -1, v.w = -1 <= 0. OK.
    #    If w[0] = 0, v.w = 0 <= 0. OK.
    #    
    #    So for a fixed v=e_i, we can include any weight-2 vector w such that w[i] <= 0.
    #    Can we include ALL weight-1 vectors?
    #    If we include v=e_i and v'=-e_i, etc.
    #    Let's check if we can add ANY weight-1 vector to the full set of weight-2 vectors.
    #    No, because for any weight-1 vector v=e_i, there exist weight-2 vectors with w[i]=1 (e.g., e_i + e_j).
    #    These would conflict.
    #    
    #    So we cannot have all weight-2s AND any weight-1.
    #    But maybe we can swap some weight-2s for weight-1s to get a larger set?
    #    Unlikely, weight-2s are "denser" in terms of count vs constraints?
    #    Actually, weight-1s have fewer constraints between themselves (any distinct weight-1s have dist^2=2, R=1, 2>=1 OK).
    #    But they constrain the weight-2s heavily.
    #    
    #    What if we use ONLY weight-1 vectors?
    #    k = 22.
    #    
    #    What if we use ONLY weight-3 vectors?
    #    Norm^2 = 3. R = sqrt(3) ≈ 1.732.
    #    Dist^2 >= 3.
    #    v, w weight 3.
    #    ||v-w||^2 = 3+3-2 dot = 6 - 2 dot >= 3 => 2 dot <= 3 => dot <= 1.
    #    Max dot for distinct weight-3s?
    #    If they share 2 indices with same sign, dot=2. Conflict.
    #    So we need a set of weight-3 vectors where no two share 2 indices with the same sign.
    #    This is a packing problem.
    #    Total weight-3s: 1320.
    #    Degree?
    #    Neighbors share 2 indices.
    #    Choose 2 of 3 (3 ways). Fix signs.
    #    Choose 1 new index (8 ways). Fix sign (2 ways).
    #    3 * 8 * 2 = 48 neighbors.
    #    Greedy MIS ~ 1320 / 49 ~ 26.
    #    Smaller than 220.
    #    
    #    So the pure weight-2 set (k=220) is very strong.
    #    
    #    Is there any set larger than 220?
    #    The kissing number in d=11 is 224.
    #    So 224 is the theoretical max.
    #    We found 220.
    #    Can we add 4 more vectors?
    #    Maybe some weight-1 or weight-3 vectors that are compatible with the weight-2s?
    #    As shown, any weight-1 vector conflicts with some weight-2s.
    #    To add a weight-1 vector v=e_i, we MUST remove all weight-2 vectors with w[i]=1.
    #    How many weight-2 vectors have w[i]=1?
    #    w[i]=1 means the index i is one of the two non-zero spots, with sign +.
    #    Choose 1 other index from 10 (10 ways).
    #    Sign of other index: 2 ways.
    #    Total 20 vectors conflict with e_i.
    #    So to add e_i, we lose 20 weight-2s and gain 1 weight-1. Net -19. Bad.
    #    
    #    What about weight-3?
    #    To add a weight-3 vector u, we must remove weight-2s that conflict with it.
    #    Conflict if dot(u, w) > 1 (since R^2=3 for weight-3? No, if we mix, R is max norm).
    #    If we mix weight-2 (norm 2) and weight-3 (norm sqrt(3)≈1.73), R=2.
    #    Condition for w (weight 2) and u (weight 3):
    #    ||w-u||^2 >= 4.
    #    4 + 3 - 2 w.u >= 4 => 3 - 2 w.u >= 0 => 2 w.u <= 3 => w.u <= 1.
    #    So we must remove any weight-2 w such that w.u >= 2.
    #    Max w.u?
    #    u has 3 non-zeros. w has 2.
    #    If they share 2 indices with same sign, dot=2.
    #    So for a given u, how many weight-2s have dot >= 2?
    #    u has 3 pairs of indices. For each pair, there are 2 weight-2 vectors matching the signs?
    #    If u has signs (1,1,1) on indices (i,j,k).
    #    Pairs: (i,j), (i,k), (j,k).
    #    For (i,j), w=(1,1,0..) at i,j. Dot=2.
    #    Also w=(1,1,0..) with other signs?
    #    w must have non-zeros at i,j.
    #    Signs must match u to get dot 2.
    #    If u[i]=1, u[j]=1, then w[i]=1, w[j]=1 gives dot 2.
    #    w[i]=1, w[j]=-1 gives dot 0.
    #    So exactly 1 weight-2 vector per pair conflicts?
    #    Total 3 conflicting weight-2s.
    #    So to add 1 weight-3, we remove 3 weight-2s. Net -2. Bad.
    #    
    #    Conclusion: The set of all 220 weight-2 vectors is likely optimal or near-optimal.
    #    Score should be 220.
    #    
    #    Let's double check if I missed any "weight 2" variations.
    #    Weight 2 means exactly two non-zero entries, each +/-1.
    #    Count: C(11,2) * 2^2 = 55 * 4 = 220.
    #    
    #    Implementation:
    #    Generate all 220 vectors.
    #    Return them.
    #    
    #    Why did the previous program not do this?
    #    It tried to mix weights and used SA. It probably got stuck in a local optimum 
    #    or the constraints for mixed sets were too tight, leading to a smaller set.
    #    The "Minimum Degree Greedy" hint was for weight-4, which is a different problem.
    #    
    #    Let's write the code to return the 220 weight-2 vectors.
    
    vectors = []
    for comb in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for idx, sign in zip(comb, signs):
                v[idx] = sign
            vectors.append(v)
            
    return np.array(vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
