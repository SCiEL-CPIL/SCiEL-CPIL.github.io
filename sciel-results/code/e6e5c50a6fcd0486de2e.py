"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Strategy:
    # 1. Generate a structured candidate pool based on known kissing number constructions.
    #    The kissing number in dimension 11 is known to be at least 504 (from Leech lattice related constructions) 
    #    and at most 592. 
    #    Actually, the best known lower bound for d=11 is 504? No, wait.
    #    d=8: 240, d=9: 366, d=10: 504? No.
    #    Let's check standard values.
    #    d=1: 2, d=2: 6, d=3: 12, d=4: 24, d=5: 40, d=6: 72, d=7: 126, d=8: 240.
    #    d=9: 366 (proved?), d=10: 504 (proved?), d=11: 504? 
    #    Actually, for d=11, the best known lower bound is 504? Or is it higher?
    #    According to tables, k(11) is between 504 and 592.
    #    Wait, the Leech lattice is in d=24.
    #    A common construction for high dimensions uses vectors with small integer entries.
    #    
    #    The current program uses weight-2 and weight-4 vectors.
    #    Norm^2 for weight-2 is 2. Norm^2 for weight-4 is 4.
    #    If we normalize to norm 2, the radius is 1.
    #    Distance squared between two vectors u, v must be >= 4 (if norm is 2, distance >= 2, dist^2 >= 4).
    #    Wait, the task says: "minimum pairwise distance is at least the maximum vector norm".
    #    If we normalize vectors to norm 2, max norm is 2.
    #    So min distance >= 2.
    #    dist(u,v)^2 = ||u||^2 + ||v||^2 - 2 u.v = 4 + 4 - 2 u.v = 8 - 2 u.v.
    #    We need 8 - 2 u.v >= 4  =>  2 u.v <= 4  =>  u.v <= 2.
    #    
    #    So the condition for normalized vectors (norm 2) is dot product <= 2.
    #    For integer vectors, we can just keep them as integers and check:
    #    ||u - v||^2 >= max(||u||^2, ||v||^2)
    #    
    #    Let's stick to integer vectors.
    #    To maximize the count, we want vectors that are "close" to each other in the sense of small dot products relative to their norms.
    #    Vectors with smaller norms are "denser" in the sphere?
    #    If I use only weight-2 vectors (norm^2=2):
    #    Condition: ||u-v||^2 >= 2.
    #    ||u-v||^2 = 2 + 2 - 2 u.v = 4 - 2 u.v.
    #    4 - 2 u.v >= 2 => 2 u.v <= 2 => u.v <= 1.
    #    Integer vectors of weight 2 have dot products that are integers.
    #    Possible dot products for weight 2 vectors:
    #    -2, -1, 0, 1, 2.
    #    If they share 2 coords with same signs: dot=2. Conflict (2 > 1).
    #    If they share 2 coords with opposite signs: dot=-2. OK.
    #    If they share 1 coord: dot=1 or -1. OK (1 <= 1).
    #    If they share 0 coords: dot=0. OK.
    #    
    #    So for weight-2 vectors, the ONLY conflict is when they share both coordinates with the SAME signs.
    #    i.e., u and v are identical, or u = -v?
    #    If u=(1,1,0..), v=(1,1,0..). u.v=2. Conflict.
    #    If u=(1,1,0..), v=(1,-1,0..). u.v=0. OK.
    #    If u=(1,1,0..), v=(1,0,1..). u.v=1. OK.
    #    
    #    So, a set of weight-2 vectors is valid if no two vectors are identical.
    #    Wait, what about u=(1,1,0..) and v=(-1,-1,0..)? u.v=-2. OK.
    #    What about u=(1,1,0..) and v=(1,1,0..)? u.v=2. Conflict.
    #    So we just can't have duplicates.
    #    How many weight-2 vectors are there?
    #    C(11, 2) * 4 = 55 * 4 = 220.
    #    So we can pick all 220 weight-2 vectors?
    #    Let's check if any pair conflicts.
    #    Conflict only if u.v > 1.
    #    Max u.v for distinct weight-2 vectors:
    #    If they share 2 coords, they are either identical (u.v=2) or opposite on one coord (u.v=0) or opposite on both (u.v=-2).
    #    Wait, if they share 2 coords, say indices i,j.
    #    u[i]=1, u[j]=1.
    #    v[i]=1, v[j]=1 => u.v=2 (Identical).
    #    v[i]=1, v[j]=-1 => u.v=0.
    #    v[i]=-1, v[j]=1 => u.v=0.
    #    v[i]=-1, v[j]=-1 => u.v=-2.
    #    So yes, distinct weight-2 vectors never have dot product > 1.
    #    Thus, ALL 220 weight-2 vectors form a valid kissing configuration!
    #    
    #    Can we add more?
    #    We can add weight-4 vectors?
    #    Condition for weight-4 (norm^2=4) and weight-2 (norm^2=2):
    #    ||u-v||^2 >= 4 (max norm is 4? No, max norm of the SET. If we mix, max norm is 4).
    #    Wait, the condition is min dist >= max norm.
    #    If we have a weight-4 vector (norm 2) and a weight-2 vector (norm sqrt(2) ~ 1.41),
    #    Max norm is 2.
    #    So min dist must be >= 2.
    #    dist^2 = 4 + 2 - 2 u.v = 6 - 2 u.v.
    #    6 - 2 u.v >= 4 => 2 u.v <= 2 => u.v <= 1.
    #    
    #    So a weight-4 vector u is compatible with a weight-2 vector v if u.v <= 1.
    #    
    #    Let's start with the 220 weight-2 vectors.
    #    Can we add weight-4 vectors?
    #    A weight-4 vector has 4 non-zero entries.
    #    It will have dot products with many weight-2 vectors.
    #    If a weight-4 vector u has a weight-2 vector v in its support, and signs align, u.v could be 1 or 2.
    #    If u.v = 2, conflict.
    #    If u.v = 1, ok.
    #    If u.v = 0, ok.
    #    
    #    Since we want to maximize k, and we already have 220, can we get more?
    #    The known kissing number for d=11 is likely higher than 220.
    #    Actually, 220 is quite small compared to 504.
    #    This implies we should use vectors of HIGHER weight?
    #    Or maybe the "norm 2" normalization is key.
    #    
    #    Let's look at the "Leech" style construction.
    #    Vectors with entries 0, 1, -1 are not enough.
    #    We need to allow entries like 2?
    #    If we allow entries 2, norm^2 increases.
    #    Let's stick to the current program's candidate pool (weight 2 and 4) but use a better algorithm.
    #    The current program got 1.89.
    #    1.89 * 220 (approx seed?) No, seed is not 220.
    #    The score is k.
    #    If the score is 1.89, and seed is 1.0, then k is 1.89 * k_seed.
    #    We don't know k_seed.
    #    But typically these tasks have a seed of a simple greedy or random set.
    #    
    #    Let's assume the goal is to find the largest possible k.
    #    The set of all 220 weight-2 vectors is valid.
    #    Can we add ANY weight-4 vector to this set?
    #    If we add a weight-4 vector u, it must satisfy u.v <= 1 for ALL 220 weight-2 vectors v.
    #    Is there such a u?
    #    If u has a 1 at index i, and a 1 at index j.
    #    There is a weight-2 vector v with 1 at i and 1 at j.
    #    Then u.v >= 1. If u has any other overlapping 1s, it might be >1.
    #    But specifically, if u contains the pair (i,j) with positive signs, it conflicts with v=(1,1,0..) at i,j?
    #    u.v = 1*1 + 1*1 + ... = 2 + ...
    #    If u has 1 at i and 1 at j, and v has 1 at i and 1 at j, u.v >= 2.
    #    Since we need u.v <= 1, u CANNOT have two 1s that match a weight-2 vector in the set.
    #    But the set contains ALL weight-2 vectors, including all sign combinations.
    #    So for any pair of indices (i,j), there is a v with v[i]=u[i] and v[j]=u[j].
    #    Then u.v >= u[i]^2 + u[j]^2 = 1+1=2.
    #    So u.v >= 2 for some v.
    #    Thus, NO weight-4 vector can be added to the full set of 220 weight-2 vectors.
    #    
    #    So, if we take all 220 weight-2 vectors, we are stuck at 220.
    #    Can we get a larger set by dropping some weight-2 vectors to add more weight-4 vectors?
    #    This is the Independent Set problem again.
    #    
    #    However, 220 is a solid baseline.
    #    The current program scores 1.89.
    #    If the seed was, say, 100, then 1.89 means k=189.
    #    If the seed was 200, k=378.
    #    
    #    Let's try to construct a set larger than 220.
    #    Maybe use vectors of weight 3?
    #    Weight 3: norm^2 = 3.
    #    Condition for two weight-3 vectors:
    #    3+3 - 2 u.v >= 3 => 6 - 2 u.v >= 3 => 2 u.v <= 3 => u.v <= 1.5.
    #    Since u.v is integer, u.v <= 1.
    #    Conflict if u.v >= 2.
    #    
    #    How many weight-3 vectors?
    #    C(11,3) * 8 = 165 * 8 = 1320.
    #    This is a much larger pool.
    #    If we can find a large independent set here, we might beat 220.
    #    
    #    Let's generate Weight-2, Weight-3, and Weight-4 vectors.
    #    And use a Greedy + Local Search approach.
    
    candidates = []
    # Weight 2
    for comb in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(comb, signs):
                v[idx] = s
            candidates.append(v)
            
    # Weight 3
    for comb in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(comb, signs):
                v[idx] = s
            candidates.append(v)
            
    # Weight 4
    for comb in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(comb, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates)
    n_cand = len(candidates)
    
    # Compute norms squared
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    # Compute Gram matrix
    G = candidates @ candidates.T
    
    # Conflict matrix
    # Condition: G[i,j] <= (norms_sq[i] + norms_sq[j] - max(norms_sq[i], norms_sq[j])) / 2
    # Which is G[i,j] <= min(norms_sq[i], norms_sq[j]) / 2
    
    min_norms_sq = np.minimum(norms_sq[:, None], norms_sq[None, :])
    limits = min_norms_sq / 2.0
    
    conflict_mat = G > limits
    
    # Greedy Independent Set
    # We want to maximize the count.
    # Heuristic: Min-degree in conflict graph.
    
    static_degrees = np.sum(conflict_mat, axis=1)
    order = np.argsort(static_degrees)
    
    avail_mask = np.ones(n_cand, dtype=bool)
    selected = []
    
    for v in order:
        if not avail_mask[v]:
            continue
        selected.append(v)
        avail_mask[conflict_mat[v]] = False
        
    current_count = len(selected)
    best_selected = selected.copy()
    best_count = current_count
    
    # Local Search: Try to swap or add.
    # Since we have a valid set, we can try to add vectors from the remaining pool.
    # Or swap one out for two in? (Hard to implement efficiently).
    # Let's try random restarts of the greedy with perturbed degrees.
    
    rng = np.random.default_rng(42)
    
    # Time check
    if time.time() > deadline - 20:
        final_vecs = candidates[best_selected]
        return final_vecs
        
    for _ in range(50):
        if time.time() > deadline - 10:
            break
            
        # Perturb degrees
        perturbed = static_degrees + rng.random(n_cand) * 10
        order = np.argsort(perturbed)
        
        avail_mask = np.ones(n_cand, dtype=bool)
        curr_selected = []
        
        for v in order:
            if not avail_mask[v]:
                continue
            curr_selected.append(v)
            avail_mask[conflict_mat[v]] = False
            
        if len(curr_selected) > best_count:
            best_count = len(curr_selected)
            best_selected = curr_selected.copy()
            
    # Final check
    if not best_selected:
        return np.array([np.ones(D, dtype=np.int64)], dtype=np.int64)
        
    final_vecs = candidates[best_selected]
    return final_vecs

# EVOLVE-BLOCK-END
