# EVOLVE-BLOCK-START
import numpy as np
import time

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    deadline = time.time() + 620  # 10s safety margin

    # Precompute projection indices
    # P is indexed as P[y, x].
    # For a given (y, x):
    # val_{-1} = x - y
    # val_0    = x
    # val_1    = x + y
    # val_2    = x + 2y
    
    # We want to maximize D / M where:
    # D = H(x - y)
    # M = max(H(x), H(x+y), H(x+2y))
    
    # To maximize D, we want x-y to be as uniform as possible.
    # To minimize M, we want at least one of H(x), H(x+y), H(x+2y) to be small.
    # Since H(x) is the marginal of X, and H(x+y) is the sum, etc.
    # Note that for uniform distribution:
    # x-y takes values -9..9 (19 values). Probabilities are triangular: 1,2,...,10,...,1.
    # H(x) is log2(10).
    # H(x+y) is similar to x-y but shifted.
    # H(x+2y) has more values.
    
    # Let's try to construct a distribution that concentrates mass on a line where x-y is constant?
    # If x-y = c, then D = 0. Bad.
    # We want x-y to have high entropy.
    # But we want x, x+y, x+2y to have low entropy.
    
    # Consider a distribution supported on a set of points.
    # If we pick a permutation matrix (one 1 per row/col), then H(x)=log2(10), H(y)=log2(10).
    # x-y depends on the permutation.
    # If the permutation is identity (x=y), x-y=0 always. D=0.
    # If the permutation is reverse (x=9-y), x-y = 9-2y. Values: 9,7,5,3,1,-1,-3,-5,-7,-9. All distinct.
    # So H(x-y) = log2(10).
    # H(x) = log2(10).
    # H(x+y) = H(9) = 0 (since x+y=9 always).
    # H(x+2y) = H(9-y+2y) = H(9+y). Since y is uniform 0..9, x+2y is uniform 9..18. H=log2(10).
    # So M = max(log2(10), 0, log2(10)) = log2(10).
    # Ratio = 1.0. This is the same as uniform! (Uniform also gives 1.0? Let's check).
    
    # Uniform:
    # x uniform 0..9. H(x) = log2(10).
    # x-y: values -9..9. Prob proportional to 10-|k|.
    # H(x-y) = sum p_k log2(1/p_k).
    # x+y: values 0..18. Prob proportional to triangular.
    # x+2y: values 0..27.
    
    # Let's calculate uniform ratio.
    # H(x) = 3.3219
    # H(x-y):
    # probs: 1,2,3,4,5,6,7,8,9,10,9,8,7,6,5,4,3,2,1 / 100
    # H = - sum p log p
    # H(x-y) is approx 4.1? No, max entropy for 19 bins is log2(19)=4.25.
    # Triangular distribution entropy is lower than uniform.
    # Let's compute H(x-y) for uniform 10x10.
    # H(x+y) is the same as H(x-y) by symmetry.
    # H(x+2y) has 28 bins.
    
    # If ratio is D/M, and for permutation reverse, D=log2(10), M=log2(10), ratio=1.
    # For uniform, D=H(x-y), M=max(H(x), H(x+y), H(x+2y)).
    # H(x) = 3.32.
    # H(x+y) = H(x-y) (symmetry).
    # H(x+2y) is likely higher than H(x) because more bins?
    # If H(x+2y) > H(x), then M = H(x+2y).
    # If H(x-y) < H(x+2y), ratio < 1.
    
    # Let's check if we can get ratio > 1.
    # We need D > M.
    # D = H(x-y).
    # M >= H(x).
    # So we need H(x-y) > H(x).
    # This is possible if x is concentrated but x-y is spread out.
    # Example: Fix x=0. Then x-y = -y. H(x-y) = H(y). H(x)=0.
    # But M = max(0, H(y), H(2y)).
    # If y is uniform, H(y)=log2(10). H(2y)=log2(10) (0,2,..18 distinct).
    # M = log2(10). D = log2(10). Ratio = 1.
    
    # What if we make x-y uniform?
    # Let u = x-y. We want u uniform.
    # And we want x, x+y, x+2y to have low entropy.
    # x+y = 2x - u.
    # x+2y = x + 2(x-u) = 3x - 2u.
    
    # Consider a distribution where x is constant?
    # If x=c, then u = c-y. H(u) = H(y).
    # x+y = c+y. H(c+y) = H(y).
    # x+2y = c+2y. H(c+2y) = H(2y).
    # If y is uniform, H(y)=log2(10), H(2y)=log2(10).
    # D=log2(10), M=log2(10). Ratio 1.
    
    # What if y is not uniform?
    # If y is such that 2y has lower entropy than y?
    # H(2y) >= H(y) is not always true?
    # If y takes values 0,1, 2y takes 0,2. Entropies same.
    # If y takes 0,1,2, 2y takes 0,2,4. Entropies same.
    # Collisions in 2y reduce entropy?
    # If y takes 0,1, 2y takes 0,2. No collision.
    # If y takes 0,1, 2y is injective on integers.
    # So H(2y) = H(y) always for integer y? No, if range wraps or something? No, just values.
    # Since mapping y -> 2y is injective for y in 0..9, H(2y)=H(y).
    
    # So if x is constant, Ratio = 1.
    
    # What if we vary x?
    # We want H(x-y) large.
    # We want H(x), H(x+y), H(x+2y) small.
    # H(x) small means x is concentrated.
    # H(x+y) small means x+y is concentrated.
    # This implies y is concentrated (since x is).
    # If x and y are both concentrated, x-y is concentrated.
    # So H(x-y) is small.
    # Contradiction?
    
    # Maybe M is dominated by H(x+2y) in the uniform case, and H(x-y) is smaller?
    # Let's calculate uniform values precisely.
    # H(x) = 3.321928
    # H(x-y):
    # p = [1,2,3,4,5,6,7,8,9,10,9,8,7,6,5,4,3,2,1]/100
    # H = - sum p log2 p
    # H(x-y) = 4.074?
    # H(x+y) = 4.074
    # H(x+2y):
    # Values 0..27.
    # Counts:
    # 0:1, 1:1, 2:2, 3:2, 4:3, 5:3, 6:4, 7:4, 8:5, 9:5, 10:6, 11:6, 12:7, 13:7, 14:8, 15:8, 16:9, 17:9, 18:10, 19:9, 20:8, 21:7, 22:6, 23:5, 24:4, 25:3, 26:2, 27:1
    # Sum = 100.
    # H(x+2y) is likely > H(x-y).
    # If M = H(x+2y), and D = H(x-y), and H(x-y) < H(x+2y), then Ratio < 1.
    
    # So uniform gives < 1?
    # Wait, the seed program (uniform) scored 1.0000.
    # This implies D/M = 1 for uniform? Or the score is normalized differently?
    # "The search score is value/seed".
    # If seed is uniform, and uniform gives value V_seed.
    # If I output uniform, score = V_seed / V_seed = 1.
    # So I need to find a distribution with value > V_seed.
    
    # Let's check if we can get D/M > 1.
    # We need H(x-y) > max(H(x), H(x+y), H(x+2y)).
    # Since H(x+2y) is usually the largest, we need H(x-y) > H(x+2y).
    # Is it possible?
    # x-y has 19 possible values.
    # x+2y has 28 possible values.
    # Max entropy for x-y is log2(19) = 4.25.
    # Max entropy for x+2y is log2(28) = 4.81.
    # So H(x-y) <= 4.25.
    # If we can make H(x+2y) close to 4.25 or less, and H(x-y) close to 4.25, we might win.
    # But H(x+2y) can be made small by making x+2y concentrated.
    
    # Strategy:
    # Make x-y uniform (max entropy).
    # Make x+2y concentrated (min entropy).
    # Make x and x+y concentrated.
    
    # Let u = x-y. We want u uniform on some set.
    # Let v = x+2y. We want v concentrated.
    # x = (u + 2v + ?)/?
    # u = x-y => y = x-u
    # v = x+2(x-u) = 3x-2u => x = (v+2u)/3
    # y = (v-u)/3
    # For x,y to be integers, v+2u and v-u must be divisible by 3.
    # v+2u ≡ 0 mod 3 => v ≡ u mod 3 (since 2 ≡ -1, v-u ≡ 0 => v≡u)
    # v-u ≡ 0 mod 3 => v ≡ u mod 3.
    # So we need v ≡ u mod 3.
    
    # We can choose pairs (u, v) such that v ≡ u mod 3.
    # And x,y in 0..9.
    # x = (v+2u)/3, y = (v-u)/3.
    # Constraints:
    # 0 <= v+2u <= 27
    # 0 <= v-u <= 27
    # 0 <= u <= 9? No, u = x-y in -9..9.
    # v = x+2y in 0..27.
    
    # Let's pick a specific u, say u=0.
    # Then v must be divisible by 3. v in {0,3,6,...,27}.
    # x = v/3, y = v/3.
    # Points: (0,0), (1,1), ..., (9,9).
    # This is the diagonal.
    # u=0 always. H(u)=0. Bad.
    
    # We want u to vary.
    # Let's pick u uniformly from a set U.
    # And v uniformly from a set V?
    # If we pick a set of points (x,y) such that u takes many values and v takes few.
    
    # Consider the set of points where v=0.
    # x+2y=0 => x=0, y=0. Only (0,0).
    # v=1: x+2y=1 => (1,0).
    # v=2: (2,0).
    # v=3: (0,1), (3,0).
    # v=4: (2,1), (4,0).
    # v=5: (1,2), (5,0).
    # v=6: (0,3), (3,1), (6,0).
    # ...
    
    # If we pick all points with small v, then v is concentrated.
    # But u = x-y.
    # For v=0, u=0.
    # v=1, u=1.
    # v=2, u=2.
    # v=3, u=-1 (0,1) or 3 (3,0).
    # If we pick (0,1), u=-1.
    # If we pick (3,0), u=3.
    
    # Let's try to construct a distribution supported on a "band" where v is small.
    # But we need u to have high entropy.
    # If we take all points with v <= K, the number of points grows with K.
    # The values of u will be spread out.
    
    # Let's try a specific construction:
    # Pick points such that x+2y = c (constant).
    # Then H(x+2y) = 0.
    # H(x) and H(x+y) depend on the line.
    # Line x+2y=c.
    # y can range from 0 to min(9, c/2).
    # x = c-2y.
    # If we pick a line with many points, say c=18.
    # y=0, x=18 (invalid, x>9).
    # y=4.5? No integers.
    # c=18:
    # y=4, x=10 (invalid).
    # y=5, x=8. (8,5). u=3.
    # y=6, x=6. (6,6). u=0.
    # y=7, x=4. (4,7). u=-3.
    # y=8, x=2. (2,8). u=-6.
    # y=9, x=0. (0,9). u=-9.
    # Points: (8,5), (6,6), (4,7), (2,8), (0,9).
    # u values: 3, 0, -3, -6, -9.
    # If uniform on these 5 points:
    # H(u) = log2(5) = 2.32.
    # H(x): x values 8,6,4,2,0. H(x)=log2(5)=2.32.
    # H(x+y): 13,12,11,10,9. H=2.32.
    # H(x+2y) = 0.
    # M = max(2.32, 2.32, 0) = 2.32.
    # D = 2.32.
    # Ratio = 1.
    
    # It seems hard to beat 1 with a single line.
    # What if we use a 2D structure?
    
    # Let's try to optimize numerically.
    # Start with uniform.
    # Use gradient ascent on log P?
    # Or just random restarts + local search.
    
    # Given time constraints, let's implement a simple local search.
    # 1. Start with uniform.
    # 2. Compute ratio.
    # 3. Try moving probability mass from one cell to another.
    # 4. Accept if ratio improves.
    # 5. Repeat.
    
    # To speed up, we can precompute the changes in entropy.
    # But entropy is non-linear.
    
    # Let's try a different initialization:
    # Concentrate mass on cells where x-y is "central" but x+2y is "central"?
    # No, we want x-y spread, x+2y concentrated.
    
    # Let's try the "reverse diagonal" again: x+y=9.
    # Points: (0,9), (1,8), ..., (9,0).
    # u = x-y: -9, -7, -5, -3, -1, 1, 3, 5, 7, 9.
    # H(u) = log2(10).
    # x: 0..9. H(x)=log2(10).
    # x+y: 9. H=0.
    # x+2y: 9-y+2y = 9+y. 9..18. H=log2(10).
    # M = log2(10).
    # Ratio = 1.
    
    # What if we mix reverse diagonal with something else?
    # If we add weight to (0,0), u=0, x=0, x+y=0, x+2y=0.
    # This adds a spike to u at 0, reducing H(u).
    # Adds spike to x at 0, reducing H(x).
    # Adds spike to x+y at 0, reducing H(x+y).
    # Adds spike to x+2y at 0, reducing H(x+2y).
    # Since H(x+2y) was log2(10) (max), reducing it helps M.
    # But reducing H(u) hurts D.
    # Which effect is stronger?
    
    # Let's simulate this mentally.
    # P = (1-eps) * Uniform(RD) + eps * Delta(0,0).
    # H(u) decreases.
    # H(x+2y) decreases.
    # If H(x+2y) was the max, M decreases.
    # D decreases.
    # If D and M decrease by similar amounts, ratio might stay 1.
    # But if M decreases faster, ratio increases.
    
    # Let's try this numerically.
    # We will implement a local search that starts with the reverse diagonal distribution.
    
    P = np.zeros((10, 10), dtype=np.float64)
    for i in range(10):
        P[i, 9-i] = 1.0/10.0
        
    def compute_ratio(P):
        # P[y, x]
        # u = x - y
        # v0 = x
        # v1 = x + y
        # v2 = x + 2y
        
        # Flatten
        p = P.flatten()
        # Indices
        ys, xs = np.mgrid[0:10, 0:10]
        u = xs - ys
        v0 = xs
        v1 = xs + ys
        v2 = xs + 2*ys
        
        # Compute entropies
        def entropy(vals, p):
            # vals: 1D array of projection values
            # p: 1D array of probabilities
            # Sum p for each unique val
            uniq, idx = np.unique(vals, return_inverse=True)
            sums = np.zeros(len(uniq))
            np.add.at(sums, idx, p)
            # Filter zeros
            sums = sums[sums > 1e-12]
            if len(sums) == 0:
                return 0.0
            return -np.sum(sums * np.log2(sums))
        
        H_u = entropy(u.flatten(), p)
        H_v0 = entropy(v0.flatten(), p)
        H_v1 = entropy(v1.flatten(), p)
        H_v2 = entropy(v2.flatten(), p)
        
        M = max(H_v0, H_v1, H_v2)
        if M < 1e-9:
            return 0.0
        return H_u / M
    
    best_P = P.copy()
    best_ratio = compute_ratio(best_P)
    
    # Local search
    # Try moving mass from i to j
    # To make it fast, we only consider a few steps or use a simpler heuristic.
    # Given 600s limit, we can do many iterations.
    
    # Let's try adding mass to (0,0) and (9,9) etc.
    # Actually, let's just do a simple random restart local search.
    
    start_time = time.time()
    iters = 0
    while time.time() - start_time < 600:
        iters += 1
        # Pick a random cell to reduce, random cell to increase
        i = np.random.randint(100)
        j = np.random.randint(100)
        if i == j:
            continue
        
        p_i = P.flatten()[i]
        if p_i < 1e-6:
            continue
            
        # Try moving a small fraction
        delta = p_i * 0.1
        # Or try moving all?
        # Let's try moving all first, then small amounts.
        
        P_new = P.copy()
        P_new.flatten()[i] -= delta
        P_new.flatten()[j] += delta
        
        # Check validity
        if np.any(P_new < -1e-9):
            continue
            
        P_new = np.clip(P_new, 0, None)
        s = P_new.sum()
        if s < 1e-9:
            continue
        P_new /= s
        
        r_new = compute_ratio(P_new)
        
        if r_new > best_ratio:
            best_ratio = r_new
            best_P = P_new.copy()
            P = P_new
        else:
            # Revert
            pass
            
    # Final check and return
    P_out = np.clip(best_P, 0, None)
    s = P_out.sum()
    if s > 1e-9:
        P_out /= s
    else:
        P_out = np.ones((10,10))/100.0
        
    return P_out

# EVOLVE-BLOCK-END
