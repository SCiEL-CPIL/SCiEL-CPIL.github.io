import numpy as np
import time

def construct():
    n = 256
    start_time = time.time()
    # Safety margin: Evaluator kills at 90s. We stop at 85s to be safe.
    deadline = start_time + 85.0
    
    # 1. Precompute all arithmetic progressions
    # An AP is defined by start a (0..n-1) and step s (1..n-1).
    # Indices: a, a+s, ..., a+(k-1)s < n
    progs = []
    for a in range(n):
        for s in range(1, n):
            # Generate indices for this AP
            idx = np.arange(a, n, s)
            if len(idx) > 0:
                progs.append(idx)
    
    Np = len(progs)
    
    # 2. Build incidence matrix M (Np x n)
    # M[p, i] = 1 if index i is in progression p, else 0
    # This allows vectorized computation of sums: sums = M @ signs
    M = np.zeros((Np, n), dtype=np.int8)
    for p, idx in enumerate(progs):
        M[p, idx] = 1
    
    # 3. Objective function
    # We want to minimize max(|M @ signs|)
    # Since signs are +/- 1, sums are integers.
    
    def get_max_abs_sum(signs):
        # signs is 1D array of shape (n,) with values +/- 1
        # M is (Np, n)
        # sums is (Np,)
        sums = M.dot(signs)
        return int(np.max(np.abs(sums)))
    
    # 4. Simulated Annealing
    # State: signs (n,)
    # Move: Flip a random subset of positions (block flip)
    #   - Single flip: flip 1 random index
    #   - Block flip: flip k random indices (k=2,3,4)
    # Acceptance: Metropolis criterion
    
    rng = np.random.RandomState(42)
    
    # Initialize with random signs
    signs = rng.choice([-1, 1], size=n)
    current_sums = M.dot(signs)
    current_cost = get_max_abs_sum(signs)
    
    best_signs = signs.copy()
    best_cost = current_cost
    
    # Annealing parameters
    # Start with high temperature to escape local minima
    # The cost function is discrete. Max possible cost is n (if all +1 in a length n AP, but max AP length is < 256).
    # Typical random cost is around sqrt(n * log(n)) or similar.
    # Let's estimate a reasonable T range.
    # If cost is ~10, T should start around 5-10 to allow moves that increase cost slightly.
    
    T_start = 10.0
    T_end = 0.01
    # Number of iterations is determined by time budget.
    # We will run as many as possible.
    
    # To speed up, we can precompute the effect of flipping a set of indices.
    # If we flip indices in set S, the change in sums for progression p is:
    # delta_sums[p] = -2 * (sum of signs[i] for i in S and i in p)
    # This is still a bit expensive if S is large.
    # For single flip i: delta = -2 * signs[i] * M[:, i]
    # For block flip S: delta = -2 * (M[:, S] * signs[S]).T
    
    # Let's implement a hybrid: mostly single flips, some block flips.
    # Single flip is O(Np) for sum update if we maintain sums array.
    # M.dot(signs) is O(Np * n) which is ~256 * 256 * 256 ~ 16M ops. 
    # Wait, Np is sum over a,s of len(arange(a,n,s)).
    # For s=1, len is n-a. Sum ~ n^2/2 = 32k.
    # For s=2, ~ n^2/4 = 16k.
    # Total Np ~ n^2 * (1 + 1/2 + 1/3 + ...) ~ n^2 * log(n) ~ 65k * 8 ~ 500k.
    # So M is 500k x 256.
    # M.dot(signs) is 500k * 256 ~ 128M multiplications.
    # NumPy can do ~100-500M ops/sec. So one evaluation is ~1-2ms.
    # In 85 seconds, we can do ~40k-80k evaluations.
    # This is enough for a decent SA run.
    
    # Optimization: Maintain current_sums explicitly to avoid full dot product for single flips.
    # current_sums = M @ signs
    # Flip i: signs[i] *= -1. 
    # new_sums = current_sums - 2 * signs[i] * M[:, i]
    # This is O(Np) vector op, much faster than O(Np*n).
    
    # For block flips of size k, we can update in O(k*Np) or just recompute if k is small?
    # Actually, for k=2, update is O(2*Np).
    
    # Let's stick to single flips for simplicity and speed, but use SA to escape.
    # The previous local search failed because it only accepted improvements.
    # SA accepts worse solutions with probability exp(-delta/T).
    
    # Precompute column norms or just use M directly.
    # M is int8. signs are int8.
    
    # To make it even faster, we can use the fact that we only need max(|sums|).
    # But we need the sums to update them.
    
    # Let's try to do more sophisticated moves: 
    # Sometimes flipping a "bad" position (one contributing to the max sum APs) is good.
    # But random SA is robust.
    
    # Add a "kick" mechanism: if no improvement for X iterations, do a random restart or large perturbation.
    
    no_improve_count = 0
    max_no_improve = 5000
    
    # We will iterate until deadline
    while time.time() < deadline:
        # Choose move
        # 90% single flip, 10% block flip of size 2-4
        if rng.random() < 0.9:
            # Single flip
            i = rng.randint(n)
            # Update sums
            # M[:, i] is a view, but we need to multiply by signs[i]
            # delta = -2 * signs[i] * M[:, i]
            # Since signs[i] is +/- 1, delta is +/- 2 * M[:, i]
            # We can avoid the multiplication by signs[i] by just flipping the sign of the contribution
            # current_sums += -2 * signs[i] * M[:, i]
            
            # Optimization: M[:, i] is mostly 0. 
            # But vectorized op is fast.
            current_sums -= 2 * signs[i] * M[:, i]
            signs[i] *= -1
            
            # Calculate cost
            # We need max(|current_sums|)
            # This is O(Np).
            new_cost = int(np.max(np.abs(current_sums)))
            
            # Acceptance
            delta = new_cost - current_cost
            if delta < 0 or rng.random() < np.exp(-delta / T_start):
                # Accept
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_signs = signs.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # Reject: revert
                signs[i] *= -1
                current_sums += 2 * signs[i] * M[:, i]
                no_improve_count += 1
        else:
            # Block flip of size 2
            i1, i2 = rng.choice(n, size=2, replace=False)
            # Update sums for i1 and i2
            # This is slightly more complex to revert if rejected.
            # Let's just do single flips for now to keep code simple and fast.
            # Block flips are harder to revert efficiently without storing old sums.
            pass
            
        # Cool down
        # Exponential cooling
        # We don't know total iterations, so use a fixed rate per iteration?
        # Or cool based on time?
        # Let's use a fixed number of iterations estimate.
        # If we do ~50k iterations, T goes from 10 to 0.01.
        # T = T_start * (T_end/T_start)^(iter/total)
        # Since we don't know total, let's just decay T by a factor.
        T_start *= 0.99995 # Slow cooling
        if T_start < T_end:
            T_start = T_end
        
        # Kick: If stuck, randomize a few bits or restart
        if no_improve_count > max_no_improve:
            # Random perturbation: flip 10 random bits
            for _ in range(10):
                i = rng.randint(n)
                current_sums -= 2 * signs[i] * M[:, i]
                signs[i] *= -1
            current_cost = int(np.max(np.abs(current_sums)))
            no_improve_count = 0
            # Maybe increase T slightly?
            T_start = max(T_start, 1.0)

    return [int(x) for x in best_signs]

# Test/Run
if __name__ == "__main__":
    signs = construct()
    print(f"Result length: {len(signs)}")
    # Quick check
    # Re-evaluate to be sure
    n = 256
    progs = []
    for a in range(n):
        for s in range(1, n):
            idx = np.arange(a, n, s)
            if len(idx) > 0:
                progs.append(idx)
    M = np.zeros((len(progs), n), dtype=np.int8)
    for p, idx in enumerate(progs):
        M[p, idx] = 1
    sums = M.dot(signs)
    print(f"Max abs sum: {np.max(np.abs(sums))}")
