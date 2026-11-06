import time
import random
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin to stay well under 90s kill

    N = 256

    # 1. Precompute ALL APs (length >= 1) and reverse mapping
    ap_indices = []
    aps_containing = [[] for _ in range(N)]

    # Iterate over all valid (start, step) pairs
    for d in range(1, N + 1):
        for s in range(1, N + 1):
            # Generate the base sequence of indices for this (s, d)
            base_indices = []
            curr = s - 1
            while curr < N:
                base_indices.append(curr)
                curr += d

            # Every prefix of base_indices is a valid AP
            for L in range(1, len(base_indices) + 1):
                idx_list = base_indices[:L]
                ap_idx = len(ap_indices)
                ap_indices.append(idx_list)
                for idx in idx_list:
                    aps_containing[idx].append(ap_idx)

    num_aps = len(ap_indices)

    # 2. Helper functions

    def thue_morse(N):
        """Generate Thue-Morse sequence as +/-1 signs."""
        seq = np.zeros(N, dtype=np.int8)
        for i in range(N):
            # Parity of number of 1-bits in binary representation of i
            # Use bit_count if available (Python 3.10+), else bin().count()
            try:
                parity = i.bit_count() & 1
            except AttributeError:
                parity = bin(i).count('1') & 1
            seq[i] = 1 if parity == 0 else -1
        return seq

    def compute_sums(signs):
        """Compute sums for all APs given a signs array. Uses vectorized approach where possible."""
        sums = np.zeros(num_aps, dtype=np.int16)
        for r, idx_list in enumerate(ap_indices):
            sums[r] = np.sum(signs[idx_list])
        return sums

    def greedy_optimize(s_init, time_budget, rng, no_improve_limit=3):
        """
        Greedy Coordinate Descent with occasional random restarts/perturbations.
        Returns best signs and best score.
        """
        s = s_init.copy()
        sums = compute_sums(s)
        current_max = int(np.max(np.abs(sums)))
        best_s = s.copy()
        best_score = current_max

        indices_list = list(range(N))

        # Track consecutive non-improving sweeps to detect local minima
        no_improve_sweeps = 0

        while time.time() - start_time < time_budget:
            rng.shuffle(indices_list)
            improved_this_sweep = False

            for i in indices_list:
                if time.time() - start_time > time_budget:
                    break

                old_val = s[i]
                delta_val = -2 * old_val
                affected = aps_containing[i]

                # Save old sums for affected APs
                old_sums = sums[affected].copy()

                # Apply flip
                sums[affected] += delta_val
                s[i] = -old_val

                new_max = int(np.max(np.abs(sums)))

                if new_max < current_max:
                    current_max = new_max
                    improved_this_sweep = True
                    if current_max < best_score:
                        best_score = current_max
                        best_s = s.copy()
                elif new_max == current_max:
                    # Accept ties with some probability to explore
                    if rng.random() < 0.05:
                        improved_this_sweep = True
                    else:
                        # Revert
                        sums[affected] = old_sums
                        s[i] = old_val
                else:
                    # Revert
                    sums[affected] = old_sums
                    s[i] = old_val

            if not improved_this_sweep:
                no_improve_sweeps += 1
                if no_improve_sweeps >= no_improve_limit:
                    break
            else:
                no_improve_sweeps = 0

        return best_s, best_score

    def continuous_optimize(s_init, time_budget, rng):
        """
        Optimize a continuous representation of the signs using SGD on a smooth surrogate loss.
        Returns a discretized signs array.
        """
        # Initialize continuous vector with noise around the initial signs
        x = s_init.astype(np.float64) + rng.uniform(-0.5, 0.5, N)
        
        # Precompute AP indices as numpy arrays for vectorized operations
        # We need a matrix A where A[r, i] = 1 if i is in AP r, else 0
        # But this matrix is sparse and potentially large.
        # Instead, we can use the list of indices.
        
        # Let's use a sparse-like approach or just loop over APs for gradient.
        # Given num_aps is ~100k, looping is slow.
        # Let's try to build a sparse matrix or use a more efficient gradient calculation.
        
        # Actually, for N=256, we can afford a dense matrix if we are careful, 
        # but 100k * 256 floats is ~100MB, which might be too much or slow to allocate.
        # Let's stick to the list approach but optimize the inner loop.
        
        # Alternative: Use the fact that sum for AP r is sum(x[i] for i in ap_indices[r]).
        # Gradient for x[i] is sum over all APs containing i of (sum_r * sign(sum_r) / len(affected_aps?))
        # Wait, the loss is sum_r (sum_r)^2.
        # d/dx_i sum_r (sum_r)^2 = sum_r 2 * sum_r * (d sum_r / d x_i)
        # d sum_r / d x_i is 1 if i is in AP r, else 0.
        # So grad_i = 2 * sum_{r in aps_containing[i]} sum_r
        
        # We can maintain an array 'sums' and update it as x changes.
        # But x changes continuously.
        
        # Let's just do a simple SGD step.
        # Loss = sum(sums^2)
        # We need to compute sums for current x.
        
        lr = 0.1
        best_x = x.copy()
        best_loss = float('inf')
        best_s = s_init.copy()
        
        # Precompute aps_containing as numpy arrays for faster iteration?
        # aps_containing is already a list of lists.
        
        for step in range(500):  # Fixed number of steps to control time
            if time.time() - start_time > time_budget:
                break
                
            # Compute current sums
            sums = np.zeros(num_aps, dtype=np.float64)
            for r, idx_list in enumerate(ap_indices):
                sums[r] = np.dot(x, np.array(idx_list)) # This is slow, np.dot with array creation
                # Better: sums[r] = x[idx_list].sum()
            
            # This loop is O(num_aps * avg_len). 
            # num_aps is ~100k. avg_len is small.
            # This might be too slow for 500 steps.
            # Let's estimate: 100k * 2 ops = 200k ops per step. 500 steps = 100M ops. 
            # In Python, this is too slow.
            
            # We need a faster way to compute sums.
            # We can precompute a sparse matrix A.
            # Let's try to build it once.
            
            break # Abort this function as it's likely too slow to implement correctly in time budget
            
        return best_s

    # 3. Main loop: multiple restarts with different initializations
    rng_global = random.Random(42)
    best_global_s = None
    best_global_score = float('inf')

    # Budget allocation:
    # - First run: Thue-Morse init (strong heuristic)
    # - Second run: Random init
    # - Remaining runs: Random inits, each with decreasing time budget

    # Run 1: Thue-Morse
    tm = thue_morse(N)
    t1 = time.time()
    s1, score1 = greedy_optimize(tm, time_limit * 0.4, rng_global)
    if score1 < best_global_score:
        best_global_score = score1
        best_global_s = s1.copy()

    # Run 2: Random init
    if time.time() - start_time < time_limit * 0.7:
        rand_init = np.array([rng_global.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
        s2, score2 = greedy_optimize(rand_init, time_limit * 0.3, rng_global)
        if score2 < best_global_score:
            best_global_score = score2
            best_global_s = s2.copy()

    # Remaining runs: more random inits with shorter budgets
    remaining_budget = time_limit - (time.time() - start_time)
    if remaining_budget > 2.0:
        # Do a few more short runs
        num_extra = min(5, int(remaining_budget / 3.0))
        for k in range(num_extra):
            if time.time() - start_time > time_limit - 1.0:
                break
            rand_init = np.array([rng_global.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
            s_k, score_k = greedy_optimize(rand_init, min(3.0, time_limit - (time.time() - start_time) - 1.0), rng_global)
            if score_k < best_global_score:
                best_global_score = score_k
                best_global_s = s_k.copy()

    # Final safety: if we haven't found anything (shouldn't happen), return random
    if best_global_s is None:
        best_global_s = np.array([rng_global.choice([-1, 1]) for _ in range(N)], dtype=np.int8)

    return [int(x) for x in best_global_s]
