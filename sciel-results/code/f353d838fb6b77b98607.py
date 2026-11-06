import time
import random
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 50.0  # Safety margin to stay well under 90s

    N = 256

    # Precompute all APs and build incidence structure
    ap_indices = []
    aps_containing = [[] for _ in range(N)]

    for d in range(1, N + 1):
        for s in range(1, N + 1):
            base_indices = []
            curr = s - 1
            while curr < N:
                base_indices.append(curr)
                curr += d

            for L in range(1, len(base_indices) + 1):
                idx_list = base_indices[:L]
                ap_idx = len(ap_indices)
                ap_indices.append(idx_list)
                for idx in idx_list:
                    aps_containing[idx].append(ap_idx)

    num_aps = len(ap_indices)
    print(f"Number of APs: {num_aps}")

    # Build affected lists as numpy arrays for fast indexing
    affected_lists = [np.array(aps_containing[i], dtype=np.int32) for i in range(N)]

    def compute_sums(signs):
        """Compute sums for all APs given a signs array."""
        sums = np.zeros(num_aps, dtype=np.int16)
        for r, idx_list in enumerate(ap_indices):
            sums[r] = np.sum(signs[idx_list])
        return sums

    def get_max_abs(sums):
        return int(np.max(np.abs(sums)))

    def greedy_optimize(s_init, time_budget, rng):
        """
        Greedy Coordinate Descent with fast vectorized updates.
        """
        s = s_init.copy()
        sums = compute_sums(s)
        current_max = get_max_abs(sums)
        best_s = s.copy()
        best_score = current_max

        indices_list = list(range(N))
        no_improve_sweeps = 0
        max_no_improve = 5
        deadline = start_time + time_budget

        while time.time() < deadline:
            rng.shuffle(indices_list)
            improved_this_sweep = False

            for i in indices_list:
                if time.time() >= deadline:
                    break

                old_val = s[i]
                delta_val = -2 * old_val
                affected = affected_lists[i]

                old_sums = sums[affected].copy()
                new_max_before = current_max

                # Apply flip
                sums[affected] += delta_val
                s[i] = -old_val

                new_max = get_max_abs(sums)

                if new_max < current_max:
                    current_max = new_max
                    improved_this_sweep = True
                    if current_max < best_score:
                        best_score = current_max
                        best_s = s.copy()
                elif new_max == current_max:
                    # Count tight constraints before and after
                    old_tight = np.sum(np.abs(old_sums) == current_max)
                    new_tight = np.sum(np.abs(sums[affected]) == current_max)
                    if new_tight < old_tight:
                        improved_this_sweep = True
                    elif new_tight == old_tight:
                        if rng.random() < 0.01:
                            improved_this_sweep = True
                        else:
                            sums[affected] = old_sums
                            s[i] = old_val
                    else:
                        sums[affected] = old_sums
                        s[i] = old_val
                else:
                    sums[affected] = old_sums
                    s[i] = old_val

            if not improved_this_sweep:
                no_improve_sweeps += 1
                if no_improve_sweeps >= max_no_improve:
                    if rng.random() < 0.5:
                        # Perturb: flip 3-10 random elements
                        num_flips = rng.randint(3, 10)
                        for _ in range(num_flips):
                            j = rng.choice(indices_list)
                            s[j] = -s[j]
                            sums[affected_lists[j]] += -2 * s[j]
                        current_max = get_max_abs(sums)
                        no_improve_sweeps = 0
                    else:
                        break
            else:
                no_improve_sweeps = 0

        return best_s, best_score

    def sat_check(D, time_budget=5.0):
        """
        Check if there exists a sign assignment with max AP sum <= D.
        Uses local search to minimize violations.
        Returns (success, signs) or (False, None).
        """
        deadline = time.time() + time_budget
        rng = random.Random(hash(D) % 2**32)
        s = np.array([rng.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
        
        # Compute initial sums
        sums = compute_sums(s)
        
        def count_violations(sums):
            return int(np.sum(np.abs(sums) > D))
        
        violations = count_violations(sums)
        if violations == 0:
            return True, s.copy()
        
        best_violations = violations
        best_s = s.copy()
        
        # Local search: repeatedly flip a sign that reduces the number of violations
        for iteration in range(200):
            if time.time() > deadline:
                break
            
            # Try flipping each element and see if it reduces violations
            improved = False
            for i in range(N):
                if time.time() > deadline:
                    break
                
                old_val = s[i]
                delta_val = -2 * old_val
                affected = affected_lists[i]
                old_sums = sums[affected].copy()
                
                # Apply flip
                sums[affected] += delta_val
                s[i] = -old_val
                
                new_violations = count_violations(sums)
                
                if new_violations < violations:
                    violations = new_violations
                    improved = True
                    if violations < best_violations:
                        best_violations = violations
                        best_s = s.copy()
                        if violations == 0:
                            return True, best_s
                else:
                    # Revert
                    sums[affected] = old_sums
                    s[i] = old_val
            
            if not improved:
                # No improvement found; try a random perturbation
                if rng.random() < 0.3:
                    num_flips = rng.randint(1, 5)
                    for _ in range(num_flips):
                        j = rng.randint(0, N - 1)
                        s[j] = -s[j]
                        sums[affected_lists[j]] += -2 * s[j]
                    violations = count_violations(sums)
                    if violations < best_violations:
                        best_violations = violations
                        best_s = s.copy()
                else:
                    break
        
        if best_violations == 0:
            return True, best_s
        return False, best_s

    rng_global = random.Random(42)
    best_global_s = None
    best_global_score = float('inf')

    # Run 1: Thue-Morse + greedy
    def thue_morse(N):
        """Generate Thue-Morse sequence as +/-1 signs."""
        seq = np.zeros(N, dtype=np.int8)
        for i in range(N):
            parity = i.bit_count() & 1
            seq[i] = 1 if parity == 0 else -1
        return seq
    
    tm = thue_morse(N)
    s1, score1 = greedy_optimize(tm, time_limit * 0.3, rng_global)
    if score1 < best_global_score:
        best_global_score = score1
        best_global_s = s1.copy()

    # Run 2: Try SAT-based approach for small D values
    for D in [2, 3, 4]:
        if time.time() - start_time > time_limit * 0.6:
            break
        
        success, sat_s = sat_check(D, time_budget=4.0)
        if success:
            sums = compute_sums(sat_s)
            actual_max = get_max_abs(sums)
            if actual_max < best_global_score:
                best_global_score = actual_max
                best_global_s = sat_s.copy()
                # Try to improve further with greedy
                refined_s, refined_score = greedy_optimize(sat_s, 3.0, rng_global)
                if refined_score < best_global_score:
                    best_global_score = refined_score
                    best_global_s = refined_s.copy()
                break

    # Run 3: Random init + greedy
    if time.time() - start_time < time_limit * 0.75:
        rand_init = np.array([rng_global.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
        s2, score2 = greedy_optimize(rand_init, time_limit * 0.15, rng_global)
        if score2 < best_global_score:
            best_global_score = score2
            best_global_s = s2.copy()

    # Remaining time: more greedy restarts
    remaining_budget = time_limit - (time.time() - start_time)
    if remaining_budget > 1.0:
        num_extra = min(5, int(remaining_budget / 2.0))
        for k in range(num_extra):
            if time.time() - start_time > time_limit - 1.0:
                break
            rand_init = np.array([rng_global.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
            budget = min(2.0, time_limit - (time.time() - start_time) - 1.0)
            if budget < 0.5:
                break
            s_k, score_k = greedy_optimize(rand_init, budget, rng_global)
            if score_k < best_global_score:
                best_global_score = score_k
                best_global_s = s_k.copy()

    # Final safety
    if best_global_s is None:
        best_global_s = np.array([rng_global.choice([-1, 1]) for _ in range(N)], dtype=np.int8)

    return [int(x) for x in best_global_s]
