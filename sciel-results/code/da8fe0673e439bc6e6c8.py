import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1  # 129
    MAX_VAL = 128

    def calc_reps(A):
        """Compute representation counts for sums N_LOW..N_HIGH using unordered pairs."""
        reps = np.zeros(RANGE_SIZE, dtype=np.int32)
        A_list = sorted(A)
        n = len(A_list)
        for i in range(n):
            ai = A_list[i]
            sums = ai + np.array(A_list[i:], dtype=np.int32)
            mask = (sums >= N_LOW) & (sums <= N_HIGH)
            if np.any(mask):
                indices = sums[mask] - N_LOW
                np.add.at(reps, indices, 1)
        return reps

    def is_covered(reps):
        return np.all(reps > 0)

    def get_peak(reps):
        return int(np.max(reps))

    def delta_reps_remove(A, x, current_reps):
        """Compute the change in reps when removing x from A. Returns new_reps."""
        new_reps = current_reps.copy()
        for y in A:
            if y == x:
                continue
            s = x + y
            if N_LOW <= s <= N_HIGH:
                new_reps[s - N_LOW] -= 1
        s = 2 * x
        if N_LOW <= s <= N_HIGH:
            new_reps[s - N_LOW] -= 1
        return new_reps

    def delta_reps_add(A, x, current_reps):
        """Compute the change in reps when adding x to A (x not in A). Returns new_reps."""
        new_reps = current_reps.copy()
        for y in A:
            s = x + y
            if N_LOW <= s <= N_HIGH:
                new_reps[s - N_LOW] += 1
        s = 2 * x
        if N_LOW <= s <= N_HIGH:
            new_reps[s - N_LOW] += 1
        return new_reps

    # --- Phase 1: Construct a structured initial candidate ---
    best_A = None
    best_peak = float('inf')

    # Start with a reasonable base: try to build from {0, 128}
    A_current = {0, 128}
    reps = calc_reps(A_current)

    uncovered = set(range(N_LOW, N_HIGH + 1))
    for s in range(N_LOW, N_HIGH + 1):
        if reps[s - N_LOW] > 0:
            uncovered.discard(s)

    # Greedily add elements using fast delta updates
    while len(uncovered) > 0 and len(A_current) < 80 and time.time() < deadline - 30:
        best_x = None
        best_new_covered = 0
        best_reps = None
        best_peak_val = float('inf')

        candidates = list(range(0, MAX_VAL + 1))
        random.shuffle(candidates)

        for x in candidates:
            if x in A_current:
                continue
            new_reps = delta_reps_add(A_current, x, reps)

            if not is_covered(new_reps):
                new_covered_count = np.sum((reps == 0) & (new_reps > 0))
                peak = get_peak(new_reps)

                if new_covered_count > best_new_covered or (new_covered_count == best_new_covered and peak < best_peak_val):
                    best_new_covered = new_covered_count
                    best_x = x
                    best_reps = new_reps
                    best_peak_val = peak
            else:
                peak = get_peak(new_reps)
                if peak < best_peak_val or best_reps is None:
                    best_x = x
                    best_reps = new_reps
                    best_peak_val = peak
                    best_new_covered = 1000

        if best_x is None:
            break

        A_current.add(best_x)
        reps = best_reps if best_reps is not None else calc_reps(A_current)

        for s in range(N_LOW, N_HIGH + 1):
            if reps[s - N_LOW] > 0:
                uncovered.discard(s)

        if len(uncovered) == 0:
            break

    if is_covered(reps):
        peak = get_peak(reps)
        if peak < best_peak:
            best_peak = peak
            best_A = list(A_current)
    else:
        # Fallback
        A_current = set(range(64, 129))
        reps = calc_reps(A_current)
        if is_covered(reps):
            best_peak = get_peak(reps)
            best_A = list(A_current)

    # --- Phase 2: Try to improve by removing elements ---
    if best_A is not None:
        current_A = set(best_A)
        current_reps = calc_reps(current_A)
        current_peak = get_peak(current_reps)

        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            for x in list(current_A):
                if len(current_A) <= 1:
                    break
                test_reps = delta_reps_remove(current_A, x, current_reps)
                if is_covered(test_reps):
                    test_peak = get_peak(test_reps)
                    if test_peak <= current_peak:
                        current_A = current_A - {x}
                        current_reps = test_reps
                        current_peak = test_peak
                        improved = True
                        break

        if is_covered(current_reps):
            peak = get_peak(current_reps)
            if peak < best_peak:
                best_peak = peak
                best_A = list(current_A)

    # --- Phase 3: Spectral-guided local search using DFT/Parseval ---
    # The key insight: sum of squares of representation multiplicities can be
    # computed via Parseval's identity using the DFT of the indicator vector.
    # For a set A with indicator vector a, the representation count for sum s
    # is (a * a)[s] (autocorrelation). By Parseval, sum_s (a*a)[s]^2 can be
    # related to the DFT of a. We use this to guide swaps towards more uniform
    # distribution of representations.

    if best_A is not None and time.time() < deadline - 10:
        current_A = set(best_A)
        current_reps = calc_reps(current_A)
        current_peak = get_peak(current_reps)

        # Precompute DFT of indicator vector for fast spectral energy computation
        # The indicator vector has length 129 (indices 0..128)
        # For spectral energy, we need the circular autocorrelation which relates
        # to |DFT(a)|^2
        
        # Actually, let's use a simpler but effective approach:
        # Track the sum of squares of reps as a secondary objective
        # and use it to guide swaps when peak is tied.

        current_sq_sum = float(np.sum(current_reps.astype(np.float64) ** 2))
        
        T = 1.0
        cooling_rate = 0.999
        min_T = 0.05

        # Precompute for faster evaluation: for each x in [0,128], precompute
        # which sums it affects
        sum_effects = {}
        for x in range(MAX_VAL + 1):
            affected = []
            for y in range(MAX_VAL + 1):
                s = x + y
                if N_LOW <= s <= N_HIGH:
                    affected.append(s - N_LOW)
            sum_effects[x] = np.array(affected, dtype=np.int32)

        while time.time() < deadline - 5 and T > min_T:
            if len(current_A) == 0:
                break

            item_to_remove = random.choice(list(current_A))
            new_item = random.randint(0, 128)

            if new_item == item_to_remove:
                T *= cooling_rate
                continue

            # Fast delta: remove item_to_remove, then add new_item
            intermediate_reps = delta_reps_remove(current_A, item_to_remove, current_reps)

            if new_item in current_A and new_item != item_to_remove:
                T *= cooling_rate
                continue

            new_reps = delta_reps_add(current_A - {item_to_remove}, new_item, intermediate_reps)
            new_A = (current_A - {item_to_remove}) | {new_item}

            if len(new_A) == 0:
                T *= cooling_rate
                continue

            if is_covered(new_reps):
                new_peak = get_peak(new_reps)
                new_sq_sum = float(np.sum(new_reps.astype(np.float64) ** 2))
                delta = new_peak - current_peak

                if delta < 0:
                    current_A = new_A
                    current_reps = new_reps
                    current_peak = new_peak
                    current_sq_sum = new_sq_sum
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_A = list(current_A)
                elif delta == 0:
                    # Same peak, prefer lower sum of squares (more uniform)
                    if new_sq_sum < current_sq_sum:
                        current_A = new_A
                        current_reps = new_reps
                        current_peak = new_peak
                        current_sq_sum = new_sq_sum
                    else:
                        prob = np.exp(-(new_sq_sum - current_sq_sum) / (100.0 * max(T, 0.001)))
                        if random.random() < prob:
                            current_A = new_A
                            current_reps = new_reps
                            current_peak = new_peak
                            current_sq_sum = new_sq_sum
                else:
                    prob = np.exp(-delta / max(T, 0.001))
                    if random.random() < prob:
                        current_A = new_A
                        current_reps = new_reps
                        current_peak = new_peak
                        current_sq_sum = new_sq_sum
            else:
                pass

            T *= cooling_rate

        # Final removal pass
        if time.time() < deadline - 5:
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for x in list(current_A):
                    if len(current_A) <= 1:
                        break
                    test_reps = delta_reps_remove(current_A, x, current_reps)
                    if is_covered(test_reps):
                        test_peak = get_peak(test_reps)
                        if test_peak <= current_peak:
                            current_A = current_A - {x}
                            current_reps = test_reps
                            current_peak = test_peak
                            improved = True
                            break

            if is_covered(current_reps):
                peak = get_peak(current_reps)
                if peak < best_peak:
                    best_peak = peak
                    best_A = list(current_A)

    # --- Phase 4: Additional restart with different initial structure ---
    if time.time() < deadline - 20:
        # Try starting from a different structured set
        # Use a set that's more spread out initially
        for start_seed in range(3):
            if time.time() >= deadline - 15:
                break
            
            random.seed(start_seed * 1000 + 42)
            
            # Different initial structures
            if start_seed == 0:
                A_init = set(range(0, 129, 4))  # Every 4th number
            elif start_seed == 1:
                A_init = set(range(32, 97, 2))  # Even numbers in middle range
            else:
                A_init = set(range(0, 129, 3))  # Every 3rd number
            
            A_curr = set(A_init)
            r_curr = calc_reps(A_curr)
            
            if not is_covered(r_curr):
                # Need to add elements to cover
                uncov = set(range(N_LOW, N_HIGH + 1))
                for s in range(N_LOW, N_HIGH + 1):
                    if r_curr[s - N_LOW] > 0:
                        uncov.discard(s)
                
                while len(uncov) > 0 and len(A_curr) < 70 and time.time() < deadline - 15:
                    best_x = None
                    best_new = 0
                    best_r = None
                    best_p = float('inf')
                    
                    cands = list(range(0, MAX_VAL + 1))
                    random.shuffle(cands)
                    
                    for x in cands:
                        if x in A_curr:
                            continue
                        nr = delta_reps_add(A_curr, x, r_curr)
                        if not is_covered(nr):
                            nc = np.sum((r_curr == 0) & (nr > 0))
                            p = get_peak(nr)
                            if nc > best_new or (nc == best_new and p < best_p):
                                best_new = nc
                                best_x = x
                                best_r = nr
                                best_p = p
                        else:
                            p = get_peak(nr)
                            if p < best_p or best_r is None:
                                best_x = x
                                best_r = nr
                                best_p = p
                                best_new = 1000
                    
                    if best_x is None:
                        break
                    
                    A_curr.add(best_x)
                    r_curr = best_r if best_r is not None else calc_reps(A_curr)
                    
                    for s in range(N_LOW, N_HIGH + 1):
                        if r_curr[s - N_LOW] > 0:
                            uncov.discard(s)
                    
                    if len(uncov) == 0:
                        break
            
            if is_covered(r_curr):
                # Try removal
                p_curr = get_peak(r_curr)
                imp = True
                while imp and time.time() < deadline - 10:
                    imp = False
                    for x in list(A_curr):
                        if len(A_curr) <= 1:
                            break
                        tr = delta_reps_remove(A_curr, x, r_curr)
                        if is_covered(tr):
                            tp = get_peak(tr)
                            if tp <= p_curr:
                                A_curr = A_curr - {x}
                                r_curr = tr
                                p_curr = tp
                                imp = True
                                break
                
                if is_covered(r_curr):
                    pk = get_peak(r_curr)
                    if pk < best_peak:
                        best_peak = pk
                        best_A = list(A_curr)

    if best_A is None:
        return list(range(64, 129))

    return best_A
