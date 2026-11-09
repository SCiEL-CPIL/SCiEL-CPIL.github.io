import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin (evaluator kills at 90s)
    random.seed(42)  # Reproducibility
    
    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1
    MAX_VAL = 128
    SUM_MAX = 256

    # Precompute: for each element x, which target sums it participates in
    # contrib[x] = array of indices into [N_LOW..N_HIGH] where x + y is in range for y in [0,128]
    # We need: for a given x, the set of y values such that x+y is in [N_LOW, N_HIGH]
    # i.e., y in [N_LOW - x, N_HIGH - x] intersect [0, 128]
    
    # Fast rep calculation using indicator vector and convolution
    # Key insight: unordered pair count for sum s:
    #   If s is odd: (ordered_count(s)) / 2
    #   If s is even: (ordered_count(s) - diag(s)) / 2 + diag(s) = ordered_count(s)/2 + diag(s)/2
    # where diag(s) = 1 if s/2 is in A and s is even, else 0
    # ordered_count(s) = sum over a in A of [s-a in A]
    
    def calc_reps_from_indicator(ind):
        """ind: bool array of size 129 (MAX_VAL+1). Returns int32 array of size RANGE_SIZE."""
        # Convolve indicator with itself to get ordered pair counts
        ind_f = ind.astype(np.float64)
        conv = np.convolve(ind_f, ind_f, mode='full')
        # conv[s] = number of ordered pairs (a,b) with a+b=s, a,b in [0,128]
        
        sums_range = np.arange(N_LOW, N_HIGH + 1)
        ordered = conv[sums_range]
        
        # Diagonal: s even and s/2 in A
        is_even = (sums_range % 2 == 0)
        diag = np.zeros(RANGE_SIZE, dtype=np.float64)
        if np.any(is_even):
            half_vals = sums_range[is_even] // 2
            # half_vals are in [64, 128] since sums_range is [128, 256]
            diag_mask = ind[half_vals]
            diag[is_even] = diag_mask.astype(np.float64)
        
        unordered = ordered / 2.0 + diag / 2.0
        return np.round(unordered).astype(np.int32)

    def calc_reps(A_set):
        """Compute reps from a set of elements."""
        ind = np.zeros(MAX_VAL + 1, dtype=bool)
        for x in A_set:
            ind[x] = True
        return calc_reps_from_indicator(ind)

    def is_covered(reps):
        return np.all(reps > 0)

    def get_peak(reps):
        return int(np.max(reps))

    # Delta update: adding element x to A changes reps by +1 for each sum s in [N_LOW, N_HIGH]
    # where s - x is in A (and s - x != x for the unordered adjustment)
    # More precisely:
    # Adding x: for each y in A (y != x), sum s = x+y gets +1 unordered rep
    #            for y = x, sum s = 2x gets +1 unordered rep (the pair (x,x))
    # So: delta_add(x, A) adds +1 to reps[s] for s = x+y where y in A and N_LOW <= s <= N_HIGH
    # But we need to be careful: if x is already in A, we shouldn't double count.
    # We only call delta when x is NOT in A (for add) or x IS in A (for remove).
    
    def delta_add(x, A_set, reps):
        """Return new reps after adding x to A_set (x not in A_set)."""
        new_reps = reps.copy()
        # For each y in A_set, sum s = x + y
        # If s in [N_LOW, N_HIGH], increment new_reps[s - N_LOW] by 1
        y_arr = np.array(sorted(A_set), dtype=np.int32)
        sums = x + y_arr
        mask = (sums >= N_LOW) & (sums <= N_HIGH)
        if np.any(mask):
            idx = (sums[mask] - N_LOW).astype(np.intp)
            new_reps[idx] += 1
        # Also the pair (x, x): sum = 2x
        s2 = 2 * x
        if N_LOW <= s2 <= N_HIGH:
            new_reps[s2 - N_LOW] += 1
        return new_reps

    def delta_remove(x, A_set, reps):
        """Return new reps after removing x from A_set (x in A_set)."""
        new_reps = reps.copy()
        # For each y in A_set, y != x, sum s = x + y loses 1
        # For y = x, sum s = 2x loses 1
        y_arr = np.array(sorted(A_set), dtype=np.int32)
        sums = x + y_arr
        mask = (sums >= N_LOW) & (sums <= N_HIGH)
        if np.any(mask):
            idx = (sums[mask] - N_LOW).astype(np.intp)
            new_reps[idx] -= 1
        # The pair (x,x) is already counted above (when y=x in y_arr)
        return new_reps

    def delta_swap(x_out, x_in, A_set, reps):
        """Return new reps after removing x_out and adding x_in from A_set.
        x_out in A_set, x_in not in A_set."""
        new_reps = reps.copy()
        # Remove x_out: for each y in A_set (including x_out itself), sum x_out+y loses 1
        y_arr = np.array(sorted(A_set), dtype=np.int32)
        sums_out = x_out + y_arr
        mask_out = (sums_out >= N_LOW) & (sums_out <= N_HIGH)
        if np.any(mask_out):
            idx_out = (sums_out[mask_out] - N_LOW).astype(np.intp)
            new_reps[idx_out] -= 1
        
        # Add x_in: for each y in A_set \ {x_out}, sum x_in+y gains 1
        # Plus the pair (x_in, x_in)
        y_arr2 = np.array(sorted(A_set - {x_out}), dtype=np.int32)
        sums_in = x_in + y_arr2
        mask_in = (sums_in >= N_LOW) & (sums_in <= N_HIGH)
        if np.any(mask_in):
            idx_in = (sums_in[mask_in] - N_LOW).astype(np.intp)
            new_reps[idx_in] += 1
        s2_in = 2 * x_in
        if N_LOW <= s2_in <= N_HIGH:
            new_reps[s2_in - N_LOW] += 1
            
        return new_reps

    best_A = None
    best_peak = float('inf')
    
    def update_best(A, reps):
        nonlocal best_A, best_peak
        if is_covered(reps):
            peak = get_peak(reps)
            if peak < best_peak:
                best_peak = peak
                best_A = list(A)

    def random_init():
        """Create a random initial set with good coverage structure."""
        A = set()
        # Ensure endpoints and their partners
        A.add(0)
        A.add(128)
        # Add structured elements for coverage
        for x in [1, 2, 3, 4, 5, 123, 124, 125, 126, 127]:
            A.add(x)
        # Mid-range
        for x in [32, 64, 96]:
            A.add(x)
        # Fill to ~40 elements
        while len(A) < 40:
            A.add(random.randint(0, 128))
        return A

    def repair_coverage(A):
        """Add elements until all sums in [128,256] are covered."""
        A = set(A)
        reps = calc_reps(A)
        tries = 0
        while not is_covered(reps) and tries < 200:
            missing = np.where(reps == 0)[0] + N_LOW
            if len(missing) == 0:
                break
            # For each missing sum, try to find an element that covers it
            best_candidate = None
            best_gain = 0
            for s in missing[:20]:  # Limit to first 20 missing sums
                # Find x such that s-x in [0,128] and x not in A
                x_min = max(0, s - 128)
                x_max = min(128, s)
                candidates = []
                for x in range(x_min, x_max + 1):
                    if x not in A:
                        candidates.append(x)
                random.shuffle(candidates)
                for c in candidates[:10]:
                    # Use delta_add for speed
                    new_reps = delta_add(c, A, reps)
                    new_missing = np.sum(new_reps == 0)
                    gain = len(missing) - new_missing
                    if gain > best_gain:
                        best_gain = gain
                        best_candidate = c
            if best_candidate is not None:
                A.add(best_candidate)
                reps = delta_add(best_candidate, A - {best_candidate}, reps)
            else:
                # Random fallback
                c = random.randint(0, 128)
                if c not in A:
                    A.add(c)
                    reps = delta_add(c, A - {c}, reps)
            tries += 1
        return A, reps

    # Phase 1: Multi-start with fast local search
    num_restarts = 80
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        current_A = random_init()
        current_A, current_reps = repair_coverage(current_A)
        
        if not is_covered(current_reps):
            continue

        update_best(current_A, current_reps)
        
        # Fast local search with delta updates
        T = 2.0
        cooling = 0.995
        min_T = 0.05
        stuck_counter = 0
        max_stuck = 300
        no_improve = 0
        max_no_improve = 500
        
        while time.time() < deadline and T > min_T:
            if stuck_counter > max_stuck or no_improve > max_no_improve:
                break
            
            current_peak = get_peak(current_reps)
            move_type = random.random()
            
            if len(current_A) > 12 and move_type < 0.35:
                # Try Remove
                # Pick element with high peak contribution
                A_list = list(current_A)
                peak_elems = []
                for x in A_list:
                    # Count how many peak sums involve x
                    y_arr = np.array(sorted(current_A), dtype=np.int32)
                    sums = x + y_arr
                    mask = (sums >= N_LOW) & (sums <= N_HIGH)
                    if np.any(mask):
                        idx = (sums[mask] - N_LOW).astype(np.intp)
                        peak_count = int(np.sum(current_reps[idx] == current_peak))
                    else:
                        peak_count = 0
                    peak_elems.append((peak_count, x))
                
                peak_elems.sort(reverse=True)
                # Weighted random selection from top elements
                top_k = min(20, len(peak_elems))
                candidates = [e[1] for e in peak_elems[:top_k]]
                weights = [e[0] + 1 for e in peak_elems[:top_k]]
                item_to_remove = random.choices(candidates, weights=weights, k=1)[0]
                
                new_reps = delta_remove(item_to_remove, current_A, current_reps)
                
                if is_covered(new_reps):
                    new_peak = get_peak(new_reps)
                    if new_peak < current_peak:
                        current_A.discard(item_to_remove)
                        current_reps = new_reps
                        update_best(current_A, current_reps)
                        stuck_counter = 0
                        no_improve = 0
                    elif new_peak == current_peak:
                        if random.random() < 0.3:
                            current_A.discard(item_to_remove)
                            current_reps = new_reps
                            stuck_counter = 0
                            no_improve += 1
                        else:
                            stuck_counter += 1
                            no_improve += 1
                    else:
                        prob = np.exp(-(new_peak - current_peak) / T)
                        if random.random() < prob:
                            current_A.discard(item_to_remove)
                            current_reps = new_reps
                            stuck_counter = 0
                            no_improve += 1
                        else:
                            stuck_counter += 1
                            no_improve += 1
                else:
                    stuck_counter += 1
                    no_improve += 1
                    
            elif move_type < 0.75:
                # Try Swap
                A_list = list(current_A)
                # Pick high-peak element to remove
                peak_elems = []
                for x in A_list:
                    y_arr = np.array(sorted(current_A), dtype=np.int32)
                    sums = x + y_arr
                    mask = (sums >= N_LOW) & (sums <= N_HIGH)
                    if np.any(mask):
                        idx = (sums[mask] - N_LOW).astype(np.intp)
                        peak_count = int(np.sum(current_reps[idx] == current_peak))
                    else:
                        peak_count = 0
                    peak_elems.append((peak_count, x))
                
                peak_elems.sort(reverse=True)
                top_k = min(15, len(peak_elems))
                candidates_rm = [e[1] for e in peak_elems[:top_k]]
                weights_rm = [e[0] + 1 for e in peak_elems[:top_k]]
                item_to_remove = random.choices(candidates_rm, weights=weights_rm, k=1)[0]
                
                new_item = random.randint(0, 128)
                if new_item == item_to_remove or new_item in current_A:
                    stuck_counter += 1
                    no_improve += 1
                    T *= cooling
                    continue
                
                new_reps = delta_swap(item_to_remove, new_item, current_A, current_reps)
                
                if is_covered(new_reps):
                    new_peak = get_peak(new_reps)
                    if new_peak < current_peak:
                        current_A.discard(item_to_remove)
                        current_A.add(new_item)
                        current_reps = new_reps
                        update_best(current_A, current_reps)
                        stuck_counter = 0
                        no_improve = 0
                    else:
                        prob = np.exp(-(new_peak - current_peak) / T)
                        if random.random() < prob:
                            current_A.discard(item_to_remove)
                            current_A.add(new_item)
                            current_reps = new_reps
                            stuck_counter = 0
                            no_improve += 1
                        else:
                            stuck_counter += 1
                            no_improve += 1
                else:
                    stuck_counter += 1
                    no_improve += 1
            else:
                # Targeted swap: replace peak element with one helping low-coverage sums
                A_list = list(current_A)
                peak_elems = []
                for x in A_list:
                    y_arr = np.array(sorted(current_A), dtype=np.int32)
                    sums = x + y_arr
                    mask = (sums >= N_LOW) & (sums <= N_HIGH)
                    if np.any(mask):
                        idx = (sums[mask] - N_LOW).astype(np.intp)
                        peak_count = int(np.sum(current_reps[idx] == current_peak))
                    else:
                        peak_count = 0
                    peak_elems.append((peak_count, x))
                
                peak_elems.sort(reverse=True)
                if not peak_elems or peak_elems[0][0] == 0:
                    stuck_counter += 1
                    T *= cooling
                    continue
                
                item_to_remove = peak_elems[0][1]
                
                # Find low-coverage sums
                low_cov = np.where(current_reps <= 2)[0] + N_LOW
                if len(low_cov) == 0:
                    stuck_counter += 1
                    T *= cooling
                    continue
                
                found = False
                for _ in range(8):
                    s = random.choice(low_cov.tolist())
                    # Find x_in such that s - x_in in current_A (and x_in not in A, x_in != item_to_remove)
                    candidates_in = []
                    for y in current_A:
                        x_in = s - y
                        if 0 <= x_in <= 128 and x_in != item_to_remove and x_in not in current_A:
                            candidates_in.append(x_in)
                    if candidates_in:
                        new_item = random.choice(candidates_in)
                        new_reps = delta_swap(item_to_remove, new_item, current_A, current_reps)
                        if is_covered(new_reps):
                            new_peak = get_peak(new_reps)
                            if new_peak <= current_peak:
                                current_A.discard(item_to_remove)
                                current_A.add(new_item)
                                current_reps = new_reps
                                update_best(current_A, current_reps)
                                stuck_counter = 0
                                no_improve = 0
                                found = True
                                break
                if not found:
                    stuck_counter += 1
                    no_improve += 1
            
            T *= cooling
    
    # Final Polish: Greedy removal and targeted swaps
    if best_A and time.time() < deadline:
        A = set(best_A)
        reps = calc_reps(A)
        
        # Greedy removal pass
        changed = True
        while changed and time.time() < deadline:
            changed = False
            A_list = sorted(A)
            for x in A_list:
                new_reps = delta_remove(x, A, reps)
                if is_covered(new_reps):
                    new_peak = get_peak(new_reps)
                    old_peak = get_peak(reps)
                    if new_peak <= old_peak:
                        A.discard(x)
                        reps = new_reps
                        changed = True
                        update_best(A, reps)
                        break
        
        # Targeted swap pass
        if time.time() < deadline:
            for _ in range(3000):
                if time.time() > deadline:
                    break
                
                peak = get_peak(reps)
                if peak <= 2:
                    break
                
                # Find peak contributors
                peak_elems = []
                for x in A:
                    y_arr = np.array(sorted(A), dtype=np.int32)
                    sums = x + y_arr
                    mask = (sums >= N_LOW) & (sums <= N_HIGH)
                    if np.any(mask):
                        idx = (sums[mask] - N_LOW).astype(np.intp)
                        pc = int(np.sum(reps[idx] == peak))
                    else:
                        pc = 0
                    peak_elems.append((pc, x))
                
                peak_elems.sort(reverse=True)
                if not peak_elems or peak_elems[0][0] == 0:
                    break
                
                improved = False
                for pc, x in peak_elems[:5]:
                    if time.time() > deadline:
                        break
                    # Try swapping with random non-members
                    for _ in range(15):
                        if time.time() > deadline:
                            break
                        new_item = random.randint(0, 128)
                        if new_item in A or new_item == x:
                            continue
                        new_reps = delta_swap(x, new_item, A, reps)
                        if is_covered(new_reps):
                            new_peak = get_peak(new_reps)
                            if new_peak < peak:
                                A.discard(x)
                                A.add(new_item)
                                reps = new_reps
                                peak = new_peak
                                update_best(A, reps)
                                improved = True
                                break
                    if improved:
                        break
                
                if not improved:
                    # Try removing zero-contribution elements
                    zero_elems = [x for pc, x in peak_elems if pc == 0]
                    if zero_elems and len(A) > 15:
                        x_rm = random.choice(zero_elems)
                        new_reps = delta_remove(x_rm, A, reps)
                        if is_covered(new_reps):
                            A.discard(x_rm)
                            reps = new_reps
                            update_best(A, reps)
                    else:
                        break

    if best_A is None:
        return list(range(64, 129))
        
    return best_A
