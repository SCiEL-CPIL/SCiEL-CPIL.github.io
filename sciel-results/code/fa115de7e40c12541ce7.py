import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1
    MAX_VAL = 128
    DOMAIN = list(range(MAX_VAL + 1))

    def calc_reps(A):
        """Returns array of counts for sums N_LOW..N_HIGH using unordered pairs."""
        A_arr = np.array(sorted(A), dtype=np.int32)
        n = len(A_arr)
        reps = np.zeros(RANGE_SIZE, dtype=np.int32)
        
        for i in range(n):
            ai = A_arr[i]
            sums = ai + A_arr[i:]
            mask = (sums >= N_LOW) & (sums <= N_HIGH)
            valid_sums = sums[mask]
            if len(valid_sums) > 0:
                indices = valid_sums - N_LOW
                np.add.at(reps, indices, 1)
        return reps

    def is_covered(reps):
        return np.all(reps > 0)

    def get_peak(reps):
        return int(np.max(reps))

    def element_contribution(x, A, reps):
        """For element x in set A, return (num_peak_contrib, coverage_risk).
        num_peak_contrib: how many sums at peak level involve x
        coverage_risk: number of sums that would drop to 0 if x removed
        """
        A_list = sorted(A)
        peak = get_peak(reps)
        peak_count = 0
        coverage_risk = 0
        
        for y in A_list:
            s = x + y
            if N_LOW <= s <= N_HIGH:
                idx = s - N_LOW
                if reps[idx] == peak:
                    peak_count += 1
                if reps[idx] == 1:
                    coverage_risk += 1
        
        return peak_count, coverage_risk

    def find_peak_elements(A, reps):
        """Find elements that contribute most to peak sums."""
        peak = get_peak(reps)
        if peak == 0:
            return []
        
        scores = {}
        for x in A:
            pc, cr = element_contribution(x, A, reps)
            scores[x] = pc
        
        # Return elements sorted by peak contribution (descending)
        sorted_elems = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return sorted_elems

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
        A = set()
        A.add(128)
        A.add(0)
        # Add structured elements to ensure good coverage
        # Include elements that help cover low and high ends
        for x in [1, 2, 3, 125, 126, 127]:
            A.add(x)
        # Fill with random
        while len(A) < 40:
            A.add(random.randint(0, 128))
        return A

    def repair_coverage(A):
        """Add elements until all sums in [128,256] are covered."""
        A = set(A)
        reps = calc_reps(A)
        tries = 0
        while not is_covered(reps) and tries < 100:
            missing = np.where(reps == 0)[0] + N_LOW
            # For each missing sum, find a good element to add
            best_candidate = None
            best_gain = 0
            for s in missing:
                # Find elements x such that s-x is in [0,128] and x not in A
                candidates = set()
                for x in range(0, 129):
                    y = s - x
                    if 0 <= y <= 128:
                        candidates.add(x)
                candidates -= A
                for c in candidates:
                    new_A = A | {c}
                    new_reps = calc_reps(new_A)
                    new_missing = np.sum(new_reps == 0)
                    gain = len(missing) - new_missing
                    if gain > best_gain:
                        best_gain = gain
                        best_candidate = c
            if best_candidate is not None:
                A.add(best_candidate)
                reps = calc_reps(A)
            else:
                # Random fallback
                A.add(random.randint(0, 128))
                reps = calc_reps(A)
            tries += 1
        return A, reps

    # Phase 1: Multi-start Local Search with peak-aware moves
    num_restarts = 25
    
    for r in range(num_restarts):
        if time.time() > deadline: break
        
        current_A = random_init()
        current_A, current_reps = repair_coverage(current_A)
        
        if not is_covered(current_reps):
            continue

        update_best(current_A, current_reps)
        
        # Local Search: Peak-aware Simulated Annealing
        T = 2.0
        cooling = 0.995
        min_T = 0.05
        stuck_counter = 0
        max_stuck = 300
        
        while time.time() < deadline and T > min_T:
            if stuck_counter > max_stuck:
                break
            
            current_peak = get_peak(current_reps)
            move_type = random.random()
            
            if len(current_A) > 15 and move_type < 0.4:
                # Try Remove: prioritize elements with high peak contribution
                peak_elems = find_peak_elements(current_A, current_reps)
                if peak_elems:
                    # Weighted selection: higher peak contribution -> higher chance
                    weights = []
                    for x in peak_elems:
                        pc, _ = element_contribution(x, current_A, current_reps)
                        weights.append(pc + 1)
                    total_w = sum(weights)
                    r = random.random() * total_w
                    cum = 0
                    item_to_remove = peak_elems[-1]
                    for i, x in enumerate(peak_elems):
                        cum += weights[i]
                        if r <= cum:
                            item_to_remove = x
                            break
                else:
                    item_to_remove = random.choice(list(current_A))
                
                new_A = set(current_A)
                new_A.remove(item_to_remove)
                new_reps = calc_reps(new_A)
                
                if is_covered(new_reps):
                    new_peak = get_peak(new_reps)
                    
                    if new_peak < current_peak:
                        current_A = new_A
                        current_reps = new_reps
                        update_best(current_A, current_reps)
                        stuck_counter = 0
                    elif new_peak == current_peak:
                        # Accept with some probability to explore
                        if random.random() < 0.3:
                            current_A = new_A
                            current_reps = new_reps
                            update_best(current_A, current_reps)
                            stuck_counter = 0
                        else:
                            stuck_counter += 1
                    else:
                        prob = np.exp(-(new_peak - current_peak) / T)
                        if random.random() < prob:
                            current_A = new_A
                            current_reps = new_reps
                            stuck_counter = 0
                        else:
                            stuck_counter += 1
                else:
                    stuck_counter += 1
            elif move_type < 0.7:
                # Try Swap: remove a high-peak element, add a random one
                peak_elems = find_peak_elements(current_A, current_reps)
                if peak_elems:
                    item_to_remove = peak_elems[0]  # Highest peak contributor
                else:
                    item_to_remove = random.choice(list(current_A))
                
                new_item = random.randint(0, 128)
                if new_item == item_to_remove:
                    continue
                    
                new_A = set(current_A)
                new_A.remove(item_to_remove)
                new_A.add(new_item)
                
                new_reps = calc_reps(new_A)
                
                if is_covered(new_reps):
                    new_peak = get_peak(new_reps)
                    
                    if new_peak < current_peak:
                        current_A = new_A
                        current_reps = new_reps
                        update_best(current_A, current_reps)
                        stuck_counter = 0
                    else:
                        prob = np.exp(-(new_peak - current_peak) / T)
                        if random.random() < prob:
                            current_A = new_A
                            current_reps = new_reps
                            stuck_counter = 0
                        else:
                            stuck_counter += 1
                else:
                    stuck_counter += 1
            else:
                # Try targeted swap: replace a peak element with one that helps low-coverage sums
                peak_elems = find_peak_elements(current_A, current_reps)
                if not peak_elems:
                    stuck_counter += 1
                    T *= cooling
                    continue
                    
                item_to_remove = peak_elems[0]
                
                # Find sums with low coverage (count == 1 or 2)
                low_cov_sums = np.where(current_reps <= 2)[0] + N_LOW
                if len(low_cov_sums) == 0:
                    stuck_counter += 1
                    T *= cooling
                    continue
                
                # Try candidates that help cover low-coverage sums
                found = False
                for _ in range(10):
                    s = random.choice(low_cov_sums.tolist())
                    # Find x such that s - x is in current_A
                    candidates = []
                    for y in current_A:
                        x = s - y
                        if 0 <= x <= 128 and x != item_to_remove:
                            candidates.append(x)
                    if candidates:
                        new_item = random.choice(candidates)
                        if new_item not in current_A:
                            new_A = set(current_A)
                            new_A.remove(item_to_remove)
                            new_A.add(new_item)
                            new_reps = calc_reps(new_A)
                            if is_covered(new_reps):
                                new_peak = get_peak(new_reps)
                                if new_peak <= current_peak:
                                    current_A = new_A
                                    current_reps = new_reps
                                    update_best(current_A, current_reps)
                                    stuck_counter = 0
                                    found = True
                                    break
                if not found:
                    stuck_counter += 1
            
            T *= cooling
            
    # Final Polish: Peak-targeted refinement
    if best_A:
        A = set(best_A)
        reps = calc_reps(A)
        
        # Phase 2a: Greedy removal of redundant elements
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for x in sorted(A, key=lambda e: -element_contribution(e, A, reps)[0]):
                new_A = set(A)
                new_A.remove(x)
                new_reps = calc_reps(new_A)
                if is_covered(new_reps):
                    new_peak = get_peak(new_reps)
                    old_peak = get_peak(reps)
                    if new_peak <= old_peak:
                        A = new_A
                        reps = new_reps
                        changed = True
                        update_best(A, reps)
                        break
        
        # Phase 2b: Targeted swaps to reduce peak
        if time.time() < deadline:
            for _ in range(200):
                if time.time() > deadline:
                    break
                
                peak = get_peak(reps)
                peak_elems = find_peak_elements(A, reps)
                if not peak_elems:
                    break
                
                improved = False
                for x in peak_elems[:3]:  # Try top 3 peak contributors
                    if time.time() > deadline:
                        break
                    # Try swapping x with various candidates
                    for _ in range(5):
                        new_item = random.randint(0, 128)
                        if new_item == x:
                            continue
                        new_A = set(A)
                        new_A.remove(x)
                        new_A.add(new_item)
                        new_reps = calc_reps(new_A)
                        if is_covered(new_reps):
                            new_peak = get_peak(new_reps)
                            if new_peak < peak:
                                A = new_A
                                reps = new_reps
                                peak = new_peak
                                update_best(A, reps)
                                improved = True
                                break
                    if improved:
                        break
                
                if not improved:
                    break

    if best_A is None:
        return list(range(64, 129))
        
    return best_A
