import time
import random
import numpy as np
from scipy.optimize import minimize, differential_evolution

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = 128
    min_sum = 128
    max_sum = 256
    S = max_sum - min_sum + 1  # 129 sums

    def get_counts(mask):
        """Fast count of unordered pair sums (a<=b) in [128,256] using FFT."""
        m = len(mask)
        # Full linear convolution of mask with itself
        N = 1 << (2 * m - 1).bit_length()
        f = np.fft.rfft(mask.astype(np.float64), n=N)
        conv = np.fft.irfft(f * f, n=N)[:2 * m - 1]
        conv = np.rint(conv).astype(np.int64)  # round to int
        
        # Vectorized computation of unordered pair counts
        sums = np.arange(min_sum, max_sum + 1)
        ordered_counts = conv[sums]
        
        # For even sums, add diagonal correction (count of x where 2x = sum)
        even_mask = (sums % 2 == 0)
        half_sums = sums // 2
        diag = np.where(even_mask, 
                        np.where((half_sums >= 0) & (half_sums <= n), 
                                mask[half_sums], 0), 
                        0)
        
        counts = (ordered_counts + diag) // 2
        return counts

    def soft_max(x, beta=40.0):
        """Smooth max via log-sum-exp."""
        x = np.asarray(x, dtype=np.float64)
        m = np.max(x)
        return m + np.log(np.sum(np.exp(beta * (x - m)))) / beta

    def get_cost(mask):
        counts = get_counts(mask)
        if np.any(counts == 0):
            return float('inf')
        return int(np.max(counts))

    def mask_to_list(mask):
        return [i for i in range(n + 1) if mask[i]]

    def list_to_mask(A_list):
        mask = np.zeros(n + 1, dtype=np.int64)
        for x in A_list:
            mask[x] = 1
        return mask

    # Precompute contribution matrix: contrib[i, s] = 1 if i can contribute to sum s
    # i contributes to sum s if there exists j in A such that i+j=s, i.e., j=s-i
    # For unordered pairs with i<=j, we need i <= s-i, so i <= s/2
    # But for the multiplicative weights, we just need: if i is in A, it contributes to sums s where s-i is also in A
    # Actually, let's think differently: for each element i, which sums does it help cover?
    # If i is in A, then for any j in A, i+j is a sum. So i helps cover sum s if s-i is in A.
    # But this depends on the current A. Let's use a different approach.
    
    # For multiplicative weights: maintain weights w[i] for each i in [0,128]
    # At each iteration:
    # 1. Sample A from weights (or use threshold)
    # 2. Compute counts for each sum
    # 3. For each element i, compute its "contribution score": how much it helps under-represented sums vs hurts over-represented sums
    # 4. Update weights: w[i] *= exp(eta * score_i)
    
    def multiplicative_weights_search(mask_init, iterations=500, eta=0.5, time_limit=None):
        """Apply multiplicative weights to balance pair sum multiplicities."""
        if time_limit is None:
            time_limit = deadline - 5
        
        # Start with initial mask
        mask = mask_init.copy()
        weights = np.ones(n + 1)  # log-weights
        
        best_mask = mask.copy()
        best_cost = get_cost(mask)
        
        for it in range(iterations):
            if time.time() > time_limit:
                break
            
            # Compute current counts
            counts = get_counts(mask).astype(np.float64)
            
            # If any sum has 0 representations, we need to fix that first
            zero_sums = np.where(counts == 0)[0]
            if len(zero_sums) > 0:
                # Prioritize fixing zero sums
                # For each zero sum s, we need to add pairs (i, s-i) where i <= s-i
                # Find elements that could help cover zero sums
                for s_idx in zero_sums:
                    s = min_sum + s_idx
                    # Find all pairs (i, j) with i+j=s, i<=j, i,j in [0,128]
                    # i ranges from max(0, s-128) to min(128, s//2)
                    i_min = max(0, s - 128)
                    i_max = min(128, s // 2)
                    for i in range(i_min, i_max + 1):
                        j = s - i
                        if j <= 128:
                            # Adding i or j would help cover this sum (if the other is present)
                            # But we need both. For now, boost both
                            weights[i] += eta * 2.0
                            weights[j] += eta * 2.0
            
            # Compute score for each element: how much it contributes to under vs over represented sums
            # For element i, it contributes to sum s if s-i is in mask
            # score_i = sum over s of (target - counts[s]) * indicator(i contributes to s)
            # Where target is the mean or median count
            
            target = np.mean(counts[counts > 0]) if np.any(counts > 0) else 1.0
            deficit = target - counts  # positive if under-represented, negative if over
            
            # For each element i, compute its contribution to each sum
            # i contributes to sum s if s-i is in mask AND i <= s-i (for unordered)
            # Actually, i contributes to sum s if s-i is in mask (regardless of order, since we count unordered)
            # But we need to be careful: if i = s-i, it's one pair, otherwise two ordered pairs become one unordered
            
            scores = np.zeros(n + 1)
            for i in range(n + 1):
                if mask[i] == 0:
                    # If i is not in mask, adding it would create new pairs
                    # For each j in mask, i+j is a new sum
                    j_indices = np.where(mask == 1)[0]
                    sums_created = i + j_indices
                    valid = (sums_created >= min_sum) & (sums_created <= max_sum)
                    if np.any(valid):
                        s_vals = sums_created[valid]
                        # Each new pair adds 1 to the count of that sum
                        # But we need to check if this creates duplicate counts
                        # For now, approximate: score = sum of deficits for created sums
                        scores[i] = np.sum(deficit[s_vals - min_sum])
                else:
                    # If i is in mask, removing it would remove pairs
                    # For each j in mask with j >= i, i+j is a sum
                    j_indices = np.where(mask == 1)[0]
                    j_indices = j_indices[j_indices >= i]  # unordered: i <= j
                    if len(j_indices) > 0:
                        sums_removed = i + j_indices
                        valid = (sums_removed >= min_sum) & (sums_removed <= max_sum)
                        if np.any(valid):
                            s_vals = sums_removed[valid]
                            # Removing i decreases count of these sums
                            # Score is negative of deficit (we want to remove if it helps over-represented)
                            scores[i] = -np.sum(deficit[s_vals - min_sum])
            
            # Update weights
            weights += eta * scores
            weights = np.clip(weights, -10, 10)  # prevent overflow
            
            # Convert weights to probabilities and sample new mask
            # Use softmax to get probabilities
            exp_weights = np.exp(weights - np.max(weights))
            probs = exp_weights / np.sum(exp_weights)
            
            # Expected size: aim for around 40-50 elements
            target_size = 45
            # Adjust threshold to get desired size
            # Use a temperature-based approach
            temperature = 1.0 + it * 0.01  # anneal
            adjusted_probs = probs ** (1.0 / temperature)
            adjusted_probs = adjusted_probs / np.sum(adjusted_probs)
            
            # Sample elements
            new_mask = np.zeros(n + 1, dtype=np.int64)
            # Use expected number of elements
            expected_count = np.sum(adjusted_probs * (n + 1))
            # Adjust to target size by scaling
            scale = target_size / max(expected_count, 1)
            adjusted_probs = adjusted_probs * scale
            adjusted_probs = np.clip(adjusted_probs, 0, 1)
            
            # Sample
            for i in range(n + 1):
                if random.random() < adjusted_probs[i]:
                    new_mask[i] = 1
            
            # Ensure at least some elements
            if np.sum(new_mask) < 10:
                # Fallback: take top elements by weight
                top_indices = np.argsort(weights)[-target_size:]
                new_mask[top_indices] = 1
            
            # Check if new mask is better
            new_cost = get_cost(new_mask)
            if new_cost < best_cost:
                best_cost = new_cost
                best_mask = new_mask.copy()
                mask = new_mask.copy()
            
            # Also try to keep mask if cost is same but different (for exploration)
            elif new_cost == best_cost and random.random() < 0.1:
                mask = new_mask.copy()
        
        return best_mask, best_cost

    best_score = float('inf')
    best_mask = None

    def update_best(mask):
        nonlocal best_score, best_mask
        if not np.any(mask):
            return
        cost = get_cost(mask)
        if cost < best_score:
            best_score = cost
            best_mask = mask.copy()

    # 1. Heuristic Initializations
    # Symmetric ranges
    for offset in range(0, 40):
        lo = max(0, 64 - offset)
        hi = min(128, 64 + offset)
        mask = np.zeros(n + 1, dtype=np.int64)
        mask[lo:hi + 1] = 1
        update_best(mask)

    # Modular
    for mod in [3, 4, 5, 6, 7, 8, 9, 10]:
        for keep in range(1, mod):
            mask = np.zeros(n + 1, dtype=np.int64)
            for x in range(n + 1):
                if x % mod != mod - keep:
                    mask[x] = 1
            update_best(mask)

    # Random initializations
    random.seed(42)
    for _ in range(80):
        if time.time() > deadline - 40:
            break
        size = random.randint(25, 65)
        A = random.sample(range(n + 1), size)
        mask = list_to_mask(A)
        update_best(mask)

    # 2. Multiplicative Weights Search (NEW)
    if best_mask is not None and time.time() < deadline - 30:
        mw_mask, mw_cost = multiplicative_weights_search(best_mask, iterations=300, eta=0.3, time_limit=deadline - 25)
        if mw_cost < best_score:
            best_score = mw_cost
            best_mask = mw_mask.copy()
        else:
            # Try from a few different initializations
            for _ in range(5):
                if time.time() > deadline - 25:
                    break
                random_mask = np.zeros(n + 1, dtype=np.int64)
                size = random.randint(35, 55)
                A = random.sample(range(n + 1), size)
                random_mask[A] = 1
                mw_mask2, mw_cost2 = multiplicative_weights_search(random_mask, iterations=200, eta=0.3, time_limit=deadline - 20)
                if mw_cost2 < best_score:
                    best_score = mw_cost2
                    best_mask = mw_mask2.copy()

    # 3. Local Search Refinement
    if best_mask is None:
        best_mask = np.zeros(n + 1, dtype=np.int64)
        best_mask[64:129] = 1
        best_score = get_cost(best_mask)

    current_mask = best_mask.copy()
    current_cost = best_score

    def try_local_search(mask, cost, time_limit):
        nonlocal best_score, best_mask
        current_mask = mask.copy()
        current_cost = cost
        no_improve_count = 0
        max_no_improve = 3

        while time.time() < time_limit:
            improved = False

            # 1. Try removing (order by least impact)
            ones = [i for i in range(n + 1) if current_mask[i] == 1]
            # Try removals in random order
            random.shuffle(ones)
            for i in ones:
                if time.time() > time_limit:
                    break
                trial_mask = current_mask.copy()
                trial_mask[i] = 0
                if not np.any(trial_mask):
                    continue
                tc = get_cost(trial_mask)
                if tc <= current_cost:
                    current_mask = trial_mask
                    current_cost = tc
                    if tc < best_score:
                        best_score = tc
                        best_mask = current_mask.copy()
                    improved = True
                    no_improve_count = 0
                    break

            # 2. Try adding
            if not improved:
                zeros = [i for i in range(n + 1) if current_mask[i] == 0]
                random.shuffle(zeros)
                for i in zeros:
                    if time.time() > time_limit:
                        break
                    trial_mask = current_mask.copy()
                    trial_mask[i] = 1
                    tc = get_cost(trial_mask)
                    if tc < current_cost:
                        current_mask = trial_mask
                        current_cost = tc
                        if tc < best_score:
                            best_score = tc
                            best_mask = current_mask.copy()
                        improved = True
                        no_improve_count = 0
                        break

            # 3. Try swapping
            if not improved:
                ones = [i for i in range(n + 1) if current_mask[i] == 1]
                zeros = [i for i in range(n + 1) if current_mask[i] == 0]
                random.shuffle(ones)
                random.shuffle(zeros)
                # Limit to speed up but be more generous
                if len(ones) > 40:
                    ones = ones[:40]
                if len(zeros) > 40:
                    zeros = zeros[:40]

                for i in ones:
                    if time.time() > time_limit:
                        break
                    for j in zeros:
                        if time.time() > time_limit:
                            break
                        trial_mask = current_mask.copy()
                        trial_mask[i] = 0
                        trial_mask[j] = 1
                        tc = get_cost(trial_mask)
                        if tc < current_cost:
                            current_mask = trial_mask
                            current_cost = tc
                            if tc < best_score:
                                best_score = tc
                                best_mask = current_mask.copy()
                            improved = True
                            no_improve_count = 0
                            break
                    if improved:
                        break

            # 4. Random perturbation to escape local minima
            if not improved and time.time() < time_limit - 3:
                no_improve_count += 1
                if no_improve_count >= max_no_improve:
                    # More aggressive perturbation
                    flip_count = random.randint(3, 7)
                else:
                    flip_count = random.randint(2, 5)
                indices = random.sample(range(n + 1), flip_count)
                trial_mask = current_mask.copy()
                for idx in indices:
                    trial_mask[idx] = 1 - trial_mask[idx]
                if np.any(trial_mask):
                    tc = get_cost(trial_mask)
                    if tc < current_cost:
                        current_mask = trial_mask
                        current_cost = tc
                        if tc < best_score:
                            best_score = tc
                            best_mask = current_mask.copy()
                        improved = True
                        no_improve_count = 0
                    elif tc == current_cost and random.random() < 0.2:
                        current_mask = trial_mask
                        if tc < best_score:
                            best_score = tc
                            best_mask = current_mask.copy()
                        improved = True
                        no_improve_count = 0

            if not improved and no_improve_count >= max_no_improve:
                break

    try_local_search(best_mask, best_score, deadline - 3)

    # Final check
    if best_mask is None or not np.any(best_mask):
        return list(range(64, 129))

    # Final polish: try to remove any redundant elements
    final_mask = best_mask.copy()
    ones = [i for i in range(n + 1) if final_mask[i] == 1]
    random.shuffle(ones)
    for i in ones:
        if time.time() > deadline - 1:
            break
        trial_mask = final_mask.copy()
        trial_mask[i] = 0
        if not np.any(trial_mask):
            continue
        tc = get_cost(trial_mask)
        if tc <= best_score:
            final_mask = trial_mask
            if tc < best_score:
                best_score = tc
                best_mask = final_mask.copy()

    return mask_to_list(final_mask)
