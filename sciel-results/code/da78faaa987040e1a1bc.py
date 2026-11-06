import time
import numpy as np
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 55.0

    # Precompute all arithmetic progressions of length >= 2 within [1, 256]
    # An AP is defined by (start, step).
    # Indices are 0-based in the array.
    # AP terms: start, start+step, ... <= n
    # We store the list of indices for each AP.
    
    ap_indices = []
    # Map from position index (0 to n-1) to list of AP indices containing it
    pos_to_aps = [[] for _ in range(n)]
    
    ap_id = 0
    for start in range(1, n + 1):
        for step in range(1, n + 1):
            # Generate terms
            terms = []
            cur = start
            while cur <= n:
                terms.append(cur - 1) # 0-based
                cur += step
            if len(terms) >= 2:
                ap_indices.append(terms)
                for idx in terms:
                    pos_to_aps[idx].append(ap_id)
                ap_id += 1
                
    num_aps = len(ap_indices)
    
    # Convert to numpy arrays for faster access if needed, 
    # but list of lists is fine for updating.
    # We will maintain an array of sums.
    
    def thue_morse_signs(n):
        signs = np.zeros(n, dtype=np.int8)
        for i in range(n):
            cnt = bin(i).count('1')
            signs[i] = 1 if cnt % 2 == 0 else -1
        return signs

    def evaluate(signs):
        """Compute max abs sum and count of APs with abs sum >= 2."""
        sums = np.zeros(num_aps, dtype=np.int32)
        for ap_id, indices in enumerate(ap_indices):
            s = 0
            for idx in indices:
                s += signs[idx]
            sums[ap_id] = s
        
        abs_sums = np.abs(sums)
        max_sum = int(np.max(abs_sums)) if num_aps > 0 else 0
        bad_count = int(np.sum(abs_sums >= 2))
        return max_sum, bad_count, sums

    def local_search(signs_init, max_iters, deadline):
        signs = signs_init.copy()
        max_sum, bad_count, sums = evaluate(signs)
        
        best_signs = signs.copy()
        best_max = max_sum
        best_bad = bad_count
        
        rng = random.Random()
        
        # Simulated Annealing / Hill Climbing hybrid
        # Objective: Minimize (bad_count, max_sum) lexicographically
        # Or just bad_count, then max_sum.
        
        current_bad = bad_count
        current_max = max_sum
        
        temperature = 2.0
        cooling = 0.999
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            i = rng.randint(0, n - 1)
            old_sign = signs[i]
            
            # Flip
            signs[i] = -old_sign
            
            # Update sums for affected APs
            affected = pos_to_aps[i]
            
            old_bad = current_bad
            old_max = current_max
            
            # We need to update bad_count and max_sum efficiently.
            # Since we only change a few APs, we can update the count.
            # However, maintaining a global max dynamically is tricky without a heap or histogram.
            # Given small num_aps (~8000), we can just recalculate max if bad_count changes or periodically.
            # Or, we can maintain a histogram of abs sums.
            
            # Let's use a histogram for speed.
            # Histogram size: max possible sum is 256.
            
            # Actually, let's just track the changes.
            # If an AP sum goes from 1 to -1, abs sum stays 1.
            # If it goes from 2 to 0, abs sum drops from 2 to 0.
            
            # To be safe and simple, let's just update the sums array and 
            # check if the specific APs changed their "bad" status.
            
            new_bad = current_bad
            # We don't easily know the new max without scanning.
            # But we can check if any AP became "more bad" or "less bad".
            
            # Let's just update the sums and then compute max/bad for the whole thing?
            # No, that's O(num_aps) per flip. 8000 * 3000 = 24M ops. Might be too slow in Python.
            
            # Better: Update histogram.
            # Let's implement histogram inside the loop.
            
            # Re-initialize histogram logic? No, let's just do a fast check.
            # If the number of affected APs is small, we can update a counter.
            
            # Let's try a simpler approach: 
            # Just flip, and if it looks promising (based on local sums), compute full stats.
            # But computing full stats is the bottleneck.
            
            # Alternative: 
            # Maintain `sums` array.
            # Maintain `bad_count`.
            # Maintain `max_sum`.
            
            # For each affected AP:
            #   old_val = sums[ap_id]
            #   new_val = old_val - 2*old_sign
            #   sums[ap_id] = new_val
            #   if abs(old_val) >= 2 and abs(new_val) < 2: bad_count -= 1
            #   if abs(old_val) < 2 and abs(new_val) >= 2: bad_count += 1
            
            # This updates bad_count in O(1) per AP.
            # For max_sum:
            #   If the AP that had the max sum changes, max_sum might change.
            #   If the new value is higher than current max, update max.
            #   If the old value was the max, and new value is lower, we might need to find the new max.
            #   To avoid O(N) scan, we can just check if new_val > max_sum.
            #   If we reduced the max, we might have a stale max.
            #   We can tolerate a stale max for a few steps, or just recalculate max every K steps.
            
            # Let's try this.
            
            changed_max = False
            for ap_id in affected:
                old_val = sums[ap_id]
                new_val = old_val - 2 * old_sign
                sums[ap_id] = new_val
                
                old_abs = abs(old_val)
                new_abs = abs(new_val)
                
                if old_abs >= 2 and new_abs < 2:
                    new_bad -= 1
                elif old_abs < 2 and new_abs >= 2:
                    new_bad += 1
                    
                if new_abs > current_max:
                    current_max = new_abs
                    changed_max = True
                # If old_abs == current_max and new_abs < current_max, we don't know the new max.
                # We'll mark that we need a recalculation soon.
            
            # Heuristic: If we reduced bad_count, accept.
            # If bad_count same, and we increased max, reject.
            # If bad_count same, and max unchanged (or we don't know), accept with prob.
            
            # To be safe, if we suspect max might have dropped (we reduced the value of a high sum),
            # we should recalculate. But let's just accept if bad_count <= current_bad.
            
            if new_bad < current_bad:
                current_bad = new_bad
                # Accept
            elif new_bad == current_bad:
                # Check if max improved?
                # We can't easily tell if max decreased.
                # Let's just accept with some probability to explore.
                if rng.random() < 0.1:
                    current_bad = new_bad
                    # Accept
                else:
                    # Revert
                    signs[i] = old_sign
                    for ap_id in affected:
                        sums[ap_id] += 2 * old_sign
                    continue
            else:
                # Revert
                signs[i] = old_sign
                for ap_id in affected:
                    sums[ap_id] += 2 * old_sign
                continue
            
            # Periodically recalculate true max and bad to correct drift
            if it % 100 == 0:
                true_max, true_bad, _ = evaluate(signs)
                if true_bad < current_bad:
                    current_bad = true_bad
                current_max = true_max
                
                if current_bad < best_bad or (current_bad == best_bad and current_max < best_max):
                    best_bad = current_bad
                    best_max = current_max
                    best_signs = signs.copy()
                    
                if best_bad == 0:
                    break

        return best_signs, best_max, best_bad

    best_overall = None
    best_overall_max = n + 1
    best_overall_bad = n + 1

    # Initializations
    inits = []
    tm = thue_morse_signs(n)
    inits.append(tm)
    inits.append(-tm)
    
    for seed in [42, 123, 456, 789, 1000, 2024, 3141, 2718, 1618, 9999]:
        rng = random.Random(seed)
        signs = np.array([rng.choice([-1, 1]) for _ in range(n)], dtype=np.int8)
        inits.append(signs)

    for signs_init in inits:
        if time.time() > deadline - 5.0:
            break
        
        signs, mx, bad = local_search(signs_init, max_iters=5000, deadline=deadline)
        
        if bad < best_overall_bad or (bad == best_overall_bad and mx < best_overall_max):
            best_overall_bad = bad
            best_overall_max = mx
            best_overall = signs.copy()
            
            if best_overall_bad == 0:
                break

    if best_overall is None:
        rng = random.Random(42)
        return [rng.choice([-1, 1]) for _ in range(n)]
    
    return [int(s) for s in best_overall]
