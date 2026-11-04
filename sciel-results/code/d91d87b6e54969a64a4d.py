import numpy as np

N = 400

def construct():
    best_peak = np.inf
    best_support = None
    best_signs = None
    
    def get_peak_and_conv(support, signs):
        M = len(support)
        conv = np.zeros(N)
        for i in range(M):
            si = support[i]
            sgi = signs[i]
            for j in range(M):
                idx = (si + support[j]) % N
                conv[idx] += sgi * signs[j]
        return np.max(np.abs(conv)), conv

    def local_search(support, signs, max_steps=100):
        M = len(support)
        current_peak, conv = get_peak_and_conv(support, signs)
        
        for step in range(max_steps):
            improved = False
            
            # Try flipping signs
            # We try all M flips, pick the best one? Or random?
            # Trying all M is M * (M + N) ops.
            # 20 * 420 = 8400 ops per step.
            # 100 steps = 840k ops.
            # This is fine.
            
            best_gain = 0
            best_i = -1
            best_new_conv = None
            best_new_peak = current_peak
            
            for i in range(M):
                # Simulate flip i
                # We don't want to modify conv permanently until we choose the best flip
                # We can calculate the delta
                # Delta at idx = (support[i] + support[j]) % N is -4 * signs[i] * signs[j] for j != i
                # We need to find the new max.
                # This is hard to do incrementally without updating the array.
                # So we update a copy? That's O(N) per flip.
                # M * N = 20 * 400 = 8000 ops per step.
                # 100 steps = 800k ops.
                
                # Let's just do the update on a copy of conv for each candidate i?
                # That's expensive.
                
                # Alternative: Just do random flips and accept if better.
                # This is simpler and faster.
                pass
            
            # Let's stick to random flips/moves with immediate acceptance (Hill Climbing).
            # It's faster per step.
            
            # 1. Random Sign Flip
            i = np.random.choice(M)
            old_sign = signs[i]
            signs[i] = -old_sign
            # Update conv incrementally
            si = support[i]
            sgi = -old_sign # New sign
            # Delta is -4 * sgi * signs[j] ?
            # Old contribution: old_sign * signs[j]
            # New contribution: -old_sign * signs[j]
            # Diff: -2 * old_sign * signs[j] for pair (i,j)
            # And -2 * old_sign * signs[j] for pair (j,i)
            # Total diff: -4 * old_sign * signs[j]
            # Wait, signs[i] is already flipped in the array?
            # If I flip signs[i] in the array, then signs[i] is new.
            # The formula for delta should use the OLD sign to calculate the change from old state to new state.
            # Let's revert signs[i] for calculation, or just use old_sign.
            
            # Let's do it cleanly:
            # 1. Save old sign
            # 2. Flip sign in array
            # 3. Update conv
            # 4. Check peak
            # 5. If worse, revert sign and conv
            
            # This is what I had.
            
            # Let's optimize the "Check peak" part.
            # Instead of np.max(np.abs(conv)), can we maintain a histogram of values?
            # Values are integers (sums of +/- 1).
            # Range of values: -M*M to M*M.
            # We can maintain a count array.
            # When conv[idx] changes from v to v+delta, decrement count[v], increment count[v+delta].
            # Then find the max index with count > 0.
            # This makes the peak check O(1) if we keep track of current max?
            # Or O(Range) to find max. Range is ~400.
            # So O(1) update + O(1) peak check (if we just check if the changed indices exceed current max? No, max might decrease).
            # If max decreases, we need to scan.
            # But usually max stays same or increases.
            # If we only accept improvements, max never increases.
            # So we only need to check if the new values at the changed indices are < current max?
            # No, the max might be at an index that didn't change.
            # So if we don't change the max index, max stays same.
            # If we change the max index, it might decrease.
            # So we can just check the changed indices.
            # If the new value at a changed index is > current max, update max.
            # If the old max index was changed and its value decreased, we need to find the new max.
            # This is rare.
            
            # This is getting complex. Let's stick to the simple O(N) peak check but reduce iterations.
            
            # Let's try 20 steps per trial, 20 trials.
            # 20 * 20 * 15 * 420 = 250k ops. Very fast.
            # But might not converge.
            
            # Let's try 100 steps, 10 trials.
            # 100 * 10 * 15 * 420 = 630k ops.
            
            # Let's go with 50 steps, 20 trials.
            
            pass # Placeholder for actual logic
            
            # Actual logic:
            # 1. Flip random sign
            i = np.random.choice(M)
            old_sign = signs[i]
            signs[i] = -old_sign
            # Update conv
            si = support[i]
            for j in range(M):
                if j == i: continue
                idx = (si + support[j]) % N
                # The change in conv[idx] is -4 * old_sign * signs[j]
                # Note: signs[j] is current (unchanged).
                conv[idx] -= 4 * old_sign * signs[j]
            
            new_peak = np.max(np.abs(conv))
            if new_peak < current_peak:
                current_peak = new_peak
                improved = True
            else:
                # Revert
                signs[i] = old_sign
                for j in range(M):
                    if j == i: continue
                    idx = (si + support[j]) % N
                    conv[idx] += 4 * old_sign * signs[j]
            
            # 2. Move support
            i = np.random.choice(M)
            old_pos = support[i]
            new_pos = np.random.choice(N)
            if new_pos not in support:
                support[i] = new_pos
                # Update conv
                # Remove contributions of old_pos
                # Add contributions of new_pos
                # This is complex.
                # Let's just rebuild conv for moves?
                # Rebuild is M^2 = 400 ops.
                # Move is rare (1 in 15).
                # So 1/15 * 400 = 26 ops.
                # Flip is 14/15 * 420 = 392 ops.
                # Total ~ 418 ops per step.
                
                # Rebuild conv
                conv_new = np.zeros(N)
                for a in range(M):
                    sa = support[a]
                    sa_sign = signs[a]
                    for b in range(M):
                        sb = support[b]
                        sb_sign = signs[b]
                        idx = (sa + sb) % N
                        conv_new[idx] += sa_sign * sb_sign
                new_peak = np.max(np.abs(conv_new))
                
                if new_peak < current_peak:
                    current_peak = new_peak
                    conv = conv_new
                    improved = True
                else:
                    support[i] = old_pos
                    # conv is unchanged in memory, but we created conv_new.
                    # We don't need to revert conv if we didn't assign it.
            
            if not improved:
                break
        
        return current_peak, support, signs

    # Main loop
    for M in range(16, 22):
        for trial in range(20):
            if trial == 0:
                # Greedy Sidon init
                support = np.array(_greedy_sidon(N, M))
            else:
                support = np.random.choice(N, size=M, replace=False)
            
            signs = np.random.choice([-1, 1], size=M)
            
            peak, sup, sig = local_search(support, signs, max_steps=100)
            
            if peak < best_peak:
                best_peak = peak
                best_support = sup
                best_signs = sig

    x = np.zeros(N)
    x[best_support] = best_signs
    norm = np.linalg.norm(x)
    if norm > 0:
        x = x / norm
    return x

def _greedy_sidon(N, M):
    support = [0]
    diffs = set()
    for k in range(1, N):
        valid = True
        for s in support:
            if (k - s) in diffs or (s - k) in diffs:
                valid = False
                break
        if valid:
            support.append(k)
            for s in support[:-1]:
                diffs.add(k - s)
                diffs.add(s - k)
            if len(support) == M:
                break
    return support
