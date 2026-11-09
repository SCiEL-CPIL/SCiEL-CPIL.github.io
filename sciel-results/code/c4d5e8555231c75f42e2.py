import numpy as np
import time
import random

def construct():
    def get_primes(limit):
        sieve = np.ones(limit + 1, dtype=bool)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = False
        return np.where(sieve)[0]

    # Primes up to 10^6
    primes = get_primes(10**6)
    
    MAX_N = 10**12
    start_time = time.time()
    
    best_k = 1
    best_s = 1
    
    def compute_phi_segment(L, R):
        N = R - L + 1
        phi_arr = np.arange(L, R + 1, dtype=np.int64)
        
        for p in primes:
            if p > R:
                break
            start = ((L + p - 1) // p) * p
            if start > R:
                continue
            offset = start - L
            last = R - (R % p)
            if last < start:
                continue
            # Vectorized update
            indices = np.arange(offset, N, p)
            phi_arr[indices] //= p
            phi_arr[indices] *= (p - 1)
            
        return phi_arr

    def scan_range(L, R, prev_tail=None, prev_R=0):
        nonlocal best_k, best_s
        if prev_tail is None:
            prev_tail = []
            prev_R = 0
            
        SEG_SIZE = 10**6
        T = 200
        
        seen = {}
        left = 0
        local_max_k = 0
        local_max_start = 0
        
        # Helper to get actual start number from virtual index
        def get_actual_start(v_idx):
            if v_idx < len(prev_tail):
                return prev_R - (len(prev_tail) - 1) + v_idx
            else:
                idx_in_curr = v_idx - len(prev_tail)
                # Need to know L of the current segment?
                # This function is called inside scan_range which processes segments.
                # We need to track the global offset.
                # Let's restructure to pass offsets.
                pass
        
        # This structure is getting complicated with the tail.
        # Simpler: Just process the whole range in one go if it fits in memory?
        # 10^8 is too big for one array.
        # But we can process segment by segment and keep a small tail.
        
        # Let's implement a robust segment scanner.
        
        current_L = L
        while current_L <= R:
            if time.time() - start_time > 115:
                break
            current_R = min(current_L + SEG_SIZE - 1, R)
            N = current_R - current_L + 1
            phi_arr = compute_phi_segment(current_L, current_R)
            
            # Process prev_tail
            for i, val in enumerate(prev_tail):
                if val in seen:
                    left = max(left, seen[val] + 1)
                seen[val] = i
                curr_k = i - left + 1
                if curr_k > local_max_k:
                    local_max_k = curr_k
                    local_max_start = left
                    if local_max_k >= 200:
                        # Calculate actual start
                        # If local_max_start < len(prev_tail), it's in prev_tail
                        # else it's in current segment
                        # This logic is fragile.
                        # Let's just return if we hit 200?
                        # We need the actual start value.
                        # We can track the actual start index directly.
                        pass
            
            # Process phi_arr
            for i, val in enumerate(phi_arr):
                v_idx = len(prev_tail) + i
                if val in seen:
                    left = max(left, seen[val] + 1)
                seen[val] = v_idx
                curr_k = v_idx - left + 1
                if curr_k > local_max_k:
                    local_max_k = curr_k
                    local_max_start = left
                    if local_max_k >= 200:
                        # Found max possible
                        # Determine actual start
                        if local_max_start < len(prev_tail):
                            actual_start = prev_R - (len(prev_tail) - 1) + local_max_start
                        else:
                            actual_start = current_L + (local_max_start - len(prev_tail))
                        return actual_start, 200
            
            # Update tail
            if N >= T:
                prev_tail = phi_arr[-T:].tolist()
            else:
                combined = prev_tail + phi_arr.tolist()
                prev_tail = combined[-T:]
                
            prev_R = current_R
            current_L = current_R + 1
            
            # Update best
            if local_max_k > best_k:
                if local_max_start < len(prev_tail):
                    actual_start = prev_R - (len(prev_tail) - 1) + local_max_start
                else:
                    actual_start = current_L - (SEG_SIZE - 1) + (local_max_start - len(prev_tail))
                    # Wait, current_L changed.
                    # We need to track the start of the segment where the run started.
                    # This is tricky.
                    
        return best_s, best_k

    # This logic for tracking actual start is error-prone.
    # Let's simplify: 
    # 1. Scan [1, 10^6].
    # 2. Scan [10^11, 10^11 + 10^8].
    # For each scan, we just find the max k and the offset.
    # Then we map offset to actual number.
    
    def scan_and_find(L, R):
        # Returns (max_k, start_offset_relative_to_L)
        SEG_SIZE = 10**6
        T = 200
        prev_tail = []
        prev_R = 0 # Relative to L? No, absolute.
        # Let's use absolute indices for the tail.
        
        seen = {}
        left = 0
        local_max_k = 0
        local_max_start_idx = 0 # Virtual index
        
        # We need to map virtual index to actual number.
        # Virtual index 0 corresponds to the first element of the first segment?
        # No, we process segments.
        # Let's keep a list of (segment_start, segment_len) to map back?
        # Or simpler:
        # We only keep the last T values in `prev_tail`.
        # If the run is longer than T, we return immediately.
        # If the run is shorter, it must be contained in the last T values + current segment?
        # No, the run can span segments.
        
        # Let's just store the actual values and their absolute positions in the tail?
        # `prev_tail` stores values.
        # We also need to know the absolute position of the start of `prev_tail`.
        # Let `tail_start` be the absolute number corresponding to `prev_tail[0]`.
        
        tail_start = 0
        prev_tail = []
        
        current_L = L
        while current_L <= R:
            if time.time() - start_time > 115:
                break
            current_R = min(current_L + SEG_SIZE - 1, R)
            N = current_R - current_L + 1
            phi_arr = compute_phi_segment(current_L, current_R)
            
            # Process tail
            for i, val in enumerate(prev_tail):
                abs_idx = tail_start + i
                if val in seen:
                    left = max(left, seen[val] + 1)
                seen[val] = i # Store virtual index in seen?
                # If we store virtual index, we need to shift when we discard tail?
                # No, we just use the current virtual indexing for the window.
                # The window `left` is in virtual indices.
                # But `seen` values are from previous segments?
                # We should clear `seen` and rebuild?
                # If we clear `seen`, we lose the history needed for the window.
                # The window size is at most 200.
                # So we only need to remember the last 200 values.
                # We can just keep a list `window_vals` and `window_pos`?
                # And a dict `val_to_pos` for the window?
                
                # Let's use a simpler approach for the sliding window:
                # Maintain a list `recent_vals` and `recent_positions` (absolute).
                # And a dict `val_to_last_pos` (absolute).
                # If `val` in `val_to_last_pos`:
                #   `last_pos = val_to_last_pos[val]`
                #   If `last_pos >= current_window_start`:
                #       `current_window_start = last_pos + 1`
                # `val_to_last_pos[val] = current_abs_pos`
                # `curr_len = current_abs_pos - current_window_start + 1`
                
                pass
            
            # This is getting too complex to patch.
            # Let's rewrite `scan_and_find` cleanly.
            
            current_L = current_R + 1
            
        return 0, 0

    # Clean implementation of scan_and_find
    def scan_range_clean(L, R):
        SEG_SIZE = 10**6
        T = 200
        
        # State
        tail_vals = []
        tail_start = 0 # Absolute start of tail_vals
        
        # Window state
        window_start = L
        val_to_pos = {}
        max_k = 0
        max_start = L
        
        current_L = L
        while current_L <= R:
            if time.time() - start_time > 115:
                break
            current_R = min(current_L + SEG_SIZE - 1, R)
            N = current_R - current_L + 1
            phi_arr = compute_phi_segment(current_L, current_R)
            
            # Process tail
            for i, val in enumerate(tail_vals):
                pos = tail_start + i
                if val in val_to_pos:
                    last_pos = val_to_pos[val]
                    if last_pos >= window_start:
                        window_start = last_pos + 1
                val_to_pos[val] = pos
                
                curr_k = pos - window_start + 1
                if curr_k > max_k:
                    max_k = curr_k
                    max_start = window_start
                    if max_k >= 200:
                        return max_k, max_start
            
            # Process current segment
            for i, val in enumerate(phi_arr):
                pos = current_L + i
                if val in val_to_pos:
                    last_pos = val_to_pos[val]
                    if last_pos >= window_start:
                        window_start = last_pos + 1
                val_to_pos[val] = pos
                
                curr_k = pos - window_start + 1
                if curr_k > max_k:
                    max_k = curr_k
                    max_start = window_start
                    if max_k >= 200:
                        return max_k, max_start
            
            # Update tail
            if N >= T:
                tail_vals = phi_arr[-T:].tolist()
                tail_start = current_R - T + 1
            else:
                # Combine
                # This case is rare
                combined = tail_vals + phi_arr.tolist()
                tail_vals = combined[-T:]
                tail_start = current_R - (len(tail_vals) - 1)
                
            # Cleanup val_to_pos?
            # We only need positions >= window_start.
            # We can't easily remove keys.
            # But we check `if last_pos >= window_start`.
            # If `last_pos < window_start`, it's ignored.
            # So we don't need to remove.
            # The dict size grows to the number of distinct values in the scanned range.
            # For 10^8 range, this is 10^8 entries.
            # 10^8 entries in a dict is ~8GB. TOO MUCH.
            
            # We MUST clean up.
            # We can only keep the last T values in `val_to_pos`?
            # No, if the window is small, we don't need old values.
            # But we don't know which keys to remove.
            
            # Alternative:
            # Since we only care about runs up to 200,
            # if the window is small, we can reset?
            # No, the window can be large.
            
            # Better:
            # We only need to remember the last 200 values.
            # If a value appears, and its last position is < window_start, it's irrelevant.
            # We can just overwrite `val_to_pos[val]`.
            # But the dict still holds old keys.
            
            # Solution:
            # Use a sliding window of the *last 200 values* explicitly.
            # `recent_vals` = list of last 200 values.
            # `recent_pos` = list of last 200 positions.
            # `val_to_idx_in_recent` = dict mapping value to index in recent.
            
            # When adding new value:
            # 1. If val in val_to_idx_in_recent:
            #    last_idx = val_to_idx_in_recent[val]
            #    last_pos = recent_pos[last_idx]
            #    if last_pos >= window_start:
            #        window_start = last_pos + 1
            #        # Update window_start in recent? No.
            #        # We need to check if any other values in recent are now < window_start?
            #        # No, we just need to know the start of the current valid window.
            # 2. Append val to recent_vals, pos to recent_pos.
            # 3. If len(recent) > 200:
            #    Pop front.
            #    If the popped value is in val_to_idx_in_recent, remove it?
            #    Or update its index?
            #    Since we pop from front, indices shift.
            #    This is O(N) to update indices.
            
            # Better:
            # Use a deque?
            # Or just a list and a dict `val_to_pos`.
            # And periodically clean `val_to_pos`?
            # If we scan 10^6 numbers, and window is small, `val_to_pos` has 10^6 entries.
            # We can clear `val_to_pos` and rebuild from `tail_vals`?
            # But `tail_vals` only has 200.
            # If the window is larger than 200, we return.
            # So the window is ALWAYS <= 200.
            # Therefore, we only need to remember the last 200 values!
            # Any value not in the last 200 positions is < window_start (since window_start >= pos - 199).
            # So we can safely ignore values not in the last 200.
            
            # Implementation:
            # `recent_vals` (list, max size 200)
            # `recent_pos` (list, max size 200)
            # `val_to_idx` (dict, max size 200)
            
            pass
        
        return max_k, max_start

    # Let's implement the "Last 200" strategy.
    
    def scan_range_optimized(L, R):
        SEG_SIZE = 10**6
        T = 200
        
        recent_vals = []
        recent_pos = []
        val_to_idx = {}
        
        window_start = L
        max_k = 0
        max_start = L
        
        current_L = L
        while current_L <= R:
            if time.time() - start_time > 115:
                break
            current_R = min(current_L + SEG_SIZE - 1, R)
            N = current_R - current_L + 1
            phi_arr = compute_phi_segment(current_L, current_R)
            
            for i, val in enumerate(phi_arr):
                pos = current_L + i
                
                if val in val_to_idx:
                    last_idx = val_to_idx[val]
                    last_pos = recent_pos[last_idx]
                    if last_pos >= window_start:
                        window_start = last_pos + 1
                
                # Add to recent
                recent_vals.append(val)
                recent_pos.append(pos)
                val_to_idx[val] = len(recent_vals) - 1
                
                # Prune if > T
                if len(recent_vals) > T:
                    old_val = recent_vals.pop(0)
                    old_pos = recent_pos.pop(0)
                    # Remove from val_to_idx?
                    # If old_val is still in recent_vals, we need to update its index.
                    # But we popped from front, so all indices decreased by 1.
                    # This is O(N) to update.
                    # Given T=200, O(200) is fine.
                    if old_val in val_to_idx:
                        # Find new index
                        # Since we pop from front, we can just rebuild?
                        # Or update all?
                        # Rebuild is O(200).
                        val_to_idx = {v: i for i, v in enumerate(recent_vals)}
                    
                    # Also update window_start if it was pointing to old_pos?
                    # window_start is an absolute number.
                    # If window_start == old_pos, it's now invalid?
                    # No, window_start is the start of the valid window.
                    # If we popped an element that was < window_start, it's fine.
                    # If we popped an element that was >= window_start, 
                    # it means the window shrank?
                    # No, the window is defined by `window_start`.
                    # The elements in `recent_vals` are the last 200.
                    # If `window_start` is very old, it might be < old_pos.
                    # But we only use `val_to_idx` to find collisions *within* the recent window.
                    # If a collision is with an element that was popped, it's < window_start?
                    # Not necessarily.
                    # If the run is length 100, and we pop the 101st element from the back?
                    # No, we pop from the front (oldest).
                    # So if the run is 100, the oldest element in the run is at index len(recent)-100.
                    # We only pop if len > 200.
                    # So we keep the last 200.
                    # Any collision must be with an element in the last 200.
                    # So this is safe.
                
                curr_k = pos - window_start + 1
                if curr_k > max_k:
                    max_k = curr_k
                    max_start = window_start
                    if max_k >= 200:
                        return max_k, max_start
            
            current_L = current_R + 1
            
        return max_k, max_start

    # Scan small range
    k1, s1 = scan_range_optimized(1, 10**6)
    if k1 > best_k:
        best_k = k1
        best_s = s1
        
    # Scan random large range
    if time.time() - start_time < 100:
        # Pick a random start
        # Avoid 0
        rand_start = random.randint(10**11, 10**12 - 10**8)
        k2, s2 = scan_range_optimized(rand_start, rand_start + 10**8 - 1)
        if k2 > best_k:
            best_k = k2
            best_s = s2
            
    return {'start': best_s, 'length': best_k}

