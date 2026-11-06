def _vectorized_max_ap_sum(s, max_step=32):
    s = np.array(s, dtype=np.int8)
    max_abs = 0
    for d in range(1, max_step + 1):
        # Reshape into d rows, each row is a residue class
        # We need to handle the ragged end (last row might be shorter)
        # Simple approach: iterate rows
        for r in range(d):
            vals = s[r::d]
            if len(vals) < 2: continue
            # Vectorized Kadane for max and min
            # cur_max[i] = max(vals[i], cur_max[i-1] + vals[i])
            # We can compute this iteratively over the array, which is O(N/d) per d, total O(N log N)
            # This is still Python loop over the length of vals.
            # Better: use cumsum and sliding window?
            # Max subarray sum = max(prefix[j] - min(prefix[0..j-1]))
            # We can compute prefix sums vectorized.
            p = np.cumsum(vals)
            # min_prefix[i] = min(p[0], ..., p[i-1])
            # We need min of p[0..i-1] for each i.
            # This is a "running min" which is not directly vectorized in basic numpy without scipy.
            # But we can use a simple loop for the running min if the length is small?
            # Or just use the standard Kadane in a tight Python loop? 
            # The previous code did this and was slow because it did it for ALL d up to 256.
            # If we limit to d=1..16, and use numpy for the cumsum part, it might be faster.
            
            # Actually, a pure Python Kadane on a list of 16 elements is very fast.
            # The bottleneck was d up to 256.
            # If we limit the *search* evaluation to d=1..16, it will be ~16x faster than before.
            # And we can do a full eval (d=1..256) at the end if time permits.
            
            # Let's stick to Python Kadane but only for d <= 16 in the inner loop.
            cur_max = vals[0]
            cur_min = vals[0]
            best_max = vals[0]
            best_min = vals[0]
            for v in vals[1:]:
                cur_max = max(v, cur_max + v)
                cur_min = min(v, cur_min + v)
                best_max = max(best_max, cur_max)
                best_min = min(best_min, cur_min)
            max_abs = max(max_abs, abs(best_max), abs(best_min))
    return max_abs
