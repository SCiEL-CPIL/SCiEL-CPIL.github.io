import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    rng = np.random.default_rng(42)
    h = rng.normal(0, 1, N)
    s = np.sum(h)
    if abs(s) < 1e-9:
        h = np.ones(N)
    else:
        h = h / s

    c = np.convolve(h, h)
    max_c = np.max(c)
    best_h = h.copy()
    best_max_c = max_c

    # Precompute shift vectors? No, they depend on i.
    # But we can compute the shift vector on the fly.
    
    step_size = 0.1
    
    for iteration in range(20000):
        if time.time() > deadline:
            break
            
        improved = False
        # Shuffle indices to avoid bias
        indices = rng.permutation(N)
        
        for i in indices:
            if time.time() > deadline:
                break
            
            h_orig = h[i]
            
            # Precompute the shift vector for this i
            # shift_vec has h at indices i to i+N-1
            # We need to update c.
            # Instead of allocating a new array every time, we can use a temporary buffer?
            # Or just do the slice addition.
            
            # Try +step
            h[i] = h_orig + step_size
            # Update c
            # c += 2*step * shift_vec
            # shift_vec is h (with new value? No, the cross term uses the OLD h?
            # (h + d e_i) * h = h*h + d e_i * h
            # e_i * h uses the OLD h.
            # So we must use h BEFORE the change for the shift vector.
            # But we changed h[i].
            # So we need to use h with h[i] = h_orig for the shift vector.
            
            # This is tricky.
            # Let's compute the shift vector using h_orig.
            # shift_vec = zeros
            # shift_vec[i:i+N] = h_current (where h[i] is h_orig)
            
            # Since we modified h[i], we need to be careful.
            # Let's restore h[i] to h_orig, compute shift, then apply.
            
            # Actually, the cross term is d * (e_i * h_old + h_old * e_i)
            # = 2 * d * (h_old shifted right by i)
            
            # So:
            # 1. Save h[i]
            # 2. Compute shift_vec from current h (which is h_old)
            # 3. Apply +d to h[i]
            # 4. Update c using shift_vec
            # 5. Check max
            # 6. If worse, revert c and h[i]
            
            # But if we try -d, we need to be careful with the order.
            
            # Let's do:
            # 1. h[i] = h_orig
            # 2. shift_vec = zeros(2N-1)
            # 3. shift_vec[i:i+N] = h
            # 4. Try +d:
            #    h[i] += d
            #    c += 2*d*shift_vec + d^2*delta(2i)
            #    max_plus = max(c)
            # 5. Try -d:
            #    # Revert +d first if we want to test -d from original
            #    # Or test -d from original directly.
            #    c -= 2*d*shift_vec + d^2*delta(2i)
            #    h[i] -= 2*d # back to h_orig
            #    h[i] -= d
            #    c -= 2*d*shift_vec - d^2*delta(2i) # Note: d is negative here?
            #    # If we use d_pos = step, d_neg = -step.
            #    # Update for d: c_new = c_old + 2*d*shift + d^2*delta
            #    # So for -d: c_new = c_old - 2*d*shift + d^2*delta
            
            # Let's just implement a helper to apply step d.
            
            def apply_step(d):
                nonlocal c, h
                # Assumes h[i] is currently h_orig
                # 1. Update h
                h[i] += d
                # 2. Update c
                # shift_vec is based on h_old
                c[i:i+N] += 2.0 * d * h[i-N:i] # This is wrong indexing.
                
            # This inline logic is error prone.
            # Let's just recompute c for each candidate?
            # O(N^2) per candidate.
            # 400 indices * 2 candidates * 1000 iterations = 800k recomputations.
            # 800k * 160k ops = 128 billion. Too slow.
            
            # We MUST use incremental update.
            
            # Correct Incremental Update:
            # Let h_old be the current h.
            # Let d be the step.
            # h_new = h_old + d * e_i
            # c_new = c_old + 2*d * (h_old shifted right by i) + d^2 * delta(2i)
            # Note: h_old shifted right by i means:
            # vec = zeros(2N-1)
            # vec[i:i+N] = h_old
            
            # So:
            h[i] = h_orig # Ensure we start from original
            
            # Create shift vector
            # We can avoid allocation by using slices if we are careful, 
            # but c is a single array.
            # We need to add 2*d*h_old to c[i:i+N].
            # But h_old[i] is h_orig.
            # The slice c[i:i+N] corresponds to the convolution terms involving e_i.
            
            # Step 1: Apply +d
            d = step_size
            # Update c
            c[i:i+N] += 2.0 * d * h
            c[2*i] += d * d
            # Update h
            h[i] += d
            
            max_plus = np.max(c)
            
            # Step 2: Apply -d (from original state)
            # First revert +d
            c[i:i+N] -= 2.0 * d * h
            c[2*i] -= d * d
            h[i] = h_orig
            
            # Now apply -d
            d_neg = -step_size
            c[i:i+N] += 2.0 * d_neg * h
            c[2*i] += d_neg * d_neg
            h[i] += d_neg
            
            max_minus = np.max(c)
            
            # Choose best
            if max_plus < max_minus and max_plus < max_c:
                # Keep +d
                # We are currently at -d state. Need to go from -d to +d?
                # No, we reverted to original, then applied -d.
                # We want +d.
                # Revert -d
                c[i:i+N] -= 2.0 * d_neg * h
                c[2*i] -= d_neg * d_neg
                h[i] = h_orig
                
                # Apply +d
                c[i:i+N] += 2.0 * d * h
                c[2*i] += d * d
                h[i] += d
                
                max_c = max_plus
                improved = True
            elif max_minus < max_c:
                # Keep -d
                # We are currently at -d state.
                # Just keep it.
                max_c = max_minus
                improved = True
            else:
                # Revert to original
                c[i:i+N] -= 2.0 * d_neg * h
                c[2*i] -= d_neg * d_neg
                h[i] = h_orig

        if not improved:
            step_size *= 0.8
            if step_size < 1e-6:
                break

    return h
