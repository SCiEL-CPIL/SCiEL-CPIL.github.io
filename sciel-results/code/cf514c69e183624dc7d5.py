import time
import numpy as np

N = 400
DEADLINE = 140.0  # safety margin before 150s limit

def _compute_conv(x):
    """Compute circular autoconvolution."""
    return np.fft.ifft(np.fft.fft(x) * np.fft.fft(x)).real

def _psr(x):
    """Compute Peak Sidelobe Ratio (max off-center abs / center)."""
    conv = _compute_conv(x)
    center = conv[0]
    if center <= 0:
        return float('inf')
    off_vals = np.concatenate([conv[1:], conv[:-1]])
    return np.max(np.abs(off_vals)) / center

def _lfsr_sequence(n, poly_bits, init):
    """Generate n bits from an LFSR."""
    seq = []
    state = init
    deg = poly_bits.bit_length() - 1
    for _ in range(n):
        seq.append(state & 1)
        feedback = 0
        for i in range(deg + 1):
            if (poly_bits >> i) & 1:
                feedback ^= (state >> i) & 1
        state = (state >> 1) | (feedback << deg)
    return seq

def _local_search(x, max_time):
    """
    Local search using O(N) convolution update for single bit flips.
    Tries to minimize the maximum absolute off-center convolution value.
    """
    x = x.copy()
    n = len(x)
    conv = _compute_conv(x)
    current_max = np.max(np.abs(np.concatenate([conv[1:], conv[:-1]])))
    start = time.time()
    
    # Precompute indices for fast access
    # We will iterate through indices and try flipping
    # To speed up, we can try all flips in a pass, keep the best, and repeat
    # Or greedy: flip if it reduces current_max
    
    improved = True
    no_improve_count = 0
    
    while improved and (time.time() - start) < max_time:
        improved = False
        for i in range(n):
            if (time.time() - start) > max_time:
                break
            
            # Calculate change in conv if x[i] flips
            # delta_c[k] = -2 * x[i] * (x[(i-k)%n] + x[(i+k)%n]) for k != 0
            # We need to check if this reduces the max.
            # Instead of calculating all, we can just do the flip, update conv, check max.
            # O(N) update:
            old_val = x[i]
            x[i] = -old_val
            
            # Update conv
            # For each k, conv[k] changes by -2 * new_x[i] * (x[(i-k)%n] + x[(i+k)%n])? 
            # Wait, the formula derived was:
            # c'_k = c_k - 2 * x_i_new * (x_{i-k} + x_{i+k}) + 2 * x_i_old * (x_{i-k} + x_{i+k})?
            # Let's re-derive carefully.
            # c_k = sum_j x_j x_{j-k}
            # Terms involving x_i:
            # j=i: x_i x_{i-k}
            # j-k=i => j=i+k: x_{i+k} x_i
            # So c_k has terms x_i x_{i-k} + x_{i+k} x_i.
            # If x_i becomes -x_i:
            # New terms: -x_i x_{i-k} - x_{i+k} x_i.
            # Change: -2 x_i (x_{i-k} + x_{i+k}).
            # Note: x_i here is the OLD value.
            
            # Vectorized update
            # We need x[(i-k)%n] and x[(i+k)%n] for all k.
            # Let's create the index arrays once outside the loop? 
            # Or just compute the delta vector.
            
            # Indices for x[(i-k)%n]
            idx1 = (i - np.arange(n)) % n
            # Indices for x[(i+k)%n]
            idx2 = (i + np.arange(n)) % n
            
            delta = -2.0 * old_val * (x[idx1] + x[idx2])
            
            # Apply delta to conv
            conv_new = conv + delta
            
            # Check new max
            # conv_new[0] should be N (unchanged)
            off_new = np.concatenate([conv_new[1:], conv_new[:-1]])
            new_max = np.max(np.abs(off_new))
            
            if new_max < current_max - 1e-9:
                # Accept flip
                conv = conv_new
                current_max = new_max
                improved = True
                no_improve_count = 0
            else:
                # Revert flip
                x[i] = old_val
                no_improve_count += 1
                
        if no_improve_count > n:
            # If we went through a whole pass without improvement, 
            # we might be stuck. Try a random perturbation or break.
            # For now, break to save time, or try a few random flips.
            # Let's try a few random flips to escape local minimum
            for _ in range(10):
                if (time.time() - start) > max_time:
                    break
                ri = np.random.randint(n)
                old_val = x[ri]
                x[ri] = -old_val
                idx1 = (ri - np.arange(n)) % n
                idx2 = (ri + np.arange(n)) % n
                delta = -2.0 * old_val * (x[idx1] + x[idx2])
                conv_new = conv + delta
                off_new = np.concatenate([conv_new[1:], conv_new[:-1]])
                new_max = np.max(np.abs(off_new))
                if new_max < current_max - 1e-9:
                    conv = conv_new
                    current_max = new_max
                    improved = True
            if not improved:
                break
            no_improve_count = 0
            
    return x, current_max

def construct():
    t_start = time.time()
    
    best_x = None
    best_val = float('inf')
    
    # 1. Generate candidates
    candidates = []
    
    # LFSR sequences (m-sequences)
    # Degree 9 (511 length), Degree 10 (1023 length)
    # We take the first 400 bits.
    configs = [
        (0b100010001, 0x1FF),  # x^9+x^4+1
        (0b100100001, 0x1FF),  # x^9+x^5+1
        (0b101101001, 0x1FF),  # x^9+x^6+x^5+x^3+1
        (0b1000001001, 0x3FF), # x^10+x^3+1
        (0b1101010001, 0x3FF), # x^10+x^8+x^5+x^4+x^3+x^2+1
    ]
    
    for poly, init in configs:
        bits = _lfsr_sequence(N, poly, init)
        x = np.array([1.0 if b == 1 else -1.0 for b in bits])
        val = _psr(x)
        candidates.append((x, val))
        
        # Also try phase shifts of the LFSR sequence?
        # Since we are taking a fixed segment, shifting the init state is equivalent to phase shift.
        # We already have different inits in configs? No, mostly 0x1FF/0x3FF.
        # Let's add a few random inits for the best poly.
    
    # Add random sequences
    rng = np.random.RandomState(42)
    for _ in range(20):
        x = rng.choice([-1.0, 1.0], size=N)
        val = _psr(x)
        candidates.append((x, val))
        
    # Sort by initial PSR
    candidates.sort(key=lambda t: t[1])
    
    # 2. Refine the top candidates
    # We have ~25 candidates.
    # Time budget: 140s.
    # Let's allocate time dynamically.
    # Try to refine the best one for a while, then the next best if time remains.
    
    # We will run local search on the top K candidates.
    K = 5
    
    for idx in range(K):
        if time.time() - t_start > DEADLINE - 10:
            break
            
        x_init, val_init = candidates[idx]
        
        # Allocate time for this candidate
        # If it's the first one, give it more time.
        remaining = DEADLINE - (time.time() - t_start)
        if idx == 0:
            alloc_time = remaining * 0.5
        else:
            alloc_time = remaining * 0.15
            
        if alloc_time < 1.0:
            break
            
        x_ref, val_ref = _local_search(x_init, alloc_time)
        
        if val_ref < best_val:
            best_val = val_ref
            best_x = x_ref
            
        # If we found a very good value, we might stop early?
        # Theoretical min for N=400? 
        # Random is ~ sqrt(N) = 20.
        # M-sequences have peak 1, but truncated/wrapped might be higher.
        # Let's see what value we get.
        
    if best_x is None:
        return np.ones(N)
        
    return best_x
