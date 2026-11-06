import time
import math
import numpy as np
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit

    N_COEFFS = 24
    sqrt_N = math.sqrt(N_COEFFS)
    
    # Known good starting points
    # Rudin-Shapiro sequence (length 24)
    def get_rs_poly(n):
        p, q = [1], [1]
        while len(p) < n:
            p, q = p + q, p + [-c for c in q]
        return p[:n]
    
    rs_poly = get_rs_poly(N_COEFFS)
    
    # A few other known low-autocorrelation sequences
    known_solutions = [
        rs_poly,
        [1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1, 1, 1],
        [1, -1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1],
        [1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1],
    ]

    # --- Fast Evaluation Machinery ---
    # We use FFT for coarse sampling and golden-section for refinement.
    # N_fft must be large enough to resolve the peaks.
    # The polynomial has degree 23.
    # P(z) = sum c_k z^k.
    # We want max |P(z)| for |z|=1.
    
    N_FFT = 4096  # Higher resolution for better coarse peak location
    
    # Precompute nothing, just use np.fft.fft on padded coeffs.
    # Note: P(e^{i theta_j}) where theta_j = 2*pi*j/N.
    # fft(c)[j] = sum_{k=0}^{N-1} c_k e^{-i 2 pi k j / N}
    # If we set c_k = coeffs[k] for k < 24 and 0 otherwise,
    # then fft(c)[j] = sum_{k=0}^{23} coeffs[k] e^{-i 2 pi k j / N}
    # This is P(e^{-i 2 pi j / N}).
    # |P(e^{-i theta})| = |P(e^{i theta})| due to conjugate symmetry if coeffs are real?
    # P(z) = sum c_k z^k. P(1/z) = sum c_k z^{-k}.
    # |P(z)|^2 = P(z) P(1/z) for |z|=1 if c_k real?
    # Actually, |P(e^{i theta})|^2 = (sum c_k e^{ik theta}) (sum c_l e^{-il theta})
    # fft gives us values at negative frequencies. The magnitudes are the same.
    
    def evaluate_fast(coeffs):
        # coeffs: list of 24 ints (+1/-1)
        # 1. Coarse sampling via FFT
        c_padded = np.zeros(N_FFT, dtype=np.float64)
        c_padded[:N_COEFFS] = coeffs
        # Use real FFT? No, complex is fine and fast.
        # Actually, since coeffs are real, we can use rfft?
        # But we need the full circle.
        # np.fft.fft is fast enough.
        vals = np.fft.fft(c_padded)
        mags = np.abs(vals)
        
        # Find global max index
        max_idx = np.argmax(mags)
        coarse_max = mags[max_idx]
        
        # 2. Refine the peak
        # The peak is near theta = 2*pi*max_idx/N_FFT
        # We refine in a window around this theta.
        # Window width: 2*pi/N_FFT is the bin spacing.
        # The true peak is within +/- pi/N_FFT of the bin center (interpolated).
        # Actually, for a smooth function, the peak is between the two highest bins?
        # Or just refine around the max bin.
        
        theta_max = 2 * np.pi * max_idx / N_FFT
        
        # Handle wrap-around for the interval
        # Interval: [theta_max - pi/N_FFT, theta_max + pi/N_FFT]
        # But if theta_max is near 0 or 2pi, we might cross the boundary.
        # Golden section search works on [a, b].
        # Let's just define the local function and search in a small interval.
        
        # Local evaluation function
        # P(z) = sum c_k z^k
        # To make it faster, we can precompute powers?
        # But theta changes.
        # Direct sum is O(24). 20-30 iterations of GS is ~600 ops. Very fast.
        
        def local_eval(theta):
            # theta is a float
            # Compute P(e^{i theta})
            # Use Horner's method? Or direct sum?
            # Direct sum with np.exp is slow due to overhead.
            # Manual loop is faster for small N.
            z = complex(math.cos(theta), math.sin(theta))
            val = 0.0 + 0.0j
            z_power = 1.0 + 0.0j
            for k in range(N_COEFFS):
                val += coeffs[k] * z_power
                z_power *= z
            return abs(val)
        
        # Golden Section Search
        # We want to maximize local_eval.
        a = theta_max - math.pi / N_FFT
        b = theta_max + math.pi / N_FFT
        
        # Ensure a, b are within [0, 2pi) for consistency? Not strictly necessary for trig,
        # but keeps numbers small.
        # Actually, cos/sin handle any real.
        
        gr = (math.sqrt(5) - 1) / 2
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        
        fc = local_eval(c)
        fd = local_eval(d)
        
        # 30 iterations of GS gives (b-a) * (2/phi)^30 ~ (b-a) * 1e-6
        # Initial width ~ 1.5e-3. Final width ~ 1.5e-9.
        # This is sufficient for 1e-9 accuracy.
        for _ in range(30):
            if fc < fd:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = local_eval(d)
            else:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = local_eval(c)
        
        refined_max = max(fc, fd)
        
        # Check if the second highest peak is close
        # If so, refine it too.
        # Sort indices by magnitude
        # This is O(N log N) which is 4096 log 4096 ~ 50k ops.
        # Doing this every eval might be slow if we do many evals.
        # Let's only do it if the top peak is not significantly higher than the second.
        # Or just check a few neighbors?
        # The task says "golden-section refinement of the highest peaks".
        # Usually, for these polynomials, the max is unique or very close.
        # Let's refine the top 2 peaks if they are within 1% of each other.
        
        # Get second max index
        # Instead of full sort, find max, mask it, find max again.
        max_idx2 = max_idx
        if max_idx > 0:
            mags_copy = mags.copy()
            mags_copy[max_idx] = 0
            max_idx2 = np.argmax(mags_copy)
        else:
            mags_copy = mags.copy()
            mags_copy[0] = 0
            max_idx2 = np.argmax(mags_copy)
            
        if mags[max_idx2] > 0.98 * coarse_max:
            theta_max2 = 2 * np.pi * max_idx2 / N_FFT
            a2 = theta_max2 - math.pi / N_FFT
            b2 = theta_max2 + math.pi / N_FFT
            c2 = b2 - gr * (b2 - a2)
            d2 = a2 + gr * (b2 - a2)
            fc2 = local_eval(c2)
            fd2 = local_eval(d2)
            
            for _ in range(30):
                if fc2 < fd2:
                    a2 = c2
                    c2 = d2
                    fc2 = fd2
                    d2 = a2 + gr * (b2 - a2)
                    fd2 = local_eval(d2)
                else:
                    b2 = d2
                    d2 = c2
                    fd2 = fc2
                    c2 = b2 - gr * (b2 - a2)
                    fc2 = local_eval(c2)
            
            refined_max = max(refined_max, max(fc2, fd2))
            
        return refined_max

    # --- Optimization Loop ---
    
    # 1. Evaluate all known seeds
    best_coeffs = None
    best_score = float('inf')
    
    for sol in known_solutions:
        score = evaluate_fast(sol) / sqrt_N
        if score < best_score:
            best_score = score
            best_coeffs = sol[:]
            
    current_coeffs = best_coeffs[:]
    current_score = best_score
    
    # 2. Local Search (Hill Climbing) with Kicks
    # The directive suggests a Genetic Algorithm strategy, but the current program
    # is a local search. The "strongest idea" from inspirations might be the robust
    # evaluation or the specific kicking strategy.
    # However, the prompt asks to COMBINE.
    # Inspiration 1 uses local search. Inspiration 2 uses known seeds.
    # Current uses known seeds + local search.
    # Let's stick to the local search family as per REFINE directive ("KEEP this design").
    # We will improve the local search by:
    # 1. Using a better kicking strategy (random restarts vs kicks).
    # 2. Possibly parallelizing? No, standard library only.
    # 3. Optimizing the evaluation speed.
    
    # Optimization: Precompute exp(i*k*theta) for the local eval?
    # The local eval calls local_eval many times.
    # local_eval does 24 complex mults.
    # We can speed up local_eval by using numpy if we vectorize the angles?
    # But GS search is sequential.
    
    # Let's try to optimize the inner loop of local_eval.
    # Instead of complex multiplication, we can use real/imag parts?
    # Or just rely on Python's complex speed.
    
    # A better evaluation for the local search:
    # We don't need 1e-9 precision for the *search* steps, only for the final score?
    # The task says "accuracy about 1e-9" for the score.
    # But during search, we just need to know if A < B.
    # If we use a coarser refinement (e.g. 10 iterations of GS), it's faster.
    # Let's make the refinement depth adaptive or fixed to a faster value during search,
    # and use high precision only for the final candidate check?
    # No, the evaluator uses its own high precision.
    # But if our search picks a point that is locally optimal at low precision but not high,
    # we might miss the true optimum.
    # However, the landscape is likely smooth enough that low-precision local search
    # finds the correct basin.
    
    # Let's create a faster evaluator for the search loop.
    def evaluate_fast_search(coeffs):
        # Use smaller FFT? No, FFT is fast.
        # Use fewer GS iterations?
        # Let's use 15 iterations of GS.
        # (2/phi)^15 ~ 1e-3.
        # Initial width 1e-3. Final width 1e-6.
        # This should be sufficient to distinguish close values.
        
        c_padded = np.zeros(N_FFT, dtype=np.float64)
        c_padded[:N_COEFFS] = coeffs
        vals = np.fft.fft(c_padded)
        mags = np.abs(vals)
        
        max_idx = np.argmax(mags)
        theta_max = 2 * np.pi * max_idx / N_FFT
        
        def local_eval_fast(theta):
            z = complex(math.cos(theta), math.sin(theta))
            val = 0.0 + 0.0j
            z_power = 1.0 + 0.0j
            for k in range(N_COEFFS):
                val += coeffs[k] * z_power
                z_power *= z
            return abs(val)
        
        a = theta_max - math.pi / N_FFT
        b = theta_max + math.pi / N_FFT
        gr = (math.sqrt(5) - 1) / 2
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = local_eval_fast(c)
        fd = local_eval_fast(d)
        
        for _ in range(15):
            if fc < fd:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = local_eval_fast(d)
            else:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = local_eval_fast(c)
        
        refined_max = max(fc, fd)
        
        # Check second peak
        max_idx2 = max_idx
        if max_idx > 0:
            mags_copy = mags.copy()
            mags_copy[max_idx] = 0
            max_idx2 = np.argmax(mags_copy)
        else:
            mags_copy = mags.copy()
            mags_copy[0] = 0
            max_idx2 = np.argmax(mags_copy)
            
        if mags[max_idx2] > 0.98 * mags[max_idx]:
            theta_max2 = 2 * np.pi * max_idx2 / N_FFT
            a2 = theta_max2 - math.pi / N_FFT
            b2 = theta_max2 + math.pi / N_FFT
            c2 = b2 - gr * (b2 - a2)
            d2 = a2 + gr * (b2 - a2)
            fc2 = local_eval_fast(c2)
            fd2 = local_eval_fast(d2)
            
            for _ in range(15):
                if fc2 < fd2:
                    a2 = c2
                    c2 = d2
                    fc2 = fd2
                    d2 = a2 + gr * (b2 - a2)
                    fd2 = local_eval_fast(d2)
                else:
                    b2 = d2
                    d2 = c2
                    fd2 = fc2
                    c2 = b2 - gr * (b2 - a2)
                    fc2 = local_eval_fast(c2)
            
            refined_max = max(refined_max, max(fc2, fd2))
            
        return refined_max

    # Re-evaluate seeds with fast search to find best start for search?
    # No, just use the best from the precise eval as the start.
    # But maybe the fast eval finds a better start?
    # Let's just run the search.
    
    indices = list(range(N_COEFFS))
    
    # Stagnation counter
    stagnation_count = 0
    
    while time.time() - start_time < deadline:
        improved = False
        random.shuffle(indices)
        
        for i in indices:
            new_coeffs = current_coeffs[:]
            new_coeffs[i] = -new_coeffs[i]
            new_score = evaluate_fast_search(new_coeffs) / sqrt_N
            
            # Accept if better (or equal with some probability? No, strict improvement)
            if new_score < current_score - 1e-10:
                current_coeffs = new_coeffs
                current_score = new_score
                improved = True
                stagnation_count = 0
                
                # Update best
                # We don't want to do precise eval every step.
                # We'll do precise eval at the end of the loop or periodically.
                # For now, track the "search score".
                if current_score < best_score:
                    # Tentative best
                    pass
        
        if not improved:
            stagnation_count += 1
            
            # If stuck, try a kick
            # Kick size: 1, 2, or 3
            kick_size = 1
            if stagnation_count > 5:
                kick_size = random.choice([2, 3])
            elif stagnation_count > 2:
                kick_size = random.choice([1, 2])
            
            kick_indices = random.sample(indices, kick_size)
            for k in kick_indices:
                current_coeffs[k] = -current_coeffs[k]
            current_score = evaluate_fast_search(current_coeffs) / sqrt_N
            
            # If the kick made it much worse, revert?
            # No, simulated annealing style: accept worse moves sometimes to escape.
            # But we are doing hill climbing with kicks.
            # If the kick is bad, we are in a bad spot.
            # Let's just continue.
            
            # Check time
            if time.time() - start_time > deadline - 10:
                break
    
    # Final Polish:
    # Run a few rounds of precise evaluation and hill climbing from the best found
    # to ensure we are at a local minimum for the precise metric.
    
    # Get the current best candidate
    candidate = current_coeffs[:]
    
    # Perform precise hill climbing from this candidate
    precise_improved = True
    while precise_improved and time.time() - start_time < deadline - 5:
        precise_improved = False
        random.shuffle(indices)
        for i in indices:
            new_coeffs = candidate[:]
            new_coeffs[i] = -new_coeffs[i]
            new_score = evaluate_fast(new_coeffs) / sqrt_N
            
            if new_score < best_score - 1e-12:
                candidate = new_coeffs
                best_score = new_score
                best_coeffs = candidate[:]
                precise_improved = True

    return best_coeffs
# EVOLVE-BLOCK-END
