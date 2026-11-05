import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # The problem asks for 24 coefficients c_0...c_23 in {+1, -1}.
    # We want to minimize max_{|z|=1} |P(z)| where P(z) = sum c_k z^k.
    
    # Symmetries:
    # 1. Global sign: P(z) -> -P(z) doesn't change |P(z)|. We can fix c_0 = 1.
    # 2. Time reversal: P(z) -> z^{23} P(1/z) = sum c_{23-k} z^k.
    #    |z^{23} P(1/z)| = |P(1/z)|. On |z|=1, 1/z = conj(z).
    #    |P(conj(z))| = conj(P(z)) if coefficients are real. So |P(conj(z))| = |P(z)|.
    #    Thus, the sup norm of P is the same as the sup norm of its time-reversed version.
    #    We can fix c_23 = 1 (or more generally, break the symmetry).
    #    Actually, if we fix c_0=1, we can still have c_23 = -1 or 1.
    #    If we fix c_0=1 and c_23=1, we cover half the orbits?
    #    Let's be careful. The group acting on the set of polynomials is {I, R} x {1, -1}.
    #    Size 4.
    #    Total polynomials: 2^24.
    #    Orbits: 2^24 / 4 = 2^22 = 4,194,304.
    #    This is too many to brute force in 120s with high accuracy evaluation.
    #    4 million evaluations * (cost of sup norm).
    #    Sup norm with M=2^15 points: 32768 points.
    #    4M * 32k * 24 flops = 3 * 10^12 flops. Too slow.
    
    # However, the task description mentions: "The exhaustive optimum is 1.2583758. The exact minimum for 24 coefficients (degree 23) is recovered; Odlyzko's online tables list the optimal polynomials."
    # This implies the search space might be smaller or there's a specific known set.
    # Wait, "exhaustive optimum" might refer to exhaustive search over a reduced set or just that the global optimum is known.
    # But 4 million is actually feasible if the evaluation is fast enough?
    # 120 seconds / 4,194,304 evaluations = ~28 microseconds per evaluation.
    # Can we do a sup-norm check in 28 microseconds?
    # FFT of length 32768 takes ~32768 * 15 * 5 ops ~ 2.5M ops.
    # 2.5M ops / 10^9 ops/sec = 2.5 ms.
    # 4M evals * 2.5 ms = 10,000 seconds. Way too slow.
    
    # So we cannot exhaustively search all orbits.
    
    # Let's re-read the "INSPIRATIONS".
    # Inspiration 3 (score 1.1198) used "targeted local search that minimizes the maximum of the top few spectral peaks simultaneously using a greedy 'peak-kill' heuristic".
    # Current program (score 1.3784) uses local search with 1-flip and 2-flip.
    # Inspiration 2 (score 1.2406) suggests "materially different approach".
    
    # Is there a smaller search space?
    # Maybe we can use the fact that the optimal polynomial is known?
    # The prompt says "Odlyzko's online tables list the optimal polynomials."
    # If I can find the specific optimal polynomial for N=24, I can just return it.
    # The score is 1.2583758.
    # Let's look for "Odlyzko tables plus-minus one polynomials 24 coefficients".
    # Since I don't have internet access, I must rely on the code or derive it.
    # However, the "exhaustive optimum" phrasing is tricky.
    
    # Let's try a smarter search.
    # The current program is stuck at 1.3784.
    # The best known is 1.258.
    # A greedy "peak-kill" strategy might work better.
    # 
    # Strategy:
    # 1. Start with a good initial guess (e.g., Golay-Rudin-Shapiro).
    # 2. Compute the sup norm and identify the frequency bins where the magnitude is maximal.
    # 3. Instead of trying all single flips, identify which flips would most significantly reduce the magnitude at these specific peak frequencies.
    # 4. The change in F[k] when flipping c_n is -2 * c_n * exp(-i * 2*pi * k * n / M).
    # 5. We want to minimize |F_new[k]| for the peak k.
    # 6. This is a local heuristic.
    
    # Let's implement a more aggressive local search:
    # 1. High-resolution FFT (M=2^16 or 2^17) to accurately locate peaks.
    # 2. Identify top K peaks.
    # 3. For each coefficient n, calculate the "potential" of flipping it to reduce the sum of squares of the top K peaks (or the max of them).
    # 4. Flip the coefficient with the best potential.
    # 5. Repeat until no improvement or time limit.
    # 6. Use random restarts to escape local minima.
    
    # To beat 1.3784, we need to get closer to 1.258.
    # The current code uses M=2^15. Let's use M=2^16 for better peak resolution.
    # And instead of brute forcing 2-flips (528 combos), we can use a greedy step.
    
    # Let's refine the "peak-kill" idea.
    # At each step, we have current coeffs c.
    # Compute F = DFT(c).
    # Find index k* of max |F[k*]|.
    # We want to flip c_n to minimize |F_new[k*]|.
    # F_new[k*] = F[k*] - 2*c_n*D[k*, n].
    # Let F[k*] = A * exp(i*theta). D[k*, n] = exp(-i*phi).
    # F_new = A*exp(i*theta) - 2*c_n*exp(-i*phi).
    # We want to minimize |F_new|.
    # This is equivalent to choosing c_n (which is +/- 1) such that the vector 2*c_n*exp(-i*phi) points most opposite to F[k*].
    # Actually, we are constrained to flip c_n, so the change is fixed in magnitude (2) and direction depends on n and current c_n.
    # For a fixed n, the new value is determined. We just pick the n that gives the smallest resulting |F_new[k*]|.
    # But reducing the single global peak might increase other peaks.
    # So we should look at the top K peaks.
    # Objective: minimize max( |F_new[k1]|, ..., |F_new[kK]| ).
    # Greedy: Pick n that minimizes this max.
    
    # Complexity:
    # 1. FFT: O(M log M).
    # 2. Find top K peaks: O(M).
    # 3. For each n (24 options), compute new values at K peaks: O(24*K).
    #    Total per step: O(M log M + 24*K).
    # With M=65536, K=10, this is very fast.
    # We can do thousands of steps in 120s.
    
    # Let's implement this.
    
    M = 2**16  # 65536 points
    k = np.arange(M)
    n_idx = np.arange(24)
    # Precompute D matrix columns for the 24 coefficients?
    # D[:, n] = exp(-2j * pi * k * n / M)
    # Memory: 65536 * 24 * 8 bytes = 12 MB. Fine.
    # Time to compute: 65536 * 24 exps. Fast.
    phases = np.outer(k, n_idx)
    D = np.exp(-2j * np.pi * phases / M)
    
    def get_supnorm_and_top_peaks(coeffs, K=10):
        F = D @ coeffs
        abs_F = np.abs(F)
        max_val = np.max(abs_F)
        # Get indices of top K peaks
        # np.argpartition is O(M)
        top_indices = np.argpartition(abs_F, -K)[-K:]
        return max_val, top_indices, abs_F

    def greedy_step(coeffs, K=10):
        max_val, top_indices, _ = get_supnorm_and_top_peaks(coeffs, K)
        
        best_n = -1
        best_new_max = max_val + 1e-9
        current_F = D @ coeffs
        
        # For each coefficient n, simulate flip
        for n in range(24):
            # F_new = F - 2 * c_n * D[:, n]
            # We only need to check the values at top_indices
            # F_new[top_indices] = current_F[top_indices] - 2 * coeffs[n] * D[top_indices, n]
            delta = 2.0 * coeffs[n] * D[top_indices, n]
            F_new_peaks = current_F[top_indices] - delta
            new_max = np.max(np.abs(F_new_peaks))
            
            if new_max < best_new_max:
                best_new_max = new_max
                best_n = n
                
        return best_n, best_new_max

    # Initial guess: Golay-Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    current_c = np.array(p[:24], dtype=float)
    
    best_c = current_c.copy()
    best_val, _, _ = get_supnorm_and_top_peaks(current_c, 10)
    
    import random
    no_improvement = 0
    while time.time() < deadline:
        n_flip, new_val = greedy_step(current_c, K=10)
        
        if n_flip != -1 and new_val < best_val:
            current_c[n_flip] *= -1.0
            best_c = current_c.copy()
            best_val = new_val
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement > 5:
                # Random restart / perturbation
                # Flip 3-5 random bits
                for _ in range(random.randint(3, 5)):
                    idx = random.randint(0, 23)
                    current_c[idx] *= -1.0
                val, _, _ = get_supnorm_and_top_peaks(current_c, 10)
                if val < best_val:
                    best_c = current_c.copy()
                    best_val = val
                no_improvement = 0
                
    return [int(x) for x in best_c]
