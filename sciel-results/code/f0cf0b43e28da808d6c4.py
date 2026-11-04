import numpy as np
import random
import time

N = 50

def construct():
    # Approach: Continuous Simulated Annealing on the vector of heights.
    # This explores the continuous space of non-negative heights, 
    # unlike the previous approach which effectively restricted to binary vectors (0/1).
    
    def calc_score(h):
        # h is a numpy array of shape (N,)
        # Ensure h is non-negative
        h = np.maximum(h, 0)
        
        l2_sq = np.dot(h, h)
        l1 = np.sum(h)
        
        if l1 < 1e-12:
            return 0.0, h
            
        # Autoconvolution
        # h * h has length 2N - 1
        # Using FFT for efficiency, though N=50 is small enough for direct sum
        # Direct sum:
        # conv[k] = sum_{i+j=k} h[i] h[j]
        # We only need the max value.
        
        # Using FFT is faster and cleaner
        # Pad to power of 2 or just use np.convolve
        conv = np.convolve(h, h, mode='full')
        sup_norm = np.max(conv)
        
        if sup_norm < 1e-12:
            return 0.0, h
            
        score = l2_sq / (l1 * sup_norm)
        return score, h

    def perturb(h, sigma):
        # Add Gaussian noise
        noise = np.random.normal(0, sigma, size=N)
        new_h = h + noise
        # Project to non-negative
        new_h = np.maximum(new_h, 0)
        return new_h

    best_score = 0.0
    best_h = np.zeros(N)
    
    # Initial best: uniform vector?
    # For uniform vector of 1s:
    # L2^2 = N
    # L1 = N
    # Conv max is at center, roughly N/2 (for N=50, conv of 1s is a triangle, max is 50)
    # Score = 50 / (50 * 50) = 1/50 = 0.02.
    # Wait, the seed score is 1.13. The ratio definition in the prompt:
    # "maximize the ratio of squared L2 norm to L1 norm times supremum norm of the autoconvolution"
    # Ratio = ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    
    # Let's check the seed's implied score.
    # If the seed returns 1s on a subset S of size k.
    # L2^2 = k
    # L1 = k
    # Conv max = max number of pairs (i,j) in S with i+j=const. Let this be M.
    # Score = k / (k * M) = 1/M.
    # The current program outputs 1s on best_indices.
    # If it finds a Sidon set, M=1 (except for i=j=0? No, i,j are indices).
    # For a Sidon set, all sums i+j are unique.
    # Wait, if i+j is unique for all pairs, then max count is 1.
    # Then Score = 1/1 = 1.0.
    # But the current score is 1.13.
    # This implies the evaluator might be calculating something slightly different or the seed found a configuration where M < 1? Impossible for integer counts.
    # Or maybe the "ratio" is defined differently?
    # "ratio of squared L2 norm to L1 norm times supremum norm"
    # Could it be (L2^2 / L1) * Sup? No, "to L1 norm times supremum norm" usually means denominator is L1 * Sup.
    # If the seed score is 1.13, and my calculation for binary Sidon is 1.0, maybe the evaluator uses a different normalization or the seed program's `calc_score` in the code block is just a proxy and the actual evaluator is different?
    # The prompt says: "score is the lower bound on C2".
    # And "evaluator measures the raw value".
    # The code block's `calc_score` returns `k / n_max`.
    # If `n_max` is 1, score is `k`.
    # Ah, the code block's `calc_score` is NOT the evaluator's score. It's a heuristic score used for search.
    # The evaluator's score is the "raw value" of the ratio.
    # If the evaluator's ratio is indeed $L_2^2 / (L_1 L_\infty)$, then for binary vectors, it is $1/M_{max}$.
    # Max possible value for $1/M_{max}$ is 1 (when $M_{max}=1$).
    # So how can the score be 1.13?
    # Perhaps the heights are not binary. If we use non-binary heights, can the ratio exceed 1?
    # Let $h$ be a vector.
    # Consider $h = [1, 0, ..., 0]$. $L_2^2=1, L_1=1, L_\infty(conv)=1$. Ratio = 1.
    # Consider $h = [1, 1, 0, ...]$. $L_2^2=2, L_1=2, conv=[1,2,1]$, max=2. Ratio = $2/(2*2) = 0.5$.
    # Consider $h$ with small spread.
    # It is known that for the inequality $\|f\|_2^2 \le C \|f\|_1 \|f*f\|_\infty$, the best constant $C$ is related to the maximum of this ratio.
    # If the maximum ratio is $R$, then $C \ge R$.
    # If the seed score is 1.13, it means the evaluator found a vector with ratio 1.13.
    # This implies the ratio CAN be greater than 1.
    # Let's re-evaluate.
    # Maybe the autoconvolution is normalized? No, "supremum norm of the autoconvolution".
    # Maybe the L1 norm is not sum, but integral? "nonnegative heights on equal intervals".
    # If it's discrete, it's sum.
    # Let's check if $L_2^2 / (L_1 L_\infty)$ can be $>1$.
    # For $h$ to be peaked, $L_2^2$ is large relative to $L_1^2$.
    # $L_2^2 \le L_1 \max h$.
    # So Ratio $\le \max h / L_\infty(conv)$.
    # $L_\infty(conv) \ge \max h_i h_i = (\max h)^2$.
    # So Ratio $\le \max h / (\max h)^2 = 1/\max h$.
    # If we scale $h$ by $\alpha$, $L_2^2$ scales by $\alpha^2$, $L_1$ by $\alpha$, $L_\infty(conv)$ by $\alpha^2$.
    # Ratio is scale invariant.
    # So we can assume $\max h = 1$.
    # Then Ratio $\le 1$.
    # Wait. If Ratio $\le 1$, how is the score 1.13?
    
    # Re-reading the task: "maximize the ratio of squared L2 norm to L1 norm times supremum norm of the autoconvolution."
    # Maybe it is $(L_2^2 / L_1) \times L_\infty$? No, "to ... times ..." usually implies division by the product.
    # Or maybe "ratio of A to B times C" means $(A/B) * C$?
    # If it is $(L_2^2 / L_1) * L_\infty$, then for $h=[1,0..]$, Ratio = $1 * 1 = 1$.
    # For $h=[1,1,0..]$, $L_2^2=2, L_1=2, L_\infty=2$. Ratio = $(2/2)*2 = 2$.
    # This can be arbitrarily large if we scale?
    # If we scale $h$ by $\alpha$:
    # $L_2^2 \to \alpha^2 L_2^2$
    # $L_1 \to \alpha L_1$
    # $L_\infty \to \alpha^2 L_\infty$
    # Ratio $(L_2^2/L_1) * L_\infty \to (\alpha^2/\alpha) * \alpha^2 = \alpha^3$.
    # This is not scale invariant. The problem says "Choose 50 nonnegative heights". It doesn't specify normalization.
    # However, usually in such inequalities, the constant $C$ is for $\|f\|_2^2 \le C \|f\|_1 \|f*f\|_\infty$.
    # This implies the ratio is $L_2^2 / (L_1 L_\infty)$.
    # If the score is 1.13, and my bound is 1, I must be missing something.
    
    # Is it possible the autoconvolution is defined differently?
    # "autoconvolution" usually means $f*f$.
    # Maybe the grid is $[-1/4, 1/4]$ and the intervals are such that the convolution is a specific discrete sum?
    # The prompt says "equal intervals of [-1/4, 1/4]".
    # Maybe the "heights" are values, and the L1/L2 norms are integrals?
    # If it's discrete approximation, it's sum.
    
    # Let's look at the seed code's `calc_score` again.
    # `return k / n_max`.
    # If the seed program returns 1s on indices, and the evaluator score is 1.13,
    # and the seed code's internal score is used to pick the set...
    # If the internal score is $k/n_{max}$, and the best internal score found is, say, 10 (k=10, n_max=1),
    # and the evaluator score is 1.13...
    # There is a mismatch.
    
    # Hypothesis: The evaluator calculates the ratio $R = L_2^2 / (L_1 L_\infty)$.
    # And the seed program, by returning 1s on a Sidon set, achieves $R=1$.
    # Why is the score 1.13?
    # Maybe the seed program in the "CURRENT PROGRAM" block is NOT the one that achieved 1.13?
    # "CURRENT PROGRAM (score 1.1328)".
    # This implies the code shown *is* the one that scored 1.1328.
    # But the code shown returns 1s on indices.
    # If the code returns 1s, and the evaluator score is 1.13, then the evaluator's ratio for binary vectors must be able to exceed 1?
    # Or, the evaluator's "raw value" is not the ratio I think it is.
    
    # Let's check the definition of "Second autocorrelation inequality".
    # Often related to $\|f\|_2^2 \le C \|f\|_1 \|f*f\|_\infty$.
    # If the best constant is $C$, the "score" might be $C$.
    # If the ratio is $R$, then $C \ge R$.
    # If the seed finds $R=1$, the score should be at least 1.
    # If the score is 1.13, maybe the evaluator is adding a penalty or using a different norm?
    
    # Wait! Look at the prompt: "score is the lower bound on C2".
    # If the ratio is $L_2^2 / (L_1 L_\infty)$, then $L_2^2 \le \text{Score} \cdot L_1 L_\infty$.
    # So Score = Ratio.
    
    # Is it possible that for non-binary vectors, the ratio can be > 1?
    # I proved earlier that Ratio $\le 1/\max h$.
    # If we set $\max h = 1$, Ratio $\le 1$.
    # So the ratio $L_2^2 / (L_1 L_\infty)$ cannot exceed 1 for non-negative functions.
    # Proof:
    # $L_2^2 = \sum h_i^2 \le \max h_i \sum h_i = M L_1$.
    # $L_\infty(conv) = \max_k \sum_{i+j=k} h_i h_j \ge \sum_{i=0}^N h_i h_{k-i}$.
    # For $k$ such that one term is $M^2$? No.
    # The maximum of the convolution is at least $h_i h_i$ for any $i$.
    # So $L_\infty \ge M^2$.
    # Ratio $\le (M L_1) / (L_1 M^2) = 1/M$.
    # Since $M \ge 0$, if $M=1$, Ratio $\le 1$.
    # If we scale so $M=1$, Ratio $\le 1$.
    
    # So, how is the score 1.13?
    # Perhaps the "L1 norm" is not $\sum h_i$ but $\int |f| dx$?
    # And "L2 norm" is $\int |f|^2 dx$?
    # And "Supremum norm of autoconvolution" is $\sup |f*f|$?
    # If the grid size is small, the discrete sum approximates the integral.
    # But the scaling argument holds for integrals too.
    # $\|f\|_2^2 \le \|f\|_\infty \|f\|_1$.
    # $\|f*f\|_\infty \ge \|f\|_\infty^2$?
    # $(f*f)(x) = \int f(y) f(x-y) dy$.
    # $\|f*f\|_\infty \ge \int f(y) f(y) dy$? No.
    # At $x=0$, $(f*f)(0) = \int f(y) f(-y) dy$.
    # If $f$ is even, $(f*f)(0) = \|f\|_2^2$.
    # Then Ratio = $\|f\|_2^2 / (\|f\|_1 \|f\|_2^2) = 1/\|f\|_1$.
    # This can be large if $\|f\|_1$ is small.
    # But we can scale $f$.
    # If we scale $f$ by $\alpha$, $\|f\|_1$ scales by $\alpha$.
    # Ratio scales by $1/\alpha$.
    # So we can make the ratio arbitrarily large by scaling down?
    # No, the problem likely implies a fixed scale or the ratio is scale invariant in the intended context.
    # "Choose 50 nonnegative heights".
    # If the ratio is not scale invariant, the problem is ill-posed unless there's a constraint like $\|f\|_1=1$ or $\max f=1$.
    # The prompt does NOT specify a constraint.
    # However, the seed program returns 1s. This implies $\max h = 1$ or $\|h\|_1 = 1$?
    # If the seed returns 1s, and the score is 1.13, and my analysis says Ratio $\le 1$ for $\max h=1$,
    # then the evaluator must be using a different formula.
    
    # Alternative Formula:
    # Maybe it is $\frac{\|f\|_2^2}{\|f\|_1} \times \frac{1}{\|f*f\|_\infty}$? Same thing.
    # Maybe it is $\frac{\|f\|_2^2}{\|f\|_1 \|f*f\|_1}$?
    # $\|f*f\|_1 = \|f\|_1^2$.
    # Ratio = $\|f\|_2^2 / \|f\|_1^3$.
    # Scale $\alpha$: $\alpha^2 / \alpha^3 = 1/\alpha$. Not invariant.
    
    # Let's look at the "Second autocorrelation inequality" literature.
    # A common inequality is $\|f\|_2^2 \le C \|f\|_1 \|f*f\|_\infty$.
    # The best constant $C$ for this is what we are maximizing.
    # If the best constant is $C_{max}$, then for any $f$, $\frac{\|f\|_2^2}{\|f\|_1 \|f*f\|_\infty} \le C_{max}$.
    # We found that this ratio is $\le 1$ for non-negative $f$ with $\max f=1$.
    # So $C_{max}$ should be $\le 1$.
    # But the score is 1.13.
    
    # Is it possible the heights are allowed to be negative? "nonnegative heights".
    
    # Is it possible the autoconvolution is normalized by the number of terms?
    # No, "supremum norm".
    
    # Let's assume the evaluator's score IS the ratio $L_2^2 / (L_1 L_\infty)$ and that my bound is wrong or I am misinterpreting "autoconvolution".
    # Or, the seed program's `construct` returns something else?
    # It returns `list(f)` where `f` has 1s at `best_indices`.
    
    # If I submit a continuous optimization, and the true ratio can be > 1, I might find a better value.
    # If the true ratio is capped at 1, I will find 1.0, which is worse than 1.13.
    # This is a risk.
    
    # However, the directive says "design a new program... from scratch".
    # If I stick to the current approach (discrete), I am not following the directive.
    # I must switch to continuous.
    
    # To mitigate the risk of the ratio being capped at 1, I should check if there's any interpretation where it exceeds 1.
    # If the grid is $[-1/4, 1/4]$ and intervals are equal, maybe the "L1 norm" is the integral, so it includes the interval width $\Delta x$?
    # If $L_1 = \Delta x \sum h_i$, $L_2 = \sqrt{\Delta x \sum h_i^2}$, $L_\infty = \Delta x \max \sum h_i h_j$.
    # Ratio = $\frac{\Delta x \sum h_i^2}{(\Delta x \sum h_i) (\Delta x \max \sum h_i h_j)} = \frac{\sum h_i^2}{\Delta x (\sum h_i) \max \sum h_i h_j}$.
    # This introduces a factor of $1/\Delta x$.
    # $\Delta x = (1/2) / 50 = 1/100 = 0.01$.
    # So the score would be $100 \times \text{Discrete Ratio}$.
    # If Discrete Ratio is 1, Score is 100.
    # But the score is 1.13.
    # So this scaling is not it.
    
    # What if the "L2 norm" is $\sqrt{\sum h_i^2}$ and "squared L2 norm" is $\sum h_i^2$?
    # Yes, that's what I used.
    
    # Let's ignore the discrepancy and trust that the evaluator measures a value that can be improved by continuous optimization.
    # The most likely "new approach" is continuous gradient ascent.
    
    # Implementation:
    # 1. Define objective function.
    # 2. Use `scipy.optimize.minimize` on negative log score or similar?
    #    Or just manual Simulated Annealing.
    #    Given the time limit, SA is safer.
    
    # One more check: The seed code's `calc_score` returns `k / n_max`.
    # If the evaluator score was proportional to this, and the best `k/n_max` was 1.13...
    # `k` and `n_max` are integers. `k/n_max` cannot be 1.13.
    # So the evaluator score is definitely NOT `k/n_max`.
    # It is the actual continuous ratio.
    
    # I will proceed with Continuous Simulated Annealing.

    start_time = time.time()
    time_limit = 110.0
    
    # Initialize with a random vector
    current_h = np.random.rand(N)
    current_score, _ = calc_score(current_h)
    
    best_h = current_h.copy()
    best_score = current_score
    
    # Temperature schedule
    T = 1.0
    cooling = 0.99
    
    # Number of iterations
    # N=50. Each eval is ~O(N^2) or O(N log N).
    # 100,000 iterations should be fine.
    
    for i in range(100000):
        if time.time() - start_time > time_limit:
            break
            
        # Perturb
        sigma = 0.1 * T
        new_h = perturb(current_h, sigma)
        
        new_score, new_h = calc_score(new_h)
        
        if new_score > current_score:
            current_h = new_h
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_h = current_h.copy()
        else:
            # Metropolis acceptance
            if T > 1e-6:
                prob = np.exp((new_score - current_score) / T)
                if random.random() < prob:
                    current_h = new_h
                    current_score = new_score
        
        T *= cooling
        
        # Occasionally restart if stuck?
        # For now, just run the schedule.
        
    return list(best_h)
