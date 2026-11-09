"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.fft import fft, ifft

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Size for linear convolution via FFT
    size = 1
    while size < 2 * N - 1:
        size <<= 1
        
    # Pre-compute all-ones vector
    ones = np.ones(N)
    
    def _objective(x):
        """Compute max |autocorrelation| / (sum x)^2. Lower is better."""
        s = np.sum(x)
        if s == 0:
            return float('inf')
        X = fft(x, size)
        ac = ifft(X * np.conj(X)).real[:2 * N - 1]
        return np.max(np.abs(ac)) / (s * s)
    
    best_score = float('inf')
    best_x = np.ones(N)
    
    # 1. Spectral Min-Max Optimization (Chebyshev-like)
    # We want to minimize max_k |ac_k| subject to x in {-1, 1}^N.
    # Relax to x in [-1, 1] and use iterative water-filling / projection.
    # Actually, we can use a simpler heuristic:
    # Minimize sum of squares of ac, or max ac.
    # Let's try to minimize the spectral flatness or similar.
    
    # Alternative: Direct minimization of max |ac| using subgradient method.
    # Objective: f(x) = max_k |ac_k(x)| / (sum x)^2
    # This is hard to differentiate.
    
    # Let's stick to the "spectral" idea but use a different search space.
    # The current program uses random signs and local flips.
    # A different strategy: Use the structure of the autocorrelation.
    # For a sequence of length N, the autocorrelation at lag 0 is sum x_i^2 = N (for signs).
    # We want to minimize the other lags.
    # This is related to "low autocorrelation sequences" or "Golay sequences" (though those are pairs).
    # Barker sequences have low sidelobes but are short.
    # For N=400, we want a "random-like" sequence with low peak sidelobe.
    
    # Strategy: Iterative Refinement via Spectral Descent
    # 1. Start with random signs.
    # 2. Compute FFT.
    # 3. Identify the frequency bins that contribute most to the peak sidelobes.
    # 4. Adjust signs to reduce those contributions.
    
    # A more robust approach for this specific metric:
    # The metric is max |ac| / S^2.
    # Note that ac[0] = N. So max |ac| >= N.
    # S = sum x. S^2 <= N^2.
    # If S is close to N (all ones), S^2 ~ N^2, so ac[0]/S^2 ~ 1.
    # If S is close to 0, S^2 is small, so ac[0]/S^2 is huge.
    # So we WANT S to be large (close to N).
    # This implies we want most x_i to be 1.
    # But if most are 1, ac[1] is large (many 1*1 pairs).
    # So we need to balance: keep S large, but make ac[k] for k>0 small.
    
    # Let's try a heuristic:
    # Start with all ones.
    # Flip signs to reduce the largest sidelobe.
    # But the current program already does local search from all ones and found 1.107.
    # 1.0 is the seed. 1.107 is an improvement (score = seed/value, so 1.0/1.107 < 1? No.
    # Wait. "score is the upper bound on C3'". "search score is seed/value".
    # If value is the raw metric (lower is better), then seed/value > 1 is better.
    # Seed program value:
    # All ones: ac[0]=400, ac[k]=400-k. max=400. S=400. S^2=160000.
    # Value = 400/160000 = 1/400 = 0.0025.
    # Score = 1.0 / 0.0025 = 400? No, the prompt says "score 1.0000" for the current program.
    # "1.0 equals the seed program".
    # So the seed program's value must be such that seed/value = 1.0.
    # Or maybe the "seed" is the value of the seed program?
    # "search score is seed/value, so 1.0 equals the seed program".
    # This implies the score is normalized such that the seed program gets 1.0.
    # So if I get a lower raw value, my score should be > 1.0.
    
    # The inspiration said: "Local search starting from all-ones ... Measured: 1 -> 1.107".
    # This means the raw value went from V_seed to V_new such that V_seed/V_new = 1.107.
    # So V_new = V_seed / 1.107.
    # V_seed = 0.0025. V_new = 0.0025 / 1.107 = 0.002258.
    
    # So the goal is to get a raw value lower than 0.0025.
    
    # Let's implement a more aggressive local search or a different global strategy.
    # Since local search from all-ones works, maybe we can speed it up or improve it.
    # Or use a different starting point?
    # If we start with random signs, S is small, so value is huge.
    # So all-ones is a very good starting point because S is maxed out.
    
    # What if we use a "block" structure that maintains high S but reduces sidelobes?
    # E.g., groups of 1s and -1s arranged to cancel sidelobes.
    
    # Let's try a "spectral shaping" approach.
    # We want to minimize max |ac[k]| for k != 0, subject to sum x_i = S (large).
    # Actually, we want to minimize max |ac[k]| / S^2.
    # Since ac[0] = N, max |ac| >= N.
    # So we want S^2 to be as large as possible, i.e., S close to N.
    # This forces x to be mostly 1s.
    # Let k be the number of -1s. Then S = N - 2k.
    # S^2 = (N-2k)^2.
    # We want to place the -1s to minimize the sidelobes.
    
    # Heuristic:
    # 1. Start with all 1s.
    # 2. Introduce -1s one by one.
    # 3. For each new -1, place it at the position that minimizes the increase in max |ac|.
    # 4. Stop when adding more -1s increases the objective.
    
    # This is a constructive heuristic.
    
    def _autocorr_full(x):
        X = fft(x, size)
        ac = ifft(X * np.conj(X)).real[:2 * N - 1]
        return ac
        
    def _try_candidate(x, best_score, best_x, deadline):
        if time.time() > deadline:
            return best_score, best_x
        s = np.sum(x)
        if s == 0:
            return best_score, best_x
        ac = _autocorr_full(x)
        sc = np.max(np.abs(ac)) / (s * s)
        if sc < best_score:
            best_score = sc
            best_x = x.copy()
        return best_score, best_x

    best_score, best_x = _try_candidate(np.ones(N), best_score, best_x, deadline)
    
    # Constructive heuristic: add -1s optimally
    x = np.ones(N)
    # We will try to add up to N/2 -1s
    for step in range(N // 2):
        if time.time() > deadline:
            break
        best_pos = -1
        best_sc = best_score
        # Evaluate all positions? O(N * FFT) is too slow.
        # We need a faster way to estimate the change.
        # Flipping x[i] from 1 to -1 changes x by -2 * e_i.
        # ac_new = ac_old - 4 * (x_old * e_i)_auto + 4 * (e_i)_auto
        # This is complex.
        
        # Let's stick to the local search but make it smarter.
        # The current local search flips one element at a time.
        # We can use a "batch" flip or a "simulated annealing" approach.
        
        # Let's try a different strategy:
        # Use the fact that the objective is convex in the continuous relaxation?
        # No, it's not.
        
        # Let's try a "genetic" approach with a specific crossover.
        # Or simply: run the local search from multiple random starts that have high S.
        # E.g., start with 90% ones and 10% random -1s.
        
        pass

    # Let's implement the "random restart with high S" strategy.
    rng = np.random.RandomState(42)
    
    for trial in range(100):
        if time.time() > deadline:
            break
        
        # Generate a sequence with high sum
        # Keep 80-95% of elements as 1, flip the rest randomly
        flip_prob = rng.uniform(0.05, 0.2)
        x = np.ones(N)
        mask = rng.random(N) < flip_prob
        x[mask] = -1
        
        best_score, best_x = _try_candidate(x, best_score, best_x, deadline)
        
        # Local search
        if time.time() < deadline:
            improved = True
            for it in range(500):
                if time.time() > deadline:
                    break
                if not improved:
                    break
                improved = False
                # Try flipping a random subset
                idxs = rng.choice(N, size=min(20, N), replace=False)
                for i in idxs:
                    if time.time() > deadline:
                        break
                    x[i] = -x[i]
                    sc = _objective(x)
                    if sc < best_score:
                        best_score = sc
                        best_x = x.copy()
                        improved = True
                    else:
                        x[i] = -x[i]
                        
    return best_x
# EVOLVE-BLOCK-END
