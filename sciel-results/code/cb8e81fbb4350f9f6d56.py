"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Construct a signed step function with N equal intervals on [-1/4, 1/4].
    We use the Logistic Map at r=4 to generate sign sequences.
    We explore a wide variety of initial conditions (irrational seeds) to find
    low-discrepancy sequences. We then refine the best candidates using an
    intensive local search (hill climbing) that flips signs to minimize the
    C3 upper bound (max absolute value of the triple autocorrelation).
    """
    
    def compute_c3_bound(s):
        """
        Compute the upper bound on C3, defined as the maximum absolute value
        of the third-order autocorrelation coefficients.
        """
        # Convolution of 3 sequences of length N has length 3N-2 = 1198.
        # Next power of 2 is 2048.
        N_pad = 2048
        s_pad = np.zeros(N_pad)
        s_pad[:N] = s
        S = np.fft.fft(s_pad)
        C3 = np.fft.ifft(S ** 3).real
        return np.max(np.abs(C3))

    # Generate a diverse set of initial conditions using irrational numbers.
    # The logistic map is chaotic, so small changes in x0 lead to different sequences.
    seeds = []
    # Use combinations of sqrt(k) mod 1 for k=2..50
    for k in range(2, 50):
        seeds.append(np.sqrt(k) % 1.0)
    # Add some other transcendental numbers mod 1
    seeds.append(np.pi % 1.0)
    seeds.append(np.e % 1.0)
    seeds.append((1 + np.sqrt(5)) / 2 % 1.0)
    seeds.append(np.log(2) % 1.0)
    seeds.append(np.log(3) % 1.0)
    
    best_signs = None
    best_bound = float('inf')
    
    # Evaluate initial candidates
    candidate_signs = []
    for seed in seeds:
        x = seed
        signs = np.zeros(N)
        for i in range(N):
            x = 4.0 * x * (1.0 - x)
            signs[i] = 1.0 if x > 0.5 else -1.0
        candidate_signs.append(signs)
        
    # Select the top 5 candidates based on initial bound to refine
    # This saves time compared to refining all 50+
    scored_candidates = []
    for signs in candidate_signs:
        bound = compute_c3_bound(signs)
        scored_candidates.append((bound, signs))
        
    scored_candidates.sort(key=lambda x: x[0])
    top_candidates = [s for _, s in scored_candidates[:5]]
    
    # Intensive local search on top candidates
    # We use a greedy hill climbing approach.
    # Since N=400 and FFT is fast, we can afford many evaluations.
    # Let's allocate a budget of 20,000 evaluations per candidate.
    evals_per_candidate = 20000
    
    for signs in top_candidates:
        current_bound = compute_c3_bound(signs)
        if current_bound < best_bound:
            best_bound = current_bound
            best_signs = signs.copy()
            
        # Local search
        # Randomize the order of indices to avoid bias
        indices = np.arange(N)
        np.random.shuffle(indices)
        
        for _ in range(evals_per_candidate):
            # Pick a random index to flip
            idx = np.random.choice(indices)
            
            signs[idx] *= -1.0
            new_bound = compute_c3_bound(signs)
            
            if new_bound < current_bound:
                current_bound = new_bound
                if current_bound < best_bound:
                    best_bound = current_bound
                    best_signs = signs.copy()
            else:
                signs[idx] *= -1.0 # Revert
                
    return best_signs
# EVOLVE-BLOCK-END
