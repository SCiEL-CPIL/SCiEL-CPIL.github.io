"""
Strategy: Deterministic Exhaustive Enumeration of the Self-Reciprocal Subfamily.

The previous strategy used heuristic local search (1-bit and 2-bit flips) on the full
2^24 coefficient space, which is prone to getting stuck in local minima and requires
heuristic tuning. This new strategy exploits the mathematical structure of the problem.

The sup-norm of a polynomial $P(z) = \sum_{k=0}^{N-1} c_k z^k$ on the unit circle is
invariant under time-reversal $z \to 1/z$ (which corresponds to reversing the coefficient
vector) and cyclic shifts (which correspond to multiplying by $z^m$).
Specifically, $|P(e^{i\theta})| = |z^{N-1} P(1/z)| = |\sum c_{N-1-k} e^{i\theta k}|$.
This means that the set of polynomials $\{ P(z), z^{N-1}P(1/z) \}$ have identical sup-norms.
We can restrict our search to polynomials that are "self-reciprocal" in the sense that
we only generate the first $N/2$ coefficients and mirror them. 
Wait, the symmetry is $c_k = c_{N-1-k}$? 
Let's check: $Q(z) = \sum c_k z^k$. $z^{N-1}Q(1/z) = \sum c_k z^{N-1-k}$.
If we set $c_k = c_{N-1-k}$, then $Q(z)$ is palindromic.
The set of palindromic polynomials is a subfamily. 
Does the global optimum lie in this subfamily? 
For $N=24$, the optimal value is 1.2583758.
Let's check if the known optimal polynomials are palindromic.
The problem statement mentions "Odlyzko's online tables list the optimal polynomials".
Usually, low-discrepancy sequences or optimal flat polynomials often exhibit symmetries.
However, restricting to palindromic sequences reduces the search space from $2^{24}$ to $2^{12}$.
$2^{12} = 4096$. We can exhaustively evaluate all 4096 palindromic polynomials.
This is a deterministic, complete search over a structured subfamily.

If the global optimum is not palindromic, this strategy might miss it.
However, the directive asks to "Exploit the cyclic symmetry and time-reversal invariance... 
to restrict the search space to polynomials satisfying $P(z) = z^{23}P(1/z)$ (self-reciprocal)".
This implies we should search this subfamily.
Given the small size ($2^{12}$), we can do this instantly.
If the score is worse than the current best, it suggests the optimum is not in this subfamily,
but the strategy is structurally different (exact enumeration of a subfamily vs heuristic search).
To hedge, we can also include the anti-palindromic or other symmetric classes if time permits,
but $2^{12}$ is so small we can just do it.

Actually, let's look closer at the symmetry.
$|P(e^{i\theta})| = | \sum c_k e^{ik\theta} |$.
Let $c_k = c_{N-1-k}$. Then $P(z) = z^{N/2-1} \sum_{j=0}^{N/2-1} c_j (z^{1/2} + z^{-1/2})^j$? No, that's for Chebyshev.
But the sup norm is the same for $P$ and its reverse.
So the "equivalence class" of a polynomial under reversal has size 1 (if palindromic) or 2.
By restricting to palindromic, we cover half the equivalence classes (representatives).
If the global optimum is not palindromic, its reverse is also not palindromic, and neither is in our set.
However, for $N=24$, it is highly likely that the optimal polynomials have some symmetry, or that the "flatness" constraint pushes towards symmetry.
Let's implement the exhaustive search over the $2^{12}$ palindromic polynomials.
We will also add a small set of high-quality non-palindromic seeds (like the current best) and run a few local search steps on them, just in case the palindromic subfamily doesn't contain the global optimum. But the primary engine is the exhaustive enumeration.

Wait, the prompt says: "This deterministic enumeration of the reduced search space guarantees finding the global optimum".
This claim is only true if the global optimum *is* in the reduced search space.
The prompt's "Strategy to try" suggests this is the intended path.
I will implement the exhaustive search over the 12-bit half-space (palindromic polynomials).
To ensure robustness, I will also include the specific hardcoded optimal candidate from the previous run if it exists, and maybe a few random starts, but the core novelty is the exhaustive palindromic search.

Actually, if I just return the best of the 4096 palindromic polynomials, and that score is 1.2583758, then I have found the optimum.
If the score is higher, I haven't.
Let's assume the strategy is valid.

Implementation details:
1. Generate all 4096 12-bit masks.
2. For each mask, construct the 24-coefficient palindromic vector.
3. Evaluate the sup-norm.
4. Keep the best.
5. Also, to be safe, take the best palindromic candidate and the hardcoded candidate from the previous program, and run a quick local search (1-bit flips) on them to see if we can improve further (escaping the palindromic constraint). This hybrid approach combines the deterministic enumeration with a local escape mechanism.

Let's refine the "local escape".
After finding the best palindromic $P_{pal}$, we can try flipping bits that break the symmetry.
Since we have plenty of time (4096 evals is fast), we can afford to do a full local search (1-bit and 2-bit) starting from $P_{pal}$ and the hardcoded seed.

Step 1: Exhaustive search over $2^{12}$ palindromic polynomials.
Step 2: Take the top 5 candidates from Step 1.
Step 3: Run exhaustive local search (1-bit and 2-bit flips) on these 5 candidates, allowing them to leave the palindromic subfamily.
Step 4: Return the best result.

This is structurally different: it starts with a complete enumeration of a structured subfamily, then refines.
"""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    N = 24
    half_N = N // 2
    
    # Pre-compute evaluation grid
    n_samples = 2048  # Fewer samples is fine for the initial screening of 4096 candidates
    theta = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    z = np.exp(1j * theta)
    
    # Pre-compute Z_powers for fast evaluation
    # We need to evaluate P(z) for many candidates.
    # Instead of dot product every time, we can precompute the basis?
    # Or just use FFT? 
    # For N=24, dot product is very fast.
    Z_powers = np.zeros((N, n_samples), dtype=complex)
    Z_powers[0] = 1.0
    for k in range(1, N):
        Z_powers[k] = Z_powers[k-1] * z

    def evaluate_sup_norm_fast(c):
        """
        Fast sup-norm evaluation using precomputed powers.
        c is a numpy array of +1/-1.
        """
        vals = np.dot(c, Z_powers)
        abs_vals = np.abs(vals)
        return np.max(abs_vals)

    def evaluate_sup_norm_refined(c):
        """
        Refined sup-norm evaluation for the final candidates.
        Uses golden section search on the peak.
        """
        vals = np.dot(c, Z_powers)
        abs_vals = np.abs(vals)
        peak_idx = np.argmax(abs_vals)
        
        step = 2 * np.pi / n_samples
        a = theta[peak_idx - 1]
        b = theta[(peak_idx + 1) % n_samples]
        if a > b:
            b += 2 * np.pi
            
        # Golden section search
        gr = (np.sqrt(5) - 1) / 2
        inv_gr = 1 / gr
        x1 = a + (b - a) * inv_gr
        x2 = a + (b - a) * gr
        
        def P_abs(theta_val):
            z_val = np.exp(1j * theta_val)
            val = 0.0 + 0.0j
            zp = 1.0 + 0.0j
            for k in range(N):
                val += c[k] * zp
                zp *= z_val
            return np.abs(val)
            
        f1 = P_abs(x1)
        f2 = P_abs(x2)
        
        for _ in range(50):
            if f1 < f2:
                a = x1
                x1 = x2
                f1 = f2
                x2 = a + (b - a) * gr
                f2 = P_abs(x2)
            else:
                b = x2
                x2 = x1
                f2 = f1
                x1 = a + (b - a) * inv_gr
                f1 = P_abs(x1)
                
            if b - a < 1e-12:
                break
                
        theta_opt = (x1 + x2) / 2
        max_val = P_abs(theta_opt)
        return max_val

    # Phase 1: Exhaustive search over palindromic polynomials
    # c[0:12] are free, c[12:24] are mirror of c[0:12]
    # c[k] = c[N-1-k]
    best_pal_score = np.inf
    best_pal_coeffs = None
    top_candidates = [] # List of (score, coeffs)
    
    # Iterate over all 2^12 masks
    for mask in range(1 << half_N):
        if time.time() > deadline - 20.0:
            break
            
        # Construct coefficients
        c = np.zeros(N, dtype=np.int8)
        for k in range(half_N):
            if (mask >> k) & 1:
                c[k] = 1
                c[N-1-k] = 1
            else:
                c[k] = -1
                c[N-1-k] = -1
        
        # Fast evaluation
        score = evaluate_sup_norm_fast(c)
        
        if score < best_pal_score:
            best_pal_score = score
            best_pal_coeffs = c.copy()
            
        # Keep track of top candidates for refinement
        if len(top_candidates) < 10:
            # Insert into sorted list
            inserted = False
            for i in range(len(top_candidates)):
                if score < top_candidates[i][0]:
                    top_candidates.insert(i, (score, c.copy()))
                    inserted = True
                    break
            if not inserted:
                top_candidates.append((score, c.copy()))
        elif score < top_candidates[-1][0]:
            top_candidates[-1] = (score, c.copy())

    # Ensure we have the absolute best palindromic candidate
    if best_pal_coeffs is not None:
        # Check if it's in top_candidates
        found = False
        for s, c in top_candidates:
            if np.array_equal(c, best_pal_coeffs):
                found = True
                break
        if not found:
            top_candidates.append((best_pal_score, best_pal_coeffs.copy()))
            top_candidates.sort(key=lambda x: x[0])
            if len(top_candidates) > 10:
                top_candidates.pop()

    # Phase 2: Refine top candidates using local search (allowing symmetry breaking)
    # We will take the top 5 palindromic candidates and run a full local search on them.
    # This allows us to escape the palindromic subfamily if it leads to a better global minimum.
    
    # Also add the hardcoded candidate from the previous program as a seed
    hardcoded_seed = np.array([1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1], dtype=np.int8)
    top_candidates.append((evaluate_sup_norm_fast(hardcoded_seed), hardcoded_seed.copy()))
    
    # Sort top candidates by fast score
    top_candidates.sort(key=lambda x: x[0])
    
    # Take top 5 for expensive local search
    seeds_for_local_search = [c for _, c in top_candidates[:5]]
    
    best_score = np.inf
    best_coeffs = None
    
    for c_init in seeds_for_local_search:
        if time.time() > deadline - 10.0:
            break
            
        c = c_init.copy()
        current_score = evaluate_sup_norm_refined(c)
        
        if current_score < best_score:
            best_score = current_score
            best_coeffs = c.copy()
            
        # Local Search Loop
        improved = True
        while improved:
            improved = False
            
            # 1. Try all 1-bit flips
            for i in range(N):
                if time.time() > deadline - 5.0:
                    break
                
                c[i] *= -1
                # Fast check first? No, just use refined for accuracy in the final steps?
                # Refined is slower. Let's use fast check to filter, then refined if promising?
                # For N=24, refined is ~0.5ms. 24*0.5ms = 12ms per pass.
                # 2-bit: 276*0.5ms = 138ms per pass.
                # Total ~150ms per full pass.
                # We have 10s budget per seed. ~60 passes. Plenty.
                
                new_score = evaluate_sup_norm_refined(c)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    if new_score < best_score:
                        best_score = new_score
                        best_coeffs = c.copy()
                else:
                    c[i] *= -1 # Revert
            
            if time.time() > deadline - 5.0:
                break
                
            # 2. Try all 2-bit flips
            for i in range(N):
                if time.time() > deadline - 5.0:
                    break
                    
                for j in range(i + 1, N):
                    if time.time() > deadline - 5.0:
                        break
                        
                    c[i] *= -1
                    c[j] *= -1
                    new_score = evaluate_sup_norm_refined(c)
                    
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        if new_score < best_score:
                            best_score = new_score
                            best_coeffs = c.copy()
                    else:
                        c[i] *= -1
                        c[j] *= -1 # Revert

    # Fallback
    if best_coeffs is None:
        if best_pal_coeffs is not None:
            best_coeffs = best_pal_coeffs
        else:
            best_coeffs = hardcoded_seed
            
    return best_coeffs.tolist()
# EVOLVE-BLOCK-END
