import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
import sys

N = 600
W = 1.0 / 1200.0
X = np.linspace(-0.25, 0.25, N, endpoint=False)

# Fine grid for accurate convolution evaluation
FINE_N = 8000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

def ratio_from_h(h):
    """Compute the normalized maximum of the autoconvolution.
    Score is the upper bound on C1. Lower is better.
    """
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    
    # Check for zero function
    if np.sum(h) < 1e-15:
        return float('inf')
        
    f = h[CELL]
    F = np.fft.rfft(f, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
    l2 = np.sum(f * f) * DX
    if l2 < 1e-15:
        return float('inf')
    
    return float(np.max(conv) / l2)

def smoothed_ratio_from_h(h, delta=0.01):
    """Smoothed objective using Huber-like loss on the convolution profile.
    This makes the objective differentiable almost everywhere, helping gradient-based
    or quasi-Newton methods, and potentially aiding DE by providing a smoother landscape
    if we were to use gradients (though DE doesn't use them, the smoothness of the
    underlying function might still help local searches or if we switch methods).
    Actually, DE doesn't use gradients. But we can use this for a local polish phase.
    For DE, we might just use the raw ratio or a slightly smoothed version to avoid
    noise, but the max is already a bit noisy. Let's stick to raw ratio for DE to be
    faithful to the task, but use smoothed for local polish.
    
    Wait, the directive says "design a new strategy... different optimization or search method".
    Using DE is the main change. The objective can remain the raw ratio.
    """
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    if np.sum(h) < 1e-15:
        return float('inf')
    f = h[CELL]
    F = np.fft.rfft(f, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
    l2 = np.sum(f * f) * DX
    if l2 < 1e-15:
        return float('inf')
    
    # Smoothed max: use log-sum-exp approximation or Huber
    # Let's use a simple Huber loss on the deviation from the max? No, we want to minimize the max.
    # A common trick is to minimize the sum of squares of the top k values, or use a smooth max.
    # Let's try a soft max: delta * log(sum(exp(conv/delta)))
    # But conv can be large.
    # Actually, let's just use the raw ratio for DE. It's robust.
    return float(np.max(conv) / l2)

def construct():
    deadline = time.time() + 140.0  # Safety margin for 150s limit
    
    # Initial guess: uniform function
    # h = np.ones(N)
    # ratio is 1.0 for uniform?
    # Let's check: f=1, conv=triangular peak 0.5, l2=0.5. ratio = 0.5/0.5 = 1.
    # We want to go below 1.0? The seed score is 1.0816, which is > 1.0.
    # Wait, the task says "minimize the normalized maximum... this gives an upper bound on C1".
    # "score is the upper bound on C1". "lower is better".
    # If uniform gives 1.0, and seed gives 1.08, then uniform is better?
    # Let's re-read carefully.
    # "Choose a nonnegative step function ... to minimize the normalized maximum of its autoconvolution"
    # "score is the upper bound on C1"
    # "search score is seed/value, so 1.0 equals the seed program and above 1.0 is an improvement"
    # This implies: Score = Seed_Value / My_Value.
    # If My_Value < Seed_Value, Score > 1.0.
    # So we want to MINIMIZE the raw value (the ratio).
    
    # Let's verify the ratio for a uniform function.
    # h = 1.
    # f = 1.
    # conv(x) = integral f(y)f(x-y) dy. On [-0.25, 0.25], support is 0.5.
    # For x=0, conv(0) = integral_{-0.25}^{0.25} 1*1 dy = 0.5.
    # l2 = integral 1^2 dy = 0.5.
    # Ratio = 0.5 / 0.5 = 1.0.
    
    # The seed program has score 1.0816.
    # Score = 1.0816 = Seed_Value / Seed_Program_Value?
    # No, "search score is seed/value".
    # Usually "seed" refers to the baseline value.
    # If Seed_Value is the value of the seed program?
    # "1.0 equals the seed program".
    # So if I run the seed program, value = V_seed. Score = V_seed / V_seed = 1.0.
    # If I find a better program with value V_new < V_seed, Score = V_seed / V_new > 1.0.
    # The current program's score is 1.0816.
    # This means V_seed / V_current = 1.0816.
    # So V_current = V_seed / 1.0816.
    # This implies V_current < V_seed.
    # So the current program IS an improvement over the seed.
    
    # What is the seed program?
    # "seed program" usually implies a simple baseline.
    # If the seed is uniform (value 1.0), then V_seed = 1.0.
    # Then V_current = 1.0 / 1.0816 ≈ 0.9245.
    # So the current program achieves a ratio of ~0.9245.
    # My goal is to get a ratio lower than 0.9245.
    
    # Strategy:
    # 1. Run Differential Evolution to find a good global candidate.
    # 2. Polish with local optimization (L-BFGS-B or Nelder-Mead).
    
    # DE Parameters
    # Bounds: [0, 1] for each height.
    # We can scale them later if needed, but 0-1 should be sufficient.
    # If the optimal function has max height > 1, we can rescale.
    # Actually, the ratio is scale-invariant!
    # ratio(c*h) = max(c^2 conv) / (c^2 l2) = ratio(h).
    # So we can fix one variable, or just let DE find the shape in [0,1].
    
    bounds = [(0.0, 1.0) for _ in range(N)]
    
    # DE is computationally expensive.
    # N=600. Population size 150? 150 * 600 * 1000 generations = 90 million evals.
    # Too slow.
    # We need to be smart.
    
    # Alternative: Use a lower-dimensional representation for DE?
    # The current program uses B-splines (M=12).
    # Inspiration 1 used Gaussians.
    # Inspiration 2 used Fourier.
    # The directive says "MATERIALLY DIFFERENT ... different design family".
    # Using DE on the full 600-dim space is different from B-spline basis.
    # But is it feasible?
    # Let's estimate time per evaluation.
    # FFT of size 16384. ~1ms?
    # 1000 evals/sec.
    # 150s * 1000 = 150,000 evals.
    # DE with pop=100, gens=1000 -> 100,000 evals. This fits!
    
    # Let's try DE with:
    # popsize = 100
    # maxiter = 1000
    # tol = 1e-6
    # mutation = (0.5, 1.0)
    # recombination = 0.9
    # seed = 42
    
    # We need to be careful with the objective function speed.
    # Precomputation is done.
    # The FFT is the bottleneck.
    
    # Let's add a time check inside the objective? No, DE handles it.
    # We will wrap the DE call in a try-except and time guard.
    
    best_val = float('inf')
    best_h = np.ones(N) / N  # Uniform, normalized? No, scale invariant. Just 1s.
    
    # Actually, let's initialize best_h to the current program's output if we could access it? No.
    # Let's just start with uniform.
    h_uniform = np.ones(N)
    val_uniform = ratio_from_h(h_uniform)
    best_val = val_uniform
    best_h = h_uniform
    
    # Run DE
    if time.time() < deadline - 20.0:
        try:
            # Use a wrapper to enforce non-negativity? DE respects bounds.
            # But we might want to allow larger values? No, scale invariant.
            
            # Callback to monitor time and update best
            def callback(xk, convergence):
                nonlocal best_val, best_h
                # xk is the current best individual
                val = ratio_from_h(xk)
                if val < best_val:
                    best_val = val
                    best_h = xk.copy()
                return time.time() > deadline - 10.0  # Stop if close to deadline
            
            result = differential_evolution(
                ratio_from_h,
                bounds,
                maxiter=1000,
                popsize=100,
                tol=1e-6,
                mutation=(0.5, 1.0),
                recombination=0.9,
                seed=42,
                callback=callback,
                polish=False  # We will do our own polish
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_h = result.x.copy()
                
        except Exception as e:
            print(f"DE failed: {e}")
            # Fallback to random search?
            pass

    # Phase 2: Local Polish
    # Use L-BFGS-B or Nelder-Mead on the best candidate found.
    # Nelder-Mead is derivative-free and robust.
    # L-BFGS-B is faster if gradients were available, but we don't have analytic gradients.
    # Numerical gradients for N=600 are expensive (600 evals per gradient).
    # Nelder-Mead is better here.
    
    if time.time() < deadline - 5.0:
        try:
            # Nelder-Mead
            # Initial simplex: perturb best_h
            h0 = best_h.copy()
            
            # Ensure h0 is not zero
            if np.sum(h0) < 1e-15:
                h0 = np.ones(N)
            
            # Scale for Nelder-Mead? It works on the values.
            # Bounds are not strictly enforced by Nelder-Mead, but we can clip in objective.
            
            result = minimize(
                ratio_from_h,
                h0,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_h = result.x.copy()
                
        except Exception as e:
            print(f"Polish failed: {e}")

    # Final check and return
    best_h = np.clip(best_h, 0.0, None)
    # Optional: Normalize? No, scale invariant.
    
    # Verify final value
    final_val = ratio_from_h(best_h)
    if final_val < best_val:
        best_val = final_val
        
    return list(best_h)

if __name__ == "__main__":
    # Quick test
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Time: {end-start:.2f}s")
    print(f"Score: {ratio_from_h(res)}")
