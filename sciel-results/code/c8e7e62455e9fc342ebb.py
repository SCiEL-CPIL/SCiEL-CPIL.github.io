import time
import numpy as np
from scipy.optimize import minimize
import random

def construct():
    n = 256
    deadline = time.time() + 55.0  # 5s margin for 60s limit
    
    # 1. Initialize with Thue-Morse sequence
    def thue_morse(n):
        seq = np.zeros(n, dtype=np.int8)
        for i in range(n):
            # Parity of number of set bits
            c = bin(i).count('1')
            seq[i] = 1 if c % 2 == 1 else -1
        return seq

    # Initial discrete signs
    signs_init = thue_morse(n)
    
    # 2. Continuous Relaxation
    # Map signs to z such that tanh(z) = sign. 
    # For sign 1, z is large positive. For sign -1, z is large negative.
    # We use a moderate magnitude to allow movement but start close to solution.
    z_mag = 5.0
    z_init = np.sign(signs_init) * z_mag
    
    def objective(z):
        # Soft signs
        x = np.tanh(z)
        
        # DFT of the sequence
        # We want to minimize the max absolute value of the DFT.
        # Note: DFT of a real sequence has symmetry.
        F = np.fft.fft(x)
        abs_F = np.abs(F)
        
        # The DC component (F[0]) is the sum of signs. 
        # For a balanced sequence, this should be small.
        # The task minimizes max absolute signed sum on ANY AP.
        # The sum of the whole sequence is an AP with step 1, start 1, length n.
        # So minimizing |F[0]| is crucial.
        
        # To make the max operation differentiable, we use a soft-max approximation
        # or simply minimize the sum of squares + a penalty on the max.
        # Minimizing the L2 norm of the DFT minimizes the total energy, which is constant for +/-1 signs.
        # However, for soft signs, minimizing L2 might drive x to 0.
        # We need to minimize the PEAKS while keeping x close to +/-1.
        
        # Strategy: Minimize (max_abs_F - target)^2 + lambda * ||x||^2?
        # Actually, for +/-1 signs, the energy is fixed. 
        # The "discrepancy" is related to how flat the spectrum is.
        # A random sequence has flat spectrum (low peaks). 
        # Thue-Morse has specific spectral nulls.
        
        # Let's minimize the 2nd moment of the DFT magnitude? No, that's constant.
        # Let's minimize the max magnitude using a smooth approximation.
        # soft_max(abs_F) approx max(abs_F).
        # But we also need to push x towards +/-1.
        
        # Regularization: Sum of (1 - x^2)^2 to keep x near +/-1
        reg = np.sum((1 - x**2)**2)
        
        # Spectral Penalty: We want the spectrum to be flat.
        # A common measure of "flatness" is minimizing the variance of the power spectrum.
        # Or simply minimizing the max peak.
        # Let's use a log-sum-exp approximation for the max to keep it smooth.
        # max(abs_F) ~ log(sum(exp(k*abs_F))) / k
        # Choose k large enough for approximation but small enough for stability.
        k = 10.0
        # Shift for numerical stability
        max_val = np.max(abs_F)
        log_sum_exp = max_val + np.log(np.sum(np.exp(k * (abs_F - max_val)))) / k
        
        # We want to minimize this peak value.
        # Also, the DC term F[0] is the total sum. We definitely want that to be 0.
        # The spectral penalty covers this if k is high enough, but let's be explicit.
        dc_penalty = abs_F[0] ** 2
        
        return log_sum_exp + 0.1 * dc_penalty + 0.5 * reg

    # Optimize
    # Use L-BFGS-B as it handles bounds well and is efficient for medium n.
    # Bounds: z should be within [-10, 10] to keep tanh stable and signs distinct.
    bounds = [(-10.0, 10.0)] * n
    
    best_z = z_init
    best_cost = objective(z_init)
    
    # Run optimization
    try:
        res = minimize(objective, z_init, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-5})
        
        if res.success or res.fun < best_cost:
            best_z = res.x
            best_cost = res.fun
    except Exception:
        pass
    
    # Convert back to discrete signs
    x_opt = np.tanh(best_z)
    signs_opt = np.sign(x_opt)
    # If any are 0 (unlikely with tanh and z bounds), default to 1
    signs_opt[signs_opt == 0] = 1
    signs_opt = signs_opt.astype(np.int8)
    
    # 3. Discrete Refinement (Hill Climbing)
    # The spectral objective is a proxy. We must refine with the true cost.
    
    def get_cost(s):
        max_disc = 0
        nn = n
        # Optimization: local variable lookup
        for d in range(1, nn + 1):
            for r in range(d):
                curr = 0
                min_v = 0
                max_v = 0
                idx = r
                while idx < nn:
                    curr += s[idx]
                    if curr < min_v:
                        min_v = curr
                    if curr > max_v:
                        max_v = curr
                    idx += d
                disc = max_v - min_v
                if disc > max_disc:
                    max_disc = disc
        return max_disc

    current_seq = signs_opt.tolist()
    current_cost = get_cost(current_seq)
    best_seq = current_seq[:]
    best_cost_disc = current_cost
    
    rng = random.Random(42)
    
    # Simple Hill Climbing with some restarts if time permits
    # Since the spectral start is good, we expect fewer steps needed.
    
    while time.time() < deadline:
        improved = False
        # Try a few random flips
        for _ in range(100):
            if time.time() > deadline:
                break
            idx = rng.randrange(n)
            old_val = current_seq[idx]
            current_seq[idx] = -old_val
            
            new_cost = get_cost(current_seq)
            
            if new_cost < current_cost:
                current_cost = new_cost
                improved = True
                if new_cost < best_cost_disc:
                    best_cost_disc = new_cost
                    best_seq = current_seq[:]
            else:
                # Revert
                current_seq[idx] = old_val
        
        if not improved:
            # Try a larger perturbation: flip 2-3 bits
            flips = rng.randint(2, 3)
            indices = rng.sample(range(n), flips)
            for idx in indices:
                current_seq[idx] *= -1
            
            new_cost = get_cost(current_seq)
            if new_cost < current_cost:
                current_cost = new_cost
                if new_cost < best_cost_disc:
                    best_cost_disc = new_cost
                    best_seq = current_seq[:]
            else:
                for idx in indices:
                    current_seq[idx] *= -1
            
            # If stuck for a while, break? 
            # For now, keep going until deadline as hill climbing is fast enough for n=256
            # Actually, get_cost is O(N^2). 256^2 = 65536 ops.
            # 100 flips * 65k ops = 6.5M ops per loop.
            # Python is slow. 6.5M ops might take ~0.1-0.2s.
            # So ~50-100 loops per second.
            # We have ~50s. So ~2500-5000 iterations.
            
            # To avoid infinite loop if no improvement, we can break after some patience
            # But simple HC often needs restarts.
            # Let's add a simple restart mechanism if no improvement in last 500 checks.
            # For simplicity and robustness within time, we just keep flipping.
            
    return best_seq

# Test locally if needed (commented out for submission)
# if __name__ == "__main__":
#     res = construct()
#     print(len(res))
#     print(res[:10])
