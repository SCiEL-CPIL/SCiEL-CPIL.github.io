"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400
import numpy as np
import time

# EVOLVE-BLOCK-START

def _compute_objective(f):
    """
    Compute max |autocorr| / (sum f)^2.
    f is a numpy array.
    """
    # Autoconvolution via FFT
    n = len(f)
    f_hat = np.fft.fft(f, n=2*n)
    # Inverse FFT of squared magnitude gives autoconvolution
    # Note: numpy fft convention: IFFT(FFT(f)^2) gives circular convolution of length 2n
    # We want linear autoconvolution.
    # autocorr[k] = sum f[i] f[i+k]
    # Using FFT:
    # F = FFT(f, 2n)
    # A = IFFT(|F|^2)
    # The linear autoconvolution is contained in A[0:n] and A[2n-n:2n] (conjugate symmetric)
    # Actually, for real f, A is real and symmetric.
    # A[0] = sum f^2
    # A[k] for k=1..n-1 is sum f[i] f[i+k]
    # A[n-k] is sum f[i] f[i-k] = sum f[i+k] f[i] = A[k]
    # So we just need to look at the first n elements of A.
    
    A = np.fft.ifft(f_hat * np.conj(f_hat)).real
    A = A[:n] # Take the first n elements which correspond to lags 0 to n-1
    
    max_abs_ac = np.max(np.abs(A))
    integral = np.sum(f)
    if integral == 0:
        return np.inf
    return max_abs_ac / (integral ** 2)

def _quantize_and_refine(f_cont):
    """
    Quantize continuous f_cont to signs (+1/-1) and perform local search.
    """
    # Quantize: threshold at 0
    # If f[i] >= 0, sign is 1, else -1
    f = np.sign(f_cont)
    # Handle zeros: if f_cont[i] == 0, sign is 0. We need +/-1.
    # Randomly assign or based on small noise?
    # Let's add tiny noise to break ties or just use np.where(f_cont >= 0, 1, -1)
    f = np.where(f_cont >= 0, 1.0, -1.0)
    
    best_f = f.copy()
    best_obj = _compute_objective(best_f)
    
    start_time = time.time()
    deadline = 100.0 # Leave 50s for safety and other overheads if any, but this is the main loop
    
    # Local Search: Flip signs
    # We can be smarter: flip the index that causes the most reduction in max_abs_ac
    # But simple random/ordered flips are okay if we have time.
    # Given N=400, N^2 autocorr is 160k ops. FFT is much faster.
    # Let's try a few passes of greedy local search.
    
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        # Try flipping each index
        # To speed up, we can just iterate through indices.
        for i in range(N):
            if (time.time() - start_time) > deadline:
                break
            if best_f[i] == 1.0:
                new_f = best_f.copy()
                new_f[i] = -1.0
            else:
                new_f = best_f.copy()
                new_f[i] = 1.0
            
            obj = _compute_objective(new_f)
            if obj < best_obj:
                best_obj = obj
                best_f = new_f
                improved = True
        # If no improvement in a full pass, we might be stuck in a local minimum.
        # We could try random restarts or larger moves, but let's stick to simple first.
        # If no improvement, break to save time? Or try random flips?
        # Let's add a simple random restart mechanism if stuck.
        if not improved:
             # Try 10 random flips
             for _ in range(10):
                 if (time.time() - start_time) > deadline:
                     break
                 i = np.random.randint(N)
                 new_f = best_f.copy()
                 new_f[i] *= -1
                 obj = _compute_objective(new_f)
                 if obj < best_obj:
                     best_obj = obj
                     best_f = new_f
                     improved = True
             if not improved:
                 break

    return best_f, best_obj

def _spectral_optimization():
    """
    Optimize continuous f to minimize max |autocorr| / (sum f)^2.
    Uses gradient descent on the objective.
    """
    # Initialize f
    # Start with random or ones?
    # Random might be better to avoid symmetry traps.
    np.random.seed(42)
    f = np.random.randn(N)
    
    # Normalize to avoid issues with scaling
    # The objective is scale invariant: (c * max_ac) / (c * sum)^2 = max_ac / (c * sum^2)
    # Wait, autocorr scales with c^2. Sum scales with c.
    # Obj(c f) = max|c^2 ac| / (c sum)^2 = c^2 max|ac| / c^2 sum^2 = Obj(f).
    # So scale doesn't matter.
    
    best_f = f.copy()
    best_obj = _compute_objective(best_f)
    
    start_time = time.time()
    deadline = 60.0 # 60 seconds for spectral opt
    
    lr = 0.01
    momentum = 0.9
    v = np.zeros_like(f)
    
    while (time.time() - start_time) < deadline:
        # Compute gradient
        # Obj = max_k |A_k| / S^2 where A_k = sum_i f_i f_{i+k}, S = sum f_i
        # Let M = max_k |A_k|, K* be the argmax.
        # dObj/df_i = (1/S^2) * dM/df_i - 2 M/S^3 * dS/df_i
        # dS/df_i = 1
        # dM/df_i = sign(A_{K*}) * dA_{K*}/df_i
        # A_{K*} = sum_j f_j f_{j+K*}
        # dA_{K*}/df_i = f_{i-K*} + f_{i+K*} (with boundary conditions 0)
        
        # Compute A
        n = len(f)
        f_hat = np.fft.fft(f, n=2*n)
        A = np.fft.ifft(f_hat * np.conj(f_hat)).real[:n]
        
        M = np.max(np.abs(A))
        K_star = np.argmax(np.abs(A))
        S = np.sum(f)
        
        if S == 0:
            break
            
        # Gradient of M w.r.t f
        # g_M = sign(A[K_star]) * (f shifted by -K_star + f shifted by K_star)
        # Note: A[k] = sum f[i] f[i+k].
        # dA[k]/df[i] = f[i-k] + f[i+k] (if indices valid)
        
        # Construct gradient array
        grad_M = np.zeros_like(f)
        sign_A = np.sign(A[K_star])
        
        # Shift 1: f[i-K]
        if K_star == 0:
            # dA[0]/df[i] = 2 f[i]
            grad_M += 2.0 * f
        else:
            # f[i-K] is non-zero for i >= K
            grad_M[K_star:] += f[:n-K_star]
            # f[i+K] is non-zero for i < n-K
            grad_M[:n-K_star] += f[K_star:]
            
        grad_M *= sign_A
        
        # Total Gradient
        grad_obj = grad_M / (S**2) - 2.0 * M / (S**3)
        
        # Update
        v = momentum * v - lr * grad_obj
        f_new = f + v
        
        # Optional: Project to some constraint? No, scale invariant.
        # But if f becomes very large/small, numerical issues?
        # Let's normalize f to have sum 1 to keep numbers stable.
        # Since objective is scale invariant, we can normalize.
        if np.sum(f_new) != 0:
            f_new = f_new / np.sum(f_new)
        
        obj = _compute_objective(f_new)
        
        if obj < best_obj:
            best_obj = obj
            best_f = f_new.copy()
            f = f_new
        else:
            # Reduce learning rate if stuck
            lr *= 0.95
            # Or revert
            f = best_f.copy()
            v = np.zeros_like(f)
            
    return best_f, best_obj

def construct():
    start_time = time.time()
    
    # 1. Spectral Optimization
    f_cont, obj_cont = _spectral_optimization()
    
    # 2. Quantize and Refine
    f_int, obj_int = _quantize_and_refine(f_cont)
    
    # 3. Also try a direct local search from all-ones as a fallback/alternative
    # If spectral fails or takes too long, we might want a fallback.
    # But we have deadline guards.
    
    # Check time
    if time.time() - start_time > 110:
         return f_int
         
    # Maybe try a few random restarts of the quantization?
    # Or just return the best found.
    
    return f_int

# EVOLVE-BLOCK-END
