import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We use 140s.
    deadline = start_time + 140.0

    def compute_value(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        c = np.convolve(h, h)
        return abs(2.0 * N * np.max(c) / (s * s))

    # Initialize with seed
    h_best = np.ones(N)
    val_best = compute_value(h_best)
    
    rng = np.random.default_rng(42)

    # Strategy: Simulated Annealing on the Soft-Max objective.
    # Objective: Minimize L = logsumexp(beta * c) / N
    # where c = conv(h,h).
    # As beta -> inf, this approaches max(c).
    # We start with small beta and increase it, using Metropolis acceptance.
    
    # Precompute FFT size for efficient convolution if needed, 
    # but N=400 is small enough for np.convolve (160k ops).
    # np.convolve is O(N^2). 160k * 10000 steps = 1.6e9 ops.
    # Python loop overhead is the bottleneck. 
    # Let's try to vectorize or use FFT if we need speed.
    # For now, standard np.convolve.
    
    # We will work with h such that sum(h) = N.
    # Let h = 1 + f, sum(f)=0.
    # Minimize max(conv(1+f, 1+f)).
    # conv(1+f, 1+f) = conv(1,1) + 2*conv(1,f) + conv(f,f)
    # conv(1,1) has peak N at lag 0.
    # We want the peak of the whole thing to be small.
    # Since we can't change the DC component (sum is fixed), 
    # we are effectively trying to make the "excess" autocorrelation negative or flat.
    
    h = np.ones(N)
    
    # Soft max parameters
    # We'll anneal beta from small to large.
    # Start beta = 1.0, end beta = 100.0?
    # Actually, the scale of c is ~N^2 = 160,000.
    # So beta should be scaled.
    # Let's use beta such that beta * c_peak is in a manageable range for exp.
    # If c_peak ~ 1000, beta=0.01 gives 10. exp(10) is fine.
    
    current_beta = 0.01
    target_beta = 1.0
    
    # Simulated Annealing parameters
    # We run a fixed number of iterations or until time limit.
    # Each iteration:
    # 1. Propose h_new = h + noise
    # 2. Enforce sum(h_new) = N
    # 3. Compute objective for h and h_new
    # 4. Accept with probability exp(-(E_new - E_old)/T)
    
    # To speed up, we can reuse c from previous step if the change is local?
    # No, conv is global.
    
    # Let's estimate steps per second.
    # np.convolve(400,400) takes ~1-2 ms in Python/NumPy?
    # Let's test mentally. 160k multiplications. NumPy is fast.
    # Probably < 1ms. So ~1000-2000 steps per second.
    # 140 seconds -> ~200,000 steps. That's a lot.
    
    # However, Python loop overhead for 200k iterations is significant.
    # Each iteration:
    # - rng.normal (fast)
    # - np.sum (fast)
    # - np.convolve (fast)
    # - np.max (fast)
    # - logsumexp (fast)
    # - comparison/acceptance (fast)
    # Overhead might be ~0.1-0.2 ms per step.
    # So ~5000 steps/sec.
    # 140s * 5000 = 700,000 steps. Very good.
    
    # Let's structure the annealing schedule.
    # We will perform blocks of steps with constant beta, then increase beta.
    
    num_blocks = 20
    steps_per_block = 10000  # Total 200,000 steps
    
    # Temperature schedule
    # Start T high enough to accept bad moves, end T low.
    # E scale: L = logsumexp(beta*c). 
    # If c ~ 1000, beta=0.01, L ~ 10.
    # Delta L can be large.
    # Let's set T0 = 1.0, Tend = 0.01.
    
    T0 = 5.0
    Tend = 0.001
    
    # We need to be careful with the objective function scale.
    # Let's normalize the objective to be around 1.
    # obj = (logsumexp(beta * c) - beta * c_min) / N ? No, just logsumexp.
    # Actually, we want to minimize max(c).
    # Using logsumexp(beta*c) is a good proxy.
    
    def soft_max_obj(h, beta):
        c = np.convolve(h, h)
        # To avoid overflow in exp, subtract max
        c_max = np.max(c)
        # If c_max is very large, beta*c_max might overflow.
        # c_max ~ 160000. beta ~ 1. 160000 is fine for exp? No, exp(160000) overflows.
        # We MUST subtract the max before exp.
        # logsumexp(x) = max(x) + log(sum(exp(x - max(x))))
        # So we compute:
        shifted = beta * c - beta * c_max
        # shifted <= 0. exp(shifted) <= 1. Safe.
        val = beta * c_max + np.log(np.sum(np.exp(shifted)))
        return val, c_max, c

    # We track the best hard max found so far.
    # The soft max is just the optimizer.
    
    h_curr = np.ones(N)
    curr_obj, curr_c_max, _ = soft_max_obj(h_curr, current_beta)
    
    # We want to minimize the hard max.
    # If curr_c_max < val_best, update best.
    if curr_c_max < val_best:
        val_best = curr_c_max
        h_best = h_curr.copy()
        
    # Annealing loop
    for b_idx in range(num_blocks):
        # Update beta
        # Linear or exponential? Exponential is better for annealing.
        # beta = T0 * (Tend/T0)^(b_idx/num_blocks)
        # But we are increasing beta (making it sharper).
        # Let's interpolate log(beta)
        log_beta_start = np.log(0.01)
        log_beta_end = np.log(1.0)
        current_beta = np.exp(log_beta_start + (log_beta_end - log_beta_start) * (b_idx / num_blocks))
        
        # Update Temperature
        # T should decrease as beta increases?
        # Or T is independent?
        # Usually T decreases.
        T_curr = T0 * (Tend / T0) ** (b_idx / num_blocks)
        
        # Run steps for this block
        for s_idx in range(steps_per_block):
            if time.time() > deadline:
                break
                
            # Propose move: Add Gaussian noise
            # Noise scale should be adaptive or fixed?
            # Fixed small noise allows local search.
            # We can mix in larger noise occasionally.
            noise_scale = 0.1
            delta = rng.normal(0, noise_scale, N)
            
            # Enforce zero sum on delta?
            # If we add delta to h, sum changes.
            # We want sum(h_new) = N.
            # So delta must have sum 0.
            delta -= np.mean(delta)
            
            h_new = h_curr + delta
            
            # Compute objective for h_new
            new_obj, new_c_max, _ = soft_max_obj(h_new, current_beta)
            
            # Acceptance probability
            delta_obj = new_obj - curr_obj
            if delta_obj < 0:
                accept = True
            else:
                # Avoid underflow in exp
                if T_curr > 1e-12:
                    log_prob = -delta_obj / T_curr
                    if log_prob > -100: # exp(-100) is ~3e-44
                        accept = rng.random() < np.exp(log_prob)
                    else:
                        accept = False
                else:
                    accept = False
            
            if accept:
                h_curr = h_new
                curr_obj = new_obj
                
                # Check hard max for global best
                if new_c_max < val_best:
                    val_best = new_c_max
                    h_best = h_curr.copy()
            else:
                # Reject, keep h_curr
                pass
        
        if time.time() > deadline:
            break

    # Final check
    final_val = compute_value(h_best)
    if final_val < val_best:
        val_best = final_val
        h_best = h_best.copy() # It's already the best, but just in case of numerical drift

    return h_best
