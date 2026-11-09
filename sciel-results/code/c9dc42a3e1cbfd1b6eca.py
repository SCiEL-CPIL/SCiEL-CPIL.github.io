"""Alternative strategy: Continuous Priority-List Optimization with CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan(order, P):
    """
    Compute makespan for a given job order on processing_times matrix P.
    """
    n_machines = P.shape[1]
    if n_machines == 0:
        return 0.0
    
    c_machines = [0.0] * n_machines
    
    for j in order:
        row = P[j]
        c_curr = 0.0
        for m in range(n_machines):
            if c_machines[m] > c_curr:
                c_curr = c_machines[m]
            c_curr += row[m]
            c_machines[m] = c_curr
            
    return c_machines[-1]

def _priority_to_order(priorities, n_jobs):
    """
    Convert a priority vector to a job order (indices).
    Lower priority value = higher priority (processed earlier).
    """
    return np.argsort(priorities).tolist()

def _objective(priorities, P):
    """
    Objective function for CMA-ES: returns the makespan for the order defined by priorities.
    """
    order = _priority_to_order(priorities, P.shape[0])
    return _makespan(order, P)

def order_jobs(processing_times):
    """
    Order jobs to minimize makespan using Priority-List + CMA-ES.
    """
    P = np.asarray(processing_times, dtype=float)
    if P.size == 0:
        return []
        
    n_jobs, n_machines = P.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
        
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
    # 1. Initialize with a heuristic priority vector (LPT-like: shorter total processing time first)
    # Lower priority value means higher priority. So we assign lower values to shorter jobs.
    totals = P.sum(axis=1)
    # Normalize totals to [0, 1] for initial scaling
    if totals.max() > totals.min():
        init_priorities = (totals - totals.min()) / (totals.max() - totals.min())
    else:
        init_priorities = np.zeros(n_jobs)
        
    # 2. CMA-ES Parameters
    # sigma: initial step size. We want to explore the space of priorities.
    # Since priorities are relative, scaling matters. We start with sigma=1.0.
    sigma = 1.0
    popsize = max(4, int(4 + 3 * np.log(n_jobs)))
    
    # Simple CMA-ES implementation (no scipy dependency to be safe, though scipy is allowed)
    # We implement a basic version:
    # x: mean of the distribution
    # C: covariance matrix (we start with identity for simplicity, or adapt)
    # For a robust simple version, we can use a standard Gaussian step and adapt sigma.
    # Full CMA-ES is complex. Let's use a simplified "Adaptive Step Size" or just standard CMA-ES logic if possible.
    # Given the time constraint and simplicity, let's implement a basic CMA-ES.
    
    x = init_priorities.copy()
    C = np.eye(n_jobs)
    # Eigen decomposition for sampling
    eigvals, eigvecs = np.linalg.eigh(C)
    
    # CMA-ES parameters
    # lambda (popsize)
    lam = popsize
    # mu (number of selected individuals)
    mu = lam // 2
    # weights
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= np.sum(weights)
    # mu_eff
    mu_eff = 1.0 / np.sum(weights**2)
    # sigma adaptation
    cs = (mu_eff + 2) / (n_jobs + mu_eff + 5)
    ds = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n_jobs + 1)) - 1) + cs
    cc = (4 - mu_eff / n_jobs) / (n_jobs + 4 - mu_eff / n_jobs)
    c1 = 2 / ((n_jobs + 1.3)**2 + mu_eff)
    cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n_jobs + 2)**2 + mu_eff))
    
    ps = np.zeros(n_jobs)
    best_order = _priority_to_order(x, n_jobs)
    best_makespan = _makespan(best_order, P)
    
    while (time.time() - start_time) < time_limit:
        # Sample population
        offspring = []
        for i in range(lam):
            # Sample z ~ N(0, I)
            z = np.random.randn(n_jobs)
            # Sample y = B * D * z
            y = (eigvecs * np.sqrt(eigvals)) @ z
            # x' = x + sigma * y
            x_new = x + sigma * y
            offspring.append((x_new, _objective(x_new, P)))
            
        # Sort by fitness (makespan)
        offspring.sort(key=lambda t: t[1])
        
        # Update best
        if offspring[0][1] < best_makespan:
            best_makespan = offspring[0][1]
            best_order = _priority_to_order(offspring[0][0], n_jobs)
            
        # Update mean
        x_new_mean = sum([w * off[0] for w, off in zip(weights, offspring[:mu])])
        
        # Update evolution path
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / (sigma * np.sqrt(np.sum(eigvals))) * (eigvecs.T @ (x_new_mean - x))
        
        # Update covariance
        hsig = (np.sqrt(1 - (1 - cs)**2) / np.sqrt(1 - (1 - cs)**(2 * lam))) if np.sum(eigvals) > 0 else 0
        hs = np.sqrt((n_jobs + 2) * cc + 1) * hsig
        delta_hs = 1.0 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) < (1.4 + 2 / (n_jobs + 1)) * np.sqrt((n_jobs + 1) / (n_jobs - 1)) else 0.0
        
        # Rank-one and rank-mu update
        # For simplicity and stability in short runs, we might simplify C update or use identity if it gets unstable
        # Let's do the standard update
        C = (1 - c1 - cmu) * C + c1 * (1 + (1 - delta_hs) * ds * cs) * np.outer(ps, ps)
        
        # Rank-mu part
        # y_k = (offspring[k][0] - x) / sigma
        y_m = [(off[0] - x) / sigma for off in offspring[:mu]]
        C += cmu * sum([w * np.outer(y, y) for w, y in zip(weights, y_m)])
        
        # Re-orthogonalize C to ensure symmetry and positive semi-definiteness
        C = (C + C.T) / 2
        # Eigen decomposition for next iteration
        try:
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip small negative eigenvalues
            eigvals = np.clip(eigvals, 1e-12, None)
        except:
            # Fallback if linear algebra fails
            break
            
        # Update sigma
        sigma *= np.exp((cs / ds) * (np.linalg.norm(ps) / np.sqrt(2 * n_jobs) - 1))
        
        # Update mean
        x = x_new_mean
        
        # Optional: restart if sigma gets too small or too large
        if sigma < 1e-8 or sigma > 1e8:
            break

    return best_order
# EVOLVE-BLOCK-END
