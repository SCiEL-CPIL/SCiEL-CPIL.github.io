"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from collections import defaultdict


def _makespan(processing_times, order):
    """Compute makespan for a given job order efficiently."""
    n_jobs = len(order)
    if n_jobs == 0:
        return 0.0
    n_machines = processing_times.shape[1]
    
    # Use vectorized computation where possible
    # C[j][m] = completion time of job j on machine m
    # We can compute this incrementally
    completion = np.zeros((n_jobs, n_machines), dtype=np.float64)
    
    for j in range(n_jobs):
        job_idx = order[j]
        for m in range(n_machines):
            p = processing_times[job_idx, m]
            if m == 0:
                if j == 0:
                    completion[j, m] = p
                else:
                    completion[j, m] = completion[j-1, m] + p
            else:
                if j == 0:
                    completion[j, m] = completion[j, m-1] + p
                else:
                    completion[j, m] = max(completion[j-1, m], completion[j, m-1]) + p
    
    return completion[-1, -1]


def _permutation_to_binary_vector(perm, n_jobs):
    """Convert a permutation to a binary precedence vector.
    For each pair (i, j) with i < j, bit[i,j] = 1 if i comes before j in perm."""
    n = n_jobs
    # Position of each job in the permutation
    pos = np.zeros(n, dtype=np.int32)
    for idx, job in enumerate(perm):
        pos[job] = idx
    
    # Build binary vector: for i < j, bit = 1 if pos[i] < pos[j]
    # We store as a flat array of length n*(n-1)/2
    n_pairs = n * (n - 1) // 2
    vec = np.zeros(n_pairs, dtype=np.float64)
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if pos[i] < pos[j]:
                vec[idx] = 1.0
            else:
                vec[idx] = 0.0
            idx += 1
    
    return vec


def _binary_vector_to_permutation(vec, n_jobs):
    """Convert a binary precedence vector back to a permutation.
    This is approximate - we use the implied ordering."""
    n = n_jobs
    n_pairs = n * (n - 1) // 2
    
    # Reconstruct pairwise preferences
    # For each pair (i, j), i < j: if vec bit is 1, i before j; else j before i
    # We compute a score for each job: number of jobs it should come before
    scores = np.zeros(n, dtype=np.float64)
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if vec[idx] > 0.5:
                # i before j
                scores[i] += 1
            else:
                # j before i
                scores[j] += 1
            idx += 1
    
    # Sort by score descending (higher score = should come earlier)
    # Break ties by original index
    order = np.argsort(-scores, kind='stable')
    return list(order)


def _neh_rule(processing_times):
    """NEH heuristic for flow-shop scheduling."""
    n_jobs, n_machines = processing_times.shape
    
    # Sort jobs by total processing time in descending order
    total_pt = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-total_pt)
    
    # Build the permutation incrementally
    permutation = [sorted_jobs[0]]
    
    for job in sorted_jobs[1:]:
        # Try inserting the job at each position
        best_pos = 0
        best_ms = float('inf')
        
        for pos in range(len(permutation) + 1):
            new_perm = permutation[:pos] + [job] + permutation[pos:]
            ms = _makespan(processing_times, new_perm)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        permutation = permutation[:best_pos] + [job] + permutation[best_pos:]
    
    return permutation


def _local_search(processing_times, order, max_iter=100):
    """2-opt style local search with adjacent swaps."""
    n_jobs = len(order)
    current_order = list(order)
    current_ms = _makespan(processing_times, current_order)
    
    improved = True
    iter_count = 0
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        
        for i in range(n_jobs - 1):
            # Try swapping adjacent jobs i and i+1
            new_order = current_order[:]
            new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
            
            new_ms = _makespan(processing_times, new_order)
            
            if new_ms < current_ms:
                current_order = new_order
                current_ms = new_ms
                improved = True
    
    return current_order, current_ms


def _simulated_annealing(processing_times, initial_order, start_temp=10.0, end_temp=0.01, 
                          cooling_rate=0.95, max_iter=500, deadline=None):
    """Simulated annealing with adjacent swaps."""
    n_jobs = len(initial_order)
    current_order = list(initial_order)
    current_ms = _makespan(processing_times, current_order)
    best_order = list(current_order)
    best_ms = current_ms
    
    temp = start_temp
    iter_count = 0
    
    while temp > end_temp and iter_count < max_iter:
        iter_count += 1
        
        # Check deadline
        if deadline is not None and time.time() > deadline:
            break
        
        # Generate a random adjacent swap
        i = np.random.randint(0, n_jobs - 1)
        j = i + 1
        
        new_order = current_order[:]
        new_order[i], new_order[j] = new_order[j], new_order[i]
        
        new_ms = _makespan(processing_times, new_order)
        delta = new_ms - current_ms
        
        # Accept if better or with probability exp(-delta/temp)
        if delta < 0 or np.random.random() < np.exp(-delta / max(temp, 1e-10)):
            current_order = new_order
            current_ms = new_ms
            
            if current_ms < best_ms:
                best_ms = current_ms
                best_order = list(current_order)
        
        temp *= cooling_rate
    
    return best_order, best_ms


def _gaussian_process_predict(X_train, y_train, X_test, length_scale=1.0, noise=1e-6, sigma_f=1.0):
    """Simple Gaussian Process regression for prediction.
    X_train: (n, d) training inputs
    y_train: (n,) training targets
    X_test: (m, d) test inputs
    Returns: mean predictions, variance predictions
    """
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    
    # Standardize y
    y_mean = np.mean(y_train)
    y_std = np.std(y_train) + 1e-10
    y_norm = (y_train - y_mean) / y_std
    
    # Compute kernel matrix
    # RBF kernel: K(x, x') = sigma_f^2 * exp(-||x - x'||^2 / (2 * length_scale^2))
    # For efficiency, compute squared distances
    diff = X_train[:, np.newaxis, :] - X_train[np.newaxis, :, :]
    sq_dist = np.sum(diff ** 2, axis=2)
    K = sigma_f ** 2 * np.exp(-sq_dist / (2.0 * length_scale ** 2))
    
    # Add noise
    K += noise * np.eye(n_train)
    
    # Solve for alpha
    try:
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
    except np.linalg.LinAlgError:
        # Fallback to direct solve
        try:
            alpha = np.linalg.solve(K, y_norm)
        except np.linalg.LinAlgError:
            # Very bad fallback: just use mean
            return np.full(n_test, y_mean), np.full(n_test, y_std ** 2)
    
    # Predict at test points
    diff_test = X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]
    sq_dist_test = np.sum(diff_test ** 2, axis=2)
    K_test = sigma_f ** 2 * np.exp(-sq_dist_test / (2.0 * length_scale ** 2))
    
    # Mean prediction
    mu = K_test @ alpha
    mu = mu * y_std + y_mean
    
    # Variance prediction (approximate)
    v = np.linalg.solve(L, K_test.T)
    var = sigma_f ** 2 - np.sum(v ** 2, axis=0)
    var = np.maximum(var, 1e-10) * y_std ** 2
    
    return mu, var


def _bayesian_optimization_guided_search(processing_times, initial_order, deadline, max_candidates=50):
    """Use Gaussian Process surrogate to guide search.
    Represents permutations as binary precedence vectors.
    """
    n_jobs = len(initial_order)
    if n_jobs < 3:
        return list(initial_order), _makespan(processing_times, initial_order)
    
    n_pairs = n_jobs * (n_jobs - 1) // 2
    
    # Initialize with some random permutations and the initial one
    X_list = []
    y_list = []
    perm_list = []
    
    # Add initial permutation
    vec = _permutation_to_binary_vector(initial_order, n_jobs)
    ms = _makespan(processing_times, initial_order)
    X_list.append(vec)
    y_list.append(ms)
    perm_list.append(list(initial_order))
    
    # Add some random permutations to build initial model
    n_init = min(20, max_candidates // 2)
    for _ in range(n_init):
        rand_perm = list(range(n_jobs))
        np.random.shuffle(rand_perm)
        vec = _permutation_to_binary_vector(rand_perm, n_jobs)
        ms = _makespan(processing_times, rand_perm)
        X_list.append(vec)
        y_list.append(ms)
        perm_list.append(rand_perm)
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    best_perm = perm_list[np.argmin(y)]
    best_ms = min(y)
    
    # GP parameters
    length_scale = 1.0
    noise = 1e-6
    sigma_f = 1.0
    
    # Iteratively propose new candidates using GP
    remaining = max_candidates - len(X_list)
    for it in range(remaining):
        if time.time() > deadline:
            break
        
        # Generate candidate points in the binary space (with some continuous relaxation)
        n_candidates = 10
        candidates = np.random.rand(n_candidates, n_pairs)
        
        # Also include some perturbations of the current best
        best_vec = _permutation_to_binary_vector(best_perm, n_jobs)
        for _ in range(5):
            perturbed = best_vec.copy()
            # Flip a few random bits
            n_flips = np.random.randint(1, min(5, n_pairs // 2) + 1)
            flip_idx = np.random.choice(n_pairs, size=n_flips, replace=False)
            perturbed[flip_idx] = 1.0 - perturbed[flip_idx]
            candidates = np.vstack([candidates, perturbed])
        
        # Predict mean and variance
        mu, var = _gaussian_process_predict(X, y, candidates, length_scale, noise, sigma_f)
        
        # Use Expected Improvement (EI) as acquisition function
        best_y = np.min(y)
        improvement = best_y - mu
        sigma = np.sqrt(var)
        z = improvement / (sigma + 1e-10)
        ei = improvement * (0.5 + 0.5 * np.sign(z)) + sigma * (0.5 * np.exp(-z ** 2 / 2.0))
        
        # Select candidate with highest EI
        best_idx = np.argmax(ei)
        selected_vec = candidates[best_idx]
        
        # Convert to permutation
        new_perm = _binary_vector_to_permutation(selected_vec, n_jobs)
        
        # Evaluate
        new_ms = _makespan(processing_times, new_perm)
        
        # Update GP training data
        X = np.vstack([X, selected_vec])
        y = np.append(y, new_ms)
        
        # Update best
        if new_ms < best_ms:
            best_ms = new_ms
            best_perm = new_perm
    
    return best_perm, best_ms


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on flow-shop instances."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n_jobs, n_machines = processing_times.shape
    
    # Handle edge cases
    if n_jobs <= 1:
        return list(range(n_jobs))
    if n_machines == 0:
        return list(range(n_jobs))
    
    # Start with NEH heuristic
    neh_order = _neh_rule(processing_times)
    best_order = list(neh_order)
    best_ms = _makespan(processing_times, best_order)
    
    # Check if we should continue
    if time.time() > deadline:
        return best_order
    
    # Local search refinement
    ls_order, ls_ms = _local_search(processing_times, best_order, max_iter=50)
    if ls_ms < best_ms:
        best_order = ls_order
        best_ms = ls_ms
    
    if time.time() > deadline:
        return best_order
    
    # Simulated annealing for further improvement
    sa_order, sa_ms = _simulated_annealing(processing_times, best_order, 
                                            start_temp=5.0, end_temp=0.05,
                                            cooling_rate=0.97, max_iter=300,
                                            deadline=deadline)
    if sa_ms < best_ms:
        best_order = sa_order
        best_ms = sa_ms
    
    # Bayesian optimization guided search using GP surrogate
    if time.time() < deadline - 10:
        bo_deadline = deadline - 5
        bo_order, bo_ms = _bayesian_optimization_guided_search(
            processing_times, best_order, deadline=bo_deadline, max_candidates=40
        )
        if bo_ms < best_ms:
            best_order = bo_order
            best_ms = bo_ms
    
    # Final local search on the best result
    if time.time() < deadline - 3:
        final_order, final_ms = _local_search(processing_times, best_order, max_iter=20)
        if final_ms < best_ms:
            best_order = final_order
            best_ms = final_ms
    
    return best_order
# EVOLVE-BLOCK-END
