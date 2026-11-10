import numpy as np
import time
import sys
from scipy.optimize import minimize

def _precompute_tables():
    """Precompute index mappings for fast bincount-based ratio evaluation."""
    n = 10
    y_idx, x_idx = np.indices((n, n))
    
    # x - y range: -9 to 9 (19 values)
    diff = x_idx - y_idx
    diff_idx = (diff + 9).ravel()
    
    # x range: 0 to 9 (10 values)
    x_idx_flat = x_idx.ravel()
    
    # x + y range: 0 to 18 (19 values)
    sum_idx = (x_idx + y_idx).ravel()
    
    # x + 2y range: 0 to 27 (28 values)
    sum2_idx = (x_idx + 2 * y_idx).ravel()
    
    return diff_idx, x_idx_flat, sum_idx, sum2_idx

# Pre-compute indices at module level for speed
_DIFF_IDX, _X_IDX, _SUM_IDX, _SUM2_IDX = _precompute_tables()

def _fast_ratio(P):
    """
    Compute D/M ratio efficiently using precomputed indices.
    P is a flat array of 100 elements (row-major, P[y, x]).
    Returns 0.0 if invalid.
    """
    # Check finite
    if not np.all(np.isfinite(P)):
        return 0.0
    
    # Clip negatives
    P = np.maximum(P, 0.0)
    
    total = P.sum()
    if total <= 1e-9:
        return 0.0
    
    # Normalize
    P = P / total
    
    # Compute distributions
    # H_{-1}: x - y
    dist_diff = np.bincount(_DIFF_IDX, weights=P, minlength=19)
    p_diff = dist_diff[dist_diff > 0]
    H_diff = -np.sum(p_diff * np.log2(p_diff))
    
    # H_0: x
    dist_x = np.bincount(_X_IDX, weights=P, minlength=10)
    p_x = dist_x[dist_x > 0]
    H_x = -np.sum(p_x * np.log2(p_x))
    
    # H_1: x + y
    dist_sum = np.bincount(_SUM_IDX, weights=P, minlength=19)
    p_sum = dist_sum[dist_sum > 0]
    H_sum = -np.sum(p_sum * np.log2(p_sum))
    
    # H_2: x + 2y
    dist_sum2 = np.bincount(_SUM2_IDX, weights=P, minlength=28)
    p_sum2 = dist_sum2[dist_sum2 > 0]
    H_sum2 = -np.sum(p_sum2 * np.log2(p_sum2))
    
    M = max(H_x, H_sum, H_sum2)
    
    if M <= 1e-9:
        return 0.0
        
    return H_diff / M

def _evaluate_candidate(logits):
    """
    Evaluate a candidate given in logit space (softmax parameterization).
    """
    x_max = np.max(logits)
    exp_x = np.exp(logits - x_max)
    P = exp_x / np.sum(exp_x)
    return _fast_ratio(P)

def _polish(logits):
    """
    Use L-BFGS-B to polish a candidate from logit space.
    """
    def objective(x):
        return -_evaluate_candidate(x)
    
    try:
        res = minimize(
            objective,
            logits,
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        return res.x, -res.fun
    except:
        return logits, _evaluate_candidate(logits)

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin
    
    # 1. Generate structured seeds in logit space
    # The idea is to create distributions that are "structured" in a way that
    # might maximize the difference between H_diff and the max of others.
    
    seeds = []
    
    # Uniform
    seeds.append(np.zeros(100))
    
    # Random seeds
    np.random.seed(42)
    for _ in range(50):
        seeds.append(np.random.randn(100) * 1.0)
        
    # Structured seeds based on arithmetic progressions
    # We want H_diff (x-y) to be high, while H_x, H_sum, H_sum2 to be low.
    # High H_diff means x-y is spread out.
    # Low H_x means x is concentrated.
    # Low H_sum means x+y is concentrated.
    # Low H_sum2 means x+2y is concentrated.
    
    # If we concentrate mass on a line x = c*y + k, then:
    # x - y = (c-1)y + k. If c != 1, this is spread out.
    # x = c*y + k. If c != 0, this is spread out (bad for low H_x).
    # Wait, we want M = max(H_x, H_sum, H_sum2) to be SMALL.
    # So we want x, x+y, x+2y to be concentrated.
    # This implies y must be concentrated, and x must be concentrated.
    # But then x-y is also concentrated.
    # This suggests the optimum is not a simple line.
    
    # Let's try seeds that are "checkerboard" or "diagonal" patterns.
    x_idx, y_idx = np.indices((n, n))
    
    # Diagonal x=y -> H_diff = 0. Bad.
    # Anti-diagonal x+y=9 -> H_sum = 0. Good for M.
    # Let's try to make H_x, H_sum, H_sum2 small but not zero, and H_diff large.
    
    # Try a "block" pattern
    for bx in range(0, 10, 2):
        for by in range(0, 10, 2):
            s = np.zeros(100)
            s[by*10 + bx] = 1.0
            seeds.append(s)
            
    # Try "L" shapes
    for i in range(10):
        s = np.zeros(100)
        s[i*10:(i+1)*10] = 1.0 # Row
        seeds.append(s)
        s = np.zeros(100)
        s[i::10] = 1.0 # Col
        seeds.append(s)
        
    # 2. Evolutionary Strategy
    # Population size 50
    pop_size = 50
    population = [s.copy() for s in seeds[:pop_size]]
    
    # Fill remaining with random
    while len(population) < pop_size:
        population.append(np.random.randn(100) * 1.0)
        
    scores = np.array([_evaluate_candidate(p) for p in population])
    best_idx = np.argmax(scores)
    best_score = scores[best_idx]
    best_logits = population[best_idx].copy()
    
    # CMA-ES like step
    # We will use a simple DE-like mutation: x_new = x_best + F * (x_r1 - x_r2)
    F = 0.8
    CR = 0.9
    sigma = 1.0 # Initial step size
    
    while time.time() < deadline:
        new_population = [best_logits.copy()] # Elitism
        new_scores = [best_score]
        
        # Generate offspring
        for i in range(1, pop_size):
            # Pick 3 distinct individuals
            r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
            
            # Mutation
            v = population[r1] + F * (population[r2] - population[r3])
            
            # Crossover
            trial = np.zeros(100)
            mask = np.random.random(100) < CR
            trial[mask] = v[mask]
            trial[~mask] = population[i][~mask]
            
            # Evaluate
            score = _evaluate_candidate(trial)
            
            if score > scores[i]:
                new_population.append(trial)
                new_scores.append(score)
            else:
                new_population.append(population[i])
                new_scores.append(scores[i])
                
        population = new_population
        scores = np.array(new_scores)
        
        # Update best
        current_best_idx = np.argmax(scores)
        if scores[current_best_idx] > best_score:
            best_score = scores[current_best_idx]
            best_logits = population[current_best_idx].copy()
            
        # Periodic Polish
        if time.time() < deadline - 60: # Leave time for final polish
            if (population.index(best_logits) if best_logits in population else 0) % 10 == 0:
                polished_logits, polished_score = _polish(best_logits)
                if polished_score > best_score:
                    best_score = polished_score
                    best_logits = polished_logits
                    # Replace best in population
                    population[0] = best_logits
                    scores[0] = best_score
        
        # Decay sigma? Not used directly in DE, but we can add noise
        # Let's just keep F and CR constant for simplicity and robustness
        
    # Final Polish
    polished_logits, polished_score = _polish(best_logits)
    if polished_score > best_score:
        best_score = polished_score
        best_logits = polished_logits
        
    # Convert back to P
    x_max = np.max(best_logits)
    exp_x = np.exp(best_logits - x_max)
    best_P = exp_x / np.sum(exp_x)
    
    return best_P.reshape((n, n))
