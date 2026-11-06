"""
Strategy: Non-Parametric Frequency Aggregation via Stochastic Greedy Trials
Instead of fitting a parametric model (like the neural surrogate in the parent) 
to a single reference set, we treat the priority function as a direct measure 
of a vector's "survival rate" across many stochastic greedy constructions.

We run multiple greedy trials where the order of vectors is determined by a 
base heuristic (e.g., weight) but with random tie-breaking and slight 
randomization in the initial selection. We count how frequently each vector 
appears in the final cap sets. Vectors that consistently appear in large cap 
sets receive high priority. This non-parametric approach avoids the 
overfitting and convergence issues of the neural surrogate and better 
captures the combinatorial structure of cap sets.
"""

import time
import random
from collections import defaultdict

# Global state
_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0  # Safety margin before 300s kill
_FREQ_COUNTS = None
_IS_INITIALIZED = False
_N_CACHE = {}

def _int_to_tuple(i, n):
    """Convert integer to tuple in Z_3^n."""
    res = []
    for _ in range(n):
        res.append(i % 3)
        i //= 3
    return tuple(res)

def _tuple_to_int(v):
    """Convert tuple in Z_3^n to integer."""
    res = 0
    for i, x in enumerate(v):
        res += x * (3 ** i)
    return res

def _is_cap_set_incremental(existing_set, new_vec, n):
    """
    Check if adding new_vec to existing_set keeps it a cap set.
    existing_set is a list of tuples.
    """
    for i in range(len(existing_set)):
        a = existing_set[i]
        for j in range(i + 1, len(existing_set)):
            b = existing_set[j]
            # Check if a, b, new_vec form an AP
            is_ap = True
            for k in range(n):
                if (a[k] + b[k] + new_vec[k]) % 3 != 0:
                    is_ap = False
                    break
            if is_ap:
                return False
    return True

def _greedy_trial(n, all_vectors, seed, time_limit):
    """
    Run a single greedy trial.
    Returns the set of vectors added.
    """
    random.seed(seed)
    
    # Base priority: weight (number of non-zero elements)
    # We want to explore different orderings.
    # Sort by weight, then randomize within weight classes.
    # Actually, for speed, we just shuffle the entire list but bias towards 
    # "good" weights? No, standard greedy with random order is a good baseline.
    # Let's use a slightly smarter base: sort by weight, then shuffle.
    # Vectors with weight ~ n/3 are often good.
    
    # For efficiency, we pre-compute weights? No, n is small enough.
    
    # Strategy: 
    # 1. Shuffle vectors.
    # 2. Iterate.
    # To make trials diverse, we can shuffle.
    
    shuffled = list(all_vectors)
    random.shuffle(shuffled)
    
    cap_set = []
    cap_set_set = set()
    
    start_t = time.time()
    
    for v in shuffled:
        if time.time() - start_t > time_limit:
            break
        if _is_cap_set_incremental(cap_set, v, n):
            cap_set.append(v)
            cap_set_set.add(v)
            
    return cap_set_set

def _initialize(n, time_budget):
    """
    Run multiple greedy trials to compute frequency counts.
    """
    global _FREQ_COUNTS, _IS_INITIALIZED, _N_CACHE
    
    if _IS_INITIALIZED and _N_CACHE.get('n') == n:
        return
        
    _N_CACHE['n'] = n
    _FREQ_COUNTS = defaultdict(int)
    
    total_vectors = 3 ** n
    all_vectors = [_int_to_tuple(i, n) for i in range(total_vectors)]
    
    # Determine number of trials based on n and time budget
    # For n=6, 3^6 = 729 vectors. Greedy is fast.
    # For n=10, 3^10 = 59049 vectors. Greedy is slower.
    
    start = time.time()
    num_trials = 0
    max_trials = 1000
    
    # Estimate time per trial
    # We'll just run trials until time runs out.
    
    while time.time() - start < time_budget * 0.8:
        seed = int(time.time() * 1000) + num_trials
        trial_set = _greedy_trial(n, all_vectors, seed, time_budget * 0.1)
        
        for v in trial_set:
            _FREQ_COUNTS[v] += 1
            
        num_trials += 1
        if num_trials >= max_trials:
            break
            
    _IS_INITIALIZED = True

def priority(vector, n):
    """
    Returns priority for a vector.
    The evaluator sorts vectors by descending priority.
    """
    global _IS_INITIALIZED, _FREQ_COUNTS, _N_CACHE
    
    # Initialize if needed
    if not _IS_INITIALIZED or _N_CACHE.get('n') != n:
        # Check deadline
        if time.time() > _DEADLINE:
            _IS_INITIALIZED = True
            _FREQ_COUNTS = defaultdict(int)
        else:
            remaining = _DEADLINE - time.time()
            # Use up to 25s for initialization
            budget = min(25.0, remaining * 0.5)
            _initialize(n, budget)
    
    # Return priority based on frequency
    if _FREQ_COUNTS and vector in _FREQ_COUNTS:
        # Add a small deterministic tie-breaker to ensure strict ordering if needed
        # The evaluator likely handles ties, but a stable order is good.
        # We want higher frequency = higher priority.
        freq = _FREQ_COUNTS[vector]
        
        # Tie-breaker: use the vector's integer representation
        # This ensures that if two vectors have the same frequency, 
        # they are ordered deterministically.
        tie_breaker = _tuple_to_int(vector)
        
        # Scale frequency to be the dominant factor
        # Max possible frequency is num_trials.
        # We just return freq + small epsilon * tie_breaker
        # But float precision might be an issue if freq is large.
        # Let's just return freq. The evaluator's sort is stable? 
        # Usually, we want a total order.
        # If we return just freq, ties are broken by input order (stable sort).
        # To be safe, we can mix in a tiny hash.
        return float(freq) + (tie_breaker * 1e-9)
    else:
        # If vector was never seen (unlikely if we iterated all), 
        # or if initialization failed, use a fallback heuristic.
        weight = sum(1 for x in vector if x != 0)
        # Prefer medium weights
        score = -abs(weight - n / 3.0)
        tie_breaker = _tuple_to_int(vector)
        return score + tie_breaker * 1e-9

# EVOLVE-BLOCK-END
