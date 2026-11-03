"""Simulated Annealing with O(1) swap evaluation via interaction matrix."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _precompute_interaction(a, b):
    """
    Precompute the interaction matrix M such that M[i, j] represents the cost
    contribution associated with facilities i and j being assigned to locations
    x and y respectively, in a way that allows O(1) delta calculation for swaps.
    
    Actually, for QAP, the cost is sum_{i,j} a[i,j] b[p[i], p[j]].
    Let's define the "interaction" between facility i and facility j given
    locations x=p[i] and y=p[j] as a[i,j] * b[x,y] + a[j,i] * b[y,x].
    Since a and b are symmetric in tai256c (and generally in QAPLIB), 
    a[i,j]=a[j,i] and b[x,y]=b[y,x], so term = 2 * a[i,j] * b[x,y].
    
    However, for a general swap delta, we need to consider how the cost changes
    when p[i] and p[j] are swapped.
    Let x = p[i], y = p[j].
    Old terms involving i or j:
    T_old = sum_{k != i,j} (a[i,k] b[x, p[k]] + a[k,i] b[p[k], x] + a[j,k] b[y, p[k]] + a[k,j] b[p[k], y])
            + a[i,j] b[x,y] + a[j,i] b[y,x]
            
    New terms (swap i and j, so p[i]=y, p[j]=x):
    T_new = sum_{k != i,j} (a[i,k] b[y, p[k]] + a[k,i] b[p[k], y] + a[j,k] b[x, p[k]] + a[k,j] b[p[k], x])
            + a[i,j] b[y,x] + a[j,i] b[x,y]
            
    Since a,b symmetric:
    T_old = sum_{k != i,j} (a[i,k] + a[k,i]) b[x, p[k]] + sum_{k != i,j} (a[j,k] + a[k,j]) b[y, p[k]] + 2 a[i,j] b[x,y]
    T_new = sum_{k != i,j} (a[i,k] + a[k,i]) b[y, p[k]] + sum_{k != i,j} (a[j,k] + a[k,j]) b[x, p[k]] + 2 a[i,j] b[x,y]
    
    Delta = T_new - T_old
    Delta = sum_{k != i,j} (a[i,k] + a[k,i]) (b[y, p[k]] - b[x, p[k]]) 
            + sum_{k != i,j} (a[j,k] + a[k,j]) (b[x, p[k]] - b[y, p[k]])
    Delta = sum_{k != i,j} [ (a[i,k] + a[k,i]) - (a[j,k] + a[k,j]) ] * (b[y, p[k]] - b[x, p[k]])
    
    This is still O(n). To get O(1), we need to precompute something else.
    
    Let's define C[i, x] = sum_{k} a[i,k] b[x, p[k]] + sum_{k} a[k,i] b[p[k], x] ? No, p changes.
    
    Alternative O(1) swap evaluation strategy:
    Maintain an array `cost_contrib[i]` which is the total cost contribution of facility i
    to the total sum, i.e., sum_{j} a[i,j] b[p[i], p[j]] + sum_{j} a[j,i] b[p[j], p[i]].
    Note: sum_{i} cost_contrib[i] = 2 * Total_Cost (since every pair is counted twice).
    
    When we swap p[i] and p[j] (x=p[i], y=p[j]):
    The cost_contrib for i changes. The cost_contrib for j changes.
    The cost_contrib for any other k changes because p[i] and p[j] changed.
    This makes maintaining cost_contrib expensive to update for all k.
    
    Let's stick to the O(n) delta but optimize it with NumPy vectorization, 
    and rely on SA's ability to do many quick evaluations.
    Actually, O(n) with n=256 is very fast in NumPy. 256 multiplications and additions.
    The bottleneck in the previous code was likely the Python loop overhead 
    and the fact that it did a full pass (O(n^2) swaps) checking all pairs.
    SA does 1 swap per step, so it's O(n) per step.
    If we can do ~100,000 steps in 120s, that's ~1000 steps/sec.
    1000 * 256 ops = 256,000 ops/sec. Very feasible.
    
    So, I will implement SA with the O(n) vectorized delta calculation.
    """
    # We don't strictly need a precomputed matrix for O(1) if we use the vectorized O(n) delta.
    # But we can precompute a_sym = a + a.T to save one addition in the delta calc.
    a_sym = a + a.T
    return a_sym

def _swap_delta(a_sym, b, p, i, j):
    """
    Compute delta for swapping positions i and j in permutation p.
    a_sym[i, k] = a[i, k] + a[k, i]
    """
    x = p[i]
    y = p[j]
    
    # diff_a[k] = a_sym[i, k] - a_sym[j, k]
    diff_a = a_sym[i, :] - a_sym[j, :]
    # diff_b[k] = b[y, k] - b[x, k]
    # We need sum_{k != i,j} diff_a[k] * diff_b[p[k]]
    
    # Vectorized:
    # term = diff_a * diff_b[p]
    # But we must exclude k=i and k=j.
    
    # diff_b is b[y, :] - b[x, :]
    diff_b = b[y, :] - b[x, :]
    
    # Calculate sum over all k first
    # temp[k] = diff_a[k] * diff_b[p[k]]
    temp = diff_a * diff_b[p]
    
    # Subtract terms for k=i and k=j
    # For k=i: diff_a[i] * diff_b[p[i]] = diff_a[i] * diff_b[x]
    # For k=j: diff_a[j] * diff_b[p[j]] = diff_a[j] * diff_b[y]
    
    delta = np.sum(temp) - diff_a[i] * diff_b[x] - diff_a[j] * diff_b[y]
    
    return delta

def _solve_tai256c(a, b, time_limit=115.0):
    n = len(a)
    start_time = time.time()
    
    # Precompute
    a_sym = _precompute_interaction(a, b)
    
    # Initial solution: Rank matching (greedy)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.copy()
    
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Simulated Annealing Parameters
    # Start temperature high enough to accept most moves
    # End temperature low enough to accept only improvements
    T_start = 1000.0
    T_end = 1.0
    # Cooling rate: T_{t+1} = T_t * cooling_rate
    # We want to run for a fixed number of steps or until time limit.
    # Let's estimate steps. 
    # Each step is ~O(n) numpy ops. ~50-100us per step?
    # Let's aim for 50,000 steps.
    # If T_start = 1000, T_end = 1, and N=50000
    # T_end = T_start * alpha^N => 1 = 1000 * alpha^50000
    # alpha = (1/1000)^(1/50000) = exp(ln(1/1000)/50000) = exp(-6.9/50000) ≈ 1 - 0.000138
    # Let's just use a fixed number of steps and dynamic cooling.
    
    max_steps = 200000  # Will be limited by time anyway
    
    current_T = T_start
    # Estimate cooling rate to reach T_end in max_steps
    cooling_rate = (T_end / T_start) ** (1.0 / max_steps)
    
    rng = np.random.default_rng(42)
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break
            
        # Pick two random distinct indices
        i, j = rng.choice(n, size=2, replace=False)
        
        # Calculate delta
        delta = _swap_delta(a_sym, b, p, i, j)
        
        # Acceptance criterion
        if delta < 0 or rng.random() < np.exp(-delta / current_T):
            # Accept move
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        # Cool down
        current_T *= cooling_rate
        if current_T < T_end:
            current_T = T_end
            
    return best_p.tolist()

def solve(a, b):
    return _solve_tai256c(a, b)
# EVOLVE-BLOCK-END
