import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def get_points_from_weights(weights):
    """
    weights: array of 20 positive numbers (arc lengths).
    Returns (20, 2) array of points on the unit circle.
    """
    w_sum = np.sum(weights)
    if w_sum <= 0:
        w_sum = 1e-6
    angles = np.cumsum(weights / w_sum * 2 * np.pi)
    angles = angles % (2 * np.pi)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.column_stack((x, y))

def params_to_weights(params):
    """
    Convert 19 parameters to 20 positive weights.
    params[0] is fixed or used for w0?
    Let's use 19 params to define 19 weights, and w20 is derived?
    Or 20 params?
    The problem has 20 vertices. We can fix one angle to 0 (rotation invariance).
    So we have 19 degrees of freedom for the relative angles.
    Let u be 19 vectors.
    We need 20 weights.
    Let u = [u1, ..., u19].
    Let w_i = exp(u_i) for i=1..19.
    Let w_0 = 1. (Or exp(u0) if we had u0, but we fixed rotation).
    Actually, it's better to have all weights variable.
    If we fix vertex 0 at angle 0, the other 19 angles are determined by cumulative sums of 19 positive steps.
    So we need 19 positive steps $s_1, \dots, s_{19}$.
    Angle 0 = 0.
    Angle 1 = s_1
    Angle 2 = s_1 + s_2
    ...
    Angle 19 = sum(s_1..s_19).
    We don't need the 20th step to close the loop?
    The distance from 19 to 0 is determined by the remaining arc.
    The "weights" $w_i$ are the arc lengths.
    $w_0$ is the arc from 19 to 0.
    $w_i$ is the arc from $i-1$ to $i$ for $i=1..19$.
    Sum $w_i = 2\pi$.
    We have 20 variables $w_i$. Constraint sum=2pi. 19 DOF.
    Let $x$ be 19 variables.
    $w_i = \exp(x_i)$ for $i=1..19$.
    $w_0 = 2\pi - \sum_{i=1}^{19} \exp(x_i)$.
    We must ensure $w_0 > 0$.
    So $\sum \exp(x_i) < 2\pi$.
    This is a constraint.
    To handle this easily, we can map $x_i$ to a range that ensures the sum is small.
    Or use a softmax-like normalization on 20 variables, but fix one?
    
    Alternative: Use 20 variables $y_i$.
    $w_i = y_i^2 + \epsilon$.
    Normalize to sum $2\pi$.
    This ensures positivity.
    We have 20 variables, but one is redundant (scale invariance is removed by normalization).
    So effectively 19 DOF.
    We can just optimize 20 variables and normalize.
    The cost function is invariant to scaling of $y$.
    This might be fine.
    """
    w = params**2 + 0.01
    w = w / np.sum(w)
    return w * 2 * np.pi

def get_points(params):
    w = params_to_weights(params)
    angles = np.cumsum(w)
    angles = angles % (2 * np.pi)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.column_stack((x, y))

def vertex_score(v_idx, points):
    dists = np.linalg.norm(points - points[v_idx], axis=1)
    dists[v_idx] = np.inf
    sorted_dists = np.sort(dists)
    if sorted_dists[0] < 0.05:
        return 0.0
    gaps = sorted_dists[1:] - sorted_dists[:-1]
    k = np.argmin(gaps)
    D = (sorted_dists[k] + sorted_dists[k+1]) / 2.0
    if D < 0.1:
        return 0.0
    if k == 0:
        d_sel = sorted_dists[0:4]
    elif k == 17:
        d_sel = sorted_dists[15:19]
    else:
        d_sel = sorted_dists[k-1:k+3]
    ratios = np.maximum(d_sel / D, D / d_sel)
    raw = np.mean(ratios)
    score = min(1.0, 1.0 / raw)
    return score

def min_vertex_score(params):
    points = get_points(params)
    scores = np.array([vertex_score(i, points) for i in range(20)])
    return -np.min(scores)

def soft_min_vertex_score(params, beta=50.0):
    """
    Smooth approximation of min_vertex_score using log-sum-exp.
    We want to maximize the minimum score.
    Let S_i be the score of vertex i.
    We want to maximize min(S_i).
    This is equivalent to minimizing -min(S_i) = max(-S_i).
    A soft-max of -S_i is: (1/beta) * log(sum(exp(-beta * S_i))).
    So we minimize this.
    """
    points = get_points(params)
    scores = np.array([vertex_score(i, points) for i in range(20)])
    # We want to maximize min(scores).
    # Equivalent to minimizing -min(scores).
    # Soft max of (-scores) is (1/beta) * log(sum(exp(-beta * scores)))
    # Wait, max(x) = lim_{beta->inf} (1/beta) log sum exp(beta x).
    # So max(-scores) = (1/beta) log sum exp(-beta scores).
    # We want to minimize this.
    # So the objective function to minimize is (1/beta) log sum exp(-beta scores).
    
    # However, scores can be 0. If any score is 0, exp(0) = 1, so the sum is at least 1.
    # The objective will be positive.
    # As beta increases, it approaches the exact max(-scores) = -min(scores).
    
    # To avoid overflow in exp, we can shift.
    # Let m = max(-scores) = -min(scores).
    # sum exp(-beta scores) = exp(beta m) * sum exp(-beta (scores - m))
    # log sum = beta m + log sum exp(-beta (scores - m))
    # obj = m + (1/beta) log sum exp(-beta (scores - m))
    
    neg_scores = -scores
    m = np.max(neg_scores)
    # If all scores are 0, m=0, exp(0)=1, sum=20, obj = (1/beta) log(20).
    # If one score is 1, others 0, m=0 (since -1 < 0), wait.
    # scores are in [0, 1].
    # neg_scores in [-1, 0].
    # m is in [-1, 0].
    
    # Let's just use the standard stable log-sum-exp.
    a = -beta * scores
    a_max = np.max(a)
    if a_max > 500:
        # Overflow risk, but scores are bounded, so -beta*scores is bounded by beta.
        # If beta=50, max is 50. exp(50) is safe.
        pass
    sum_exp = np.sum(np.exp(a - a_max))
    log_sum = a_max + np.log(sum_exp)
    obj = log_sum / beta
    return obj

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590
    
    n = 20
    # Initial guess: regular polygon
    # weights all equal 2pi/20
    # params y such that y^2 ~ 1
    x0 = np.ones(20)
    
    best_params = x0
    best_score = -min_vertex_score(x0)
    
    # Phase 1: Differential Evolution
    # Bounds for y: [0.1, 10]
    bounds = [(0.1, 10.0) for _ in range(20)]
    
    if time.time() < deadline - 100: # Leave time for refinement
        try:
            res_de = differential_evolution(
                min_vertex_score,
                bounds,
                maxiter=200,
                popsize=15,
                tol=1e-8,
                update_interval=1,
                mutation=(0.5, 1.5),
                recombination=0.9
            )
            if res_de.fun < best_score:
                best_score = -res_de.fun
                best_params = res_de.x
        except Exception:
            pass

    # Phase 2: Local Refinement with L-BFGS-B using soft-min
    # This helps navigate the non-smooth min landscape.
    rng = np.random.default_rng(42)
    
    restarts = [best_params]
    # Add some random restarts
    for i in range(5):
        if time.time() > deadline - 20:
            break
        # Random restart
        x_rand = rng.uniform(0.5, 2.0, size=20)
        restarts.append(x_rand)
        
    # Also try perturbations of the best solution
    for i in range(5):
        if time.time() > deadline - 20:
            break
        x_pert = best_params + rng.normal(0, 0.1, size=20)
        restarts.append(x_pert)

    current_beta = 10.0
    for x_init in restarts:
        if time.time() > deadline - 10:
            break
        try:
            # Annealing beta
            for beta in [5.0, 10.0, 20.0, 50.0, 100.0, 200.0]:
                if time.time() > deadline - 5:
                    break
                res_lm = minimize(
                    soft_min_vertex_score,
                    x_init,
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res_lm.fun < 1e-10: # If soft-min is close to 0, it means scores are close to 1?
                    # Actually, soft_min returns positive value.
                    # We track best_params based on exact score.
                    pass
                
                # Evaluate exact score for the result
                exact_score = -min_vertex_score(res_lm.x)
                if exact_score > best_score:
                    best_score = exact_score
                    best_params = res_lm.x
                    x_init = res_lm.x # Use this as starting point for higher beta
        except Exception:
            pass

    # Phase 3: Final polish with exact min (or high beta)
    # Try a few more steps with the exact objective to fine-tune
    for _ in range(3):
        if time.time() > deadline - 5:
            break
        try:
            res_final = minimize(
                min_vertex_score,
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res_final.fun < best_score:
                best_score = -res_final.fun
                best_params = res_final.x
        except Exception:
            break

    return get_points(best_params)
