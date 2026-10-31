import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve

N = 50

def compute_ratio(h):
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    conv = fftconvolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-15:
        return 0.0
    return h2_sq / conv_inf

def is_sidon_set(indices):
    """Check if a set of indices forms a B2 (Sidon) set: all pairwise sums are distinct."""
    indices = sorted(indices)
    sums = set()
    for i in range(len(indices)):
        for j in range(i, len(indices)):
            s = indices[i] + indices[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def build_addition_graph(N):
    """
    Build the addition graph on {0, ..., N-1}.
    Two indices a, b are adjacent if there exists another pair (c, d) with c+d = a+b,
    where (c,d) != (a,b) and (c,d) != (b,a). This includes self-sums: a+a collides
    with c+d if c+d=2a and {c,d} != {a,a}.
    An independent set in this graph is a Sidon set.
    """
    # For each sum s, collect all unordered pairs (i,j) with i<=j and i+j=s
    sum_to_pairs = {}
    for i in range(N):
        for j in range(i, N):
            s = i + j
            if s not in sum_to_pairs:
                sum_to_pairs[s] = []
            sum_to_pairs[s].append((i, j))
    
    # Build adjacency: a and b conflict if there's a sum s where both (a,b) and some other pair exist
    # Actually, we need: a and b are "incompatible" if adding b to a set containing a creates a collision.
    # More precisely: in a Sidon set, all pairwise sums (i,j) with i<=j must be distinct.
    # Two elements a, b (a<b) conflict if there exists c,d in the set such that a+b = c+d with {c,d} != {a,b}.
    # But for graph-based MIS, we want: an edge between a and b means they CANNOT both be in a Sidon set.
    # a and b cannot both be in a Sidon set if:
    #   - There exist c != a, c != b such that a + c = b + c' for some c'... 
    #   Actually simpler: a and b conflict if 2a = a + b (i.e., a=b, impossible for distinct) 
    #   or if there's a third element c such that a+c = b+c (impossible) or a+a = b+c or a+b = c+c etc.
    # 
    # The correct condition: a and b (a < b) are incompatible in a Sidon set if there exists c (c != a, c != b)
    # such that a + c = b + c (no), or a + a = b + c (c = 2a - b, need c != a, c != b, c >= 0, c < N),
    # or a + b = c + c (c = (a+b)/2, need c integer, c != a, c != b, c >= 0, c < N).
    #
    # Wait, that's not quite right either. In a Sidon set, the condition is that ALL pairwise sums are distinct.
    # If we have elements {a, b} plus potentially others, the constraint is that no two different pairs give the same sum.
    # For just {a, b}: sums are a+a, a+b, b+b. These are all distinct as long as a != b.
    # The conflict arises when we try to ADD a third element c. But for a graph-based approach,
    # we can define: a and b are "adjacent" (conflicting) if they CANNOT coexist in ANY Sidon set.
    # a and b cannot coexist if: there exists c such that {a,b,c} is not Sidon.
    # But that's not a pairwise property...
    #
    # Better approach: Use the standard result. A set is Sidon iff all pairwise sums i+j (i<=j) are distinct.
    # Two elements a, b (a<b) are "2-dependent" if 2a = a+b (impossible) or a+b = 2b (impossible) 
    # or 2a = 2b (impossible). So any two distinct elements can coexist in a Sidon set of size 2.
    # The constraint is higher-order. So a simple graph MIS won't directly work.
    #
    # Alternative: Use a greedy approach with backtracking, or use the addition table more directly.
    # Let me instead implement a greedy Sidon set construction with random restarts.
    return None

def generate_sidon_sets():
    """Generate several Sidon sets of various sizes within [0, N)."""
    sidon_sets = []
    base_sets = [
        [0, 1, 4, 10, 12, 17, 20, 24, 28, 30, 35, 40],
        [0, 1, 4, 9, 15, 21, 27, 33, 39, 44, 48],
        [0, 2, 5, 11, 14, 20, 26, 32, 38, 43, 47],
        [0, 1, 3, 7, 12, 18, 25, 31, 38, 44, 49],
        [0, 1, 4, 10, 12, 17, 20, 24, 28, 30],
        [0, 1, 4, 9, 15, 21, 27, 33, 39],
        [0, 1, 4, 10, 12, 17, 20, 24],
        [0, 1, 4, 9, 15, 21, 27],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 9, 15, 21],
        [0, 1, 4, 10, 12],
        [0, 1, 4, 9, 15],
        [0, 1, 4, 10],
        [0, 1, 4, 9],
        [0, 1, 4],
        [0, 1, 3],
    ]
    
    for s in base_sets:
        if all(0 <= x < N for x in s) and is_sidon_set(s):
            sidon_sets.append(list(s))
    
    np.random.seed(123)
    for _ in range(5000):
        k = np.random.randint(4, 11)
        indices = sorted(np.random.choice(N, k, replace=False).tolist())
        if is_sidon_set(indices):
            found = False
            for existing in sidon_sets:
                if len(existing) == k and set(existing) == set(indices):
                    found = True
                    break
            if not found:
                sidon_sets.append(indices)
                if len(sidon_sets) > 50:
                    break
    
    return sidon_sets

def optimize_weights_for_support(support, beta=100.0, maxiter=1000, ftol=1e-12, n_restarts=3):
    """
    Optimize weights h_i for i in support to minimize max convolution coefficient.
    Uses smooth max surrogate: (1/beta) * log(sum(exp(beta * conv_k)))
    Parameterization: h = softmax(x) ensures positivity and sum=1.
    Multiple random restarts to escape local minima.
    """
    k = len(support)
    if k == 0:
        return np.zeros(N), 0.0
    
    def softmax(x):
        x_shifted = x - np.max(x)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x)
    
    def objective_and_grad(x):
        h = softmax(x)
        h_full = np.zeros(N)
        h_full[support] = h
        
        conv = fftconvolve(h_full, h_full, mode='full')
        max_c = np.max(conv)
        if max_c < 1e-15:
            return 0.0, np.zeros(k)
        
        shifted = conv - max_c
        exp_shifted = np.exp(beta * shifted)
        sum_exp = np.sum(exp_shifted)
        weights = exp_shifted / sum_exp
        
        obj_val = max_c + (1.0 / beta) * np.log(sum_exp)
        
        grad_h = np.zeros(N)
        for m in support:
            hm = h_full[m]
            if hm > 1e-15:
                grad_h[:N] += 2.0 * hm * weights[m : m + N]
        
        grad_h_support = grad_h[support]
        dot_prod = np.dot(grad_h_support, h)
        grad_x = h * (grad_h_support - dot_prod)
        
        return obj_val, grad_x
    
    best_h_full = None
    best_sc = 0.0
    
    np.random.seed(42)
    for restart in range(n_restarts):
        if restart == 0:
            x0 = np.zeros(k)
        else:
            x0 = np.random.randn(k) * 0.5
        
        try:
            res = minimize(objective_and_grad, x0, method='L-BFGS-B', 
                           jac=True, options={'maxiter': maxiter, 'ftol': ftol, 'gtol': 1e-10})
            x_opt = res.x
            h_opt_local = softmax(x_opt)
            
            h_full = np.zeros(N)
            h_full[support] = h_opt_local
            sc = compute_ratio(h_full)
            if sc > best_sc:
                best_sc = sc
                best_h_full = h_full.copy()
        except Exception:
            pass
    
    if best_h_full is None:
        h_full = np.zeros(N)
        h_full[support] = 1.0 / k
        best_sc = compute_ratio(h_full)
        best_h_full = h_full
    
    return best_h_full, best_sc

def local_search_polish(h, support, deadline=None):
    """
    Local search polish: try small perturbations of weights to improve the ratio.
    Uses coordinate descent and random restarts on the simplex.
    """
    best_h = h.copy()
    best_score = compute_ratio(best_h)
    best_support = sorted(support)
    k = len(support)
    
    def check_time():
        if deadline is not None:
            return time.time() < deadline
        return True
    
    # Coordinate descent: try adjusting each weight slightly
    for iteration in range(50):
        if not check_time():
            break
        improved = False
        
        for i in range(k):
            if not check_time():
                break
            # Try increasing weight i and decreasing others proportionally
            for delta in [0.01, 0.05, 0.1, 0.2]:
                h_new = best_h.copy()
                h_new[support[i]] += delta
                # Renormalize
                l1 = np.sum(h_new)
                if l1 < 1e-15:
                    continue
                h_new /= l1
                # Ensure non-negativity
                h_new = np.maximum(h_new, 0)
                l1 = np.sum(h_new)
                if l1 < 1e-15:
                    continue
                h_new /= l1
                
                sc_new = compute_ratio(h_new)
                if sc_new > best_score + 1e-8:
                    best_h = h_new
                    best_score = sc_new
                    improved = True
                    break
            
            # Try decreasing weight i
            for delta in [0.01, 0.05, 0.1, 0.2]:
                h_new = best_h.copy()
                h_new[support[i]] -= delta
                h_new = np.maximum(h_new, 0)
                l1 = np.sum(h_new)
                if l1 < 1e-15:
                    continue
                h_new /= l1
                
                sc_new = compute_ratio(h_new)
                if sc_new > best_score + 1e-8:
                    best_h = h_new
                    best_score = sc_new
                    improved = True
                    break
        
        if not improved:
            break
    
    # Random perturbations on the simplex
    np.random.seed(123)
    for _ in range(20):
        if not check_time():
            break
        # Generate random direction on simplex
        x = np.random.randn(k)
        x -= np.mean(x)  # Zero mean to stay on simplex
        for step_size in [0.1, 0.05, 0.01]:
            h_new = best_h.copy()
            for i in range(k):
                h_new[support[i]] += step_size * x[i]
            h_new = np.maximum(h_new, 0)
            l1 = np.sum(h_new)
            if l1 < 1e-15:
                continue
            h_new /= l1
            
            sc_new = compute_ratio(h_new)
            if sc_new > best_score + 1e-8:
                best_h = h_new
                best_score = sc_new
                break
    
    return best_h, best_score, best_support

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin
    
    best_h = np.zeros(N, dtype=np.float64)
    best_h[0] = 1.0
    best_score = compute_ratio(best_h)
    best_support = [0]
    
    sidon_sets = generate_sidon_sets()
    
    # Sort by size descending, then by score potential
    sidon_sets.sort(key=lambda x: (-len(x), sum(x)))
    
    # Phase 1: Optimize weights for each Sidon set with multiple restarts
    for s in sidon_sets:
        if time.time() > deadline - 30.0:
            break
        
        # Use a range of betas with multiple restarts
        for beta in [30.0, 60.0, 100.0]:
            if time.time() > deadline - 30.0:
                break
            
            h_opt, sc = optimize_weights_for_support(s, beta=beta, maxiter=600, n_restarts=3)
            if sc > best_score:
                best_score = sc
                best_h = h_opt.copy()
                best_support = list(s)
    
    # Phase 2: Local search polish from the best solution
    if time.time() < deadline - 20.0:
        h_opt, sc_opt, support_opt = local_search_polish(best_h, best_support, deadline=deadline - 10.0)
        if sc_opt > best_score:
            best_score = sc_opt
            best_h = h_opt.copy()
            best_support = support_opt
    
    # Phase 3: High-precision polish with multiple restarts
    if time.time() < deadline - 10.0:
        for beta in [150.0, 300.0, 500.0]:
            if time.time() > deadline - 10.0:
                break
            h_polish, sc_polish = optimize_weights_for_support(best_support, beta=beta, maxiter=1500, ftol=1e-14, n_restarts=2)
            if sc_polish > best_score:
                best_score = sc_polish
                best_h = h_polish.copy()
    
    # Phase 4: Final coordinate polish
    if time.time() < deadline - 5.0:
        h_final, sc_final, _ = local_search_polish(best_h, best_support, deadline=deadline - 2.0)
        if sc_final > best_score:
            best_score = sc_final
            best_h = h_final.copy()
    
    return list(best_h)
