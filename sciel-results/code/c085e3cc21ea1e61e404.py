import numpy as np
import time

def construct():
    t0 = time.time()
    N = 12
    M = 40
    total = 1 << N  # 4096

    masks = np.arange(total, dtype=np.int64)
    a_2d = masks[:, None]
    b_2d = masks[None, :]
    is_sup = ((a_2d & b_2d) == a_2d) & (a_2d != b_2d)
    
    # Symmetric comparability matrix
    comp = is_sup | is_sup.T

    def get_score(x):
        # x is binary vector
        return float(x @ (comp @ x) / 2)

    def simulated_annealing(initial_x, max_iter=50000, t_start=50.0, t_end=1.0):
        x = initial_x.copy()
        current_score = get_score(x)
        v = comp @ x
        best_x = x.copy()
        best_score = current_score
        
        for i in range(max_iter):
            if time.time() - t0 > 55:
                break
            T = t_start * (t_end / t_start) ** (i / max_iter)
            
            # Pick random a (1) and b (0)
            ones = np.where(x > 0)[0]
            zeros = np.where(x == 0)[0]
            if len(ones) == 0 or len(zeros) == 0:
                break
            a = int(ones[np.random.randint(len(ones))])
            b = int(zeros[np.random.randint(len(zeros))])
            
            # Calculate delta
            # Delta = Gain(b) - Loss(a)
            # Loss(a) = v[a]
            # Gain(b) = v[b] - comp[b, a]
            delta = (v[b] - comp[b, a]) - v[a]
            
            if delta > 0 or np.random.rand() < np.exp(delta / T):
                # Accept
                x[a] = 0
                x[b] = 1
                current_score += delta
                
                # Update v in place
                v -= comp[:, a]
                v += comp[:, b]
                
                if current_score > best_score:
                    best_score = current_score
                    best_x = x.copy()
                    
        return best_x, best_score

    def greedy_init():
        sel = np.zeros(total, dtype=np.float64)
        for step in range(M):
            if time.time() - t0 > 10:
                break
            vec1 = comp @ sel
            g = vec1
            g[sel > 0] = -1
            c = int(np.argmax(g))
            sel[c] = 1
        return sel

    best_sel = None
    best_score = -1
    
    # Run SA from greedy start
    sel = greedy_init()
    sel, sc = simulated_annealing(sel, max_iter=20000)
    if sc > best_score:
        best_score = sc
        best_sel = sel.copy()
        
    # Run SA from random starts
    for trial in range(5):
        if time.time() - t0 > 50:
            break
        rng = np.random.RandomState(trial)
        sel = np.zeros(total, dtype=np.float64)
        indices = rng.choice(total, M, replace=False)
        sel[indices] = 1
        sel, sc = simulated_annealing(sel, max_iter=20000)
        if sc > best_score:
            best_score = sc
            best_sel = sel.copy()

    # Convert to lists
    result = []
    for mask in np.where(best_sel > 0)[0]:
        result.append([b for b in range(N) if (int(mask) >> b) & 1])

    return result
