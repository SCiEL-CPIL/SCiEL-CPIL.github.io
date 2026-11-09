import numpy as np
import time
import math

def construct():
    t0 = time.time()
    N = 12
    M = 40
    total = 1 << N  # 4096

    masks = np.arange(total, dtype=np.int64)
    a_2d = masks[:, None]
    b_2d = masks[None, :]
    is_sup = ((a_2d & b_2d) == a_2d) & (a_2d != b_2d)
    
    # A[i,j] = 1 if i and j are comparable (one is strict subset of other)
    # Since sets are distinct, at most one direction holds.
    A = is_sup.astype(np.float64) + is_sup.T.astype(np.float64)
    
    def get_score(x):
        return float(x @ (A @ x))

    def greedy_init():
        sel = np.zeros(total, dtype=np.float64)
        for _ in range(M):
            if time.time() - t0 > 40:
                break
            gains = A @ sel
            gains[sel > 0] = -np.inf
            c = int(np.argmax(gains))
            sel[c] = 1
        return sel

    def simulated_annealing(initial_sel, t_start=1.0, t_end=0.01, steps=50000):
        x = initial_sel.copy()
        current_score = get_score(x)
        best_x = x.copy()
        best_score = current_score
        
        ax = A @ x
        
        selected_list = list(np.where(x > 0)[0])
        unselected_list = list(np.where(x == 0)[0])
        
        for step in range(steps):
            if time.time() - t0 > 55:
                break
            
            T = t_start * (t_end / t_start) ** (step / steps)
            
            if not selected_list or not unselected_list:
                break
                
            i = selected_list[np.random.randint(len(selected_list))]
            j = unselected_list[np.random.randint(len(unselected_list))]
            
            if i == j:
                continue
                
            delta = 2 * (ax[j] - ax[i] - A[i, j])
            
            if delta > 0 or np.random.random() < math.exp(delta / T):
                # Update state
                x[i] = 0
                x[j] = 1
                
                # Update ax
                ax -= A[:, i]
                ax += A[:, j]
                
                # Update lists
                selected_list.remove(i)
                unselected_list.append(i)
                unselected_list.remove(j)
                selected_list.append(j)
                
                current_score += delta
                
                if current_score > best_score:
                    best_score = current_score
                    best_x = x.copy()
            
        return best_x, best_score

    # Initial Greedy
    sel = greedy_init()
    best_x, best_score = simulated_annealing(sel, t_start=1.0, t_end=0.01, steps=50000)
    
    # Restart with perturbation if time permits
    if time.time() - t0 < 45:
        sel2 = best_x.copy()
        sel_idx = np.where(sel2 > 0)[0]
        unsel_idx = np.where(sel2 == 0)[0]
        if len(sel_idx) > 0 and len(unsel_idx) > 0:
            i = int(sel_idx[np.random.randint(len(sel_idx))])
            j = int(unsel_idx[np.random.randint(len(unsel_idx))])
            sel2[i] = 0
            sel2[j] = 1
            x2, s2 = simulated_annealing(sel2, t_start=0.5, t_end=0.01, steps=30000)
            if s2 > best_score:
                best_x = x2
                best_score = s2

    # Convert to lists
    result = []
    for mask in np.where(best_x > 0)[0]:
        result.append([b for b in range(N) if (int(mask) >> b) & 1])

    return result
