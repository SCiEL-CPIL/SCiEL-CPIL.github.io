import numpy as np
import time

def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    adj = np.array(adjacency, dtype=np.int8)
    np.fill_diagonal(adj, 0)
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    S_total = np.sum(adj)
    
    def get_cut(x):
        xAx = x @ (adj @ x)
        return 0.25 * (S_total - xAx)
    
    # Spectral initialization
    D = np.diag(adj.sum(axis=1))
    L = D - adj
    
    rng = np.random.default_rng(42)
    
    def local_search_1flip(x, max_iters=10000):
        S = adj @ x
        C = get_cut(x)
        iters = 0
        while iters < max_iters:
            gains = x * S
            i = np.argmax(gains)
            max_gain = gains[i]
            if max_gain <= 0:
                break
            v = x[i]
            x[i] = -v
            C += max_gain
            S -= 2 * v * adj[:, i]
            iters += 1
        return x, C, iters
    
    def local_search_2swap(x, max_iters=5000):
        S = adj @ x
        C = get_cut(x)
        iters = 0
        while iters < max_iters:
            gains_1 = x * S
            i = np.argmax(gains_1)
            if gains_1[i] <= 0:
                found = False
                for _ in range(20):
                    j = rng.integers(0, n)
                    if j == i:
                        continue
                    gain = gains_1[i] + gains_1[j] - 2 * adj[i, j] * x[i] * x[j]
                    if gain > 0:
                        vi = x[i]
                        vj = x[j]
                        x[i] = -vi
                        x[j] = -vj
                        C += gain
                        S -= 2 * vi * adj[:, i]
                        S -= 2 * vj * adj[:, j]
                        gains_1 = x * S
                        iters += 1
                        found = True
                        break
                if not found:
                    break
            else:
                j_gains = gains_1.copy()
                interaction = 2 * x[i] * x * adj[i, :]
                j_gains[i] = -1e9
                j_gains -= interaction
                j = np.argmax(j_gains)
                if j == i or j_gains[j] <= 0:
                    v = x[i]
                    x[i] = -v
                    C += gains_1[i]
                    S -= 2 * v * adj[:, i]
                    gains_1 = x * S
                    iters += 1
                else:
                    vi = x[i]
                    vj = x[j]
                    x[i] = -vi
                    x[j] = -vj
                    gain = gains_1[i] + gains_1[j] - 2 * adj[i, j] * vi * vj
                    C += gain
                    S -= 2 * vi * adj[:, i]
                    S -= 2 * vj * adj[:, j]
                    gains_1 = x * S
                    iters += 1
        return x, C, iters
    
    def simulated_annealing(x, start_temp=5.0, cooling=0.95, min_temp=0.01, max_iters=5000):
        """SA polish to escape local optima."""
        S = adj @ x
        C = get_cut(x)
        temp = start_temp
        iters = 0
        best_C = C
        best_x = x.copy()
        
        while temp > min_temp and iters < max_iters:
            if time.time() > deadline - 5:
                break
            
            # Pick random vertex to flip
            i = rng.integers(0, n)
            v = x[i]
            
            # Compute gain for flipping i
            gain = x[i] * S[i]
            
            # Accept if gain > 0 or with probability exp(gain/temp) if gain < 0
            if gain > 0 or (temp > 0 and rng.random() < np.exp(gain / temp)):
                x[i] = -v
                C += gain
                S -= 2 * v * adj[:, i]
                iters += 1
                
                if C > best_C:
                    best_C = C
                    best_x = x.copy()
            
            temp *= cooling
        
        return best_x, best_C, iters
    
    # Generate diverse initializations
    inits = []
    
    # 1. Spectral init (Fiedler vector)
    try:
        vals, vecs = np.linalg.eigh(L)
        fiedler = vecs[:, 1]
        x_spec = np.sign(fiedler)
        x_spec[x_spec == 0] = 1
        inits.append(x_spec.astype(np.int8))
        
        # 2. Second Fiedler vector variant
        if len(vecs) > 2:
            fiedler2 = vecs[:, 2]
            x_spec2 = np.sign(fiedler2)
            x_spec2[x_spec2 == 0] = 1
            inits.append(x_spec2.astype(np.int8))
    except:
        x_spec = np.ones(n, dtype=np.int8)
        inits.append(x_spec.copy())
    
    # 3. Alternating init
    alt = np.ones(n, dtype=np.int8)
    alt[1::2] = -1
    inits.append(alt)
    
    # 4. Random inits
    for _ in range(8):
        inits.append(rng.choice([-1, 1], size=n).astype(np.int8))
    
    # 5. Perturbed spectral inits
    for _ in range(4):
        x_pert = x_spec.copy()
        num_flips = max(1, n // 5)
        flip_idx = rng.choice(n, size=num_flips, replace=False)
        x_pert[flip_idx] *= -1
        inits.append(x_pert.astype(np.int8))
    
    best_C = -1
    best_x = None
    
    # Phase 1: Local search on all initializations
    for x_init in inits:
        if time.time() > deadline - 30:
            break
        
        x_curr = x_init.copy()
        x_ls, C_ls, _ = local_search_1flip(x_curr)
        if C_ls > best_C:
            best_C = C_ls
            best_x = x_ls.copy()
    
    if time.time() > deadline - 20:
        return ((best_x + 1) // 2).astype(int)
    
    # Phase 2: 2-swap polish on best
    x_curr = best_x.copy()
    x_ls2, C_ls2, _ = local_search_2swap(x_curr, max_iters=3000)
    if C_ls2 > best_C:
        best_C = C_ls2
        best_x = x_ls2.copy()
    
    if time.time() > deadline - 15:
        return ((best_x + 1) // 2).astype(int)
    
    # Phase 3: SA polish on best to escape local optima
    x_sa, C_sa, _ = simulated_annealing(best_x.copy(), start_temp=5.0, cooling=0.95, min_temp=0.01, max_iters=3000)
    if C_sa > best_C:
        best_C = C_sa
        best_x = x_sa.copy()
    
    # Local search after SA
    x_ls3, C_ls3, _ = local_search_1flip(best_x.copy())
    if C_ls3 > best_C:
        best_C = C_ls3
        best_x = x_ls3.copy()
    
    if time.time() > deadline - 10:
        return ((best_x + 1) // 2).astype(int)
    
    # Phase 4: More diverse restarts with local search
    num_restarts = 100
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        if i % 4 == 0:
            x_curr = rng.choice([-1, 1], size=n).astype(np.int8)
        elif i % 4 == 1:
            # Perturb best: flip random 10% of nodes
            x_curr = best_x.copy()
            num_flips = max(1, n // 10)
            flip_idx = rng.choice(n, size=num_flips, replace=False)
            x_curr[flip_idx] *= -1
        elif i % 4 == 2:
            # Perturb spectral: flip random 20% of nodes
            x_curr = x_spec.copy()
            num_flips = max(1, n // 5)
            flip_idx = rng.choice(n, size=num_flips, replace=False)
            x_curr[flip_idx] *= -1
        else:
            # Near-best perturbation: flip random 5% of nodes
            x_curr = best_x.copy()
            num_flips = max(1, n // 20)
            flip_idx = rng.choice(n, size=num_flips, replace=False)
            x_curr[flip_idx] *= -1
        
        x_ls, C_ls, _ = local_search_1flip(x_curr)
        if C_ls > best_C:
            best_C = C_ls
            best_x = x_ls.copy()
    
    # Phase 5: Final 2-swap + SA polish if time permits
    if time.time() < deadline - 5:
        x_curr = best_x.copy()
        x_ls2, C_ls2, _ = local_search_2swap(x_curr, max_iters=2000)
        if C_ls2 > best_C:
            best_C = C_ls2
            best_x = x_ls2.copy()
        
        if time.time() < deadline - 3:
            x_sa, C_sa, _ = simulated_annealing(best_x.copy(), start_temp=3.0, cooling=0.95, min_temp=0.01, max_iters=1500)
            if C_sa > best_C:
                best_C = C_sa
                best_x = x_sa.copy()
            
            # Final local search
            x_ls_final, C_ls_final, _ = local_search_1flip(best_x.copy())
            if C_ls_final > best_C:
                best_C = C_ls_final
                best_x = x_ls_final.copy()
    
    if best_x is None:
        best_x = rng.choice([-1, 1], size=n).astype(np.int8)
    
    labels = (best_x + 1) // 2
    return labels.astype(int)
