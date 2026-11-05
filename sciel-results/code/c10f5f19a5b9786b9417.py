"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def partition(adjacency):
    # Convert to numpy int matrix for fast vectorized ops
    adj = np.asarray(adjacency, dtype=np.int8)
    n = adj.shape[0]
    if n == 0:
        return np.array([], dtype=int)

    # Deadline guard
    start_time = time.time()
    DEADLINE = start_time + 140.0

    def time_left():
        return DEADLINE - time.time()

    E = int(np.sum(adj)) // 2
    degrees = adj.sum(axis=1)
    
    # Precompute neighbors for fast updates
    neighbors = [np.where(adj[i] == 1)[0] for i in range(n)]

    def cut_size_from_labels(labels_01):
        """Compute cut size from 0/1 labels."""
        x = 2 * labels_01.astype(np.float64) - 1
        return E - 0.5 * (x @ (adj @ x))

    def cut_size_from_x(x):
        """Compute cut size from +/- 1 labels."""
        return E - 0.5 * (x @ (adj @ x))

    def local_search(x):
        """
        Fast local search.
        x: array of +/- 1
        Returns: improved x
        """
        Ax = adj @ x
        # Ax is float array
        for _ in range(100):  # Max passes
            if time_left() < 2.0:
                break
            improved = False
            for v in range(n):
                gain = -x[v] * Ax[v]
                if gain > 1e-9:
                    old_x = x[v]
                    x[v] = -old_x
                    # Update Ax for neighbors
                    if len(neighbors[v]) > 0:
                        Ax[neighbors[v]] += -2 * old_x
                    improved = True
            if not improved:
                break
        return x

    def spectral_init():
        """Initialize using Fiedler vector."""
        try:
            D = np.diag(degrees)
            L = D - adj.astype(np.float64)
            w, v = np.linalg.eigh(L)
            idx = np.argsort(w)
            # Fiedler vector is the one with smallest non-zero eigenvalue
            # Usually index 1
            fiedler = v[:, idx[1]]
            x = np.sign(fiedler)
            x[x == 0] = 1.0
            return x
        except Exception:
            return np.random.choice([-1.0, 1.0], size=n)

    def goe_init():
        """Greedy Ordering Elimination initialization."""
        x = np.zeros(n, dtype=np.float64)
        assigned = np.zeros(n, dtype=bool)
        order = np.argsort(degrees)[::-1]
        for v in order:
            nb = neighbors[v]
            assigned_nbs = nb[assigned[nb]]
            if len(assigned_nbs) == 0:
                x[v] = 1.0 if np.random.random() < 0.5 else -1.0
            else:
                # Count neighbors in -1 and 1
                c_neg = np.sum(x[assigned_nbs] == -1.0)
                c_pos = np.sum(x[assigned_nbs] == 1.0)
                # We want to maximize cut.
                # If we set x[v] = 1, we cut edges to -1 neighbors.
                # If we set x[v] = -1, we cut edges to 1 neighbors.
                if c_neg >= c_pos:
                    x[v] = 1.0
                else:
                    x[v] = -1.0
            assigned[v] = True
        return x

    def simulated_annealing(x, max_iters=50000, T_start=10.0, T_end=0.1, alpha=0.999):
        """Simulated Annealing."""
        Ax = adj @ x
        best_x = x.copy()
        best_cut = cut_size_from_x(x)
        current_cut = best_cut
        T = T_start
        
        for i in range(max_iters):
            if time_left() < 5.0:
                break
            
            v = np.random.randint(n)
            gain = -x[v] * Ax[v]
            
            if gain >= 0:
                # Accept
                old_x = x[v]
                x[v] = -old_x
                if len(neighbors[v]) > 0:
                    Ax[neighbors[v]] += -2 * old_x
                current_cut += gain
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_x = x.copy()
            else:
                # Accept with probability exp(gain / T)
                if T > 0.01:
                    # Optimization: if gain is very negative, exp is 0
                    if gain > -100 * T:
                        if np.random.random() < np.exp(gain / T):
                            old_x = x[v]
                            x[v] = -old_x
                            if len(neighbors[v]) > 0:
                                Ax[neighbors[v]] += -2 * old_x
                            current_cut += gain
            T *= alpha
            if T < T_end:
                T = T_end

        return best_x, best_cut

    # Strategy
    best_x = None
    best_cut = -1

    # 1. Spectral Start
    if time_left() > 10.0:
        x_spec = spectral_init()
        x_spec = local_search(x_spec)
        cut_spec = cut_size_from_x(x_spec)
        if cut_spec > best_cut:
            best_cut = cut_spec
            best_x = x_spec.copy()

    # 2. GOE Start
    if time_left() > 10.0:
        x_goe = goe_init()
        x_goe = local_search(x_goe)
        cut_goe = cut_size_from_x(x_goe)
        if cut_goe > best_cut:
            best_cut = cut_goe
            best_x = x_goe.copy()

    # 3. Random Restarts
    if time_left() > 10.0:
        for _ in range(10):
            if time_left() < 5.0:
                break
            x_rand = np.random.choice([-1.0, 1.0], size=n)
            x_rand = local_search(x_rand)
            cut_rand = cut_size_from_x(x_rand)
            if cut_rand > best_cut:
                best_cut = cut_rand
                best_x = x_rand.copy()

    # 4. Simulated Annealing from best
    if time_left() > 15.0 and best_x is not None:
        x_sa, cut_sa = simulated_annealing(best_x, max_iters=100000)
        if cut_sa > best_cut:
            best_cut = cut_sa
            best_x = x_sa.copy()

    # 5. Final Polish
    if time_left() > 5.0 and best_x is not None:
        best_x = local_search(best_x)
        best_cut = cut_size_from_x(best_x)

    # Convert back to 0/1
    if best_x is None:
        return np.zeros(n, dtype=int)
    
    labels = (best_x + 1) // 2
    return labels.astype(int)

# EVOLVE-BLOCK-END
