import numpy as np
import time

def partition(adjacency):
    adj = np.array(adjacency, dtype=np.int8)
    n = len(adj)
    
    def get_cut_size(x):
        val = np.dot(x, adj.dot(x))
        m = np.sum(adj) / 2.0
        return (2*m - val) / 4.0

    def local_search(x):
        while True:
            improved = False
            # 1-flip pass
            row_sums = adj.dot(x)
            for i in range(n):
                # Gain condition: x[i] * row_sums[i] > 0
                if x[i] * row_sums[i] > 0:
                    x[i] *= -1
                    improved = True
                    # Update row_sums
                    # delta_k = -2 * A_ki * x_i_old = -2 * A_ki * (-x_i_new) = 2 * A_ki * x_i_new
                    # Wait, x_i_old = -x_i_new.
                    # Change in x_i is -2 x_i_old.
                    # row_sum_k += A_ki * (-2 x_i_old) = A_ki * (-2 (-x_i_new)) = 2 A_ki x_i_new.
                    row_sums += 2 * adj[:, i] * x[i]
            
            if not improved:
                # Try 2-flips
                row_sums = adj.dot(x)
                two_flip_found = False
                for u in range(n):
                    for v in range(u + 1, n):
                        # Condition: x_u row_sum_u + x_v row_sum_v - 2 A_uv x_u x_v < 0
                        val = x[u] * row_sums[u] + x[v] * row_sums[v] - 2 * adj[u, v] * x[u] * x[v]
                        if val < 0:
                            x[u] *= -1
                            x[v] *= -1
                            two_flip_found = True
                            break
                    if two_flip_found:
                        break
                
                if not two_flip_found:
                    break
                # If a 2-flip was found, loop back to 1-flip pass
                # (1-flip pass might find more improvements)
            # else:
            #    # 1-flip improved, continue to next 1-flip pass (or check 2-flips?)
            #    # The knowledge says "Interleaving".
            #    # Let's check 2-flips even if 1-flip improved?
            #    # Or just loop back to top.
            #    # Looping back to top runs 1-flip pass again.
            #    # This is fine.
            pass
        return x

    best_cut = -1
    best_labels = None
    
    deadline = time.time() + 110.0
    
    # Spectral Init
    deg = adj.sum(axis=1)
    L = np.diag(deg) - adj
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        fiedler = eigvecs[:, 1]
        x_spec = np.where(fiedler > 0, 1, -1)
        x_opt = local_search(x_spec.copy())
        cut = get_cut_size(x_opt)
        if cut > best_cut:
            best_cut = cut
            best_labels = (x_opt + 1) // 2
    except Exception:
        pass

    rng = np.random.default_rng(42)
    while time.time() < deadline:
        x_rand = rng.choice([-1, 1], size=n)
        x_opt = local_search(x_rand)
        cut = get_cut_size(x_opt)
        if cut > best_cut:
            best_cut = cut
            best_labels = (x_opt + 1) // 2

    if best_labels is None:
        best_labels = np.zeros(n, dtype=int)
        
    return best_labels
