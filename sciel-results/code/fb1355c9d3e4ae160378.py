import time
import numpy as np
import random
import math

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s/150s kill
    n = 256
    
    # Set random seed for reproducibility within a single run
    random.seed(42)
    np.random.seed(42)

    # Precompute all 4-term APs as index tuples (0-indexed)
    aps = []
    for d in range(1, 86):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            i0 = a - 1
            i1 = a + d - 1
            i2 = a + 2 * d - 1
            i3 = a + 3 * d - 1
            aps.append((i0, i1, i2, i3))
            
    M = len(aps)
    aps_arr = np.array(aps, dtype=np.int32)

    # Build per-position list of AP indices and the indices of the OTHER elements in those APs
    pos_aps = [[] for _ in range(n)]
    pos_others = [[] for _ in range(n)]
    
    for idx, ap in enumerate(aps):
        for k in range(4):
            p = ap[k]
            pos_aps[p].append(idx)
            others = [ap[j] for j in range(4) if j != k]
            pos_others[p].append(others)

    # Precompute flat arrays for vectorized delta computation
    pos_others_arrays = [None] * n
    for p in range(n):
        if pos_others[p]:
            pos_others_arrays[p] = np.array(pos_others[p], dtype=np.int32)
        else:
            pos_others_arrays[p] = np.empty((0, 3), dtype=np.int32)

    # Build incidence matrix A (n x M) where A[i, j] = 1 if position i is in AP j
    A = np.zeros((n, M), dtype=np.float64)
    for idx, ap in enumerate(aps):
        for k in range(4):
            A[ap[k], idx] = 1.0

    # Compute SVD to get top singular vectors (position-space)
    # We want the top few left singular vectors to identify high-energy modes
    try:
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        # Top 10 singular vectors in position space
        top_k = min(10, U.shape[1])
        top_U = U[:, :top_k]
        # Energy of each position: sum of squared projections onto top singular vectors
        spectral_energy = np.sum(top_U ** 2, axis=1)  # shape (n,)
        # Normalize to [0, 1]
        spectral_energy = spectral_energy / (spectral_energy.max() + 1e-12)
    except Exception:
        # Fallback: use degree-based energy
        spectral_energy = np.array([len(pos_aps[p]) for p in range(n)], dtype=np.float64)
        spectral_energy = spectral_energy / (spectral_energy.max() + 1e-12)

    # Thue-Morse initialization: c[i] = popcount(i) % 2
    c = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
    
    def count_mono(col):
        a0 = col[aps_arr[:, 0]]
        a1 = col[aps_arr[:, 1]]
        a2 = col[aps_arr[:, 2]]
        a3 = col[aps_arr[:, 3]]
        mono = (a0 == a1) & (a1 == a2) & (a2 == a3)
        return int(np.sum(mono))

    def get_delta(pos, col):
        """Calculate delta in mono count if we flip col[pos]."""
        delta = 0
        others_list = pos_others[pos]
        for others in others_list:
            c1 = col[others[0]]
            c2 = col[others[1]]
            c3 = col[others[2]]
            if c1 == c2 and c2 == c3:
                if col[pos] == c1:
                    delta -= 1
                else:
                    delta += 1
        return delta

    def vectorized_deltas(col):
        """Compute delta for flipping each position, vectorized using precomputed arrays."""
        all_deltas = np.zeros(n, dtype=np.int32)
        
        for pos in range(n):
            others_arr = pos_others_arrays[pos]
            if len(others_arr) == 0:
                continue
            
            o1 = col[others_arr[:, 0]]
            o2 = col[others_arr[:, 1]]
            o3 = col[others_arr[:, 2]]
            
            all_same = (o1 == o2) & (o2 == o3)
            
            match = (col[pos] == o1)
            
            delta = np.where(all_same, np.where(match, -1, 1), 0)
            all_deltas[pos] = int(np.sum(delta))
        
        return all_deltas

    def greedy_polish(col, time_limit=None):
        """Run greedy local search until no improvement or time limit."""
        col = col.copy()
        cur_count = count_mono(col)
        improved = True
        while improved and (time.time() - start_time) < (deadline - 5.0 if time_limit is None else time_limit):
            improved = False
            deltas = vectorized_deltas(col)
            min_delta = int(np.min(deltas))
            if min_delta < 0:
                pos = int(np.argmin(deltas))
                col[pos] = 1 - col[pos]
                cur_count += min_delta
                improved = True
        return col, cur_count
    
    def spectral_perturb(col, num_flips=5):
        """Apply spectral-based perturbation: flip positions with high spectral energy.
        Positions with high projection onto top singular vectors are more likely
        to be involved in monochromatic AP patterns."""
        col = col.copy()
        # Use spectral energy to weight flip probabilities
        # Higher energy positions are more likely to be flipped
        weights = spectral_energy + 0.1  # Add small offset to avoid zero weights
        weights = weights / weights.sum()
        
        for _ in range(num_flips):
            pos = np.random.choice(n, p=weights)
            col[pos] = 1 - col[pos]
        return col

    def spectral_project(col, num_vectors=10):
        """Project the coloring vector onto the subspace orthogonal to top singular vectors.
        This suppresses high-energy modes that contribute to monochromatic APs."""
        col_float = col.astype(np.float64)
        # Compute projection onto top singular vectors
        proj = top_U @ (top_U.T @ col_float)
        # Subtract projection to get component orthogonal to top singular vectors
        projected = col_float - proj
        # Round back to {0, 1}
        projected_col = (projected >= 0.5).astype(np.int8)
        return projected_col

    # Phase 0: Spectral projection initialization
    # Start with Thue-Morse, project to low-energy subspace, then polish
    c_projected = spectral_project(c, num_vectors=10)
    c_proj_polished, c_proj_count = greedy_polish(c_projected)
    
    # Also keep original Thue-Morse polished version
    c_tm_polished, c_tm_count = greedy_polish(c)
    
    # Choose the better initial solution
    if c_proj_count < c_tm_count:
        best_c = c_proj_polished
        best_count = c_proj_count
    else:
        best_c = c_tm_polished
        best_count = c_tm_count
    
    # Phase 1.5: Try spectral perturbation from best
    if (time.time() - start_time) < deadline - 10.0:
        for sp_attempt in range(3):
            if (time.time() - start_time) > deadline - 10.0:
                break
            perturbed = spectral_perturb(best_c, num_flips=3 + sp_attempt)
            polished, polished_count = greedy_polish(perturbed, time_limit=deadline - 10.0)
            if polished_count < best_count:
                best_c = polished
                best_count = polished_count

    # Phase 2: Simulated Annealing with improved restart strategy
    T = 2.0
    cooling = 0.9997
    min_T = 0.02
    no_improve_steps = 0
    max_no_improve = 1200
    restart_count = 0
    max_restarts = 20
    
    while (time.time() - start_time) < deadline - 3.0:
        pos = random.randint(0, n - 1)
        d = get_delta(pos, c)

        accept = False
        if d < 0:
            accept = True
            no_improve_steps = 0
        elif d == 0:
            accept = True
        else:
            if T > min_T:
                prob = math.exp(-d / T)
                if random.random() < prob:
                    accept = True
                    no_improve_steps = 0

        if accept:
            c[pos] = 1 - c[pos]
            best_count += d
            if best_count < count_mono(best_c):
                best_c = c.copy()
                no_improve_steps = 0
            else:
                no_improve_steps += 1
        else:
            no_improve_steps += 1

        T *= cooling

        if no_improve_steps > max_no_improve or restart_count >= max_restarts:
            c = best_c.copy()
            best_count = count_mono(c)
            T = 1.5
            no_improve_steps = 0
            restart_count += 1
            
            num_flips = 3 + min(restart_count, 7)
            c = spectral_perturb(c, num_flips=num_flips)
            
            c, best_count = greedy_polish(c, time_limit=deadline - 10.0)
            if best_count < count_mono(best_c):
                best_c = c.copy()
            
            if restart_count >= max_restarts:
                break

    # Phase 3: Multiple final polish attempts with different perturbations
    if (time.time() - start_time) < deadline - 2.0:
        for attempt in range(5):
            if (time.time() - start_time) > deadline - 2.0:
                break
            if attempt < 3:
                perturbed = spectral_perturb(best_c, num_flips=2 + attempt)
            else:
                perturbed = best_c.copy()
                num_flips = 2 + attempt
                for _ in range(num_flips):
                    p2 = random.randint(0, n - 1)
                    perturbed[p2] = 1 - perturbed[p2]
            polished, polished_count = greedy_polish(perturbed, time_limit=deadline - 2.0)
            if polished_count < best_count:
                best_c = polished
                best_count = polished_count

    # Final greedy polish
    if (time.time() - start_time) < deadline - 1.0:
        final_c, final_count = greedy_polish(best_c)
        if final_count < best_count:
            best_c = final_c
            best_count = final_count

    # Verify and return
    actual_count = count_mono(best_c)
    if actual_count < best_count:
        best_count = actual_count

    return best_c.tolist()
