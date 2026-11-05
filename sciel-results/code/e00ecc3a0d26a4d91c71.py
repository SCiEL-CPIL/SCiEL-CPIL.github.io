"""Improved local-search optimization of greedy decoder for MIS."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    start_time = time.time()
    DEADLINE = 143.0  # Safety margin before 150s limit

    A = np.asarray(adjacency, dtype=float)
    N = A.shape[0]

    if N == 0:
        return np.array([])

    # Ensure symmetry and remove self-loops
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)

    # Convert to boolean for faster neighbor lookups
    A_bool = A > 0.5

    # Precompute neighbor lists and degree
    neighbors_list = [np.where(A_bool[i])[0] for i in range(N)]
    deg = A.sum(axis=1)

    # --- 1. Markov Chain Absorption Time Heuristic ---
    # Goal: Prioritize vertices that are "peripheral" (far from high-degree hubs).
    # High absorption time to hubs => Good candidate for early selection (low blocking cost).
    
    if N > 0:
        # Identify hubs: top 10% degree nodes
        num_hubs = max(1, int(0.1 * N))
        hub_indices = np.argsort(-deg)[:num_hubs]
        is_hub = np.zeros(N, dtype=bool)
        is_hub[hub_indices] = True
        
        # Transition matrix P = D^-1 A
        # Avoid division by zero for isolated nodes
        safe_deg = np.where(deg > 0, deg, 1.0)
        P = A / safe_deg[:, np.newaxis]
        
        # Solve for expected absorption time to hit any hub.
        # Let t_i be expected steps from i to hit H.
        # t_i = 0 if i in H
        # t_i = 1 + sum_j P_ij t_j if i not in H
        # This is a linear system. For speed, use power iteration or simple relaxation.
        # Since N=200, we can solve (I - P_sub) t_sub = 1 efficiently.
        
        # Mask for non-hubs
        non_hub_mask = ~is_hub
        if np.sum(non_hub_mask) > 0:
            indices_non_hub = np.where(non_hub_mask)[0]
            P_sub = P[np.ix_(indices_non_hub, indices_non_hub)]
            
            # Solve (I - P_sub) t = 1
            I = np.eye(len(indices_non_hub))
            try:
                t_sub = np.linalg.solve(I - P_sub, np.ones(len(indices_non_hub)))
            except np.linalg.LinAlgError:
                # Fallback to power iteration if singular (shouldn't happen if hubs reachable)
                t_sub = np.zeros(len(indices_non_hub))
                for _ in range(100):
                    t_sub_new = 1 + P_sub.dot(t_sub)
                    if np.allclose(t_sub, t_sub_new):
                        break
                    t_sub = t_sub_new
            
            absorption_times = np.zeros(N)
            absorption_times[indices_non_hub] = t_sub
        else:
            absorption_times = np.zeros(N)
        
        # Normalize absorption times to [0, 1]
        # Higher time = more peripheral = better score
        if absorption_times.max() > absorption_times.min():
            mc_scores = (absorption_times - absorption_times.min()) / (absorption_times.max() - absorption_times.min())
        else:
            mc_scores = np.ones(N) / N
            
        # Combine with degree heuristic? 
        # The directive says "Model the vertex priority as...". Let's use MC scores primarily.
        # But degree is a strong signal. Let's blend.
        # Low degree is good.
        if deg.max() > deg.min():
            deg_scores = 1.0 - (deg - deg.min()) / (deg.max() - deg.min())
        else:
            deg_scores = np.ones(N) / N
            
        # Blend: 70% MC, 30% Degree
        scores = 0.7 * mc_scores + 0.3 * deg_scores
        
    else:
        scores = np.ones(N)

    def simulate_fast(s):
        """
        Fast greedy simulation using vectorized operations.
        Returns the size of the independent set.
        """
        order = np.argsort(-s, kind='stable')
        active = np.ones(N, dtype=bool)
        count = 0

        for v in order:
            if active[v]:
                count += 1
                nbrs = neighbors_list[v]
                if len(nbrs) > 0:
                    active[nbrs] = False
        return count

    # --- 2. Baseline ---
    best_val = simulate_fast(scores)
    best_scores = scores.copy()

    # --- 3. Local Search with Simulated Annealing ---
    rng = np.random.default_rng(42)

    num_restarts = 20
    current_temp = 1.0
    cooling = 0.95

    for restart in range(num_restarts):
        if time.time() - start_time > DEADLINE - 2:
            break

        if restart == 0:
            current_scores = best_scores.copy()
            current_val = best_val
        else:
            # Perturb the best score found so far
            noise_scale = 0.08 + 0.03 * restart
            noise = rng.normal(0, noise_scale, N)
            current_scores = best_scores + noise
            current_val = simulate_fast(current_scores)

        # Local search with simulated annealing
        no_improve = 0
        max_no_improve = 12000
        temp = current_temp

        while time.time() - start_time < DEADLINE - 3 and no_improve < max_no_improve:
            # Choose move type based on temperature
            r = rng.random()
            if r < 0.4:
                # Random swap
                i, j = rng.integers(0, N, size=2)
                if i == j:
                    continue
            elif r < 0.7:
                # Rank-adjacent swap
                order = np.argsort(-current_scores, kind='stable')
                i_idx = rng.integers(0, N)
                offset = rng.integers(1, min(15, N))
                j_idx = (i_idx + offset) % N
                i, j = order[i_idx], order[j_idx]
                if i == j:
                    continue
            else:
                # Score perturbation: add small noise to a random vertex
                i = rng.integers(0, N)
                j = rng.integers(0, N)
                if i == j:
                    continue
                # Perturb scores rather than swap
                delta = rng.normal(0, 0.1)
                current_scores[i] += delta
                new_val = simulate_fast(current_scores)
                if new_val > current_val:
                    current_val = new_val
                    no_improve = 0
                    if current_val > best_val:
                        best_val = current_val
                        best_scores = current_scores.copy()
                else:
                    current_scores[i] -= delta
                    no_improve += 1
                continue

            # Swap-based move
            si, sj = current_scores[i], current_scores[j]
            current_scores[i], current_scores[j] = sj, si
            new_val = simulate_fast(current_scores)

            delta_val = new_val - current_val
            if delta_val > 0 or (temp > 0.01 and rng.random() < np.exp(delta_val / temp)):
                current_val = new_val
                no_improve = 0
                if current_val > best_val:
                    best_val = current_val
                    best_scores = current_scores.copy()
            else:
                current_scores[i], current_scores[j] = si, sj
                no_improve += 1

            # Cool down
            temp *= cooling

        # Reset temperature for next restart
        current_temp = max(0.1, current_temp * 0.8)

    return best_scores
# EVOLVE-BLOCK-END
