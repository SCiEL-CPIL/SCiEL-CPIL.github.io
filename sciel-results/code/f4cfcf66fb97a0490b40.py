"""Kissing number in dimension 11: construct nonzero integer vectors in R^11
whose minimum pairwise distance is at least the maximum vector norm.
Interface: construct() -> integer array of shape (k, d)."""

import itertools
import time
import numpy as np

D = 11
DEADLINE_S = 118.0  # hard wall-clock safety margin before the 120 s limit


def construct():
    t0 = time.perf_counter()

    def elapsed():
        return time.perf_counter() - t0

    def is_deadline():
        return elapsed() > DEADLINE_S

    # ------------------------------------------------------------------
    # 1) Deterministic baseline: all norm^2 = 2 vectors
    # ------------------------------------------------------------------
    base = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i], v[j] = si, sj
                base.append(v)
    base_arr = np.array(base, dtype=np.int64)
    best = base_arr
    best_k = base_arr.shape[0]

    # ------------------------------------------------------------------
    # 2) Generate all vectors with norm^2 = m
    # ------------------------------------------------------------------
    def gen_all_vectors_m(m):
        if m < 1:
            return []
        results = []
        if m == 2:
            for i, j in itertools.combinations(range(D), 2):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i], v[j] = si, sj
                        results.append(v)
            for i in range(D):
                for s in (2, -2):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = s
                    results.append(v)
        elif m == 3:
            for combo in itertools.combinations(range(D), 3):
                for signs in itertools.product([1, -1], repeat=3):
                    v = np.zeros(D, dtype=np.int64)
                    for idx, s in zip(combo, signs):
                        v[idx] = s
                    results.append(v)
        elif m == 4:
            for combo in itertools.combinations(range(D), 4):
                for signs in itertools.product([1, -1], repeat=4):
                    v = np.zeros(D, dtype=np.int64)
                    for idx, s in zip(combo, signs):
                        v[idx] = s
                    results.append(v)
            for i in range(D):
                for s in (2, -2):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = s
                    results.append(v)
        elif m == 5:
            for combo in itertools.combinations(range(D), 5):
                for signs in itertools.product([1, -1], repeat=5):
                    v = np.zeros(D, dtype=np.int64)
                    for idx, s in zip(combo, signs):
                        v[idx] = s
                    results.append(v)
            for i, j in itertools.combinations(range(D), 2):
                for si in (1, -1):
                    for sj in (2, -2):
                        v = np.zeros(D, dtype=np.int64)
                        v[i], v[j] = si, sj
                        results.append(v)
        return results

    # ------------------------------------------------------------------
    # 3) Core insight: For norm^2=4 vectors with support size 4,
    #    any two such vectors with supports intersecting in <= 2
    #    are compatible for ANY sign choice.
    #    So we need to find a large set of 4-subsets of {0..10}
    #    such that any two intersect in at most 2.
    #    Then include ALL 16 sign vectors for each selected support.
    #    Additionally, include all norm^2=2 vectors (22 of them).
    # ------------------------------------------------------------------
    
    # Generate all 4-subsets
    supports_4 = list(itertools.combinations(range(D), 4))
    N_SUP = len(supports_4)
    
    # Precompute support masks as integers
    sup_masks = []
    for s in supports_4:
        mask = 0
        for i in s:
            mask |= (1 << i)
        sup_masks.append(mask)
    
    # Conflict: two supports conflict if intersection >= 3
    # Build adjacency list
    adj = [set() for _ in range(N_SUP)]
    for i in range(N_SUP):
        for j in range(i + 1, N_SUP):
            inter = sup_masks[i] & sup_masks[j]
            if inter.bit_count() >= 3:
                adj[i].add(j)
                adj[j].add(i)
    
    # ------------------------------------------------------------------
    # 4) Find maximum independent set via simulated annealing / local search
    # ------------------------------------------------------------------
    def find_mis_sa(n_nodes, adj_list, seed=42, max_iters=50000, t0_val=None):
        """Simulated annealing to find a large independent set."""
        rng = np.random.RandomState(seed)
        
        def random_greedy():
            x = np.zeros(n_nodes, dtype=bool)
            order = rng.permutation(n_nodes)
            blocked = np.zeros(n_nodes, dtype=bool)
            for v in order:
                if not x[v] and not blocked[v]:
                    x[v] = True
                    for nb in adj_list[v]:
                        blocked[nb] = True
            return x
        
        # Initial solution
        x = random_greedy()
        best_x = x.copy()
        best_size = x.sum()
        
        # Precompute neighbor lists for fast access
        neighbor_lists = [list(adj_list[i]) for i in range(n_nodes)]
        
        def count_conflicts(x):
            """Count number of conflicting pairs in x."""
            cnt = 0
            for i in range(n_nodes):
                if x[i]:
                    for j in neighbor_lists[i]:
                        if j > i and x[j]:
                            cnt += 1
            return cnt
        
        # Simulated annealing
        T = 1.0
        cooling = 0.9999
        min_T = 1e-6
        
        for it in range(max_iters):
            if t0_val is not None and (time.perf_counter() - t0_val) > DEADLINE_S:
                break
            if T < min_T:
                break
            
            # Random move: add, remove, or swap
            move_type = rng.randint(0, 3)
            
            if move_type == 0:
                # Add a random node not in x
                candidates = [i for i in range(n_nodes) if not x[i]]
                if not candidates:
                    continue
                v = candidates[rng.randint(0, len(candidates))]
                x[v] = True
                delta = 1  # size change
                # Count new conflicts
                new_conf = 0
                for nb in neighbor_lists[v]:
                    if x[nb]:
                        new_conf += 1
                # Energy: -size + penalty * conflicts
                # We want to maximize size while minimizing conflicts
                # Let's use: score = size - 10 * conflicts
                # Delta in score = delta_size - 10 * delta_conflicts
                # But we need to track conflicts properly.
                # Simpler: just evaluate after move.
                
            elif move_type == 1:
                # Remove a random node in x
                candidates = [i for i in range(n_nodes) if x[i]]
                if not candidates:
                    continue
                v = candidates[rng.randint(0, len(candidates))]
                x[v] = False
                delta = -1
                
            else:
                # Swap: remove one in x, add one not in x
                in_x = [i for i in range(n_nodes) if x[i]]
                not_in_x = [i for i in range(n_nodes) if not x[i]]
                if not in_x or not not_in_x:
                    continue
                v_out = in_x[rng.randint(0, len(in_x))]
                v_in = not_in_x[rng.randint(0, len(not_in_x))]
                x[v_out] = False
                x[v_in] = True
                delta = 0
            
            # Evaluate: use a simple acceptance based on size change
            # For simplicity, accept if size increases, or with probability exp(-delta/T) if decreases
            # But we also want to avoid too many conflicts.
            # Let's just use size as the objective and accept moves that increase size,
            # or with some probability accept moves that maintain or slightly decrease size
            # to escape local optima.
            
            cur_size = x.sum()
            
            if cur_size > best_size:
                best_x = x.copy()
                best_size = cur_size
            
            # Acceptance: accept if size didn't decrease, or with probability exp(-1/T)
            if cur_size >= best_size or rng.random() < np.exp(-1.0 / T):
                pass  # accept
            else:
                # revert
                if move_type == 0:
                    x[v] = False
                elif move_type == 1:
                    x[v] = True
                else:
                    x[v_out] = True
                    x[v_in] = False
            
            T *= cooling
        
        return best_x, best_size

    # Use a faster local search approach
    def local_search_mis(n_nodes, adj_list, seed=42, max_iters=20000):
        """Local search: add, remove, or swap to maximize independent set size."""
        rng = np.random.RandomState(seed)
        neighbor_lists = [list(adj_list[i]) for i in range(n_nodes)]
        
        # Start with random greedy
        x = np.zeros(n_nodes, dtype=bool)
        order = rng.permutation(n_nodes)
        blocked = np.zeros(n_nodes, dtype=bool)
        for v in order:
            if not x[v] and not blocked[v]:
                x[v] = True
                for nb in neighbor_lists[v]:
                    blocked[nb] = True
        
        best_x = x.copy()
        best_size = x.sum()
        
        for it in range(max_iters):
            if is_deadline():
                break
            
            improved = False
            move_type = rng.randint(0, 3)
            
            if move_type == 0:
                # Try to add a node
                candidates = [i for i in range(n_nodes) if not x[i]]
                if not candidates:
                    continue
                v = candidates[rng.randint(0, len(candidates))]
                # Check if v conflicts with any node in x
                conflict = False
                for nb in neighbor_lists[v]:
                    if x[nb]:
                        conflict = True
                        break
                if not conflict:
                    x[v] = True
                    improved = True
            
            elif move_type == 1:
                # Try to remove a node that allows adding more
                in_x = [i for i in range(n_nodes) if x[i]]
                if not in_x:
                    continue
                v = in_x[rng.randint(0, len(in_x))]
                x[v] = False
                # Now try to add as many as possible
                # Simple: just check if we can add any node that was blocked by v
                added = 0
                for nb in neighbor_lists[v]:
                    if not x[nb]:
                        # Check if nb conflicts with current x (excluding v which is now removed)
                        can_add = True
                        for nb2 in neighbor_lists[nb]:
                            if x[nb2]:
                                can_add = False
                                break
                        if can_add:
                            x[nb] = True
                            added += 1
                # Only keep the removal if we added at least as many
                if added <= 0:
                    x[v] = True  # revert
                else:
                    improved = True
            
            else:
                # Swap
                in_x = [i for i in range(n_nodes) if x[i]]
                not_in_x = [i for i in range(n_nodes) if not x[i]]
                if not in_x or not not_in_x:
                    continue
                v_out = in_x[rng.randint(0, len(in_x))]
                # Try to find a v_in that doesn't conflict with x \ {v_out}
                x[v_out] = False
                found = False
                for _ in range(10):  # Try a few random candidates
                    v_in = not_in_x[rng.randint(0, len(not_in_x))]
                    if not x[v_in]:
                        can_add = True
                        for nb in neighbor_lists[v_in]:
                            if x[nb]:
                                can_add = False
                                break
                        if can_add:
                            x[v_in] = True
                            found = True
                            improved = True
                            break
                if not found:
                    x[v_out] = True  # revert
            
            if improved:
                cur_size = x.sum()
                if cur_size > best_size:
                    best_x = x.copy()
                    best_size = cur_size
        
        return best_x, best_size

    # ------------------------------------------------------------------
    # 5) Run local search to find large independent set of supports
    # ------------------------------------------------------------------
    if not is_deadline():
        rng_main = np.random.RandomState(7)
        best_sup_set = None
        best_sup_size = 0
        
        n_trials = 20
        for trial in range(n_trials):
            if is_deadline():
                break
            seed = rng_main.randint(0, 1000000)
            sup_x, sup_size = local_search_mis(N_SUP, adj, seed=seed, max_iters=15000)
            if sup_size > best_sup_size:
                best_sup_size = sup_size
                best_sup_set = sup_x
        
        if best_sup_set is not None and best_sup_size > 0:
            # Build vector set: all 16 sign vectors for each selected support, plus all norm^2=2 vectors
            vecs = []
            
            # Add all norm^2=2 vectors (the base)
            for i, j in itertools.combinations(range(D), 2):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i], v[j] = si, sj
                        vecs.append(v)
            for i in range(D):
                for s in (2, -2):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = s
                    vecs.append(v)
            
            # Add all 16 sign vectors for each selected support
            for idx in range(N_SUP):
                if best_sup_set[idx]:
                    support = supports_4[idx]
                    for signs in itertools.product([1, -1], repeat=4):
                        v = np.zeros(D, dtype=np.int64)
                        for pos, s in zip(support, signs):
                            v[pos] = s
                        vecs.append(v)
            
            vecs_arr = np.array(vecs, dtype=np.int64)
            k = vecs_arr.shape[0]
            
            # Verify the configuration is valid
            # Check: min pairwise distance >= max norm
            if k > 0 and not is_deadline():
                norms_sq = np.array([np.dot(v, v) for v in vecs_arr], dtype=np.int64)
                max_norm_sq = np.max(norms_sq)
                
                # Check all pairs
                valid = True
                for i in range(k):
                    if is_deadline():
                        break
                    for j in range(i + 1, k):
                        diff = vecs_arr[i] - vecs_arr[j]
                        dist_sq = np.dot(diff, diff)
                        if dist_sq < max_norm_sq:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid and k > best_k:
                    best = vecs_arr
                    best_k = k
        
        # If the support-based approach didn't improve, fall back to the greedy approach
        if best_k <= 220 and not is_deadline():
            # Original greedy approach as fallback
            all_vecs = []
            for m in [2, 3, 4]:
                vecs = gen_all_vectors_m(m)
                for v in vecs:
                    all_vecs.append(v)
            
            if len(all_vecs) > 0:
                V_all = np.array(all_vecs, dtype=np.int64)
                n_all = len(all_vecs)
                norms_all = np.array([np.dot(v, v) for v in all_vecs], dtype=np.int64)
                G_all = V_all.astype(np.int64) @ V_all.astype(np.int64).T
                allowed_all = np.minimum(norms_all[:, None], norms_all[None, :]) // 2
                
                rng2 = np.random.RandomState(123)
                for trial in range(10):
                    if is_deadline():
                        break
                    if trial == 0:
                        order = list(range(n_all))
                    else:
                        order = rng2.permutation(n_all).tolist()
                    
                    chosen_mask = np.zeros(n_all, dtype=bool)
                    max_viol = np.full(n_all, -999999, dtype=np.int64)
                    chosen = []
                    
                    for idx in order:
                        if is_deadline():
                            break
                        if chosen_mask[idx]:
                            continue
                        if max_viol[idx] <= 0:
                            chosen.append(idx)
                            chosen_mask[idx] = True
                            new_viol = G_all[:, idx] - allowed_all[:, idx]
                            max_viol = np.maximum(max_viol, new_viol)
                            max_viol[chosen_mask] = -999999
                    
                    if len(chosen) > best_k:
                        best = V_all[chosen]
                        best_k = len(chosen)
    
    # ------------------------------------------------------------------
    # 6) Final safety: return best found
    # ------------------------------------------------------------------
    if best is None or best.shape[0] == 0:
        return base_arr
    return best
