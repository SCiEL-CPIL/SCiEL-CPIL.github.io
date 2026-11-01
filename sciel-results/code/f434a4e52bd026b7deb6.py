"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
import scipy.sparse.linalg as sla

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    # Generate W4 vectors (weight 4, entries in {-1,0,1}, norm^2=4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate W1 vectors (weight 1, entries in {-2,0,2}, norm^2=4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    
    if N4 == 0:
        return vectors_v1
        
    # Compute conflict matrix: conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = (dots > 2).astype(np.int8)
    np.fill_diagonal(conflicts, 0)
    
    # Convert to sparse for efficient operations
    conflicts_sparse = csr_matrix(conflicts)
    
    # Precompute neighbor lists
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    # Compute spectral lower bound using Laplacian eigenvalues
    # For a graph with n vertices, the independence number alpha >= n / (1 + d_max)
    # but we can do better with spectral methods
    L = laplacian(conflicts_sparse)
    
    # Use a few eigenvalues for a better bound
    try:
        # Get smallest eigenvalues (related to connectivity)
        eigvals, eigvecs = sla.eigsh(L, k=5, which='SM')
        # Spectral bound: alpha >= n * lambda_min / (lambda_min + lambda_max)
        # But we need lambda_max too
        eigvals_max, _ = sla.eigsh(L, k=1, which='LM')
        lambda_min = np.min(eigvals)
        lambda_max = np.max(eigvals_max)
        if lambda_max > 0:
            spectral_lb = int(N4 * lambda_min / (lambda_min + lambda_max))
        else:
            spectral_lb = N4
    except:
        spectral_lb = int(N4 / (1 + np.max(neighbor_counts)))
    
    # Simple degree-based upper bound for clique (lower bound for independence)
    # alpha <= n * (1 - 1/(1+d_avg)) roughly
    d_avg = np.mean(neighbor_counts)
    degree_ub = int(N4 * (1 - 1/(1+d_avg)))
    
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)
    
    def greedy_min_degree():
        """Greedy independent set with min-degree heuristic."""
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = neighbor_counts.copy()
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set
    
    def branch_and_bound(alive_set, current_in_set, current_size, best_size_ref, max_time=5.0):
        """
        Branch and bound for maximum independent set.
        alive_set: boolean array of vertices still available
        current_in_set: boolean array of vertices already in the solution
        current_size: number of vertices in current solution
        Returns: (best_in_set, best_size)
        """
        bnb_deadline = time.time() + max_time
        best_in_set_local = current_in_set.copy()
        best_size_local = current_size
        
        # Get the subgraph induced by alive vertices
        alive_indices = np.where(alive_set)[0]
        if len(alive_indices) == 0:
            return best_in_set_local, best_size_local
        
        # Upper bound: current_size + number of alive vertices
        # But we can do better with coloring or spectral bounds
        # Simple bound: current_size + len(alive_indices)
        if current_size + len(alive_indices) <= best_size_ref:
            return best_in_set_local, best_size_local
        
        # Choose a pivot vertex: the one with highest degree in the alive subgraph
        alive_degrees = np.array([np.sum(conflicts[v][alive_set]) for v in alive_indices])
        pivot_idx = alive_indices[np.argmax(alive_degrees)]
        pivot = pivot_idx
        
        # Branch 1: Include pivot
        # Remove pivot and all its neighbors from alive
        new_alive_1 = alive_set.copy()
        new_alive_1[pivot] = False
        nbrs = neighbors[pivot]
        new_alive_1[nbrs] = False
        
        new_in_set_1 = current_in_set.copy()
        new_in_set_1[pivot] = True
        
        # Branch 2: Exclude pivot
        new_alive_2 = alive_set.copy()
        new_alive_2[pivot] = False
        
        # Recurse on branch 1 (include) first since it might give a better bound
        if time.time() < bnb_deadline:
            result1, size1 = branch_and_bound(new_alive_1, new_in_set_1, current_size + 1, 
                                              best_size_ref, max_time/2)
            if size1 > best_size_local:
                best_size_local = size1
                best_in_set_local = result1.copy()
        
        # Recurse on branch 2 (exclude) if still promising
        if time.time() < bnb_deadline and current_size + np.sum(new_alive_2) > best_size_local:
            result2, size2 = branch_and_bound(new_alive_2, current_in_set, current_size,
                                              best_size_local, max_time/2)
            if size2 > best_size_local:
                best_size_local = size2
                best_in_set_local = result2.copy()
        
        return best_in_set_local, best_size_local
    
    def sa_local_search(in_set_init, max_time=10.0):
        """Min-conflict simulated annealing with add/remove/swap moves."""
        in_set = in_set_init.copy()
        current_size = in_set.sum()
        
        T = 5.0
        cooling = 0.999
        min_T = 0.1
        
        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1
        
        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        
        while time.time() < sa_deadline:
            not_in_set = np.where(~in_set)[0]
            in_set_indices = np.where(in_set)[0]
            
            if len(not_in_set) == 0 and len(in_set_indices) == 0:
                break
            
            move_type = rng.random()
            
            if move_type < 0.6 and len(not_in_set) > 0:
                # ADD move
                counts = count_conflicts[not_in_set]
                min_count = np.min(counts)
                min_cands = not_in_set[counts == min_count]
                v = min_cands[rng.integers(len(min_cands))]
                
                k = min_count
                delta = 1 - k
                
                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.3
                else:
                    accept = rng.random() < np.exp(delta / T)
                
                if accept:
                    to_remove = np.where(conflicts[v] & in_set)[0]
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                    in_set[v] = True
                    current_size += delta
                    count_conflicts[neighbors[v]] += 1
                else:
                    stuck_counter += 1
                    
            elif move_type < 0.8 and len(in_set_indices) > 0:
                # REMOVE move
                v = in_set_indices[rng.integers(len(in_set_indices))]
                delta = -1
                accept = rng.random() < np.exp(delta / T)
                
                if accept:
                    in_set[v] = False
                    current_size += delta
                    count_conflicts[neighbors[v]] -= 1
                else:
                    stuck_counter += 1
                    
            else:
                # SWAP move
                if len(in_set_indices) > 0 and len(not_in_set) > 0:
                    v_remove = in_set_indices[rng.integers(len(in_set_indices))]
                    v_add = not_in_set[rng.integers(len(not_in_set))]
                    
                    conflicts_with = np.where(conflicts[v_add] & in_set)[0]
                    other_conflicts = conflicts_with[conflicts_with != v_remove]
                    
                    if len(other_conflicts) == 0:
                        in_set[v_remove] = False
                        in_set[v_add] = True
                        count_conflicts[neighbors[v_remove]] -= 1
                        count_conflicts[neighbors[v_add]] += 1
                    else:
                        to_remove = np.unique(np.concatenate(([v_remove], other_conflicts)))
                        delta = 1 - len(to_remove)
                        accept = delta > 0 or (rng.random() < np.exp(delta / T))
                        if accept:
                            in_set[to_remove] = False
                            for u in to_remove:
                                count_conflicts[neighbors[u]] -= 1
                            in_set[v_add] = True
                            current_size += delta
                            count_conflicts[neighbors[v_add]] += 1
                        else:
                            stuck_counter += 1
                else:
                    stuck_counter += 1
            
            if current_size > local_best_size:
                local_best_size = current_size
                local_best_in_set = in_set.copy()
                stuck_counter = 0
            
            if stuck_counter > 2000:
                T = max(T, 3.0)
                stuck_counter = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        return local_best_in_set, local_best_size
    
    # Multiple restarts with B&B and SA
    n_restarts = 0
    max_restarts = 30
    
    while time.time() < deadline - 10 and n_restarts < max_restarts:
        n_restarts += 1
        
        # 1. Greedy Min-Degree Init
        in_set = greedy_min_degree()
        current_size = in_set.sum()
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        # 2. Try branch-and-bound on a reduced subgraph
        remaining_time = deadline - 10 - time.time()
        if remaining_time > 5:
            # Focus on vertices not in the current set but with low conflict count
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) > 0:
                # Select a subset of promising vertices for B&B
                # Vertices with fewest conflicts with current set
                conflict_counts = np.array([np.sum(conflicts[v] & in_set) for v in not_in_set])
                # Sort by conflict count (fewest first)
                sorted_idx = np.argsort(conflict_counts)
                # Take top 200 most promising vertices
                num_bnb = min(200, len(not_in_set))
                bnb_candidates = not_in_set[sorted_idx[:num_bnb]]
                
                # Create a subgraph on these candidates + their neighbors
                bnb_alive = np.zeros(N4, dtype=bool)
                bnb_alive[bnb_candidates] = True
                
                bnb_time = min(remaining_time * 0.3, 8.0)
                if bnb_time > 1.0:
                    bnb_result, bnb_size = branch_and_bound(bnb_alive, in_set, current_size,
                                                           best_size, max_time=bnb_time)
                    if bnb_size > best_size:
                        best_size = bnb_size
                        best_in_set = bnb_result.copy()
        
        # 3. SA Local Search
        remaining_time = deadline - 8 - time.time()
        if remaining_time < 3:
            break
        
        sa_time = min(remaining_time * 0.5, 12.0)
        if sa_time < 2:
            sa_time = 2.0
            
        sa_result, sa_size = sa_local_search(in_set, max_time=sa_time)
        
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()
        
        # 4. Polish: try to add any non-conflicting vertex
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 3:
            polish_set = best_in_set.copy()
            polish_size = best_size
            
            changed = True
            while changed and time.time() < deadline - 3:
                changed = False
                not_in = np.where(~polish_set)[0]
                if len(not_in) == 0:
                    break
                
                conflict_counts = conflicts[not_in] @ polish_set.astype(np.float64)
                addable_indices = np.where(conflict_counts == 0)[0]
                
                if len(addable_indices) > 0:
                    v_idx = not_in[addable_indices[0]]
                    polish_set[v_idx] = True
                    polish_size += 1
                    changed = True
            
            if polish_size > best_size:
                best_size = polish_size
                best_in_set = polish_set.copy()
    
    # Final result
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

# EVOLVE-BLOCK-END
