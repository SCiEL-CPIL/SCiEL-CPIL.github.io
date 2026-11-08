"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Build candidate pool: all integer vectors with L2 norm <= some bound
    # We'll use weight-4 vectors (4 non-zero entries of ±1)
    # Total: C(11,4) * 2^4 = 330 * 16 = 5280
    
    candidates = []
    for idx in itertools.combinations(range(D), 4):
        for signs in itertools.product([-1, 1], repeat=4):
            v = [0] * D
            for i, s in zip(idx, signs):
                v[i] = s
            candidates.append(tuple(v))
    
    N = len(candidates)  # 5280
    
    # Convert to numpy array
    V = np.array(candidates, dtype=np.float64)
    
    # All weight-4 vectors have norm^2 = 4
    # Constraint: ||v_i - v_j||^2 >= 4 (max norm squared)
    # ||v_i - v_j||^2 = 8 - 2*v_i·v_j >= 4
    # => v_i·v_j <= 2
    # Conflict if dot >= 3
    
    # Build conflict graph using chunked matrix multiplication
    chunk_size = 256
    conflicts = [set() for _ in range(N)]
    
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        if time.time() > deadline - 40:
            break
        V_chunk = V[start:end]
        dots = V_chunk @ V.T  # shape (chunk_size, N)
        for local_i in range(end - start):
            i = start + local_i
            mask = dots[local_i] > 2
            mask[i] = False  # exclude self
            conflict_indices = np.where(mask)[0]
            for j in conflict_indices:
                j = int(j)
                conflicts[i].add(j)
                conflicts[j].add(i)
    
    # QUBO formulation: maximize sum(x_i) - lambda * sum(conflict_pairs)
    # Use multi-start local search with simulated annealing
    
    def local_search(initial_selection, max_iter=5000):
        """Local search: try adding/removing vertices to improve objective."""
        selected = set(initial_selection)
        current_obj = len(selected)
        
        # Compute conflict count for current selection
        conflict_count = 0
        for i in selected:
            for j in conflicts[i]:
                if j in selected and i < j:
                    conflict_count += 1
        current_obj = len(selected) - 10 * conflict_count
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            
            # Try adding each unselected vertex
            for i in range(N):
                if i in selected:
                    continue
                # Count conflicts if we add i
                new_conflicts = sum(1 for j in conflicts[i] if j in selected)
                delta_obj = 1 - 10 * new_conflicts
                if delta_obj > 0:
                    selected.add(i)
                    current_obj += delta_obj
                    improved = True
            
            # Try removing vertices that cause many conflicts
            for i in list(selected):
                conf_in_sel = sum(1 for j in conflicts[i] if j in selected)
                if conf_in_sel > 0:
                    delta_obj = -1 + 10 * conf_in_sel
                    if delta_obj > 0:
                        selected.discard(i)
                        current_obj += delta_obj
                        improved = True
        
        return selected, current_obj
    
    def simulated_annealing(initial_selection, max_iter=10000, T0=1.0, T_end=0.01):
        """Simulated annealing to escape local optima."""
        selected = set(initial_selection)
        current_obj = len(selected) - 10 * sum(1 for i in selected for j in conflicts[i] if j in selected and i < j)
        
        T = T0
        cooling = (T_end / T0) ** (1.0 / max_iter)
        
        best_selected = set(selected)
        best_obj = current_obj
        
        for it in range(max_iter):
            T *= cooling
            if time.time() > deadline - 10:
                break
            
            # Random move: add or remove a random vertex
            i = np.random.randint(N)
            if i in selected:
                # Remove i
                conf_in_sel = sum(1 for j in conflicts[i] if j in selected)
                delta_obj = -1 + 10 * conf_in_sel
                selected.discard(i)
                current_obj += delta_obj
            else:
                # Add i
                new_conflicts = sum(1 for j in conflicts[i] if j in selected)
                delta_obj = 1 - 10 * new_conflicts
                selected.add(i)
                current_obj += delta_obj
            
            if current_obj > best_obj:
                best_obj = current_obj
                best_selected = set(selected)
            
            # Accept or reject
            if delta_obj >= 0:
                pass  # accept
            else:
                if T > 0 and np.random.random() < np.exp(delta_obj / T):
                    pass  # accept
                else:
                    # Reject: undo the move
                    if i in selected:
                        conf_in_sel = sum(1 for j in conflicts[i] if j in selected)
                        selected.discard(i)
                        current_obj -= (-1 + 10 * conf_in_sel)
                    else:
                        new_conflicts = sum(1 for j in conflicts[i] if j in selected)
                        selected.add(i)
                        current_obj -= (1 - 10 * new_conflicts)
        
        return best_selected, best_obj
    
    # Multi-start: try several different initial solutions
    best_result = set()
    best_size = 0
    
    # Start 1: Greedy by degree (lowest degree first)
    degrees = [len(c) for c in conflicts]
    order = sorted(range(N), key=lambda x: degrees[x])
    greedy_sel = set()
    for idx in order:
        can_add = all(nb not in greedy_sel for nb in conflicts[idx])
        if can_add:
            greedy_sel.add(idx)
    
    if len(greedy_sel) > best_size:
        best_result = set(greedy_sel)
        best_size = len(greedy_sel)
    
    # Local search from greedy
    if time.time() < deadline - 30:
        ls_result, _ = local_search(greedy_sel, max_iter=3000)
        # Verify it's a valid independent set
        valid_ls = all(all(nb not in ls_result for nb in conflicts[i]) for i in ls_result)
        if valid_ls and len(ls_result) > best_size:
            best_result = set(ls_result)
            best_size = len(ls_result)
    
    # Simulated annealing from greedy
    if time.time() < deadline - 20:
        sa_result, _ = simulated_annealing(greedy_sel, max_iter=5000, T0=2.0, T_end=0.01)
        # Verify validity
        valid_sa = all(all(nb not in sa_result for nb in conflicts[i]) for i in sa_result)
        if valid_sa and len(sa_result) > best_size:
            best_result = set(sa_result)
            best_size = len(sa_result)
    
    # Start 2: Random initial selection
    if time.time() < deadline - 15:
        np.random.seed(42)
        random_sel = set(np.random.choice(N, size=N//10, replace=False))
        # Make it independent by removing conflicts
        for i in list(random_sel):
            for nb in conflicts[i]:
                if nb in random_sel:
                    random_sel.discard(nb)
        
        ls_result2, _ = local_search(random_sel, max_iter=2000)
        valid_ls2 = all(all(nb not in ls_result2 for nb in conflicts[i]) for i in ls_result2)
        if valid_ls2 and len(ls_result2) > best_size:
            best_result = set(ls_result2)
            best_size = len(ls_result2)
    
    # Start 3: Another random seed
    if time.time() < deadline - 10:
        np.random.seed(123)
        random_sel2 = set(np.random.choice(N, size=N//8, replace=False))
        for i in list(random_sel2):
            for nb in conflicts[i]:
                if nb in random_sel2:
                    random_sel2.discard(nb)
        
        ls_result3, _ = local_search(random_sel2, max_iter=2000)
        valid_ls3 = all(all(nb not in ls_result3 for nb in conflicts[i]) for i in ls_result3)
        if valid_ls3 and len(ls_result3) > best_size:
            best_result = set(ls_result3)
            best_size = len(ls_result3)
    
    # Final local search on best result
    if time.time() < deadline - 5 and best_result:
        final_ls, _ = local_search(best_result, max_iter=5000)
        valid_final = all(all(nb not in final_ls for nb in conflicts[i]) for i in final_ls)
        if valid_final and len(final_ls) > best_size:
            best_result = set(final_ls)
            best_size = len(final_ls)
    
    if not best_result:
        # Fallback to seed
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    result_vectors = np.array([candidates[i] for i in best_result], dtype=np.int64)
    
    return result_vectors
# EVOLVE-BLOCK-END
