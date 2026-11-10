<<<<<<< SEARCH
import numpy as np
import time
from itertools import combinations
from collections import defaultdict

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 580
    
    # Generate all 4-subsets of {0, ..., 10}
    four_subsets = list(combinations(range(d), 4))
    n_subsets = len(four_subsets)  # 330
    
    # For each pair of 4-subsets, check if they share at most 2 elements
    # i.e., |A ∩ B| ≤ 2, which means they don't share a 3-subset
    # Precompute compatibility: compatible[i][j] = True if |S_i ∩ S_j| <= 2
    compatible = np.ones((n_subsets, n_subsets), dtype=bool)
    
    for i in range(n_subsets):
        si = set(four_subsets[i])
        for j in range(i + 1, n_subsets):
            sj = set(four_subsets[j])
            inter = len(si & sj)
            if inter > 2:
                compatible[i, j] = False
                compatible[j, i] = False
    
    # Find maximum independent set in the incompatibility graph
    # Incompatibility graph: edge between i,j if they share ≥3 elements
    # We want max set of vertices with no incompatibility edges
    # This is maximum independent set = maximum clique in complement
    
    # Use greedy with multiple restarts + local search
    best_set = set()
    best_size = 0
    rng = np.random.default_rng(12345)
    
    def try_greedy(order, time_limit):
        """Greedy: add vertices in order if compatible with all already selected"""
        selected = set()
        for idx in order:
            if time.time() > time_limit:
                break
            # Check if idx is compatible with all in selected
            ok = True
            for s in selected:
                if not compatible[idx, s]:
                    ok = False
                    break
            if ok:
                selected.add(idx)
        return selected
    
    def local_search(initial_set, time_limit):
        """Try to improve by swapping or adding"""
        selected = set(initial_set)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try adding any vertex
            for idx in range(n_subsets):
                if idx in selected:
                    continue
                ok = True
                for s in selected:
                    if not compatible[idx, s]:
                        ok = False
                        break
                if ok:
                    selected.add(idx)
                    improved = True
                    break
            # Try swapping: remove one, add one or more
            if time.time() > time_limit:
                break
            for remove in list(selected):
                if time.time() > time_limit:
                    break
                new_set = selected - {remove}
                # Try to add multiple vertices
                for add in range(n_subsets):
                    if add in new_set or add == remove:
                        continue
                    ok = True
                    for s in new_set:
                        if not compatible[add, s]:
                            ok = False
                            break
                    if ok:
                        new_set.add(add)
                        # Try to add more
                        for add2 in range(n_subsets):
                            if add2 in new_set:
                                continue
                            ok2 = True
                            for s in new_set:
                                if not compatible[add2, s]:
                                    ok2 = False
                                    break
                            if ok2:
                                new_set.add(add2)
                        if len(new_set) > len(selected):
                            selected = new_set
                            improved = True
                            break
                if improved:
                    break
        return selected
    
    # Phase 1: Multiple greedy restarts
    num_restarts = 0
    while time.time() < deadline - 120:
        num_restarts += 1
        order = rng.permutation(n_subsets)
        candidate_set = try_greedy(order, deadline - 110)
        if len(candidate_set) > best_size:
            best_size = len(candidate_set)
            best_set = candidate_set
        
        # Local search on best found
        if num_restarts % 5 == 0 and best_size > 0:
            improved = local_search(best_set, deadline - 100)
            if len(improved) > best_size:
                best_size = len(improved)
                best_set = improved
        
        if best_size >= 25:
            break
    
    # Phase 2: More intensive local search
    if time.time() < deadline - 30 and best_size > 0:
        final = local_search(best_set, deadline - 20)
        if len(final) > best_size:
            best_size = len(final)
            best_set = final
=======
import numpy as np
import time
from itertools import combinations
from collections import defaultdict

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 580
    
    # Generate all 4-subsets of {0, ..., 10}
    four_subsets = list(combinations(range(d), 4))
    n_subsets = len(four_subsets)  # 330
    
    # Precompute compatibility: compatible[i][j] = True if |S_i ∩ S_j| <= 2
    compatible = np.ones((n_subsets, n_subsets), dtype=bool)
    
    for i in range(n_subsets):
        si = set(four_subsets[i])
        for j in range(i + 1, n_subsets):
            sj = set(four_subsets[j])
            inter = len(si & sj)
            if inter > 2:
                compatible[i, j] = False
                compatible[j, i] = False
    
    # Compute degree in incompatibility graph for heuristic ordering
    incompat_degree = np.sum(~compatible, axis=1)
    
    # Find maximum independent set using improved greedy + local search
    best_set = set()
    best_size = 0
    rng = np.random.default_rng(12345)
    
    def try_greedy_heuristic(order, time_limit):
        """Greedy: add vertices in order if compatible with all already selected"""
        selected = set()
        selected_arr = np.zeros(n_subsets, dtype=bool)
        for idx in order:
            if time.time() > time_limit:
                break
            # Check if idx is compatible with all in selected using vectorized check
            if len(selected) == 0:
                selected.add(idx)
                selected_arr[idx] = True
            else:
                # Vectorized compatibility check
                compat_with_selected = compatible[idx, selected_arr]
                if compat_with_selected.all():
                    selected.add(idx)
                    selected_arr[idx] = True
        return selected
    
    def local_search_fast(initial_set, time_limit):
        """Fast local search using vectorized operations"""
        selected = set(initial_set)
        selected_arr = np.zeros(n_subsets, dtype=bool)
        for idx in selected:
            selected_arr[idx] = True
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try adding any vertex using vectorized check
            candidates = np.where(~selected_arr)[0]
            if len(candidates) == 0:
                break
            # Check which candidates are compatible with all selected
            if len(selected) > 0:
                compat_matrix = compatible[candidates, selected_arr]
                valid_mask = compat_matrix.all(axis=1)
                valid_candidates = candidates[valid_mask]
            else:
                valid_candidates = candidates
            
            if len(valid_candidates) > 0:
                # Add the first valid candidate
                add_idx = valid_candidates[0]
                selected.add(add_idx)
                selected_arr[add_idx] = True
                improved = True
                continue
            
            # Try 1-for-1 swaps: remove one from selected, add one from candidates
            for remove in list(selected):
                if time.time() > time_limit:
                    break
                temp_arr = selected_arr.copy()
                temp_arr[remove] = False
                temp_selected = selected - {remove}
                
                # Find candidates compatible with temp_selected
                candidates2 = np.where(~temp_arr)[0]
                if len(candidates2) == 0:
                    continue
                if len(temp_selected) > 0:
                    compat_matrix2 = compatible[candidates2, temp_arr]
                    valid_mask2 = compat_matrix2.all(axis=1)
                    valid_candidates2 = candidates2[valid_mask2]
                else:
                    valid_candidates2 = candidates2
                
                if len(valid_candidates2) > 0:
                    add_idx = valid_candidates2[0]
                    selected = temp_selected | {add_idx}
                    selected_arr = temp_arr.copy()
                    selected_arr[add_idx] = True
                    improved = True
                    break
        return selected
    
    def local_search_swap(initial_set, time_limit):
        """Local search with 2-for-1 and 1-for-2 swaps"""
        selected = set(initial_set)
        selected_arr = np.zeros(n_subsets, dtype=bool)
        for idx in selected:
            selected_arr[idx] = True
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Try 1-for-2 swaps: remove one, add two
            for remove in list(selected):
                if time.time() > time_limit:
                    break
                temp_arr = selected_arr.copy()
                temp_arr[remove] = False
                temp_selected = selected - {remove}
                
                # Find first candidate compatible with temp
                candidates = np.where(~temp_arr)[0]
                if len(candidates) < 2:
                    continue
                if len(temp_selected) > 0:
                    compat_matrix = compatible[candidates, temp_arr]
                    valid_mask = compat_matrix.all(axis=1)
                    valid_candidates = candidates[valid_mask]
                else:
                    valid_candidates = candidates
                
                if len(valid_candidates) < 2:
                    continue
                
                # Try adding first two valid candidates that are compatible with each other
                for i in range(len(valid_candidates)):
                    for j in range(i+1, len(valid_candidates)):
                        c1, c2 = valid_candidates[i], valid_candidates[j]
                        if compatible[c1, c2]:
                            new_set = temp_selected | {c1, c2}
                            if len(new_set) > len(selected):
                                selected = new_set
                                selected_arr = np.zeros(n_subsets, dtype=bool)
                                for idx in selected:
                                    selected_arr[idx] = True
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        return selected
    
    # Phase 1: Multiple greedy restarts with heuristic ordering
    num_restarts = 0
    # Use degree-based ordering: lower degree vertices first (more compatible)
    degree_order = np.argsort(incompat_degree)
    
    while time.time() < deadline - 120:
        num_restarts += 1
        # Mix random and degree-based ordering
        if num_restarts % 3 == 0:
            order = degree_order.copy()
            # Shuffle slightly
            for i in range(len(order) - 1, 0, -1):
                j = rng.integers(0, i + 1)
                order[i], order[j] = order[j], order[i]
        else:
            order = rng.permutation(n_subsets)
        
        candidate_set = try_greedy_heuristic(order, deadline - 110)
        if len(candidate_set) > best_size:
            best_size = len(candidate_set)
            best_set = candidate_set
        
        # Local search on best found
        if num_restarts % 3 == 0 and best_size > 0:
            improved = local_search_fast(best_set, deadline - 100)
            if len(improved) > best_size:
                best_size = len(improved)
                best_set = improved
        
        if best_size >= 30:
            break
    
    # Phase 2: More intensive local search with different strategies
    if time.time() < deadline - 60 and best_size > 0:
        # Try fast local search
        final = local_search_fast(best_set, deadline - 40)
        if len(final) > best_size:
            best_size = len(final)
            best_set = final
        
        # Try swap-based local search
        if time.time() < deadline - 20:
            final2 = local_search_swap(best_set, deadline - 15)
            if len(final2) > best_size:
                best_size = len(final2)
                best_set = final2
>>>>>>> REPLACE

