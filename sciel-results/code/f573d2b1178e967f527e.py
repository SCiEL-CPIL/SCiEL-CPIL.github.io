import time
import random
import numpy as np

N = 64
M = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin for 300s limit

    # Precompute all points
    all_points = [(x, y) for y in range(N) for x in range(N)]
    n_points = len(all_points)
    
    # Precompute squared distance matrix
    # Shape: (M, M)
    # Using int32 to save memory, max dist^2 is 2*63^2 = 7938 which fits in int32
    coords = np.array(all_points, dtype=np.int32)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_mat = np.sum(diff * diff, axis=2).astype(np.int32)
    
    def _greedy_construct(order, time_limit):
        """
        Fast greedy construction.
        Returns S (list of indices) and D (dict: idx -> set of distances to other points in S)
        """
        S = []
        S_set = set()
        D = {}
        
        for p_idx in order:
            if time.time() > time_limit:
                break
                
            if p_idx in S_set:
                continue
            
            if not S:
                S.append(p_idx)
                S_set.add(p_idx)
                D[p_idx] = set()
                continue
            
            S_arr = np.array(S, dtype=np.int32)
            
            # Distances from p to all points in S
            d_to_S = dist_mat[p_idx, S_arr]
            
            # Check 1: All distances from p to S must be distinct
            # Sort and check for duplicates
            d_sorted = np.sort(d_to_S)
            if np.any(d_sorted[1:] == d_sorted[:-1]):
                continue
            
            # Check 2: For each point s in S, d(p, s) must not be in D[s]
            # D[s] contains distances from s to all other points in S
            valid = True
            for i, s_idx in enumerate(S):
                d_val = int(d_to_S[i])
                if d_val in D[s_idx]:
                    valid = False
                    break
            
            if valid:
                S.append(p_idx)
                S_set.add(p_idx)
                # Store distances from p to all existing points in S
                D[p_idx] = set(int(d) for d in d_to_S)
                
                # Update D for existing points
                for i, s_idx in enumerate(S[:-1]):
                    D[s_idx].add(int(d_to_S[i]))
        
        return S, D

    def _local_search(S, D, time_limit):
        """
        Local search: try to add points, and if stuck, try swaps.
        S: list of indices
        D: dict idx -> set of distances
        Returns best S, D found
        """
        S = S.copy()
        D = {k: set(v) for k, v in D.items()}
        S_set = set(S)
        
        best_S = S.copy()
        best_D = {k: set(v) for k, v in D.items()}
        best_size = len(S)
        
        no_improve_count = 0
        max_no_improve = 200
        
        # Candidate pool: all points not in S
        # We maintain a list of candidates to try adding
        # Initially all points not in S
        candidates = [p for p in range(n_points) if p not in S_set]
        random.shuffle(candidates)
        cand_idx = 0
        
        while time.time() < time_limit:
            improved = False
            
            # Phase A: Try to add a point
            while cand_idx < len(candidates):
                if time.time() > time_limit:
                    break
                    
                p_idx = candidates[cand_idx]
                cand_idx += 1
                
                if p_idx in S_set:
                    continue
                
                S_arr = np.array(S, dtype=np.int32)
                if len(S_arr) == 0:
                    S.append(p_idx)
                    S_set.add(p_idx)
                    D[p_idx] = set()
                    improved = True
                    break
                    
                d_to_S = dist_mat[p_idx, S_arr]
                
                # Check 1: distinct distances
                d_sorted = np.sort(d_to_S)
                if np.any(d_sorted[1:] == d_sorted[:-1]):
                    continue
                
                # Check 2: no conflict
                valid = True
                for i, s_idx in enumerate(S):
                    d_val = int(d_to_S[i])
                    if d_val in D[s_idx]:
                        valid = False
                        break
                
                if valid:
                    # Add p_idx
                    S.append(p_idx)
                    S_set.add(p_idx)
                    D[p_idx] = set(int(d) for d in d_to_S)
                    
                    for i, s_idx in enumerate(S[:-1]):
                        D[s_idx].add(int(d_to_S[i]))
                    
                    improved = True
                    no_improve_count = 0
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = S.copy()
                        best_D = {k: set(v) for k, v in D.items()}
                    # After adding, we can continue scanning candidates
                    # But the set of valid candidates changed.
                    # For efficiency, we just continue. If we get stuck, we'll reshuffle.
                    # We do NOT reset cand_idx to 0 immediately to avoid O(N^2) overhead per add
                    # unless we are stuck.
                    break # Restart candidate scan from current position? Or break to outer loop?
                         # Breaking to outer loop allows us to check time and reset cand_idx if needed.
                         # But if we break, we lose the rest of the current candidate list scan.
                         # Let's just continue the inner loop.
            
            if not improved:
                no_improve_count += 1
                
                if no_improve_count > max_no_improve:
                    # Perturbation: Try a swap
                    # Remove one random point from S
                    if len(S) > 5:
                        # Pick a random point to remove
                        remove_idx = random.choice(S)
                        
                        # Remove it
                        S.remove(remove_idx)
                        S_set.discard(remove_idx)
                        D.pop(remove_idx, None)
                        
                        # Update D for remaining points
                        # We need to remove distance d(remove_idx, s) from D[s]
                        for s_idx in S:
                            d_val = int(dist_mat[remove_idx, s_idx])
                            if d_val in D[s_idx]:
                                D[s_idx].discard(d_val)
                        
                        # Now try to add a new point from candidates
                        # Reshuffle candidates to ensure we try different ones
                        candidates = [p for p in range(n_points) if p not in S_set]
                        random.shuffle(candidates)
                        cand_idx = 0
                        no_improve_count = 0
                    else:
                        # If set is small, just reshuffle candidates
                        candidates = [p for p in range(n_points) if p not in S_set]
                        random.shuffle(candidates)
                        cand_idx = 0
                        no_improve_count = 0
                else:
                    # If we just finished scanning all candidates without improvement,
                    # reshuffle and reset index to try a different order
                    if cand_idx >= len(candidates):
                        candidates = [p for p in range(n_points) if p not in S_set]
                        random.shuffle(candidates)
                        cand_idx = 0
        
        return best_S, best_D

    # Phase 1: Random Greedy Restarts
    # Goal: Find a good initial solution.
    best_size = 0
    best_S = []
    best_D = {}
    
    greedy_solutions = []
    
    t_phase1_end = start_time + 60
    
    while time.time() < t_phase1_end:
        order = list(range(n_points))
        random.shuffle(order)
        
        S, D = _greedy_construct(order, deadline)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = S.copy()
            best_D = {k: set(v) for k, v in D.items()}
        
        greedy_solutions.append((S, D))
        
        # Keep only top 10 greedy solutions to save memory/time
        if len(greedy_solutions) > 10:
            greedy_solutions.sort(key=lambda x: len(x[0]), reverse=True)
            greedy_solutions = greedy_solutions[:10]

    # Phase 2: Local Search on best solutions
    # Sort by size
    greedy_solutions.sort(key=lambda x: len(x[0]), reverse=True)
    
    # Take top K for local search
    top_k = min(5, len(greedy_solutions))
    
    for i in range(top_k):
        if time.time() > deadline - 30:
            break
            
        S_init = greedy_solutions[i][0]
        D_init = greedy_solutions[i][1]
        
        # Run local search
        # Use multiple restarts of local search with different seeds?
        # Or just one long run?
        # Let's do one long run with the time limit.
        
        S_ls, D_ls = _local_search(S_init, D_init, deadline - 10)
        
        if len(S_ls) > best_size:
            best_size = len(S_ls)
            best_S = S_ls.copy()
            best_D = {k: set(v) for k, v in D_ls.items()}
            
        # If we found a very good solution, we can stop
        if best_size >= 60:
            break

    # Phase 3: If time remains, try more greedy restarts or local search on best
    if time.time() < deadline - 10:
        # Try local search on the current best again with a fresh random seed/shuffle
        # This might help escape local optima
        S_ls, D_ls = _local_search(best_S, best_D, deadline - 5)
        if len(S_ls) > best_size:
            best_size = len(S_ls)
            best_S = S_ls.copy()
            best_D = {k: set(v) for k, v in D_ls.items()}

    # Return the best solution found
    return [all_points[i] for i in best_S]
