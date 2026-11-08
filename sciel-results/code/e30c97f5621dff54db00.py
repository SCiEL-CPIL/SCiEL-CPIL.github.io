"""Flow-shop makespan minimizer: NEH + Or-opt local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _makespan_fast(perm, pt):
    """Fast makespan: last machine cumulative sum of the selected rows."""
    return float(pt[perm].sum(axis=0)[-1])


def _neh(processing_times):
    """NEH heuristic: order jobs by total processing time desc, insert each optimally."""
    n_jobs, n_machines = processing_times.shape
    order = list(np.argsort(-processing_times.sum(axis=1)))
    perm = np.empty(n_jobs, dtype=np.int64)
    perm[0] = order[0]
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_cost = float('inf')
        for pos in range(i + 1):
            # build candidate
            cand = np.empty(i + 1, dtype=np.int64)
            if pos == 0:
                cand[0] = job
                cand[1:] = perm[:i]
            elif pos == i:
                cand[:i] = perm[:i]
                cand[i] = job
            else:
                cand[:pos] = perm[:pos]
                cand[pos] = job
                cand[pos + 1:] = perm[pos:i]
            cost = float(processing_times[cand].sum(axis=0)[-1])
            if cost < best_cost:
                best_cost = cost
                best_pos = pos
        # commit
        new = np.empty(i + 1, dtype=np.int64)
        if best_pos == 0:
            new[0] = job
            new[1:] = perm[:i]
        elif best_pos == i:
            new[:i] = perm[:i]
            new[i] = job
        else:
            new[:best_pos] = perm[:best_pos]
            new[best_pos] = job
            new[best_pos + 1:] = perm[best_pos:i]
        perm[:i + 1] = new
    return perm


def _p_spt(processing_times):
    """Palmer's SPT rule (shortest processing time on first machine)."""
    return np.argsort(processing_times[:, 0])


def _oropt_search(perm, processing_times, deadline, margin=2.0):
    """
    Or-opt local search: moves blocks of 1, 2, or 3 jobs.
    Uses best-improvement strategy with time guard.
    """
    n_jobs = len(perm)
    if n_jobs < 2:
        return perm, _makespan_fast(perm, processing_times)
    
    best = perm.copy()
    best_cost = _makespan_fast(best, processing_times)
    improved = True
    
    # Or-opt moves: block length L in {1, 2, 3}
    # For each L, iterate start i, and end position j (insertion index in the reduced sequence)
    # To avoid duplicates and invalid moves, we define the move as:
    # Remove block best[i:i+L]
    # Insert it at position j in the remaining sequence (0 to n_jobs-L)
    # If j == i or j == i+L, it's the same position (no move).
    
    while improved:
        improved = False
        best_move = None
        best_new_cost = best_cost
        
        for L in [1, 2, 3]:
            if L > n_jobs:
                continue
            for i in range(n_jobs - L + 1):
                # Remove block
                # The remaining sequence length is n_jobs - L
                # We can insert the block at any position 0 .. n_jobs - L
                for j in range(n_jobs - L + 1):
                    # Skip if inserting back at the same place
                    if j == i or j == i + L:
                        continue
                    
                    # Construct candidate
                    # new = best[0:i] + block + best[i+L:j] + best[j:] ? 
                    # Wait, the indices change when we remove.
                    # Let's build it carefully.
                    # Original: A (0..i-1), B (i..i+L-1), C (i+L..n-1)
                    # We want to insert B at position j in A + C.
                    # A + C is best[0:i] concatenated with best[i+L:]
                    # Let's call this rem.
                    # rem_len = n_jobs - L
                    # If j <= i:
                    #   new = best[0:j] + B + best[j:i] + best[i+L:]
                    # If j > i:
                    #   The insertion index in the original array would be j + L ?
                    #   Let's just use np.concatenate for clarity, it's fast enough for small N.
                    
                    # Optimization: Pre-compute slices? No, just concat.
                    # rem = np.concatenate([best[:i], best[i+L:]])
                    # new = np.concatenate([rem[:j], best[i:i+L], rem[j:]])
                    
                    # Actually, doing this inside the triple loop is O(N) per move.
                    # Total moves ~ N^2 * 3. Total ops ~ 3 * N^3.
                    # For N=30, 3 * 27000 = 81000 ops per pass. Very fast.
                    
                    # Let's do it without intermediate concat to save a tiny bit of time
                    # But clarity is better.
                    if j <= i:
                        # Insert before i
                        # New array: best[0:j] + best[i:i+L] + best[j:i] + best[i+L:]
                        cand = np.empty(n_jobs, dtype=np.int64)
                        cand[:j] = best[:j]
                        cand[j:j+L] = best[i:i+L]
                        cand[j+L:j+L+(i-j)] = best[j:i]
                        cand[j+L+(i-j):] = best[i+L:]
                    else:
                        # Insert after i (in the shifted coordinates)
                        # j is index in rem. 
                        # In original, the part before j in rem is best[0:i] + best[i+L:j] ?? No.
                        # rem = best[0:i] + best[i+L:]
                        # rem[:j] = best[0:i] + best[i+L:j] (if j > i)
                        # Wait, if j > i, then the first i elements of rem are best[0:i].
                        # The next j-i elements are best[i+L : i+L + (j-i)] = best[i+L : j]
                        # So rem[:j] is best[0:i] + best[i+L:j]
                        # rem[j:] is best[j:] (since j > i, j in rem corresponds to j in original? 
                        # Let's check: rem index k corresponds to original index k if k < i, 
                        # and k+L if k >= i.
                        # So rem[j] is original j+L.
                        # So rem[j:] is best[j+L:]
                        
                        # So new = best[0:i] + best[i+L:j] + best[i:i+L] + best[j+L:]
                        cand = np.empty(n_jobs, dtype=np.int64)
                        cand[:i] = best[:i]
                        cand[i:i+(j-i)] = best[i+L:j]
                        cand[i+(j-i):i+(j-i)+L] = best[i:i+L]
                        cand[i+(j-i)+L:] = best[j+L:]
                    
                    cost = _makespan_fast(cand, processing_times)
                    if cost < best_new_cost - 1e-12:
                        best_new_cost = cost
                        best_move = cand
                        best_move_L = L # not strictly needed but for debug
                        best_move_i = i
                        best_move_j = j
            
            # Time check inside L loop? Better outside.
            if time.monotonic() > deadline - margin:
                return best, best_cost
        
        if best_move is not None:
            best = best_move
            best_cost = best_new_cost
            improved = True
        else:
            improved = False
            
    return best, best_cost


def order_jobs(processing_times):
    processing_times = np.asarray(processing_times, dtype=float)
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]

    t0 = time.monotonic()
    deadline = t0 + 115.0  # safety margin vs 150s kill
    margin = 3.0

    # Constructive candidates
    cands = []
    cands.append(_neh(processing_times))
    cands.append(_p_spt(processing_times))
    
    # Add a few random starts for diversity
    rng = np.random.default_rng(42)
    for _ in range(2):
        cands.append(rng.permutation(n_jobs))

    best = cands[0]
    best_cost = _makespan_fast(best, processing_times)
    for c in cands[1:]:
        cost = _makespan_fast(c, processing_times)
        if cost < best_cost:
            best_cost = cost
            best = c

    # Run local search on the best candidate
    best, best_cost = _oropt_search(best, processing_times, deadline, margin)

    # If time remains, try local search on other constructive starts
    # We sort candidates by initial cost to prioritize better starts
    # But we already picked the best. Let's try the second best, etc.
    # To avoid re-evaluating the same one, we can just iterate through cands
    # and skip if it's the same as current best (by identity or value)
    
    # Simple approach: try each other candidate
    for c in cands:
        if time.monotonic() > deadline - margin - 10.0:
            break
        # Check if this candidate is significantly different or just try it
        # If it's the same array as best, skip
        if np.array_equal(c, best):
            continue
        
        # Run local search on this candidate
        cand_best, cand_cost = _oropt_search(c, processing_times, deadline, margin)
        if cand_cost < best_cost:
            best_cost = cand_cost
            best = cand_best

    return list(best)
# EVOLVE-BLOCK-END
