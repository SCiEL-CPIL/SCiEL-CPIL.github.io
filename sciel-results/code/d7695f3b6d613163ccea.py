import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    # Handle the specific high-interest case
    if q == 3 and n == 6 and radius == 1:
        return _construct_3_6_1()
    
    # Fallback for other parameters
    if q == 3:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Simple fallback: return all words (valid, just not minimal)
    return [list(w) for w in itertools.product(range(q), repeat=n)]

def _construct_3_6_1():
    q = 3
    n = 6
    N = q ** n
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Generate all words and map to indices
    words = [list(w) for w in itertools.product(range(q), repeat=n)]
    
    # Build neighbor matrix: neighbor_matrix[i, j] is True if dist(i, j) <= 1
    # N=729, so 729x729 boolean matrix is ~0.5 MB. Very fast.
    neighbor_matrix = np.zeros((N, N), dtype=bool)
    for i in range(N):
        neighbor_matrix[i, i] = True
        w_i = words[i]
        for pos in range(n):
            orig_val = w_i[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                # Create new word tuple to find index
                # Efficiently: calculate index directly? 
                # Index is base-3 number.
                # Let's just use the word list lookup for simplicity as N is small
                new_w = list(w_i)
                new_w[pos] = new_val
                j = words.index(new_w) # This is O(N^2) setup, 729*729*6 ~ 3M ops. Acceptable.
                neighbor_matrix[i, j] = True
                neighbor_matrix[j, i] = True # Symmetric

    # Greedy Set Cover
    uncovered = np.ones(N, dtype=bool)
    selected_mask = np.zeros(N, dtype=bool)
    
    while uncovered.any():
        # Count uncovered neighbors for each candidate
        # counts[i] = number of True in neighbor_matrix[i] where uncovered is True
        # Using matrix multiplication: neighbor_matrix @ uncovered_vector
        # But we need to exclude those already selected? No, greedy picks best from all.
        # Actually, standard greedy picks the one covering most *new* (uncovered) points.
        
        # Vectorized count:
        # counts = neighbor_matrix.dot(uncovered.astype(np.int8))
        # This gives total covered by each i among uncovered.
        counts = neighbor_matrix.dot(uncovered)
        
        # Pick max
        best_idx = np.argmax(counts)
        if counts[best_idx] == 0:
            break
            
        selected_mask[best_idx] = True
        uncovered[neighbor_matrix[best_idx]] = False
        
        if time.time() > deadline:
            break
            
    # Local Search
    # coverage[i] = number of selected codewords covering word i
    coverage = np.zeros(N, dtype=np.int32)
    for i in range(N):
        if selected_mask[i]:
            coverage[neighbor_matrix[i]] += 1
            
    # Helper to check if a selected index is redundant
    # i is redundant if for all j in neighbors[i], coverage[j] > 1
    # i.e., min(coverage[neighbors[i]]) > 1
    
    def try_reduce(selected_mask, coverage):
        changed = True
        while changed:
            changed = False
            # Iterate through selected indices
            selected_indices = np.where(selected_mask)[0]
            for i in selected_indices:
                # Check if redundant
                # If min coverage of neighbors is > 1, it's redundant
                if np.min(coverage[neighbor_matrix[i]]) > 1:
                    # Remove i
                    selected_mask[i] = False
                    coverage[neighbor_matrix[i]] -= 1
                    changed = True
                    # Break to restart? Or continue?
                    # Continuing is fine, but indices might change if we used a list.
                    # Since we use mask, we just mark False.
                    # But `selected_indices` is a snapshot. 
                    # If we remove i, other indices in snapshot are still valid.
        return selected_mask, coverage

    selected_mask, coverage = try_reduce(selected_mask, coverage)
    
    current_size = np.sum(selected_mask)
    best_mask = selected_mask.copy()
    best_cov = coverage.copy()
    best_size = current_size
    
    random.seed(42)
    
    # Local Search: 1-Exchange
    # We try to swap one selected codeword for one unselected codeword.
    # Condition for swap i -> k:
    # 1. U_i = { j | j in N(i) and coverage[j] == 1 } must be covered by N(k).
    # 2. Ideally, we want to reduce size, but 1-exchange keeps size same.
    #    However, a swap might enable a subsequent reduction.
    #    So we accept swaps that allow reduction, or just do random swaps to escape local minima?
    #    Better: Accept swap if it leads to a state where reduction decreases size.
    
    max_iters = 1000
    iter_count = 0
    no_improve_count = 0
    
    while time.time() < deadline - 2.0 and iter_count < max_iters:
        iter_count += 1
        
        # Get selected and unselected indices
        sel_idx = np.where(selected_mask)[0]
        un_sel_idx = np.where(~selected_mask)[0]
        
        if len(sel_idx) == 0 or len(un_sel_idx) == 0:
            break
            
        # Shuffle to avoid bias
        random.shuffle(sel_idx.tolist())
        swap_made = False
        
        for i in sel_idx:
            if time.time() > deadline - 2.0:
                break
                
            # Critical set for i
            # U = neighbors of i where coverage is 1
            U_mask = neighbor_matrix[i].copy()
            U_mask[coverage != 1] = False
            
            if not U_mask.any():
                # Redundant, should have been removed, but maybe coverage changed?
                # Remove it
                selected_mask[i] = False
                coverage[neighbor_matrix[i]] -= 1
                current_size -= 1
                swap_made = True
                # Try to reduce further
                selected_mask, coverage = try_reduce(selected_mask, coverage)
                new_size = np.sum(selected_mask)
                if new_size < best_size:
                    best_size = new_size
                    best_mask = selected_mask.copy()
                    best_cov = coverage.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                break # Restart loop with new state
            
            # Find candidates k in un_sel_idx that cover U
            # We need N(k) to contain all indices where U_mask is True.
            # Equivalent to: (neighbor_matrix[k] & ~U_mask).any() is False? 
            # No: We need U_mask <= neighbor_matrix[k].
            # So (U_mask & ~neighbor_matrix[k]).any() must be False.
            
            # Vectorized check for all un_sel_idx:
            # For each k, check if U_mask is subset of neighbor_matrix[k]
            # This is: ~np.any(U_mask[None, :] & ~neighbor_matrix[un_sel_idx, :], axis=1)
            
            # This creates a (len(un_sel_idx) x N) boolean array.
            # 729 * 729 is small.
            
            # Optimization: Only check if k covers the "hardest" part of U?
            # Just do the full check.
            
            # Compute subset condition
            # U_mask is shape (N,)
            # neighbor_matrix[un_sel_idx] is shape (M, N)
            # We want to know for each row m, if U_mask is subset of row m.
            # i.e., (U_mask & ~row_m).sum() == 0
            
            # Use broadcasting
            # U_mask[None, :] & ~neighbor_matrix[un_sel_idx, :]
            # This is (1, N) & (M, N) -> (M, N)
            # Then sum along axis 1.
            
            # To save memory/time, we can pick a random element from U and filter candidates first?
            # Let's try direct vectorization first.
            
            # candidates_ok = np.array([np.all(neighbor_matrix[k][U_mask]) for k in un_sel_idx])
            # The vectorized version:
            
            # Check if U is subset of N(k)
            # If U is empty, handled above.
            
            # We want k such that N(k) covers U.
            # Let's find all such k.
            
            # If U is large, this is expensive. 
            # But N=729.
            
            # Let's use the property: if k covers U, then for any u in U, k is in N(u).
            # So k must be in intersection of N(u) for all u in U.
            # Common neighbors of U.
            
            # Let's compute the intersection of neighbor rows for all u in U.
            # U_indices = np.where(U_mask)[0]
            # If U_indices is empty, handled.
            
            U_indices = np.where(U_mask)[0]
            
            # Start with all unselected
            possible_k = un_sel_idx.copy()
            
            # Intersect with neighbors of each u in U
            # We can do this iteratively or with a matrix.
            # Since |U| is likely small (critical words), iterative intersection is fast.
            
            for u in U_indices:
                # Filter possible_k to those that are neighbors of u
                # possible_k = possible_k[neighbor_matrix[u, possible_k]]
                # This is fast.
                if len(possible_k) == 0:
                    break
                possible_k = possible_k[neighbor_matrix[u, possible_k]]
                
            if len(possible_k) == 0:
                continue
                
            # Now possible_k contains all unselected codewords that cover U.
            # We want to pick one that is "best".
            # Heuristic: Pick one that covers the most other words with coverage 1 (to enable more reductions).
            # Or just pick randomly from possible_k to diversify.
            
            # Let's score them:
            # Score = sum( (coverage == 1) & neighbor_matrix[k] ) for k in possible_k
            # This counts how many critical words (other than U? or including U?) it covers.
            # Since it covers U, it covers at least |U| critical words.
            # We want to cover *other* critical words to make other codewords redundant.
            
            # Let's just pick randomly from possible_k for now to escape local minima.
            # Or pick the one with max coverage of critical words.
            
            # Vectorized score:
            # critical_mask = (coverage == 1)
            # scores = neighbor_matrix[possible_k].dot(critical_mask)
            
            critical_mask = (coverage == 1)
            scores = neighbor_matrix[possible_k].dot(critical_mask)
            
            best_k = possible_k[np.argmax(scores)]
            
            # Perform Swap: Remove i, Add best_k
            selected_mask[i] = False
            coverage[neighbor_matrix[i]] -= 1
            
            selected_mask[best_k] = True
            coverage[neighbor_matrix[best_k]] += 1
            
            # Try to reduce
            selected_mask, coverage = try_reduce(selected_mask, coverage)
            new_size = np.sum(selected_mask)
            
            if new_size < best_size:
                best_size = new_size
                best_mask = selected_mask.copy()
                best_cov = coverage.copy()
                no_improve_count = 0
                swap_made = True
                break # Restart
            else:
                # Accept swap even if size doesn't decrease immediately?
                # Yes, to escape local minima.
                # But we must ensure we don't cycle.
                # Since we track best_mask, we can always revert to best if stuck.
                # For now, accept if it's a different configuration.
                # To avoid cycling, we could keep a history, but simple acceptance with randomization is often enough.
                no_improve_count += 1
                swap_made = True
                break # Restart with new state to re-evaluate reductions
                
        if not swap_made:
            # No swap found in this pass
            no_improve_count += 1
            if no_improve_count > 50:
                # Perturb: Remove a random selected codeword and re-greedy?
                # Or just break.
                # Let's try a random restart from best?
                # Or just stop.
                pass

    # Final reduction
    best_mask, best_cov = try_reduce(best_mask, best_cov)
    
    # Convert mask to list of words
    selected_indices = np.where(best_mask)[0]
    codewords = [words[i] for i in selected_indices]
    
    return codewords
