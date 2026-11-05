import time
import random
import numpy as np
from collections import deque

def construct(q, n, radius):
    # Fallback for general parameters
    if q != 3 or n != 8 or radius != 1:
        total_words = q ** n
        
        def word_to_idx(w):
            idx = 0
            for val in w:
                idx = idx * q + val
            return idx

        def idx_to_word(idx):
            w = [0] * n
            temp = idx
            for i in range(n - 1, -1, -1):
                w[i] = temp % q
                temp //= q
            return list(w)

        # Precompute neighbors
        neighbors = [None] * total_words
        for idx in range(total_words):
            w = idx_to_word(idx)
            nbrs = [idx]
            for i in range(n):
                for val in range(q):
                    if val != w[i]:
                        nw = w[:]
                        nw[i] = val
                        nbrs.append(word_to_idx(tuple(nw)))
            neighbors[idx] = nbrs

        uncovered = np.ones(total_words, dtype=bool)
        code = []
        
        # Simple Greedy Set Cover
        while any(uncovered):
            best_idx = -1
            best_count = -1
            for idx in range(total_words):
                count = sum(1 for nbr in neighbors[idx] if uncovered[nbr])
                if count > best_count:
                    best_count = count
                    best_idx = idx
            if best_idx == -1:
                break
            code.append(best_idx)
            for nbr in neighbors[best_idx]:
                uncovered[nbr] = False
                
        return [idx_to_word(idx) for idx in code]

    # Specific optimization for q=3, n=8, radius=1
    total_words = 3 ** 8  # 6561
    N = total_words
    
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s buffer for 150s limit

    def word_to_idx(w):
        idx = 0
        for val in w:
            idx = idx * 3 + val
        return idx

    def idx_to_word(idx):
        w = [0] * 8
        temp = idx
        for i in range(7, -1, -1):
            w[i] = temp % 3
            temp //= 3
        return w

    # Precompute neighbors efficiently
    # For q=3, n=8, radius=1: size is 1 + 8*2 = 17
    neighbors_matrix = np.empty((N, 17), dtype=np.int32)
    
    for idx in range(N):
        w = [0] * 8
        temp = idx
        for i in range(7, -1, -1):
            w[i] = temp % 3
            temp //= 3
        
        neighbors_matrix[idx, 0] = idx
        k = 1
        for i in range(8):
            for val in range(3):
                if val != w[i]:
                    nw = w.copy()
                    nw[i] = val
                    neighbors_matrix[idx, k] = word_to_idx(tuple(nw))
                    k += 1

    # Data structures
    uncovered = np.ones(N, dtype=bool)
    cover_count = np.zeros(N, dtype=np.int32)
    in_code = np.zeros(N, dtype=bool)
    code_list = []
    
    # --- Phase 1: Greedy Construction ---
    # To make this faster and better, we can use a priority queue or just vectorized counts.
    # Vectorized counts are O(N * 17) per step. N=6561, 17 is small.
    # Number of steps is roughly K (size of code). K ~ 400-500.
    # 500 * 6561 * 17 ~ 55M ops. In numpy this is fast.
    
    while uncovered.any():
        if time.time() > deadline - 60:  # Reserve 60s for local search
            break
            
        # Count uncovered neighbors for all candidates
        # neighbors_matrix shape (N, 17), uncovered shape (N,)
        # result shape (N,)
        counts = uncovered[neighbors_matrix].sum(axis=1)
        
        # Mask out already selected
        counts[in_code] = -1
        
        max_count = counts.max()
        if max_count <= 0:
            break
            
        # Random tie-breaking to escape local optima
        candidates = np.where(counts == max_count)[0]
        best_idx = candidates[random.randint(0, len(candidates) - 1)]
        
        code_list.append(best_idx)
        in_code[best_idx] = True
        uncovered[neighbors_matrix[best_idx]] = False
        cover_count[neighbors_matrix[best_idx]] += 1

    # --- Phase 2: Local Search ---
    # Objective: Minimize |code_list|
    # Strategy:
    # 1. Remove redundant words (words whose removal doesn't break coverage).
    # 2. 1-for-1 swaps: Replace word A with word B if it allows future removals or leads to a state with more redundancy.
    #    Specifically, try to swap A with B such that after swap, we can remove another word C.
    #    Or simply: Swap A with B if B covers "newer" area better or reduces "fragility".
    
    # Helper: Check if word idx is redundant
    def is_redundant(idx):
        # A word is redundant if all its neighbors are covered by at least 2 codewords
        # i.e., cover_count[neighbors] >= 2 for all neighbors
        return np.all(cover_count[neighbors_matrix[idx]] >= 2)

    def try_remove():
        removed = False
        # Try to remove any redundant word
        # To be fast, we can check all or random subset. 
        # Since N is small, checking all is ~6561 ops.
        # But we can just iterate code_list.
        for i in range(len(code_list) - 1, -1, -1):
            c = code_list[i]
            if is_redundant(c):
                # Remove c
                code_list.pop(i)
                in_code[c] = False
                cover_count[neighbors_matrix[c]] -= 1
                removed = True
                return True # Removed one, return to loop again to catch chains
        return removed

    # Iterative improvement loop
    # We alternate between removal and swaps.
    
    # Precompute adjacency for faster swap candidate generation?
    # For a given codeword c, the "critical" neighbors are those with cover_count == 1.
    # Let U_c = { u in neighbors(c) | cover_count[u] == 1 }.
    # If U_c is empty, c is redundant (handled above).
    # If we remove c, U_c becomes uncovered.
    # We need a new codeword d such that d covers U_c.
    # So d must be in intersection of neighbors(U_c).
    # i.e., d in Intersection_{u in U_c} neighbors(u).
    # Also d must not be in code (or if it is, it's a no-op or replacement which we already handle by redundancy).
    # Actually, if d is in code, swapping c with d just removes c (if d was already there, we just remove c, which is removal).
    # So we look for d not in code.
    
    # Optimization: 
    # Instead of iterating all code words, we can prioritize words with large U_c (more critical, harder to swap?) 
    # or small U_c (easier to find d)?
    # If U_c is small, intersection is likely to have candidates.
    # If U_c is large, intersection might be empty.
    
    # Let's use a simple random restart / hill climbing approach.
    
    # We will run multiple passes.
    passes = 0
    last_size = len(code_list)
    stall_count = 0
    
    while time.time() < deadline - 5:
        passes += 1
        improved = False
        
        # 1. Remove redundants
        while try_remove():
            improved = True
            
        if len(code_list) < last_size:
            last_size = len(code_list)
            stall_count = 0
        else:
            stall_count += 1
            
        # If stalled, try swaps
        if not code_list:
            break
            
        # Pick a random word to try swapping
        # Heuristic: Pick words that have "low" redundancy potential?
        # Or just random.
        c_idx = random.randint(0, len(code_list) - 1)
        c = code_list[c_idx]
        
        # Get critical neighbors
        nbrs_c = neighbors_matrix[c]
        # cover_count is current state WITH c included
        # Critical neighbors are those with cover_count == 1
        critical_mask = (cover_count[nbrs_c] == 1)
        
        if not np.any(critical_mask):
            # Should have been removed in try_remove, but maybe we missed it?
            # Just continue
            continue
            
        critical_nbrs = nbrs_c[critical_mask]
        
        # Find candidates d that cover all critical_nbrs
        # d must be in neighbors of each u in critical_nbrs
        # Initialize candidates with neighbors of first critical neighbor
        # Note: d can be any word, not just non-code, but we filter later.
        
        # Efficient intersection:
        # Start with set of neighbors of critical_nbrs[0]
        # Intersect with neighbors of critical_nbrs[1], etc.
        
        # To avoid Python set overhead for every swap attempt, 
        # we can use a boolean array or just numpy sets if small.
        # Size of critical_nbrs is usually small (1-5).
        
        if len(critical_nbrs) == 1:
            candidates = set(neighbors_matrix[critical_nbrs[0]].tolist())
        else:
            candidates = set(neighbors_matrix[critical_nbrs[0]].tolist())
            for u in critical_nbrs[1:]:
                candidates.intersection_update(neighbors_matrix[u].tolist())
                if not candidates:
                    break
                    
        if not candidates:
            continue
            
        # Filter out current code words (swapping with existing code is just removal, already handled)
        # Actually, if d is in code, and we replace c with d, c is removed. 
        # This is valid only if d covers critical_nbrs.
        # If d is in code, it already covers its own neighbors.
        # But if we "swap", we just remove c.
        # This is equivalent to checking if c is redundant given d is present?
        # No, if d is present, c might not be redundant because c covers other things.
        # But if we remove c, critical_nbrs are covered by d.
        # So if d is in code, we can just remove c.
        # This is a valid move.
        
        # So we consider all candidates.
        # However, if d is in code, we just remove c.
        # If d is NOT in code, we add d and remove c.
        
        # Score candidates:
        # We want to pick a d that leads to a "better" state.
        # Metric: Number of neighbors of d that are now covered by >= 2 codewords?
        # Or: Number of new redundancies created?
        
        # Let's evaluate a few random candidates from the set to save time.
        cand_list = list(candidates)
        random.shuffle(cand_list)
        
        # Limit evaluation to top K candidates
        K_eval = 5
        best_d = None
        best_score = -1
        
        for d in cand_list[:K_eval]:
            # Simulate move: remove c, add d (if d not in code)
            # Calculate "redundancy gain" or "potential for further reduction"
            
            # Current state: cover_count
            # New state cover_count':
            # cover_count' = cover_count - 1 (for neighbors of c) + 1 (for neighbors of d, if d new)
            
            # We want to maximize the number of words that BECOME redundant?
            # Or minimize "fragility"?
            
            # Let's count how many codewords become redundant in the new state.
            # This is expensive to check for all.
            
            # Proxy: 
            # Score = sum over neighbors of d of (new_cover_count >= 2)
            # This encourages adding a word that overlaps well with existing code, creating redundancies.
            
            # Calculate new cover counts locally for neighbors of c and d
            # Note: neighbors of c and d might overlap.
            
            # For simplicity, let's just use a heuristic score:
            # Score = number of neighbors of d that have cover_count >= 2 (in current state, before removal of c)
            # This is a proxy for "how well d fits into the existing cover".
            # If d fits well, it likely creates redundancies for other words.
            
            score = np.sum(cover_count[neighbors_matrix[d]] >= 2)
            
            # Bonus if d is not in code (new word) vs in code (removal)
            # If d is in code, we are just removing c. Score should reflect that c is "useless".
            # If d is in code, it means critical_nbrs are already covered by d.
            # So c is redundant! Why wasn't it removed?
            # Ah, is_redundant checks if ALL neighbors are covered by >= 2.
            # If critical_nbrs are covered by d (which is in code), then for those neighbors, cover_count >= 2.
            # But other neighbors of c might have cover_count == 1.
            # So c is NOT redundant.
            # But if we remove c, only critical_nbrs are at risk. If d covers them, we are safe.
            # So replacing c with d (which is already there) just removes c.
            
            if score > best_score:
                best_score = score
                best_d = d
                
        if best_d is None:
            continue
            
        # Perform the move
        # Remove c
        cover_count[neighbors_matrix[c]] -= 1
        in_code[c] = False
        code_list.pop(c_idx)
        
        # Add best_d if not already in code
        if not in_code[best_d]:
            in_code[best_d] = True
            cover_count[neighbors_matrix[best_d]] += 1
            code_list.append(best_d)
            
        # After move, try to remove any new redundants
        while try_remove():
            improved = True
            
        if len(code_list) < last_size:
            last_size = len(code_list)
            stall_count = 0
        else:
            stall_count += 1

        # If we are stuck for too long, we can try to restart local search or just break.
        # Given the small size, we can just continue until time is up.

    return [idx_to_word(c) for c in code_list]
