import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    # Problem specific parameters
    # q=3, n=6, radius=1
    # Space size
    N = q ** n
    
    # Precompute neighbors
    # idx_to_word and word_to_idx
    def word_to_idx(w):
        idx = 0
        for val in w:
            idx = idx * q + val
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Generate all words
    all_words = [idx_to_word(i) for i in range(N)]
    
    # Precompute neighbor lists and masks
    neighbors = [[] for _ in range(N)]
    neighbor_masks = [0] * N
    
    for i in range(N):
        w = all_words[i]
        neighbors[i].append(i)
        neighbor_masks[i] |= (1 << i)
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(q):
                if new_val != original_val:
                    w[pos] = new_val
                    ni = word_to_idx(w)
                    neighbors[i].append(ni)
                    neighbor_masks[i] |= (1 << ni)
            w[pos] = original_val
            
    # Initial Code: Direct Sum of Tetracode and Full Space of length n-4
    # Tetracode for q=3
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    tails = list(itertools.product(range(q), repeat=n - 4))
    
    initial_codewords = []
    for h in tetracode:
        for tail in tails:
            initial_codewords.append(h + list(tail))
            
    initial_indices = [word_to_idx(c) for c in initial_codewords]

    def optimize(initial_indices, deadline):
        # State
        selected_mask = 0
        for idx in initial_indices:
            selected_mask |= (1 << idx)
            
        counts = np.zeros(N, dtype=np.int32)
        for idx in initial_indices:
            for nb in neighbors[idx]:
                counts[nb] += 1
                
        # Helper: Update counts and masks
        def add_vertex(idx):
            nonlocal selected_mask
            selected_mask |= (1 << idx)
            for nb in neighbors[idx]:
                counts[nb] += 1
                
        def remove_vertex(idx):
            nonlocal selected_mask
            selected_mask &= ~(1 << idx)
            for nb in neighbors[idx]:
                counts[nb] -= 1

        # Local Search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Phase 1: Remove Redundant
            # Get list of selected vertices
            selected_list = [i for i in range(N) if (selected_mask >> i) & 1]
            
            for u in selected_list:
                if time.time() > deadline:
                    break
                # Check if any neighbor has count 1
                can_remove = True
                for nb in neighbors[u]:
                    if counts[nb] == 1:
                        can_remove = False
                        break
                if can_remove:
                    remove_vertex(u)
                    improved = True
                    # Restart loop to pick up new redundancies
                    break
            
            if improved:
                continue
                
            # Phase 2: Swap
            selected_list = [i for i in range(N) if (selected_mask >> i) & 1]
            
            for u in selected_list:
                if time.time() > deadline:
                    break
                    
                # Find essential neighbors (count 1)
                essential = [nb for nb in neighbors[u] if counts[nb] == 1]
                
                if not essential:
                    # Should have been removed in Phase 1
                    continue
                
                # Find candidates v such that N[v] covers all essential
                # v must be in intersection of N[essential]
                cand_mask = (1 << N) - 1
                for x in essential:
                    cand_mask &= neighbor_masks[x]
                
                # v must not be in selected_mask
                cand_mask &= ~selected_mask
                
                if cand_mask == 0:
                    continue
                    
                # Select best candidate
                # Heuristic: Maximize coverage of single-count vertices
                # single_mask: bitmask of vertices with count 1
                single_mask = 0
                for i in range(N):
                    if counts[i] == 1:
                        single_mask |= (1 << i)
                        
                candidates = []
                temp_mask = cand_mask
                while temp_mask:
                    # Extract lowest set bit
                    lsb = temp_mask & -temp_mask
                    v = lsb.bit_length() - 1
                    candidates.append(v)
                    temp_mask &= temp_mask - 1
                    
                if not candidates:
                    continue
                    
                # Score candidates
                best_v = None
                best_score = -1
                for v in candidates:
                    # Score: number of single-count vertices in N[v]
                    # Note: Essential vertices are in N[v] and will become count 2.
                    # Other single vertices in N[v] will also become count 2.
                    # This is good.
                    score = bin(neighbor_masks[v] & single_mask).count('1')
                    if score > best_score:
                        best_score = score
                        best_v = v
                        
                if best_v is not None:
                    # Perform swap
                    remove_vertex(u)
                    add_vertex(best_v)
                    improved = True
                    break
                    
        # Extract final codewords
        final_indices = [i for i in range(N) if (selected_mask >> i) & 1]
        return [idx_to_word(i) for i in final_indices]

    best_codewords = None
    best_size = float('inf')
    
    # Run optimization multiple times with slight variations if possible
    # Since initial is fixed, we can try randomizing the order of processing in local search
    # or just run once as it's deterministic.
    # To add diversity, we can try a few random restarts of the "Swap" phase logic?
    # Or just rely on the strength of the bitmask local search.
    
    # Let's try a few random permutations of the initial set to break symmetry in the search order
    # Although the set is the same, the order of checking might matter if we break on first improvement.
    
    for restart in range(5):
        if time.time() > deadline:
            break
            
        # Shuffle initial indices to change processing order
        shuffled_indices = list(initial_indices)
        random.shuffle(shuffled_indices)
        
        codewords = optimize(shuffled_indices, deadline)
        
        if len(codewords) < best_size:
            best_size = len(codewords)
            best_codewords = codewords
            
        if best_size <= 71:
            break
            
    if best_codewords is None:
        best_codewords = initial_codewords
        
    return best_codewords
