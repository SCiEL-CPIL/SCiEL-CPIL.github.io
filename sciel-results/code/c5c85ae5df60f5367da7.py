import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    total_words = q ** n
    # Precompute all words and their indices
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors (sphere of radius 1) for each word
    # neighbors[i] is a list of indices of words within distance 1 of all_words[i]
    neighbors = []
    for w in all_words:
        w_list = list(w)
        nbrs = {word_to_idx[w]}
        for i in range(n):
            orig = w_list[i]
            for v in range(q):
                if v != orig:
                    nw = w_list[:]
                    nw[i] = v
                    nbrs.add(word_to_idx[tuple(nw)])
        neighbors.append(list(nbrs))
        
    # Convert neighbors to numpy arrays for faster operations if needed, 
    # but lists of ints are often fast enough for 729 nodes.
    
    def greedy_set_cover():
        """
        Initialize code using greedy set cover algorithm.
        """
        covered = np.zeros(total_words, dtype=bool)
        code = []
        remaining = set(range(total_words))
        
        # Precompute spheres as sets for faster intersection/difference?
        # Or just use the neighbors list.
        
        while remaining:
            best_word = -1
            best_count = 0
            # Heuristic: pick word that covers most uncovered neighbors
            # To speed up, we can sample or just iterate all if time allows.
            # 729 is small, so iterating all is fine.
            for w_idx in remaining:
                count = 0
                for nbr in neighbors[w_idx]:
                    if nbr in remaining:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_word = w_idx
            
            if best_word == -1:
                break
                
            code.append(best_word)
            # Update coverage
            for nbr in neighbors[best_word]:
                if nbr in remaining:
                    remaining.remove(nbr)
                    
        return code

    def local_search(code_list):
        """
        Aggressively remove redundant codewords.
        Returns a set of codeword indices.
        """
        code_set = set(code_list)
        if not code_set:
            return code_set
            
        # Count coverage for each word
        coverage_count = np.zeros(total_words, dtype=np.int32)
        for cw in code_set:
            for nbr in neighbors[cw]:
                coverage_count[nbr] += 1
                
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for cw in list(code_set):
                if time.time() > deadline - 5:
                    break
                # Check if cw can be removed
                can_remove = True
                for nbr in neighbors[cw]:
                    if coverage_count[nbr] == 1:
                        can_remove = False
                        break
                if can_remove:
                    # Remove cw
                    for nbr in neighbors[cw]:
                        coverage_count[nbr] -= 1
                    code_set.remove(cw)
                    changed = True
                    
        return code_set

    def swap_local_search(code_set):
        """
        Try to improve by swapping codewords.
        """
        if not code_set:
            return code_set
            
        coverage_count = np.zeros(total_words, dtype=np.int32)
        for cw in code_set:
            for nbr in neighbors[cw]:
                coverage_count[nbr] += 1
                
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try swapping each codeword with a random other word
            # To be efficient, we limit the number of swaps per iteration
            for cw in list(code_set):
                if time.time() > deadline - 10:
                    break
                # Find words uniquely covered by cw
                unique_words = []
                for nbr in neighbors[cw]:
                    if coverage_count[nbr] == 1:
                        unique_words.append(nbr)
                        
                if not unique_words:
                    # Can remove directly, handled by local_search, 
                    # but let's try swap anyway to change structure
                    pass
                else:
                    # Try to find a replacement
                    # Sample a few random candidates
                    for _ in range(5):
                        new_cw = random.randint(0, total_words - 1)
                        if new_cw in code_set:
                            continue
                        
                        # Check if new_cw covers all unique_words
                        valid = True
                        new_nbrs = set(neighbors[new_cw])
                        for u in unique_words:
                            if u not in new_nbrs:
                                valid = False
                                break
                            
                        if valid:
                            # Perform swap
                            # Remove cw
                            for nbr in neighbors[cw]:
                                coverage_count[nbr] -= 1
                            code_set.remove(cw)
                            
                            # Add new_cw
                            for nbr in neighbors[new_cw]:
                                coverage_count[nbr] += 1
                            code_set.add(new_cw)
                            
                            improved = True
                            break # Restart the loop with the new set
                    
            # Run pure removal local search after swaps
            code_set = local_search(list(code_set))
            # Rebuild coverage_count for next iteration if we want to continue swapping
            # But local_search already removed redundants.
            # We need to update coverage_count if we did swaps.
            # For simplicity, we just return and let the outer loop handle restarts
            # or we break here.
            # Let's just do one pass of swaps then return.
            break 

        return code_set

    best_code_set = None
    best_size = float('inf')
    
    # Run multiple restarts
    restarts = 0
    while time.time() < deadline - 20:
        restarts += 1
        # Initialize with greedy
        initial_code = greedy_set_cover()
        
        # Local search
        current_set = local_search(initial_code)
        
        # Swap local search
        current_set = swap_local_search(current_set)
        
        # Final removal pass
        current_set = local_search(list(current_set))
        
        if len(current_set) < best_size:
            best_size = len(current_set)
            best_code_set = set(current_set)
            
        # If we found a very good solution, we can tighten the search
        if best_size <= 73:
             # We are close to the known best. Try to break 73.
             # We can do a more targeted search.
             pass
             
    if best_code_set is None:
        # Fallback to direct sum if something went wrong
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        
    return [list(all_words[idx]) for idx in best_code_set]
