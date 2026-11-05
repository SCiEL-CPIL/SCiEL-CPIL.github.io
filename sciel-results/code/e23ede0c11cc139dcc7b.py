# EVOLVE-BLOCK-START
import time
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    if n <= 6:
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
    
    if n == 7:
        # Build S_6 standard construction
        word = [1]
        for m in range(2, 7):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        
        # Extract all 720 permutations of 1..6 in order of appearance
        perms, seen = [], set()
        for i in range(len(word) - 5):
            window = tuple(word[i:i + 6])
            if len(set(window)) == 6 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        N = len(perms)
        
        # Precompute overlap matrix
        ov = [[0]*N for _ in range(N)]
        for i in range(N):
            p = perms[i]
            for j in range(N):
                if i == j:
                    ov[i][j] = 6
                    continue
                q = perms[j]
                best = 0
                for k in range(5, 0, -1):
                    if p[-k:] == q[:k]:
                        best = k
                        break
                ov[i][j] = best
        
        def calc_total_overlap(path):
            total = 0
            for i in range(len(path) - 1):
                total += ov[path[i]][path[i+1]]
            return total
        
        def build_word_from_path(path):
            new_word = []
            for idx in path:
                p = perms[idx]
                block = list(p) + [7] + list(p)
                overlap = 0
                min_len = min(len(new_word), len(block))
                for o in range(min_len, -1, -1):
                    if new_word[len(new_word)-o:] == block[:o]:
                        overlap = o
                        break
                new_word += block[overlap:]
            return new_word
        
        # Initialize with standard order
        best_path = list(range(N))
        best_overlap = calc_total_overlap(best_path)
        
        # Local search: Or-Opt (move 1, 2, or 3 elements)
        # For each segment length L, try moving a segment from position i to position k
        # Calculate the delta in total overlap efficiently
        
        def or_opt_delta(path, i, k, L, ov):
            """
            Calculate delta in total overlap when moving segment path[i:i+L] to position k.
            Returns the change in total overlap (negative is improvement).
            """
            N = len(path)
            if k == i:
                return 0
            
            # Original edges that will be removed:
            # 1. Edge before segment: (i-1, i) if i > 0
            # 2. Edge after segment: (i+L-1, i+L) if i+L < N
            # 3. Internal edges of segment: (i+j, i+j+1) for j in 0..L-2 (these stay internal)
            
            # New edges that will be created:
            # After removing segment, we connect (i-1, i+L) if both exist
            # After inserting at k, we break edge (k-1, k) if both exist
            # and create (k-1, block_start) and (block_end, k)
            
            # Let's compute step by step
            
            # Cost of original configuration in the affected region
            orig_cost = 0
            # Edge before
            if i > 0:
                orig_cost += ov[path[i-1]][path[i]]
            # Edge after
            if i + L < N:
                orig_cost += ov[path[i+L-1]][path[i+L]]
            # Internal edges (these don't change, so we can ignore them for delta)
            
            # Cost of new configuration
            new_cost = 0
            
            if k < i:
                # Inserting before the removal point
                # Remove segment from i: connect (i-1, i+L)
                if i > 0 and i + L < N:
                    new_cost += ov[path[i-1]][path[i+L]]
                elif i > 0:
                    new_cost += 0  # no right edge
                elif i + L < N:
                    new_cost += 0  # no left edge
                else:
                    new_cost += 0  # segment was at both ends
                
                # Insert at k: break (k-1, k)
                if k > 0:
                    new_cost += ov[path[k-1]][path[i]]  # k-1 to block start
                if k + L < N:
                    new_cost += ov[path[i+L-1]][path[k]]  # block end to k
                
            elif k > i:
                # Inserting after the removal point
                # Need to be careful with indices
                # After removing L elements from i, elements at position >= i+L shift left by L
                # So if we want to insert at position k in the NEW array, 
                # the corresponding position in OLD array is k+L if k >= i+L, or k if k < i+L
                # But since k > i, and we're inserting after removal...
                
                # Let's think differently:
                # Original: ... path[i-1] | path[i]...path[i+L-1] | path[i+L] ... path[k-1] | path[k] ...
                # After removing segment: ... path[i-1] | path[i+L] ... path[k-1] | path[k] ...
                #   (if k >= i+L, then path[k] in new array is path[k+L] in old)
                #   (if k < i+L, this case doesn't happen since k > i and we removed from i)
                #   Actually if k > i and k < i+L, that's inside the removed segment, invalid
                #   So we require k >= i+L or k < i. For k > i, we need k >= i+L.
                
                if k < i + L:
                    return 0  # invalid move, skip
                
                # Remove segment from i: connect (i-1, i+L)
                if i > 0 and i + L < N:
                    new_cost += ov[path[i-1]][path[i+L]]
                
                # Insert at position k in new array
                # In old array, position k in new corresponds to:
                # If k < i: not this case
                # If k >= i+L: old index is k (since we're inserting before old[k] which is now at new[k-L])
                # Wait, let me reconsider.
                
                # New array after removal: indices 0..i-1, then i..N-L-1 correspond to old i+L..N-1
                # So new index j corresponds to old index j if j < i, or j+L if j >= i
                # We want to insert at new index k (where k > i, so k >= i+L for valid)
                # Actually k can be i, i+1, ..., N-L
                # If k == i: inserting right after removal point, which is same as original
                # If k > i: 
                #   The element that was at old index k+L is now at new index k
                #   So we're inserting before new[k] which is old[k+L]
                #   And after new[k-1] which is old[k-1+L] if k-1 >= i, or old[k-1] if k-1 < i
                
                if k == i:
                    # Same position, no change
                    return 0
                
                # k > i
                # Left neighbor in new array at position k-1:
                if k - 1 < i:
                    left_old = path[k-1]
                else:
                    left_old = path[k-1+L]
                
                # Right neighbor in new array at position k:
                if k < N - L:
                    right_old = path[k+L]
                else:
                    right_old = None
                
                # Insert block between left_old and right_old
                if k > 0:
                    # There is a left neighbor
                    if k - 1 < i:
                        new_cost += ov[path[k-1]][path[i]]
                    else:
                        new_cost += ov[path[k-1+L]][path[i]]
                
                if k < N - L:
                    # There is a right neighbor
                    new_cost += ov[path[i+L-1]][path[k+L]]
            
            return new_cost - orig_cost
        
        improved = True
        pass_count = 0
        while improved and time.time() < deadline - 10:
            improved = False
            pass_count += 1
            current_path = best_path[:]
            
            for length in [1, 2, 3]:
                if time.time() > deadline - 10:
                    break
                for i in range(N - length + 1):
                    if time.time() > deadline - 10:
                        break
                    block_start = current_path[i]
                    block_end = current_path[i + length - 1]
                    
                    # Try all insertion positions
                    for k in range(N - length + 1):
                        if k == i:
                            continue
                        
                        delta = or_opt_delta(current_path, i, k, length, ov)
                        if delta < 0:
                            # Perform the move
                            block = current_path[i:i+length]
                            del current_path[i:i+length]
                            # Adjust k if k > i
                            if k > i:
                                k -= length
                            current_path[k:k] = block
                            best_overlap += delta
                            improved = True
                            break  # Restart scanning from beginning for this length
            
            if improved:
                best_path = current_path
        
        # Build final word
        final_word = build_word_from_path(best_path)
        return "".join(map(str, final_word))
    
    # Fallback for n > 7
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
