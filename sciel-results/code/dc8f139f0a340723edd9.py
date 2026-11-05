import time
import random
import sys

def construct(n):
    start_time = time.time()
    deadline = 110.0  # 10s margin
    
    if n == 1:
        return "1"
    
    word = [1]
    
    for k in range(2, n + 1):
        # Check time
        if time.time() > deadline:
            # Fallback: just extend with standard construction
            # This might be invalid if we break mid-way, but we are at the start of the loop.
            # We need to complete the construction for n.
            # If we can't finish the optimization, we use the standard order for this k and subsequent.
            pass

        # Extract unique (k-1)-permutations from current word
        perms = []
        seen = set()
        for i in range(len(word) - k + 2):
            window = tuple(word[i:i + k - 1])
            if len(set(window)) == k - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        num_perms = len(perms)
        
        if num_perms == 0:
            return "".join(map(str, word))

        # Determine strategy based on time and size
        time_remaining = deadline - time.time()
        
        # If small, we can do better. If large, be careful.
        # For n=7, num_perms=5040.
        
        best_order = list(range(num_perms))
        best_len = float('inf')
        
        # If we have enough time and num_perms is manageable, try optimization
        if time_remaining > 10.0 and num_perms > 10:
            # Precompute overlaps
            # overlap[i][j] = length of longest suffix of perms[i] that is prefix of perms[j]
            # Max overlap is k-2
            # We store the "cost" = (k-1) - overlap? 
            # Transition cost from i to j in the block construction:
            # Block is p, k, p.
            # Overlap between block i and block j?
            # The seed calculates overlap between perms to determine order.
            # Actually, the cost is determined by how well the end of block i matches start of block j.
            # Block i ends with perms[i]. Block j starts with perms[j].
            # So we need overlap between perms[i] and perms[j].
            
            # Precompute overlap matrix
            # To save memory and time, we can compute on the fly if N is small,
            # but for N=5040, precomputation is better.
            
            # However, 5040x5040 matrix of ints is ~20MB.
            # Let's try to compute it.
            
            overlap_matrix = [[0] * num_perms for _ in range(num_perms)]
            
            # Optimization: Only compute if we have time
            if time.time() - start_time < 5.0: # Quick check
                 for i in range(num_perms):
                     pi = perms[i]
                     for j in range(num_perms):
                         pj = perms[j]
                         # Max overlap is k-2
                         max_ov = k - 2
                         ov = 0
                         # Check from max_ov down to 0
                         # Since k is small, this is fast
                         for o in range(max_ov, -1, -1):
                             if o == 0:
                                 break
                             if pi[-o:] == pj[:o]:
                                 ov = o
                                 break
                         overlap_matrix[i][j] = ov
            
            # Now run Greedy TSP with multiple restarts
            # Time budget for TSP: 60s?
            tsp_deadline = start_time + 100.0
            
            # Number of restarts depends on time
            # Estimate time per restart: O(N^2) ~ 25M ops. ~1-2s in Python.
            # So we can do ~30-50 restarts.
            
            max_restarts = 50
            if num_perms > 2000:
                max_restarts = 20
            elif num_perms > 500:
                max_restarts = 50
            else:
                max_restarts = 100

            for r in range(max_restarts):
                if time.time() > tsp_deadline:
                    break
                
                # Random start
                current = random.randint(0, num_perms - 1)
                visited = [False] * num_perms
                visited[current] = True
                order = [current]
                
                # Greedy step
                for _ in range(num_perms - 1):
                    best_next = -1
                    best_ov = -1
                    
                    # Find next with max overlap
                    # We can use a list of unvisited nodes
                    # For speed, we iterate through all and check visited
                    # This is O(N) per step, O(N^2) total.
                    
                    for next_idx in range(num_perms):
                        if visited[next_idx]:
                            continue
                        ov = overlap_matrix[current][next_idx]
                        if ov > best_ov:
                            best_ov = ov
                            best_next = next_idx
                            
                    if best_next == -1:
                        break
                    visited[best_next] = True
                    order.append(best_next)
                    current = best_next
                    
                if len(order) < num_perms:
                    continue
                
                # Calculate total length
                # Base length of first block: 2*(k-1) + 1
                # Each subsequent block adds: (2*(k-1) + 1) - overlap
                # Wait, the overlap between blocks is determined by the perms.
                # Block i: perms[i] + [k] + perms[i]
                # Block j: perms[j] + [k] + perms[j]
                # End of Block i is perms[i]. Start of Block j is perms[j].
                # Overlap is overlap_matrix[perms[i]_idx][perms[j]_idx]
                
                total_len = 2 * (k - 1) + 1
                for i in range(1, len(order)):
                    ov = overlap_matrix[order[i-1]][order[i]]
                    total_len += (2 * (k - 1) + 1) - ov
                    
                if total_len < best_len:
                    best_len = total_len
                    best_order = order
        
        # Construct new word using best_order
        new_word = []
        for idx, p_idx in enumerate(best_order):
            p = perms[p_idx]
            block = list(p) + [k] + list(p)
            
            if idx == 0:
                new_word = block
            else:
                # We know the overlap from the matrix if we used it, 
                # but best_order might be from fallback (identity).
                # Let's just recalculate overlap for safety, it's fast for the final construction.
                # Or we can store the overlaps in the order loop above.
                # For simplicity and safety, recalculate.
                max_overlap = min(len(new_word), len(block))
                overlap = 0
                for o in range(min(k, max_overlap), -1, -1):
                    if o == 0:
                        break
                    if new_word[-o:] == block[:o]:
                        overlap = o
                        break
                new_word += block[overlap:]
                
        word = new_word

    return "".join(map(str, word))
