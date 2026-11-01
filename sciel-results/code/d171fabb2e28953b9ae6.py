import time
import random
import sys

# EVOLVE-BLOCK-START

def construct(n):
    """
    Constructs a superpermutation for symbols 1..n.
    Uses the standard recursive construction (P + n + P blocks) but optimizes the
    order of blocks for m >= 5 to maximize overlaps, thereby reducing total length.
    """
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit

    # Base cases (hardcoded for speed and correctness, though the loop handles them)
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "123121321412314123214121431214231214321"

    word = [1]
    
    for m in range(2, n + 1):
        if time.time() > deadline - 5:
            break

        # 1. Extract unique (m-1)-permutations in order of appearance
        perms = []
        seen = set()
        # Optimization: Use a set for O(1) lookup. 
        # For m=7, len(word) is ~5919, so ~5900 windows to check.
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        num_perms = len(perms)
        
        # 2. Create blocks
        # Block structure: P + [m] + P
        # Length of block = 2*(m-1) + 1 = 2m - 1
        blocks = []
        for p in perms:
            block = list(p) + [m] + list(p)
            blocks.append(block)
        
        # 3. Determine order of blocks
        if m <= 4:
            # Standard order is optimal or near-optimal and fast
            order = list(range(num_perms))
        else:
            # For m >= 5, we have significant freedom in ordering to maximize overlap.
            # We model this as a Max TSP path problem on the blocks.
            # Cost matrix C[i][j] = overlap(blocks[i], blocks[j])
            # We want to maximize sum(C[order[k]][order[k+1]])
            
            # Precompute overlap matrix
            # Note: Overlap is directed. suffix of A matches prefix of B.
            # Since m is small (<=7), block length is small (<=13).
            # num_perms for m=5 is 24, m=6 is 120, m=7 is 720.
            
            if num_perms <= 2:
                order = list(range(num_perms))
            else:
                # Build overlap matrix
                # Using a list of lists for speed.
                # ov[i][j] is the max overlap of suffix of blocks[i] and prefix of blocks[j]
                ov_matrix = [[0] * num_perms for _ in range(num_perms)]
                
                # Optimization: Only compute for i != j? No, self-overlap might be relevant if we allow repeats, 
                # but we must visit each block exactly once. So i!=j.
                
                # Vectorized-ish overlap calculation? 
                # For small strings, simple loop is often faster in Python due to overhead of numpy.
                # However, 720x720 = 518k comparisons. 
                # Each comparison is up to 13 chars. 
                # 518k * 13 ops ~ 6.7M ops. This is fast (<1s).
                
                for i in range(num_perms):
                    bi = blocks[i]
                    for j in range(num_perms):
                        if i == j:
                            continue
                        bj = blocks[j]
                        # Max possible overlap is len(bi)-1? 
                        # Actually, if bi == bj, overlap could be len(bi). 
                        # But blocks are distinct permutations P, so bi != bj.
                        # Max overlap is len(bi) - 1 (since first char of bj is P[0], last of bi is P[-1]).
                        # Wait, if P[0] == P[-1], then overlap can be larger?
                        # Example: P=12, bi=12312. P=21, bj=21321.
                        # Suffix of bi: 2, 12, 312, 2312, 12312
                        # Prefix of bj: 2, 21, 213, 2132, 21321
                        # Match at '2'. Overlap 1.
                        
                        # Let's just compute max overlap.
                        max_ov = 0
                        # We can limit search to len(bi) - 1 because if they match fully, they are the same block (impossible for distinct perms)
                        # Actually, if bi ends with X and bj starts with X, and X is the whole block, then bi=bj.
                        # So max_ov < len(bi).
                        limit = min(len(bi), len(bj)) - 1 
                        if limit < 0: limit = 0
                        
                        # Check from largest to smallest
                        for k in range(limit, 0, -1):
                            if bi[-k:] == bj[:k]:
                                max_ov = k
                                break
                        ov_matrix[i][j] = max_ov

                # Solve Max TSP Path
                # Heuristic: Nearest Neighbor with Random Restarts + 2-Opt/3-Opt local search
                # Given time constraints, we'll do NN + limited local search.
                
                best_order = list(range(num_perms))
                best_score = -1
                
                # Calculate initial score for default order (just in case)
                curr_score = sum(ov_matrix[best_order[k]][best_order[k+1]] for k in range(num_perms-1))
                best_score = curr_score
                
                num_restarts = 10
                if num_perms > 100:
                    num_restarts = 5 # Fewer restarts for larger problems to fit time
                
                time_limit_for_search = 60 # seconds
                search_start = time.time()
                
                for r in range(num_restarts):
                    if time.time() - search_start > time_limit_for_search:
                        break
                        
                    # Nearest Neighbor
                    visited = [False] * num_perms
                    current = random.randint(0, num_perms - 1)
                    order = [current]
                    visited[current] = True
                    score = 0
                    
                    for _ in range(num_perms - 1):
                        next_node = -1
                        max_ov = -1
                        # Find unvisited neighbor with max overlap
                        # Optimization: We can pre-sort neighbors? No, dynamic.
                        for j in range(num_perms):
                            if not visited[j]:
                                ov = ov_matrix[current][j]
                                if ov > max_ov:
                                    max_ov = ov
                                    next_node = j
                        order.append(next_node)
                        visited[next_node] = True
                        score += max_ov
                        current = next_node
                    
                    if score > best_score:
                        best_score = score
                        best_order = order[:]
                        
                    # Local Search: 2-Opt for Max TSP Path
                    # Reversing a segment [i..j] in a path.
                    # For directed graphs (overlap is directed), 2-opt is tricky.
                    # Standard 2-opt reverses the path segment.
                    # Old edges: (u, v) where u=order[i-1], v=order[i] ... (x, y) where x=order[j], y=order[j+1]
                    # New edges: (u, y) and (x, v)? No.
                    # Reversing order[i..j] changes the internal edges to their reverses.
                    # Since overlap(A,B) != overlap(B,A), we must recompute the score of the reversed segment.
                    
                    # Simpler Local Search: Swap two non-adjacent nodes?
                    # Or "Or-opt": Move a small segment (length 1 or 2) to a different position.
                    
                    # Let's try Or-opt (move 1 node)
                    improved = True
                    max_iterations = 50
                    iter_count = 0
                    while improved and iter_count < max_iterations:
                        improved = False
                        iter_count += 1
                        if time.time() - search_start > time_limit_for_search:
                            break
                            
                        for i in range(num_perms):
                            if time.time() - search_start > time_limit_for_search:
                                break
                            # Move node order[i] to position j
                            # Calculate delta score
                            # This is O(N) to find best j for a fixed i.
                            # Total O(N^2) per pass. 720^2 = 500k. 50 passes = 25M. Might be slow in Python.
                            # We'll limit this to smaller N or fewer iterations.
                            if num_perms > 100 and r > 0:
                                continue # Skip heavy local search for large N if not first restart? Or just skip entirely.
                                
                            node_to_move = order[i]
                            # Remove node
                            # Edges affected:
                            # Left neighbor L = order[i-1] if i>0 else None
                            # Right neighbor R = order[i+1] if i<num_perms-1 else None
                            
                            # We need to calculate the gain of removing it and inserting it at j.
                            # This is complex. Given the time budget, NN is the primary driver.
                            # For n=7, the improvement from NN vs random is significant.
                            # We'll stick to NN + a very light 2-swap if time permits, but NN is likely sufficient 
                            # to beat the baseline (which is essentially a specific fixed order).
                            
                            # Actually, the baseline "order of appearance" is not random.
                            # NN should find a much better order.
                            
                            pass # Skipping complex local search to ensure robustness and speed.
                            
                order = best_order

        # 4. Construct new word
        new_word = []
        if not order:
            word = new_word
            continue
            
        # Add first block
        new_word.extend(blocks[order[0]])
        
        for idx in range(1, len(order)):
            b = blocks[order[idx]]
            # Find max overlap with current new_word suffix
            # We can limit search to len(b) - 1
            max_ov = 0
            limit = min(len(new_word), len(b) - 1)
            for k in range(limit, 0, -1):
                if new_word[-k:] == b[:k]:
                    max_ov = k
                    break
            new_word.extend(b[max_ov:])
        
        word = new_word

    return "".join(map(str, word))

# EVOLVE-BLOCK-END
