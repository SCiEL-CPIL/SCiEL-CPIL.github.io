import time
import random
import itertools
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    if n <= 1:
        return "".join(str(i) for i in range(1, n + 1))
    
    # For n=7, we use a specialized approach based on the known structure
    # The shortest known superpermutation for n=7 has length 5905.
    # The standard recursive construction gives:
    # S(n) = S(n-1) with 'n' inserted, length = n! + (n-1)! + ... + 1
    # For n=7: 5040 + 720 + 120 + 24 + 6 + 2 + 1 = 5913
    # We can do better with optimization.
    
    # Strategy: Use the recursive construction but optimize the ordering
    # at each step to maximize overlap.
    
    word = [1]
    
    for m in range(2, n + 1):
        # Extract distinct (m-1) permutations from word
        perms, seen = [], set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # For m >= 4, use optimized greedy with multiple restarts
        if len(perms) > 5:
            # Check time
            if time.time() > deadline - 30:
                # Not enough time for optimization, use simple order
                ordered_perms = perms
            else:
                # Compute overlap matrix efficiently
                # overlap[i][j] = max k such that suffix of perms[i] of length k == prefix of perms[j] of length k
                # For permutations of length m-1, max overlap is m-1
                # But typically overlap is small (1 or 2)
                
                # For large m, use a faster approach
                # Build a dictionary for quick lookup
                # For each permutation, store its suffixes and prefixes
                
                # Actually, for m=7, we have 720 permutations.
                # Computing full overlap matrix is 720*720 = 518400 pairs.
                # For each pair, we check up to 6 overlaps. That's ~3M operations.
                # This should be fast enough.
                
                if time.time() > deadline - 25:
                    ordered_perms = perms
                else:
                    # Compute overlap matrix
                    N = len(perms)
                    overlap_matrix = [[0] * N for _ in range(N)]
                    
                    # Precompute suffixes and prefixes for efficiency
                    # For each permutation, store all suffixes as tuples
                    # and all prefixes as tuples, with a dictionary mapping to indices
                    
                    # Actually, simpler: for each pair (i,j), check overlaps from max to 1
                    # Max overlap is m-1
                    max_ov = m - 1
                    
                    for i in range(N):
                        pi = perms[i]
                        for j in range(N):
                            pj = perms[j]
                            best = 0
                            # Check from max_ov down to 1
                            for k in range(max_ov, 0, -1):
                                if pi[-k:] == pj[:k]:
                                    best = k
                                    break
                            overlap_matrix[i][j] = best
                    
                    # Greedy path with multiple restarts
                    # Determine number of restarts based on time remaining
                    time_remaining = deadline - time.time() - 10  # Leave 10s for construction
                    if time_remaining < 5:
                        n_restarts = 10
                    elif time_remaining < 15:
                        n_restarts = 50
                    elif time_remaining < 30:
                        n_restarts = 100
                    else:
                        n_restarts = 200
                    
                    best_path = None
                    best_score = -1
                    
                    for r in range(n_restarts):
                        # Random start
                        start = random.randint(0, N - 1)
                        path = [start]
                        visited = [False] * N
                        visited[start] = True
                        current = start
                        current_score = 0
                        
                        for _ in range(N - 1):
                            # Find next node with max overlap
                            best_next = -1
                            best_ov = -1
                            for j in range(N):
                                if not visited[j]:
                                    ov = overlap_matrix[current][j]
                                    if ov > best_ov:
                                        best_ov = ov
                                        best_next = j
                            
                            if best_next == -1:
                                for j in range(N):
                                    if not visited[j]:
                                        best_next = j
                                        break
                            
                            visited[best_next] = True
                            path.append(best_next)
                            current_score += overlap_matrix[current][best_next]
                            current = best_next
                        
                        if current_score > best_score:
                            best_score = current_score
                            best_path = path
                        
                        # Check time
                        if time.time() > deadline - 15:
                            break
                    
                    if best_path is not None:
                        ordered_perms = [perms[i] for i in best_path]
                    else:
                        ordered_perms = perms
        
        else:
            ordered_perms = perms
        
        # Build the new word
        # For each permutation p in ordered_perms, we add: p + [m] + p
        # But we can overlap consecutive blocks
        new = []
        for idx in range(len(ordered_perms)):
            p = ordered_perms[idx]
            block = list(p) + [m] + list(p)
            
            if idx == 0:
                new += block
            else:
                # Find maximum overlap between end of new and start of block
                # new ends with previous p
                # block starts with current p
                # Max possible overlap is len(p) = m-1
                max_ov = min(len(new), len(block))
                ov = 0
                for k in range(max_ov, 0, -1):
                    if new[-k:] == block[:k]:
                        ov = k
                        break
                new += block[ov:]
        
        word = new
        
        if time.time() > deadline:
            break
    
    return "".join(map(str, word))
