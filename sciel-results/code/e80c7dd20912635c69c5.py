import itertools
import time
import sys

def construct(n):
    # Deadline guard
    start_time = time.time()
    timeout = 115.0  # 5 sec margin
    
    # Classic construction
    def classic_superpermutation(n):
        if n == 1:
            return [1]
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
                # Find max overlap
                max_ov = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        max_ov = o
                        break
                new += block[max_ov:]
            word = new
        return word

    # Get classic result
    classic_word = classic_superpermutation(n)
    classic_len = len(classic_word)
    
    # For small n, classic is optimal
    if n <= 5:
        return "".join(map(str, classic_word))
        
    # Try Greedy TSP for n=6, 7
    if n in [6, 7]:
        try:
            # Generate all permutations
            perms = list(itertools.permutations(range(1, n + 1)))
            N = len(perms)
            perm_to_idx = {p: i for i, p in enumerate(perms)}
            
            # Precompute overlaps?
            # Overlap(i, j) is max k such that perms[i][n-k:] == perms[j][:k]
            # We want to maximize overlap.
            
            # Optimization: 
            # For each perm i, the best next perm j is one that starts with the longest suffix of i.
            # We can build a map: suffix -> list of perms starting with that suffix.
            # But we need the max overlap.
            
            # Let's just do O(N^2) check but optimized.
            # Actually, for N=5040, N^2 is 25M.
            # In Python, 25M iterations might take 5-10s.
            # Let's try to be efficient.
            
            visited = [False] * N
            current_idx = 0
            visited[0] = True
            current_word = list(perms[0])
            
            # Precompute for each index i, the list of indices j sorted by overlap descending?
            # This is expensive to precompute.
            # Instead, at each step, find the best unvisited j.
            
            # To speed up, we can precompute for each i, the top few candidates?
            # Or just scan.
            
            # Let's try scanning.
            # We can break early if we find overlap n-1 (perfect).
            
            # Check time periodically
            steps = 0
            for _ in range(N - 1):
                best_idx = -1
                best_ov = 0
                curr_perm = perms[current_idx]
                
                # Scan all unvisited
                for j in range(N):
                    if visited[j]:
                        continue
                    # Calculate overlap
                    # Max possible overlap is n-1
                    # Check k from n-1 down to 1
                    # if curr_perm[n-k:] == perms[j][:k]
                    
                    # Optimization: 
                    # The first symbol of perms[j] must match curr_perm[n-k]
                    # We can group by first symbol?
                    # No, just check.
                    
                    # Quick check: 
                    # If perms[j][0] != curr_perm[n-1] and perms[j][1] != curr_perm[n-2] ...
                    # Just loop k.
                    for k in range(n - 1, 0, -1):
                        if curr_perm[n-k:] == perms[j][:k]:
                            if k > best_ov:
                                best_ov = k
                                best_idx = j
                            break
                            
                if best_idx == -1:
                    # Should not happen in complete graph
                    break
                    
                visited[best_idx] = True
                # Append new symbols
                # Overlap is best_ov
                # Append perms[best_idx][best_ov:]
                current_word.extend(perms[best_idx][best_ov:])
                current_idx = best_idx
                
                steps += 1
                if time.time() - start_time > timeout:
                    break
                    
            if steps == N - 1:
                greedy_len = len(current_word)
                if greedy_len < classic_len:
                    return "".join(map(str, current_word))
            else:
                # Timeout or failure, fallback
                pass
                
        except Exception:
            pass
            
    return "".join(map(str, classic_word))
