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
        
    # Try Improved Greedy TSP for n=6, 7
    if n in [6, 7]:
        try:
            # Generate all permutations
            perms = list(itertools.permutations(range(1, n + 1)))
            N = len(perms)
            
            # Precompute prefix maps for fast overlap lookup
            # prefix_map[k][prefix_tuple] = list of indices
            prefix_maps = [None] * (n + 1)
            for k in range(1, n):
                pm = {}
                for i, p in enumerate(perms):
                    pref = p[:k]
                    if pref not in pm:
                        pm[pref] = []
                    pm[pref].append(i)
                prefix_maps[k] = pm
            
            def run_greedy(start_idx=0):
                visited = [False] * N
                visited[start_idx] = True
                current_idx = start_idx
                current_word = list(perms[start_idx])
                
                for _ in range(N - 1):
                    best_idx = -1
                    best_ov = 0
                    curr_perm = perms[current_idx]
                    
                    # Check overlaps from largest to smallest
                    for k in range(n - 1, 0, -1):
                        suffix = curr_perm[n - k:]
                        candidates = prefix_maps[k].get(suffix, [])
                        for j in candidates:
                            if not visited[j]:
                                best_idx = j
                                best_ov = k
                                break
                        if best_idx != -1:
                            break
                            
                    if best_idx == -1:
                        # Fallback: find any unvisited (should not happen for connected graph)
                        # But if it does, we just pick the first one
                        for j in range(N):
                            if not visited[j]:
                                best_idx = j
                                best_ov = 0
                                break
                        if best_idx == -1:
                            return None # Failure

                    visited[best_idx] = True
                    current_word.extend(perms[best_idx][best_ov:])
                    current_idx = best_idx
                    
                    if time.time() - start_time > timeout:
                        return None

                return current_word

            # Try a few random starts to see if we get lucky
            # But deterministic is better for reproducibility if we can't guarantee improvement
            # Let's try start 0 first
            greedy_word = run_greedy(0)
            if greedy_word and len(greedy_word) < classic_len:
                return "".join(map(str, greedy_word))
            
            # If time remains, try a simple local search on the best found?
            # Or try a different start?
            # Given the time constraint, just one good greedy run is a start.
            # But let's try to optimize the greedy choice slightly.
            # If there are multiple candidates with same overlap, pick the one with best *next* potential?
            # That's expensive.
            
            # Let's just return the greedy result if it's valid and better.
            # If not, return classic.
            if greedy_word:
                return "".join(map(str, greedy_word))
                
        except Exception:
            pass
            
    return "".join(map(str, classic_word))
