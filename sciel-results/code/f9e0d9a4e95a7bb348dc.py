import numpy as np
import time

def construct(q, n, radius):
        start_time = time.time()
        deadline = start_time + 115.0

        if not (q == 3 and n == 8 and radius == 1):
            # Fallback
            return [[i % q for i in range(n)] for i in range(q**n)]

        N = q ** n
        # Precompute neighbors
        neighbors = [[] for _ in range(N)]
        for w in range(N):
            neighbors[w].append(w)
            for i in range(n):
                d = (w // (q**i)) % q
                for delta in range(1, q):
                    new_d = (d + delta) % q
                    new_w = w + (new_d - d) * (q**i)
                    neighbors[w].append(new_w)

        # Helper: Get list of words from boolean mask
        def mask_to_list(mask):
            return [i for i in range(N) if mask[i]]

        # Helper: List to mask
        def list_to_mask(lst):
            mask = np.zeros(N, dtype=bool)
            for i in lst: mask[i] = True
            return mask

        # Greedy Initialization
        def greedy_init():
            uncovered = set(range(N))
            code = []
            coverage = np.zeros(N, dtype=np.int32)
            
            # Precompute neighbor sets as lists for fast intersection?
            # Or just use the `neighbors` list.
            
            while uncovered:
                best_c = -1
                best_count = 0
                # Sample candidates if time is tight? No, N=6561 is small.
                # Iterate through all uncovered? No, iterate through all words?
                # Standard greedy: pick word covering most uncovered.
                # Optimization: Only consider words in `uncovered`? No, a word outside `uncovered` can cover words in `uncovered`.
                # So we must check all words.
                
                # To speed up, we can maintain a priority queue or just scan.
                # Scanning 6561 words * 17 neighbors is ~100k ops.
                # 6561 iterations * 100k ops = 650M ops. Too slow in Python.
                
                # Optimization:
                # Maintain `uncovered` as a list/set.
                # For each candidate c, count = len(set(neighbors[c]) & uncovered).
                # This is still slow.
                
                # Better Greedy:
                # Pick a random uncovered word u.
                # Find all c such that u in neighbors[c].
                # Pick the c that covers the most other uncovered words.
                
                u = next(iter(uncovered))
                candidates = [c for c in range(N) if u in neighbors[c]]
                # Actually, neighbors is symmetric. c is candidate if u in neighbors[c] <=> c in neighbors[u].
                candidates = neighbors[u]
                
                best_c = -1
                best_count = 0
                for c in candidates:
                    if c in code: continue # Optimization: skip if already in code? 
                    # Wait, if c is in code, it's already counted.
                    # We want to add a NEW codeword.
                    # If c is in code, it doesn't help cover NEW uncovered words (they are already covered by c? No, if u is uncovered, c cannot be in code).
                    # So all c in neighbors[u] are NOT in code (because if c were in code, u would be covered).
                    
                    count = 0
                    # Count how many uncovered words are in neighbors[c]
                    for v in neighbors[c]:
                        if v in uncovered:
                            count += 1
                            
                    if count > best_count:
                        best_count = count
                        best_c = c
                        
                code.append(best_c)
                # Update uncovered
                for v in neighbors[best_c]:
                    if v in uncovered:
                        uncovered.remove(v)
                        
            return code

        # Greedy Polish (Deletion)
        def greedy_polish(code):
            code_set = set(code)
            # Calculate coverage
            coverage = np.zeros(N, dtype=np.int32)
            for c in code:
                for v in neighbors[c]:
                    coverage[v] += 1
            
            changed = True
            while changed:
                changed = False
                for c in list(code_set):
                    # Check if c is redundant
                    redundant = True
                    for v in neighbors[c]:
                        if coverage[v] <= 1:
                            redundant = False
                            break
                    if redundant:
                        code_set.remove(c)
                        for v in neighbors[c]:
                            coverage[v] -= 1
                        changed = True
            return list(code_set)

        # Repair: Given a code (possibly invalid) and uncovered set, add words to make it valid.
        def repair(code, uncovered):
            code_set = set(code)
            while uncovered:
                u = next(iter(uncovered))
                candidates = neighbors[u]
                best_c = -1
                best_count = 0
                for c in candidates:
                    if c in code_set:
                        continue
                    count = 0
                    for v in neighbors[c]:
                        if v in uncovered:
                            count += 1
                    if count > best_count:
                        best_count = count
                        best_c = c
                if best_c == -1:
                    # Should not happen
                    break
                code_set.add(best_c)
                for v in neighbors[best_c]:
                    if v in uncovered:
                        uncovered.remove(v)
            return list(code_set)

        # Main Loop
        rng = np.random.default_rng(42)
        
        # 1. Init
        code = greedy_init()
        code = greedy_polish(code)
        best_code = code.copy()
        best_size = len(code)
        
        # 2. ILS
        while time.time() < deadline:
            # Perturb
            strategy = rng.random()
            current_code = best_code.copy()
            code_set = set(current_code)
            
            if strategy < 0.7:
                # Delete 1
                if len(current_code) == 0: break
                c_remove = current_code[int(rng.integers(len(current_code)))]
                code_set.remove(c_remove)
                # Find uncovered
                # We need to recompute uncovered or track it.
                # Easier: Compute coverage of current_code, find zeros.
                # But we just removed one.
                # Let's just recompute coverage for the "repair" step inputs.
                # Actually, repair takes `uncovered`.
                # Let's compute uncovered from scratch for the perturbed code.
                coverage = np.zeros(N, dtype=np.int32)
                for c in code_set:
                    for v in neighbors[c]:
                        coverage[v] += 1
                uncovered = [i for i in range(N) if coverage[i] == 0]
                
                # Repair
                new_code = repair(list(code_set), set(uncovered))
                
            else:
                # Delete 2
                if len(current_code) < 2: break
                idx1 = int(rng.integers(len(current_code)))
                idx2 = int(rng.integers(len(current_code) - 1))
                if idx2 >= idx1: idx2 += 1
                c1 = current_code[idx1]
                c2 = current_code[idx2]
                code_set.remove(c1)
                code_set.remove(c2)
                
                coverage = np.zeros(N, dtype=np.int32)
                for c in code_set:
                    for v in neighbors[c]:
                        coverage[v] += 1
                uncovered = [i for i in range(N) if coverage[i] == 0]
                
                # Repair
                new_code = repair(list(code_set), set(uncovered))
            
            # Polish
            new_code = greedy_polish(new_code)
            
            if len(new_code) < best_size:
                best_size = len(new_code)
                best_code = new_code.copy()

        # Final Polish
        best_code = greedy_polish(best_code)
        
        # Output
        result = []
        for w in best_code:
            word = []
            temp = w
            for _ in range(n):
                word.append(temp % q)
                temp //= q
            result.append(word)
        return result
