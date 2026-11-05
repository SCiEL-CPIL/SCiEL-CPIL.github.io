import time
import random
import sys
import numpy as np

def construct(n):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    if n <= 6:
        # Standard construction is optimal for n <= 6
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
        # Generate S_6 standard
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
        
        # Extract perms of 1..6 in the order they appear in S_6
        perms, seen = [], set()
        for i in range(len(word) - 5):
            window = tuple(word[i:i + 6])
            if len(set(window)) == 6 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        N = len(perms)  # 720
        
        # Precompute overlap matrix using NumPy for speed
        # Convert perms to a 2D array
        perm_arr = np.array(perms, dtype=np.int8)  # shape (N, 6)
        
        # ov[i][j] = overlap(perms[i], perms[j])
        # Check k from 5 down to 1
        ov = np.zeros((N, N), dtype=np.int8)
        
        # For each k from 5 down to 1, find pairs where suffix of length k matches prefix
        for k in range(5, 0, -1):
            # suffix of length k: perm_arr[:, 6-k:]
            # prefix of length k: perm_arr[:, :k]
            # We want ov[i][j] = k if suffix_k[i] == prefix_k[j] and ov[i][j] was 0
            # Use broadcasting - but this is N^2 * k which is large
            # Instead, use hashing approach
            
            # Create a hash for each suffix of length k
            # Map each suffix to an index
            from collections import defaultdict
            suffix_to_indices = defaultdict(list)
            for i in range(N):
                sfx = tuple(perm_arr[i, 6-k:])
                suffix_to_indices[sfx].append(i)
            
            prefix_to_indices = defaultdict(list)
            for j in range(N):
                pfx = tuple(perm_arr[j, :k])
                prefix_to_indices[pfx].append(j)
            
            for key in suffix_to_indices:
                if key in prefix_to_indices:
                    for i in suffix_to_indices[key]:
                        for j in prefix_to_indices[key]:
                            if i != j and ov[i, j] == 0:
                                ov[i, j] = k
        
        ov = ov.tolist()
        
        # Greedy nearest-neighbor for initial solution
        path = [0]
        visited = [False] * N
        visited[0] = True
        remaining = set(range(1, N))
        
        while remaining:
            current = path[-1]
            best_j = -1
            best_ov = -1
            for j in remaining:
                if ov[current][j] > best_ov:
                    best_ov = ov[current][j]
                    best_j = j
            path.append(best_j)
            visited[best_j] = True
            remaining.remove(best_j)
        
        # Calculate total overlap
        def calc_total_overlap(path):
            total = 0
            for i in range(len(path) - 1):
                total += ov[path[i]][path[i+1]]
            return total
        
        current_overlap = calc_total_overlap(path)
        best_overlap = current_overlap
        best_path = path[:]
        
        # 2-opt local search
        improved = True
        iteration = 0
        while improved and time.time() < deadline - 15:
            improved = False
            iteration += 1
            
            for i in range(N - 1):
                if time.time() > deadline - 15:
                    break
                for j in range(i + 1, N):
                    # Reverse path[i+1..j]
                    if j == N - 1:
                        old_ov = ov[path[i]][path[i+1]]
                        new_ov = ov[path[i]][path[j]]
                        if new_ov > old_ov:
                            path[i+1:j+1] = path[i+1:j+1][::-1]
                            current_overlap += new_ov - old_ov
                            improved = True
                            if current_overlap > best_overlap:
                                best_overlap = current_overlap
                                best_path = path[:]
                            break
                    else:
                        old_ov = ov[path[i]][path[i+1]] + ov[path[j]][path[j+1]]
                        new_ov = ov[path[i]][path[j]] + ov[path[i+1]][path[j+1]]
                        if new_ov > old_ov:
                            path[i+1:j+1] = path[i+1:j+1][::-1]
                            current_overlap += new_ov - old_ov
                            improved = True
                            if current_overlap > best_overlap:
                                best_overlap = current_overlap
                                best_path = path[:]
                            break
                if improved:
                    break
        
        # Simulated annealing with 2-opt moves
        if time.time() < deadline - 10:
            sa_start_temp = 2.0
            sa_end_temp = 0.1
            path = best_path[:]
            current_overlap = best_overlap
            best_overlap_sa = current_overlap
            best_path_sa = path[:]
            
            n_iter = 0
            max_iter = 50000
            temp = sa_start_temp
            cooling_rate = (sa_end_temp / sa_start_temp) ** (1.0 / max_iter)
            
            while n_iter < max_iter and time.time() < deadline - 10:
                n_iter += 1
                temp *= cooling_rate
                
                # Random 2-opt move
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                
                if j == N - 1:
                    old_ov = ov[path[i]][path[i+1]]
                    new_ov = ov[path[i]][path[j]]
                    delta = new_ov - old_ov
                else:
                    old_ov = ov[path[i]][path[i+1]] + ov[path[j]][path[j+1]]
                    new_ov = ov[path[i]][path[j]] + ov[path[i+1]][path[j+1]]
                    delta = new_ov - old_ov
                
                if delta > 0 or random.random() < np.exp(delta / temp):
                    path[i+1:j+1] = path[i+1:j+1][::-1]
                    current_overlap += delta
                    if current_overlap > best_overlap_sa:
                        best_overlap_sa = current_overlap
                        best_path_sa = path[:]
            
            if best_overlap_sa > best_overlap:
                best_overlap = best_overlap_sa
                best_path = best_path_sa
        
        # Use the best path found
        path = best_path
        
        # Construct S_7 using the optimized path
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
            
        return "".join(map(str, new_word))
    
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
