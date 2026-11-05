import time
import random
import numpy as np
from collections import defaultdict

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
                # Find max overlap
                overlap = 0
                min_len = min(len(new), len(block))
                for o in range(min_len, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
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
                overlap = 0
                min_len = min(len(new), len(block))
                for o in range(min_len, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
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
        
        # Precompute overlap matrix
        # ov[i][j] = overlap(perms[i], perms[j])
        ov = np.zeros((N, N), dtype=np.int8)
        
        # Use hashing for speed
        for k in range(5, 0, -1):
            suffix_to_indices = defaultdict(list)
            prefix_to_indices = defaultdict(list)
            for i in range(N):
                sfx = tuple(perms[i][6-k:])
                suffix_to_indices[sfx].append(i)
                pfx = tuple(perms[i][:k])
                prefix_to_indices[pfx].append(i)
            
            for key in suffix_to_indices:
                if key in prefix_to_indices:
                    for i in suffix_to_indices[key]:
                        for j in prefix_to_indices[key]:
                            if i != j and ov[i, j] == 0:
                                ov[i, j] = k
        
        ov_list = ov.tolist()
        
        # Greedy nearest-neighbor for initial solution
        def greedy_tsp(ov, N, start_node=0):
            path = [start_node]
            visited = [False] * N
            visited[start_node] = True
            remaining = set(range(N))
            remaining.remove(start_node)
            
            while remaining:
                current = path[-1]
                best_j = -1
                best_ov = -1
                # Find neighbor with max overlap
                # We can optimize this by pre-sorting neighbors, but N is small enough
                for j in remaining:
                    if ov[current][j] > best_ov:
                        best_ov = ov[current][j]
                        best_j = j
                path.append(best_j)
                visited[best_j] = True
                remaining.remove(best_j)
            return path

        # 2-opt improvement
        def two_opt_improve(path, ov, N, time_limit):
            improved = True
            current_overlap = 0
            for i in range(N - 1):
                current_overlap += ov[path[i]][path[i+1]]
            
            while improved and time.time() < time_limit:
                improved = False
                for i in range(N - 1):
                    if time.time() > time_limit:
                        break
                    for j in range(i + 1, N):
                        if j == N - 1:
                            old_ov = ov[path[i]][path[i+1]]
                            new_ov = ov[path[i]][path[j]]
                            if new_ov > old_ov:
                                path[i+1:j+1] = path[i+1:j+1][::-1]
                                current_overlap += new_ov - old_ov
                                improved = True
                                break
                        else:
                            old_ov = ov[path[i]][path[i+1]] + ov[path[j]][path[j+1]]
                            new_ov = ov[path[i]][path[j]] + ov[path[i+1]][path[j+1]]
                            if new_ov > old_ov:
                                path[i+1:j+1] = path[i+1:j+1][::-1]
                                current_overlap += new_ov - old_ov
                                improved = True
                                break
                    if improved:
                        break
            return path, current_overlap

        # Simulated Annealing
        def simulated_annealing(path, ov, N, start_temp, end_temp, steps, time_limit):
            current_overlap = 0
            for i in range(N - 1):
                current_overlap += ov[path[i]][path[i+1]]
            
            best_overlap = current_overlap
            best_path = path[:]
            
            temp = start_temp
            cooling_rate = (end_temp / start_temp) ** (1.0 / steps)
            
            for _ in range(steps):
                if time.time() > time_limit:
                    break
                temp *= cooling_rate
                
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
                    if current_overlap > best_overlap:
                        best_overlap = current_overlap
                        best_path = path[:]
            
            return best_path, best_overlap

        # Main optimization loop
        best_path = None
        best_overlap = -1
        
        # Run a few restarts
        num_restarts = 5
        for r in range(num_restarts):
            if time.time() > deadline - 20:
                break
            
            # Greedy start
            path = greedy_tsp(ov_list, N)
            
            # 2-Opt
            path, overlap = two_opt_improve(path, ov_list, N, deadline - 10)
            
            # Simulated Annealing
            if time.time() < deadline - 5:
                path, overlap = simulated_annealing(path, ov_list, N, 2.0, 0.1, 5000, deadline - 2)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_path = path[:]
        
        # Construct final word
        if best_path is None:
            # Fallback to standard
            new_word = []
            for idx in range(N):
                p = perms[idx]
                block = list(p) + [7] + list(p)
                overlap = 0
                min_len = min(len(new_word), len(block))
                for o in range(min_len, -1, -1):
                    if new_word[len(new_word)-o:] == block[:o]:
                        overlap = o
                        break
                new_word += block[overlap:]
        else:
            new_word = []
            for idx in best_path:
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
            overlap = 0
            min_len = min(len(new), len(block))
            for o in range(min_len, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return "".join(map(str, word))
