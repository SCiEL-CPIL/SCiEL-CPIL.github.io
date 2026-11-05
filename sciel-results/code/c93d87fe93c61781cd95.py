import time
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe
    
    # Base case
    if n == 1:
        return "1"
    
    # W is a list of integers
    W = [1]
    
    for m in range(2, n + 1):
        # 1. Extract unique (m-1)-permutations from W
        # A window of length m-1 in W is a permutation if it has distinct elements
        seen = set()
        perms = []
        # Optimization: only check windows that could be permutations
        # For m-1 < n, all windows in W_{m-1} are permutations of 1..m-1?
        # Yes, by induction.
        for i in range(len(W) - (m - 1) + 1):
            window = tuple(W[i:i + m - 1])
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        num_perms = len(perms)
        
        # 2. Compute overlap matrix
        # overlap[i][j] = length of longest suffix of perms[i] matching prefix of perms[j]
        # Max overlap is m-2
        overlap = [[0] * num_perms for _ in range(num_perms)]
        for i in range(num_perms):
            pi = perms[i]
            for j in range(num_perms):
                pj = perms[j]
                # Check suffix of pi against prefix of pj
                # Max length m-2
                max_ov = m - 2
                for k in range(max_ov, -1, -1):
                    if k == 0:
                        overlap[i][j] = 0
                        break
                    if pi[-k:] == pj[:k]:
                        overlap[i][j] = k
                        break
        
        # 3. Find a good ordering (Hamiltonian Path) maximizing overlap
        # Use Greedy + 2-Opt
        # Greedy: Start at 0, always go to neighbor with max overlap
        # Then 2-Opt: Try to improve by reversing segments
        
        # Initialize path
        path = [0]
        visited = {0}
        current = 0
        
        # Greedy construction
        while len(path) < num_perms:
            best_next = -1
            best_ov = -1
            for next_node in range(num_perms):
                if next_node not in visited:
                    ov = overlap[current][next_node]
                    if ov > best_ov:
                        best_ov = ov
                        best_next = next_node
            if best_next == -1:
                break # Should not happen
            path.append(best_next)
            visited.add(best_next)
            current = best_next
        
        # 2-Opt improvement
        # We want to maximize sum of overlaps between consecutive nodes in path
        # Score = sum(overlap[path[i]][path[i+1]])
        
        def get_score(p):
            s = 0
            for i in range(len(p) - 1):
                s += overlap[p[i]][p[i+1]]
            return s
        
        current_score = get_score(path)
        improved = True
        iterations = 0
        max_iterations = 1000 # Limit iterations to ensure speed
        
        while improved and iterations < max_iterations and time.time() < deadline - 5:
            improved = False
            iterations += 1
            # Try all 2-opt moves
            for i in range(len(path) - 1):
                for j in range(i + 1, len(path)):
                    # Reverse path[i+1 : j+1]
                    new_path = path[:i+1] + path[i+1:j+1][::-1] + path[j+1:]
                    new_score = get_score(new_path)
                    if new_score > current_score:
                        path = new_path
                        current_score = new_score
                        improved = True
        
        # 4. Construct new W
        new_W = []
        for idx, p_idx in enumerate(path):
            p = perms[p_idx]
            block = list(p) + [m] + list(p)
            if idx == 0:
                new_W.extend(block)
            else:
                # Find overlap with end of new_W
                # new_W ends with previous block's suffix
                # block starts with p
                # We need max k such that new_W[-k:] == block[:k]
                max_ov = min(len(new_W), len(block))
                ov = 0
                for k in range(max_ov, -1, -1):
                    if k == 0:
                        break
                    if new_W[-k:] == block[:k]:
                        ov = k
                        break
                new_W.extend(block[ov:])
        
        W = new_W
        
        if time.time() > deadline - 10:
            break
    
    return "".join(map(str, W))
