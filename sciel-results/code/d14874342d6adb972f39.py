# EVOLVE-BLOCK-START
import time
import random
import numpy as np

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
        
        # Precompute overlap matrix using NumPy for speed
        # ov[i][j] = max overlap between perms[i] and perms[j]
        ov = np.zeros((N, N), dtype=np.int8)
        # Convert perms to numpy array for faster comparison
        perms_arr = np.array(perms)  # shape (N, 6)
        
        for k in range(5, 0, -1):
            # For each k, check if suffix of length k of i matches prefix of length k of j
            # suffix_k[i] = perms[i][6-k:6]
            # prefix_k[j] = perms[j][0:k]
            # We want ov[i][j] = k if suffix_k[i] == prefix_k[j] and not already set to higher value
            
            # Build a mapping from tuple to list of indices for efficiency
            # Actually, let's use a different approach: for each i, compute its suffix of length k
            # and for each j, compute its prefix of length k, then match.
            
            # For speed, let's iterate and use numpy comparison
            # suffix_i = perms_arr[:, 6-k:]  shape (N, k)
            # prefix_j = perms_arr[:, :k]    shape (N, k)
            # We need ov[i,j] = k where suffix_i[i] == prefix_j[j]
            
            # This is an O(N^2) comparison which is 720^2 = 518400, times 5 = 2.5M comparisons
            # Each comparison is k elements. In Python this is slow. Let's use a dict approach.
            
            # For each i, the suffix of length k is a tuple. Group j's by their prefix of length k.
            prefix_to_indices = {}
            for j in range(N):
                pref = tuple(perms[j][:k])
                if pref not in prefix_to_indices:
                    prefix_to_indices[pref] = []
                prefix_to_indices[pref].append(j)
            
            for i in range(N):
                suff = tuple(perms[i][6-k:])
                if suff in prefix_to_indices:
                    for j in prefix_to_indices[suff]:
                        if ov[i][j] < k:
                            ov[i][j] = k
        
        ov_list = ov.tolist()  # Convert to list for faster Python access in some cases
        
        def calc_total_overlap(path):
            total = 0
            for i in range(len(path) - 1):
                total += ov[path[i]][path[i+1]]
            return total
        
        def build_word_from_path(path):
            # Fast word building using precomputed overlaps
            new_word = []
            for idx in path:
                p = perms[idx]
                block = list(p) + [7] + list(p)
                if not new_word:
                    new_word = block[:]
                else:
                    # Find max overlap
                    max_ov = min(len(new_word), len(block))
                    found = 0
                    for o in range(max_ov, -1, -1):
                        if new_word[len(new_word)-o:] == block[:o]:
                            found = o
                            break
                    new_word += block[found:]
            return new_word
        
        # === Bridge-and-Tunnel Heuristic ===
        # Step 1: Build MST on permutation graph with edge weight = overlap
        # Use Prim's algorithm
        in_mst = [False] * N
        in_mst[0] = True
        mst_edges = []  # list of (u, v, weight)
        
        # Distance from MST to each node
        dist = [0] * N
        for j in range(1, N):
            dist[j] = ov[0][j]
        
        for _ in range(N - 1):
            # Find node with max dist not in MST
            best_node = -1
            best_dist = -1
            for j in range(N):
                if not in_mst[j] and dist[j] > best_dist:
                    best_dist = dist[j]
                    best_node = j
            
            if best_node == -1:
                break
                
            # Find which MST node gives this max dist
            parent = -1
            for i in range(N):
                if in_mst[i] and ov[i][best_node] == best_dist:
                    parent = i
                    break
            
            in_mst[best_node] = True
            mst_edges.append((parent, best_node, best_dist))
            
            # Update distances
            for j in range(N):
                if not in_mst[j] and ov[best_node][j] > dist[j]:
                    dist[j] = ov[best_node][j]
        
        # Step 2: DFS traversal of MST to get initial ordering
        # Build adjacency list
        adj = [[] for _ in range(N)]
        for u, v, w in mst_edges:
            adj[u].append((v, w))
            adj[v].append((u, w))
        
        # DFS from node 0
        path = []
        visited = [False] * N
        
        def dfs(node):
            visited[node] = True
            path.append(node)
            # Sort neighbors by overlap (desc) for better ordering
            neighbors = sorted(adj[node], key=lambda x: -x[1])
            for nb, w in neighbors:
                if not visited[nb]:
                    dfs(nb)
        
        dfs(0)
        
        # Verify all nodes visited
        if len(path) != N:
            # Add any unvisited nodes
            for i in range(N):
                if not visited[i]:
                    path.append(i)
        
        best_path = path[:]
        best_overlap = calc_total_overlap(best_path)
        
        # Step 3: Local search refinement with 1-opt and 2-opt-relocate
        improved = True
        passes = 0
        max_passes = 200
        
        while improved and time.time() < deadline - 15 and passes < max_passes:
            improved = False
            passes += 1
            current_path = best_path[:]
            current_overlap = best_overlap
            
            # 1-Opt (Relocate single element) - Best Improvement
            best_gain = 0
            best_i = -1
            best_j = -1
            
            for i in range(N):
                if i > 0:
                    left = current_path[i-1]
                else:
                    left = -1
                if i < N - 1:
                    right = current_path[i+1]
                else:
                    right = -1
                
                gain_remove = 0
                if left != -1 and right != -1:
                    gain_remove = ov[left][right] - ov[left][i] - ov[i][right]
                elif left != -1:
                    gain_remove = 0 - ov[left][i]
                elif right != -1:
                    gain_remove = 0 - ov[i][right]
                else:
                    gain_remove = 0
                    
                item = current_path[i]
                
                for j in range(N):
                    if j == i or j == i + 1:
                        continue
                    
                    if j < i:
                        left_orig = j - 1 if j > 0 else -1
                        right_orig = j
                    else:
                        left_orig = j
                        right_orig = j + 1 if j < N - 1 else -1
                    
                    gain_insert = 0
                    if left_orig != -1 and right_orig != -1:
                        gain_insert = ov[left_orig][item] + ov[item][right_orig] - ov[left_orig][right_orig]
                    elif left_orig != -1:
                        gain_insert = ov[left_orig][item]
                    elif right_orig != -1:
                        gain_insert = ov[item][right_orig]
                    else:
                        gain_insert = 0
                    
                    total_gain = gain_remove + gain_insert
                    if total_gain > best_gain:
                        best_gain = total_gain
                        best_i = i
                        best_j = j
            
            if best_gain > 0:
                i = best_i
                j = best_j
                item = current_path[i]
                del current_path[i]
                if i < j:
                    j -= 1
                current_path.insert(j, item)
                current_overlap += best_gain
                best_path = current_path
                best_overlap = current_overlap
                improved = True
                continue
            
            # 2-Opt-Relocate (Move block of 2)
            if not improved:
                best_gain = 0
                best_i = -1
                best_j = -1
                for i in range(N - 1):
                    block = current_path[i:i+2]
                    if i > 0:
                        left = current_path[i-1]
                    else:
                        left = -1
                    if i + 2 < N:
                        right = current_path[i+2]
                    else:
                        right = -1
                    
                    gain_remove = 0
                    if left != -1 and right != -1:
                        gain_remove = ov[left][right] - ov[left][block[0]] - ov[block[0]][block[1]] - ov[block[1]][right]
                    elif left != -1:
                        gain_remove = 0 - ov[left][block[0]] - ov[block[0]][block[1]]
                    elif right != -1:
                        gain_remove = 0 - ov[block[0]][block[1]] - ov[block[1]][right]
                    else:
                        gain_remove = 0 - ov[block[0]][block[1]]
                    
                    for j in range(N - 1):
                        if j == i or j == i + 1:
                            continue
                        
                        if j < i:
                            left_orig = j - 1 if j > 0 else -1
                            right_orig = j
                        else:
                            left_orig = j + 1
                            right_orig = j + 2 if j < N - 2 else -1
                            
                        gain_insert = 0
                        if left_orig != -1 and right_orig != -1:
                            gain_insert = ov[left_orig][block[0]] + ov[block[0]][block[1]] + ov[block[1]][right_orig] - ov[left_orig][right_orig]
                        elif left_orig != -1:
                            gain_insert = ov[left_orig][block[0]] + ov[block[0]][block[1]]
                        elif right_orig != -1:
                            gain_insert = ov[block[0]][block[1]] + ov[block[1]][right_orig]
                        else:
                            gain_insert = ov[block[0]][block[1]]
                        
                        total_gain = gain_remove + gain_insert
                        if total_gain > best_gain:
                            best_gain = total_gain
                            best_i = i
                            best_j = j
                
                if best_gain > 0:
                    i = best_i
                    j = best_j
                    block = current_path[i:i+2]
                    del current_path[i:i+2]
                    if i < j:
                        j -= 2
                    current_path.insert(j, block[0])
                    current_path.insert(j+1, block[1])
                    current_overlap += best_gain
                    best_path = current_path
                    best_overlap = current_overlap
                    improved = True
        
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
