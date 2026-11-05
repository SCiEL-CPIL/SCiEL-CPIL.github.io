"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import heapq

def construct(n):
    """
    Constructs the shortest superpermutation for n symbols using the standard
    recursive construction, with a bridge-and-tunnel heuristic for n=7
    to reorder the permutations of 1..6 to minimize the total length.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin for safety
    
    if n == 1:
        return "1"
    
    word = [1]
    
    for m in range(2, n + 1):
        # Extract all (m-1)-permutations in order of appearance
        perms = []
        seen = set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        
        # Optimization for m=7 (adding 7 to perms of 1..6)
        if m == 7:
            num_perms = len(perms)
            
            # Precompute overlap matrix
            def get_overlap(p1, p2):
                for k in range(6, 0, -1):
                    if p1[-k:] == p2[:k]:
                        return k
                return 0
            
            # Overlap from start (end of S_6) to each perm
            start_end = tuple(word[-6:])
            start_ov = [0] * num_perms
            for i in range(num_perms):
                start_ov[i] = get_overlap(start_end, perms[i])
            
            # Pairwise overlaps
            ov_matrix = [[0] * num_perms for _ in range(num_perms)]
            for i in range(num_perms):
                for j in range(num_perms):
                    if i != j:
                        ov_matrix[i][j] = get_overlap(perms[i], perms[j])
            
            # Fitness function: maximize total overlap
            def fitness(order):
                total = start_ov[order[0]]
                for i in range(len(order) - 1):
                    total += ov_matrix[order[i]][order[i+1]]
                return total
            
            # Bridge-and-tunnel heuristic:
            # 1. Build a minimum spanning tree over the permutation graph
            # 2. Use DFS traversal to generate an ordering
            # 3. Refine with local search
            
            # Step 1: Build MST using Prim's algorithm
            # Edge weight = -overlap (we want to maximize overlap, so minimize -overlap)
            in_mst = [False] * num_perms
            in_mst[0] = True
            mst_edges = []  # list of (u, v, weight)
            
            # Priority queue: (-overlap, u, v)
            heap = []
            for v in range(1, num_perms):
                heapq.heappush(heap, (-ov_matrix[0][v], 0, v))
            
            while len(mst_edges) < num_perms - 1:
                if not heap:
                    break
                neg_w, u, v = heapq.heappop(heap)
                if in_mst[v]:
                    continue
                in_mst[v] = True
                mst_edges.append((u, v, -neg_w))
                for w in range(num_perms):
                    if not in_mst[w]:
                        heapq.heappush(heap, (-ov_matrix[v][w], v, w))
            
            # Step 2: DFS traversal of the MST
            # Build adjacency list from MST
            adj = [[] for _ in range(num_perms)]
            for u, v, w in mst_edges:
                adj[u].append((v, w))
                adj[v].append((u, w))
            
            # DFS to get an ordering
            order = []
            visited = [False] * num_perms
            
            def dfs(node):
                visited[node] = True
                order.append(node)
                # Sort neighbors by overlap (descending) to prioritize high-overlap transitions
                neighbors = sorted(adj[node], key=lambda x: -x[1])
                for neighbor, w in neighbors:
                    if not visited[neighbor]:
                        dfs(neighbor)
            
            # Start from the node with best start overlap
            start_node = max(range(num_perms), key=lambda i: start_ov[i])
            dfs(start_node)
            
            # Step 3: Local search refinement
            current_score = fitness(order)
            
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for i in range(len(order) - 1):
                    if time.time() > deadline - 5:
                        break
                    a = order[i-1] if i > 0 else -1
                    b = order[i]
                    c = order[i+1]
                    d = order[i+2] if i+2 < len(order) else -1
                    
                    # Old contribution
                    old_contrib = 0
                    if a != -1:
                        old_contrib += ov_matrix[a][b]
                    else:
                        old_contrib += start_ov[b]
                    old_contrib += ov_matrix[b][c]
                    if d != -1:
                        old_contrib += ov_matrix[c][d]
                    
                    # New contribution
                    new_contrib = 0
                    if a != -1:
                        new_contrib += ov_matrix[a][c]
                    else:
                        new_contrib += start_ov[c]
                    new_contrib += ov_matrix[c][b]
                    if d != -1:
                        new_contrib += ov_matrix[b][d]
                    
                    if new_contrib > old_contrib:
                        order[i], order[i+1] = order[i+1], order[i]
                        current_score += new_contrib - old_contrib
                        improved = True
            
            # Use the optimized order
            for idx in order:
                p = perms[idx]
                block = list(p) + [m] + list(p)
                
                # Find overlap
                max_possible_overlap = min(len(new_word), len(block), m - 1)
                overlap = 0
                for k in range(max_possible_overlap, -1, -1):
                    if new_word[len(new_word)-k:] == block[:k]:
                        overlap = k
                        break
                
                new_word.extend(block[overlap:])
        else:
            # Standard construction for m < 7
            for p in perms:
                block = list(p) + [m] + list(p)
                
                max_possible_overlap = min(len(new_word), len(block), m - 1)
                overlap = 0
                for k in range(max_possible_overlap, -1, -1):
                    if new_word[len(new_word)-k:] == block[:k]:
                        overlap = k
                        break
                
                new_word.extend(block[overlap:])
        
        word = new_word
        
        if time.time() > deadline:
            break
            
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
