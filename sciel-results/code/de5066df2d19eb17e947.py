"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import itertools

def construct(n):
    if n <= 6:
        # Use standard recursive construction for small n
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
                max_overlap = min(len(new), len(block))
                overlap = 0
                for o in range(max_overlap, -1, -1):
                    if o == 0:
                        overlap = 0
                        break
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
    
    # For n=7, use a beam search approach to find a good ordering of permutations of 1..6
    # to minimize the length of the superpermutation.
    
    # 1. Generate permutations of 1..6
    perms_6 = list(itertools.permutations(range(1, 7)))
    N = len(perms_6) # 720
    
    # 2. Precompute overlaps between all pairs of permutations.
    #    overlap[i][j] = max k such that suffix of perms_6[i] of length k
    #    equals prefix of perms_6[j] of length k.
    overlaps = [[0]*N for _ in range(N)]
    for i in range(N):
        pi = perms_6[i]
        for j in range(N):
            pj = perms_6[j]
            # Check suffix of pi vs prefix of pj
            for k in range(5, 0, -1):
                if pi[6-k:] == pj[:k]:
                    overlaps[i][j] = k
                    break
    
    # 3. Beam search to find a path visiting all nodes with minimum cost.
    #    Cost of edge i->j is (13 - overlaps[i][j]).
    #    We want to minimize sum of costs.
    
    #    Initialize beam with multiple starting points
    deadline = time.time() + 130.0
    beam_width = 50
    best_order = None
    best_cost = float('inf')
    
    # Start with several random starting points
    import random
    random.seed(42)
    
    # Initialize beam with diverse starting points
    beam = []
    for start in range(min(beam_width, N)):
        order = [start]
        visited = [False]*N
        visited[start] = True
        cost = 0
        beam.append((order, visited, cost))
    
    # Expand beam
    for step in range(1, N):
        if time.time() > deadline - 10:
            break
            
        new_beam = []
        for order, visited, cost in beam:
            last = order[-1]
            # Find best next nodes
            candidates = []
            for j in range(N):
                if not visited[j]:
                    edge_cost = 13 - overlaps[last][j]
                    # Lower bound on remaining cost
                    # At minimum, we need to visit N - step - 1 more nodes
                    # Each edge costs at least 13 - 5 = 8
                    lb = cost + edge_cost + (N - step - 1) * 8
                    if lb < best_cost:
                        candidates.append((edge_cost, j))
            
            # Sort by edge cost and keep top few
            candidates.sort()
            for edge_cost, j in candidates[:beam_width//2]:
                new_order = order + [j]
                new_visited = visited[:]
                new_visited[j] = True
                new_cost = cost + edge_cost
                if new_cost < best_cost:
                    new_beam.append((new_order, new_visited, new_cost))
        
        # Keep best beam_width candidates
        new_beam.sort(key=lambda x: x[2])
        beam = new_beam[:beam_width]
        
        # Update best solution
        for order, visited, cost in beam:
            if cost < best_cost:
                best_cost = cost
                best_order = order
    
    # If beam search didn't find a full path, use greedy fallback
    if not best_order or len(best_order) != N:
        # Greedy nearest neighbor
        order = [0]
        visited = [False]*N
        visited[0] = True
        
        for _ in range(N-1):
            last = order[-1]
            best_next = -1
            best_edge_cost = float('inf')
            for j in range(N):
                if not visited[j]:
                    edge_cost = 13 - overlaps[last][j]
                    if edge_cost < best_edge_cost:
                        best_edge_cost = edge_cost
                        best_next = j
            if best_next == -1:
                break
            order.append(best_next)
            visited[best_next] = True
        
        best_order = order
    
    # 4. Construct the superpermutation using the optimized order.
    word = []
    for idx, p_idx in enumerate(best_order):
        p = perms_6[p_idx]
        block = list(p) + [7] + list(p)
        
        if idx == 0:
            word.extend(block)
        else:
            # Find overlap with current word
            k = overlaps[best_order[idx-1]][p_idx]
            word.extend(block[k:])
            
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
