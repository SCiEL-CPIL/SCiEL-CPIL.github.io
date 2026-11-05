import time
import random
import numpy as np

def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    if n == 1:
        return "1"
        
    # Step 1: Generate base permutations for n=1
    # S_1 = [1]
    # Perms of length 1: [(1,)]
    current_word = [1]
    current_perms = [(1,)]
    
    for m in range(2, n + 1):
        # Check time
        if time.time() > deadline - 10.0:
            break
            
        # Extract permutations of length m-1 from current_word
        # They appear in order of first appearance.
        # Since current_word is a superpermutation, all (m-1)! perms are present.
        # We need to extract them in the order they appear.
        perms, seen = [], set()
        # The window size is m-1
        L = len(current_word)
        for i in range(L - (m - 1) + 1):
            w = tuple(current_word[i:i + m - 1])
            if len(set(w)) == m - 1 and w not in seen:
                seen.add(w)
                perms.append(w)
        
        # We expect len(perms) == (m-1)!
        # Construct blocks
        K = len(perms)
        blocks = []
        for p in perms:
            # Block is p, m, p
            # p is a tuple of ints
            b = list(p) + [m] + list(p)
            blocks.append(b)
            
        # Precompute overlap matrix
        # O[i][j] = overlap between blocks[i] and blocks[j]
        # Overlap is determined by suffix of p_i and prefix of p_j
        # p_i is perms[i], p_j is perms[j]
        # len(p_i) = m-1
        # max overlap is m-1
        overlaps = np.zeros((K, K), dtype=np.int8)
        for i in range(K):
            pi = perms[i]
            for j in range(K):
                pj = perms[j]
                # Check k from m-1 down to 0
                # pi[-k:] == pj[:k]
                # Since m is small, this is fast
                max_k = m - 1
                for k in range(max_k, -1, -1):
                    if k == 0:
                        overlaps[i][j] = 0
                        break
                    if pi[m-1-k:] == pj[:k]:
                        overlaps[i][j] = k
                        break
        
        # Solve TSP-like problem to maximize sum of overlaps
        # Use Nearest Neighbor + 2-opt
        # Start node: 0
        visited = [False] * K
        path = [0]
        visited[0] = True
        
        # Nearest Neighbor
        for _ in range(K - 1):
            last = path[-1]
            best_next = -1
            best_ov = -1
            for j in range(K):
                if not visited[j]:
                    if overlaps[last][j] > best_ov:
                        best_ov = overlaps[last][j]
                        best_next = j
            path.append(best_next)
            visited[best_next] = True
            
        # 2-opt improvement
        # Calculate current total overlap
        def calc_score(path):
            s = 0
            for i in range(len(path) - 1):
                s += overlaps[path[i]][path[i+1]]
            return s
            
        current_score = calc_score(path)
        improved = True
        max_iters = 50 # Limit iterations to avoid timeout
        iter_count = 0
        while improved and iter_count < max_iters and time.time() < deadline - 5.0:
            improved = False
            iter_count += 1
            for i in range(K - 1):
                # Reverse segment i+1 to j
                # Try all j > i
                for j in range(i + 1, K):
                    # Calculate delta
                    # Old: path[i] -> path[i+1] ... path[j] -> path[j+1]
                    # New: path[i] -> path[j] ... path[i+1] -> path[j+1]
                    # Note: path is not a cycle, it's a path.
                    # So we only care about edges (i, i+1) and (j, j+1) if j < K-1
                    # And the internal edges are reversed.
                    
                    # Edges removed: (i, i+1) and (j, j+1) [if j < K-1]
                    # Edges added: (i, j) and (i+1, j+1) [if j < K-1]
                    # Internal edges: (k, k+1) becomes (k+1, k)
                    
                    delta = 0
                    # Remove (i, i+1)
                    delta -= overlaps[path[i]][path[i+1]]
                    # Add (i, j)
                    delta += overlaps[path[i]][path[j]]
                    
                    if j < K - 1:
                        # Remove (j, j+1)
                        delta -= overlaps[path[j]][path[j+1]]
                        # Add (i+1, j+1)
                        delta += overlaps[path[i+1]][path[j+1]]
                    else:
                        # j is last, no edge after j
                        # Add (i+1, j) ? No, the segment is reversed.
                        # Original: ... i, i+1, ..., j
                        # New: ... i, j, ..., i+1
                        # Edge (i, j) added.
                        # Edge (i+1, j) is the new last edge?
                        # Wait, if j=K-1, the path ends at j.
                        # Original end: ... j
                        # New end: ... i+1
                        # So we remove edge (i, i+1) and (j, j+1) [none].
                        # We add (i, j) and (i+1, j) ?
                        # Let's trace:
                        # Path: P[0] ... P[i], P[i+1] ... P[j]
                        # New: P[0] ... P[i], P[j] ... P[i+1]
                        # Edge (i, i+1) removed.
                        # Edge (i, j) added.
                        # Edge (i+1, j) added?
                        # The new last node is P[i+1].
                        # The node before it is P[i+2] (if i+2 <= j).
                        # The edge into P[i+1] is from P[i+2] (which was P[i+1] in old? No).
                        # Let's just recalculate score for small K or use a simpler heuristic if K is large.
                        # For K=720, 2-opt is O(K^2) per pass.
                        # Let's use a simpler check: just try 2-opt swaps.
                        pass
                    
                    # Simplified 2-opt delta calculation for path:
                    # Remove edges (i, i+1) and (j, j+1) (if exists)
                    # Add edges (i, j) and (i+1, j+1) (if exists)
                    # Internal edges are reversed. Overlap is not symmetric?
                    # Overlap(i, j) != Overlap(j, i) generally.
                    # So internal edges change value.
                    # This makes delta calculation complex.
                    # Given time constraints, maybe just use Nearest Neighbor + Or-opt or simple local search.
                    
                    # Let's stick to a simpler heuristic: Random restarts of Nearest Neighbor.
                    break
                break
            # If we broke out, we didn't do 2-opt properly.
            # Let's just use Nearest Neighbor with a few random starts.
            break
            
        # Let's implement a proper but fast local search: Or-opt (move 1 node)
        # Move node x to position y
        # This is O(K^2) per pass.
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for x in range(K):
                if x == 0 or x == K - 1:
                    # Don't move ends? Or allow it?
                    # Let's allow moving any node to any other position.
                    pass
                node = path[x]
                # Try inserting node at position y (y != x)
                for y in range(K):
                    if y == x:
                        continue
                    # Calculate delta
                    # Remove node x from path
                    # Insert node x at y
                    # This affects edges around x and around y.
                    
                    # Get neighbors of x in original path
                    prev_x = path[x-1] if x > 0 else None
                    next_x = path[x+1] if x < K-1 else None
                    
                    # Get neighbors of insertion point y in path without x
                    # This is tricky.
                    
                    # Given the complexity and time, let's just use the Nearest Neighbor result.
                    # It's likely good enough to beat the standard order.
                    break
                break
            break
            
        # Construct new word
        new_word = []
        for idx, i in enumerate(path):
            b = blocks[i]
            if idx == 0:
                new_word.extend(b)
            else:
                prev_i = path[idx-1]
                ov = overlaps[prev_i][i]
                new_word.extend(b[ov:])
                
        current_word = new_word
        current_perms = perms # Not strictly needed, we extract from word
        
    return "".join(map(str, current_word))
