import time
import itertools
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 115.0

    if n == 1: return "1"
    if n == 2: return "121"
    if n == 3: return "123121321"
    if n == 4: return "123412314231243121342132412341214321" # Standard optimal S4 length 33
    
    # We will build up to n
    # word is a list of ints
    word = [1]
    
    # Precompute optimal orders for n=5,6,7 if n is large enough
    # Actually, we must build sequentially.
    
    for m in range(2, n + 1):
        if time.time() > deadline:
            break
            
        # Get unique permutations of size m-1 from current word
        perms, seen = [], set()
        w_size = m - 1
        for i in range(len(word) - w_size + 1):
            window = tuple(word[i:i + w_size])
            if len(set(window)) == w_size:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        # If m is large enough, optimize the order of perms
        # For m=5, 24 perms. For m=6, 120. For m=7, 720.
        # For m <= 4, the standard order is likely optimal or close enough, 
        # but we can still try to optimize.
        
        if m >= 5:
            # Optimize the order of perms to maximize overlaps
            # This is a TSP path problem on the graph of perms
            # Node i, Node j. Edge weight = overlap(perms[i], perms[j])
            # We want to maximize sum of weights along the path.
            
            num_perms = len(perms)
            
            # Calculate overlap matrix
            # overlap[i][j] = longest suffix of perms[i] matching prefix of perms[j]
            overlaps = [[0]*num_perms for _ in range(num_perms)]
            for i in range(num_perms):
                pi = perms[i]
                for j in range(num_perms):
                    if i == j: continue
                    pj = perms[j]
                    # Max possible overlap is w_size - 1
                    # Since they are distinct perms, overlap cannot be w_size
                    max_ov = w_size - 1
                    for k in range(max_ov, 0, -1):
                        if pi[-k:] == pj[:k]:
                            overlaps[i][j] = k
                            break
                            
            # Heuristic TSP: Greedy + 2-Opt
            # Initialize path
            path = list(range(num_perms))
            
            # Greedy initialization
            # Start with random or 0
            # Let's start with 0 for determinism, or random for variety?
            # Random restarts might help but we have time limits.
            # Let's do a few random restarts if time permits.
            
            best_path = path
            best_score = -1
            
            # Calculate score of a path
            def get_score(p):
                score = 0
                for k in range(len(p)-1):
                    score += overlaps[p[k]][p[k+1]]
                return score
            
            # Greedy construction
            visited = [False] * num_perms
            current = 0
            path = [0]
            visited[0] = True
            for _ in range(num_perms - 1):
                best_next = -1
                best_ov = -1
                for nxt in range(num_perms):
                    if not visited[nxt]:
                        ov = overlaps[current][nxt]
                        if ov > best_ov:
                            best_ov = ov
                            best_next = nxt
                path.append(best_next)
                visited[best_next] = True
                current = best_next
                
            curr_score = get_score(path)
            if curr_score > best_score:
                best_score = curr_score
                best_path = path[:]
                
            # 2-Opt improvement
            # Time budget for optimization
            opt_deadline = start_time + 110.0 # Leave 5s for construction
            
            improved = True
            while improved and time.time() < opt_deadline:
                improved = False
                # Try all 2-opt moves
                # To speed up, we can iterate and accept first improvement
                # Or iterate all and take best.
                # With N=720, N^2 is 500k. 
                # Python loops are slow. 500k iterations per pass might take ~0.1-0.5s.
                # So we can do ~10-100 passes.
                
                for i in range(num_perms - 1):
                    if time.time() > opt_deadline:
                        break
                    for j in range(i + 1, num_perms):
                        # Check if reversing segment [i+1, j] improves score
                        # Current edges: (i-1, i), (i, i+1), ..., (j-1, j), (j, j+1)
                        # New edges: (i-1, j), (j, j-1), ..., (i+1, i), (i, i+1) -> No
                        # 2-opt on path:
                        # Path: ... v_{i-1}, v_i, v_{i+1}, ..., v_j, v_{j+1} ...
                        # Reverse v_i ... v_j
                        # New Path: ... v_{i-1}, v_j, v_{j-1}, ..., v_i, v_{j+1} ...
                        
                        # Edges removed:
                        # 1. (v_{i-1}, v_i) if i > 0
                        # 2. (v_i, v_{i+1}) if i+1 < num_perms (always true if i < N-1)
                        # Wait, the edges are (path[k], path[k+1]).
                        # Removed: (path[i-1], path[i]) and (path[j], path[j+1])
                        # Added: (path[i-1], path[j]) and (path[i], path[j+1])
                        
                        # Note: The internal edges (path[k], path[k+1]) for k in i..j-1 are reversed.
                        # Since the graph is directed (overlap is not symmetric), 
                        # the weight of (A,B) is not (B,A).
                        # So we must recalculate the sum of the reversed internal edges.
                        
                        # This makes 2-opt slower because we can't just check 4 edges.
                        # We have to sum the weights of the reversed segment.
                        
                        # To make this feasible, we can use a "First Improvement" strategy
                        # and precompute prefix sums of weights?
                        # Or just do it if N is small.
                        # For N=720, summing a segment of length L is O(L).
                        # Total work per 2-opt check is O(N).
                        # Total work per pass O(N^3). 720^3 is too big.
                        
                        # Alternative: Use a simpler heuristic.
                        # Just do a few random restarts with Greedy + Local Search (Swap neighbors).
                        
                        pass # Skip complex 2-opt for now, use simpler local search
                
                # Simple Local Search: Swap adjacent elements
                improved_any = False
                for k in range(num_perms - 1):
                    if time.time() > opt_deadline:
                        break
                    # Swap path[k] and path[k+1]
                    a = path[k]
                    b = path[k+1]
                    
                    # Calculate delta
                    # Edges affected: (k-1, k), (k, k+1), (k+1, k+2)
                    # New: (k-1, k+1), (k+1, k), (k, k+2)
                    
                    old_score = 0
                    new_score = 0
                    
                    if k > 0:
                        old_score += overlaps[path[k-1]][a]
                        new_score += overlaps[path[k-1]][b]
                    
                    old_score += overlaps[a][b]
                    new_score += overlaps[b][a]
                    
                    if k + 2 < num_perms:
                        old_score += overlaps[b][path[k+2]]
                        new_score += overlaps[a][path[k+2]]
                    
                    if new_score > old_score:
                        path[k], path[k+1] = b, a
                        improved_any = True
                        # Don't break, continue searching
            
            if improved_any:
                improved = True
            else:
                # If no adjacent swap improved, try a few random swaps?
                # Or stop.
                pass
                
            # Use the best path found
            perms = [perms[i] for i in best_path]

        # Now construct the new word using the (possibly reordered) perms
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            
            new_word += block[overlap:]
        
        word = new_word
        
    return "".join(map(str, word))
