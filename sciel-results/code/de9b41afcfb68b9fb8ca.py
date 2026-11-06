import time
import random
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s limit
    
    # Strategy:
    # 1. Hardcode a known valid n=17 coloring if we can construct one.
    #    Since we cannot easily verify a hardcoded n=17, we will attempt a fast search for n=12, 13, etc.
    #    Actually, the best known is 17. R_4(C6) > 16 is known, >17 is open?
    #    Wait, the prompt says "The best known colouring has 17 vertices".
    #    So we should aim for n=17.
    #
    #    A pure backtracking search for n=17 is likely too slow in Python within 120s.
    #    We need a smarter approach or a hardcoded valid coloring.
    #
    #    Let's try to construct a valid n=17 coloring using a heuristic or local search.
    #    Or, we can use the algebraic construction if one exists.
    #    
    #    Known construction for R_4(C6):
    #    There is a construction based on finite fields or groups.
    #    However, implementing a verified n=17 construction from scratch is risky.
    #
    #    Alternative: Use a random restart local search (simulated annealing or hill climbing) to fix C6s.
    #    Start with a random coloring for n=17.
    #    Objective: Minimize the number of C6s in all color classes.
    #    If we reach 0, we have a valid coloring.
    #    
    #    Counting C6s is the bottleneck.
    #    For a fixed color c, the number of C6s can be estimated or counted using adjacency matrices.
    #    A cycle v0-v1-...-v5-v0 is a C6.
    #    We can count the number of paths of length 5 between i and j, but that's not exactly C6 count.
    #    Actually, we just need to detect if ANY C6 exists.
    #    
    #    Let's implement a fast C6 detector.
    #    For each color c, maintain the adjacency matrix (as bitsets).
    #    To check if a C6 exists, we can iterate over pairs of non-adjacent vertices (i,j) and check if there is a path of length 5?
    #    No, the cycle must be closed.
    #    
    #    Let's use a different approach:
    #    Randomly permute vertices? No.
    #    
    #    Let's try a backtracking search with strong pruning for n=12 first, then try to extend?
    #    Or just use the known n=9 seed and try to extend to n=10, 11...
    #    
    #    Actually, let's look at the "inspiration" or known results.
    #    If I can't find n=17, I should at least try to beat n=9.
    #    n=10, 11, 12 are likely achievable with fast backtracking + bitsets.
    #    
    #    Let's implement a fast backtracking solver for n up to 12 or 13.
    #    Key optimization:
    #    - Use bitsets for adjacency.
    #    - Precompute "forbidden" conditions?
    #    - When adding edge (u,v) with color c, check if it creates a C6.
    #    A C6 is created if there exists a path u-w1-w2-w3-w4-v of length 5 in color c in the existing graph.
    #    We can check this by computing the set of vertices reachable from u in exactly 5 steps in color c,
    #    and checking if v is in that set.
    #    However, we must ensure the path is simple (no repeated vertices).
    #    For small n and sparse graphs, walks might approximate paths, but for correctness we need simple paths.
    #    But wait: if there is a walk of length 5, it might not be a simple path.
    #    However, if the graph is simple and we are building it up, can we use a DFS check?
    #    DFS for path of length 5 is O(n^5) worst case, but with bitsets and pruning it's fast.
    #    
    #    Let's implement a helper: has_simple_path_len_5(u, v, c, adj_c)
    #    Using recursion with memoization or just DFS with bitsets for visited.
    #    
    #    To make it faster, we can precompute reachability sets for each step?
    #    R_0 = {u}
    #    R_1 = N(u)
    #    R_2 = N(R_1) \ R_0
    #    R_3 = N(R_2) \ (R_0 ∪ R_1)
    #    R_4 = N(R_3) \ (R_0 ∪ R_1 ∪ R_2)
    #    R_5 = N(R_4) \ (R_0 ∪ ... ∪ R_4)
    #    If v is in R_5, there is a simple path of length 5?
    #    This iterative set expansion gives vertices reachable by a *walk* that doesn't immediately backtrack?
    #    Actually, this "frontier" expansion ensures that the path is simple in terms of not revisiting vertices in the *set* of previous layers.
    #    Does it guarantee a simple path?
    #    Yes, if we define R_k as vertices reachable from u by a simple path of length k, this recurrence works IF we track the exact history, which we don't.
    #    However, for small n, the probability of false positive (walk exists but no simple path) is low, and we can verify with DFS if needed.
    #    But for correctness, we must be sure.
    #    
    #    Let's use a precise DFS with bitsets for the visited set, but optimized.
    #    Given the time limit, let's try to solve n=12.
    
    def solve(n, time_limit):
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        m = len(edges)
        colors = [-1] * m
        adj = [[0] * n for _ in range(4)]
        
        # Precompute neighbors for each color? No, adj is dynamic.
        
        def has_path_5(u, v, c):
            # Check if there is a simple path of length 5 from u to v in color c
            # DFS with pruning
            # visited is a bitmask
            # path length remaining
            # We use a recursive function with bitset visited.
            
            def dfs(cur, target, depth, visited):
                if depth == 0:
                    return cur == target
                if depth == 1:
                    # Check if target is neighbor of cur and not visited
                    return (adj[c][cur] & (1 << target)) and not (visited & (1 << target))
                
                nexts = adj[c][cur] & ~visited
                # Optimization: order neighbors by degree? Or just iterate.
                # For speed, we can check if target is in the "reachable in depth-1" set?
                # Let's just iterate.
                res = False
                while nexts:
                    lsb = nexts & -nexts
                    w = lsb.bit_length() - 1
                    nexts ^= lsb
                    if dfs(w, target, depth - 1, visited | lsb):
                        return True
                return False
            
            return dfs(u, v, 5, 1 << u)

        def backtrack(idx):
            if time.time() > deadline - 2:
                return False
            
            if idx == m:
                return True
            
            u, v = edges[idx]
            
            # Heuristic: Try colors that are less constrained?
            # Or just 0,1,2,3.
            for c in range(4):
                if not has_path_5(u, v, c):
                    adj[c][u] |= (1 << v)
                    adj[c][v] |= (1 << u)
                    colors[idx] = c
                    
                    if backtrack(idx + 1):
                        return True
                    
                    adj[c][u] &= ~(1 << v)
                    adj[c][v] &= ~(1 << u)
                    colors[idx] = -1
            
            return False

        if backtrack(0):
            return {'n': n, 'colors': colors}
        return None

    # Try n=12 first
    # If n=12 is too slow, try n=11, etc.
    # n=12 has 66 edges. 4^66 is huge, but pruning is strong.
    # Let's try n=10 first to ensure we beat the seed (n=9).
    
    best_result = {'n': 9, 'colors': [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]}
    
    for n_try in [12, 11, 10]:
        if time.time() > deadline - 10:
            break
        res = solve(n_try, deadline)
        if res:
            best_result = res
            break # We want the largest n, so we try from high to low.
            # Actually, if we find n=12, we are done.
            # If we fail n=12, we try n=11.
            # The loop above does this.
    
    return best_result
