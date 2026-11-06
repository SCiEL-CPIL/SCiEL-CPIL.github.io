import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Set recursion limit high just in case, though we use iterative/bounded DFS
    sys.setrecursionlimit(10000)

    def is_triangle_free_circulant(n, S):
        """
        Check if the circulant graph defined by difference set S is triangle-free.
        This is equivalent to checking if S is sum-free modulo n.
        """
        # S should be symmetric and not contain 0 or n/2 (if n even, n/2 is self-inverse)
        # Actually, for triangle-free, we need (a+b)%n not in S for all a,b in S.
        S_list = list(S)
        for a in S_list:
            for b in S_list:
                if (a + b) % n in S:
                    return False
        return True

    def has_independent_set_10(adj, n):
        """
        Check if the graph with adjacency bitmasks 'adj' has an independent set of size 10.
        Returns True if such a set exists.
        """
        mask_all = (1 << n) - 1
        
        # Quick greedy lower bound
        candidates = mask_all
        size = 0
        while candidates and size < 10:
            best_v = -1
            min_deg = n + 1
            temp = candidates
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                deg = (adj[v] & candidates).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
            
            if best_v == -1:
                break
            size += 1
            candidates &= ~adj[best_v] & ~(1 << best_v)
        
        if size >= 10:
            return True

        # Exact branch and bound
        def backtrack(candidates, size):
            if size == 10:
                return True
            if candidates == 0:
                return False
            
            remaining = candidates.bit_count()
            if size + remaining < 10:
                return False
            
            # Pick vertex with max degree in candidates to prune effectively
            best_v = -1
            max_deg = -1
            temp = candidates
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                deg = (adj[v] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
            
            if best_v == -1:
                return False
            
            # Branch 1: Include best_v in independent set
            # Neighbors of best_v cannot be in the set
            if backtrack(candidates & ~adj[best_v] & ~(1 << best_v), size + 1):
                return True
            
            # Branch 2: Exclude best_v
            if backtrack(candidates & ~(1 << best_v), size):
                return True
            
            return False

        return backtrack(mask_all, 0)

    def build_graph_from_S(n, S):
        """
        Build adjacency bitmasks and edge list for circulant graph with difference set S.
        """
        adj = [0] * n
        for i in range(n):
            for d in S:
                j = (i + d) % n
                if i != j:
                    adj[i] |= (1 << j)
        
        # Ensure symmetry in representation (though circulant is inherently symmetric)
        # adj[i] has bit j set iff adj[j] has bit i set for circulant
        # But let's verify and clean up
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
                    # Ensure adjacency is symmetric in the bitmask for safety
                    adj[j] |= (1 << i)
        return adj, edges

    def find_circulant_graph(n):
        """
        Search for a circulant graph on n vertices that is triangle-free and has alpha < 10.
        """
        if time.time() > deadline:
            return None
        
        # We want a symmetric sum-free set S.
        # S subset of {1, ..., n-1}. If d in S, n-d in S.
        # We can just define D subset of {1, ..., n//2} and S = D union {n-d for d in D}.
        # Note: if n is even, n/2 is its own inverse.
        
        max_d = n // 2
        candidates_base = list(range(1, max_d + 1))
        
        # Try multiple random orders to find a good maximal sum-free set
        # We want S to be large to minimize alpha.
        
        for attempt in range(500):
            if time.time() > deadline:
                return None
            
            random.shuffle(candidates_base)
            D = set()
            
            # Greedily add elements to D
            for d in candidates_base:
                # Check if adding d (and n-d) keeps the set sum-free
                new_D = set(D)
                new_D.add(d)
                if n % 2 == 0 and d == n//2:
                    S_new = set(new_D)
                else:
                    S_new = set(new_D)
                    S_new.add(n - d)
                
                # Check sum-free property for S_new
                valid = True
                S_list = list(S_new)
                # Optimization: only check sums involving new elements?
                # Actually, if S_old was sum-free, we only need to check:
                # 1. new + new not in S_new
                # 2. new + old not in S_new
                # But a full check is O(|S|^2) which is small (max 40^2 = 1600).
                for a in S_list:
                    for b in S_list:
                        if (a + b) % n in S_new:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    D = new_D
            
            # Construct S from D
            S = set(D)
            for d in D:
                if d != n - d:
                    S.add(n - d)
            
            # Check if S is empty (trivial graph, alpha=n, bad)
            if not S:
                continue
                
            # Build graph
            adj, edges = build_graph_from_S(n, S)
            
            # Check triangle-free (should be true by construction, but verify)
            # is_triangle_free_circulant is redundant if we built S correctly, 
            # but let's assume the construction is correct.
            
            # Check independence number
            if not has_independent_set_10(adj, n):
                return {'n': n, 'edges': edges}
        
        return None

    # Iterate n downwards from 40 to 25
    # R(3,10) <= 41, so max valid n is 40.
    # Known best is 39. We aim for 40, then 39, etc.
    
    # To improve success rate, we might want to try specific structures first.
    # But random greedy on sum-free sets is a strong baseline.
    
    for n in range(40, 24, -1):
        if time.time() > deadline:
            break
            
        result = find_circulant_graph(n)
        if result is not None:
            return result
            
    # Fallback: Return a smaller known valid graph if we run out of time
    # A circulant graph on n=26 with specific S might work.
    # Or just return a small valid graph to ensure score > 0.
    # Let's try n=20 with a known good S or just a random one that works.
    
    # Simple fallback: Complete bipartite graph K_{10,10} is triangle-free, alpha=10 (fail).
    # We need alpha < 10.
    # K_{9,9} on 18 vertices: alpha=9. Valid.
    n = 18
    edges = []
    for i in range(9):
        for j in range(9, 18):
            edges.append([i, j])
    
    return {'n': n, 'edges': edges}

if __name__ == "__main__":
    # Self-test
    res = construct()
    print(f"Result: n={res['n']}, edges={len(res['edges'])}")
