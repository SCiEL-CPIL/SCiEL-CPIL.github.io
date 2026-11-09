import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Strategy: Greedy removal from K16 to eliminate all Q3 subgraphs
    # K16 has 120 edges. We remove edges to break all Q3 subgraphs.
    
    # Precompute all Q3 subgraphs in K16
    # A Q3 is a 3-cube: 8 vertices, 12 edges, each vertex degree 3 in the subgraph
    # We need to find all sets of 8 vertices that can form a Q3.
    
    # For efficiency, we'll use the property that Q3 is bipartite with parts of size 4.
    # For each pair of 4-sets (A, B) with no common vertices, check if they can form Q3.
    
    # But that's too slow. Instead, let's use a direct approach:
    # Enumerate all 8-vertex subsets (C(16,8) = 12870) and for each, check if it contains Q3.
    # For an 8-vertex subset, check all 70 ways to split into two 4-sets, and for each split,
    # check if the bipartite graph contains Q3.
    
    # To speed up, we'll precompute for each 8-vertex subset whether it contains Q3.
    
    all_8_subsets = list(itertools.combinations(range(16), 8))
    
    # Precompute permutations of 4 elements
    perms4 = list(itertools.permutations(range(4)))
    combos_4_of_8 = list(itertools.combinations(range(8), 4))
    
    def has_q3_in_subset(subset):
        """Check if the 8-vertex subset contains Q3 as a subgraph."""
        s = list(subset)
        for (i, j, k, l) in combos_4_of_8:
            A = [s[i], s[j], s[k], s[l]]
            B = [s[x] for x in range(8) if x not in (i, j, k, l)]
            
            # Check orientation 1: A = even parity, B = odd parity
            # Required edges (using the fixed labeling from before):
            # a0(000) -> b0(001), b1(010), b3(100)
            # a1(011) -> b0(001), b1(010), b2(111)
            # a2(101) -> b0(001), b2(111), b3(100)
            # a3(110) -> b1(010), b2(111), b3(100)
            for p in perms4:
                # p maps position in odd-parity label to index in B
                # Required: 
                # A[0] connected to B[p[0]], B[p[1]], B[p[3]]
                # A[1] connected to B[p[0]], B[p[1]], B[p[2]]
                # A[2] connected to B[p[0]], B[p[2]], B[p[3]]
                # A[3] connected to B[p[1]], B[p[2]], B[p[3]]
                # In K16, all these edges exist, so this is always true.
                return True
            
            # Since we're checking K16, all bipartite edges exist, so if the
            # combinatorial structure allows Q3, it's there.
            # The structure is: we need a bijection such that the pattern matches.
            # For any 4-4 split, there exists such a bijection iff the bipartite graph
            # is complete (which it is in K16). So every 8-vertex subset in K16 contains Q3.
            return True
        
        return False
    
    # Wait - in K16, EVERY 8-vertex subset contains Q3 as a subgraph because K16 is complete.
    # So we need to remove edges to break ALL Q3 subgraphs.
    # This is a hitting set problem: find the minimum number of edges to remove such that
    # every Q3 has at least one edge removed.
    
    # Since C(16,8) = 12870 and each Q3 has 12 edges, we have 12870 constraints.
    # Each edge removal breaks multiple Q3s.
    
    # Greedy: repeatedly remove the edge that is in the most remaining Q3s.
    
    # First, precompute for each edge, how many Q3s it belongs to.
    # An edge (u,v) is in a Q3 iff there exists a Q3 containing both u and v.
    
    # For edge (u,v), count the number of 8-vertex subsets containing u and v that have Q3.
    # Since all 8-subsets in K16 have Q3, the count is C(14,6) = 3003 for each edge.
    # So all edges are symmetric. Any removal sequence of the same length gives the same count.
    
    # But after removal, the structure changes. We need to track which Q3s are still "alive".
    
    # Let's use a different approach: random greedy with verification.
    
    # Actually, let me think about this differently. The known extremal number ex(n, Q3) 
    # grows like n^(3/2) * log(n) or similar. For n=16, a good construction might be
    # around 40-60 edges.
    
    # Let me try a random approach: start with K16, randomly remove edges until Q3-free.
    
    import random
    random.seed(42)
    
    def count_q3(edges_set):
        """Count the number of Q3 subgraphs in the graph."""
        count = 0
        for subset in all_8_subsets:
            if has_q3_in_subset(edges_set, subset):
                count += 1
        return count
    
    def has_q3_in_subset(edges_set, subset):
        """Check if the induced subgraph on subset contains Q3."""
        s = list(subset)
        for (i, j, k, l) in combos_4_of_8:
            A = [s[i], s[j], s[k], s[l]]
            B = [s[x] for x in range(8) if x not in (i, j, k, l)]
            
            # Check all permutations for orientation A=even
            for p in perms4:
                # Required edges:
                req = [
                    (A[0], B[p[0]]), (A[0], B[p[1]]), (A[0], B[p[3]]),
                    (A[1], B[p[0]]), (A[1], B[p[1]]), (A[1], B[p[2]]),
                    (A[2], B[p[0]]), (A[2], B[p[2]]), (A[2], B[p[3]]),
                    (A[3], B[p[1]]), (A[3], B[p[2]]), (A[3], B[p[3]])
                ]
                if all((min(u,v), max(u,v)) in edges_set for u, v in req):
                    return True
            
            # Check all permutations for orientation A=odd
            for p in perms4:
                req = [
                    (B[0], A[p[0]]), (B[0], A[p[1]]), (B[0], A[p[3]]),
                    (B[1], A[p[0]]), (B[1], A[p[1]]), (B[1], A[p[2]]),
                    (B[2], A[p[0]]), (B[2], A[p[2]]), (B[2], A[p[3]]),
                    (B[3], A[p[1]]), (B[3], A[p[2]]), (B[3], A[p[3]])
                ]
                if all((min(u,v), max(u,v)) in edges_set for u, v in req):
                    return True
        
        return False
    
    # Start with K16
    edges_set = set()
    for u in range(16):
        for v in range(u+1, 16):
            edges_set.add((u, v))
    
    # Greedy removal: repeatedly remove the edge in the most Q3s
    # For efficiency, track which Q3s are still active
    
    # Precompute: for each 8-subset, the 12 edges of each Q3 it contains
    # This is too much. Instead, use a simpler heuristic.
    
    # Random restart: try multiple random removal orders
    best_edges = None
    best_count = 0
    
    for trial in range(50):
        if time.time() > deadline - 15:
            break
        
        random.seed(trial)
        edges_set = set()
        for u in range(16):
            for v in range(u+1, 16):
                edges_set.add((u, v))
        
        # Randomly remove edges until Q3-free
        edge_list = list(edges_set)
        random.shuffle(edge_list)
        
        removed = set()
        for e in edge_list:
            if time.time() > deadline - 15:
                break
            if e in removed:
                continue
            # Try removing e
            edges_set.remove(e)
            removed.add(e)
            
            # Check if Q3-free
            is_q3_free = True
            for subset in all_8_subsets:
                if has_q3_in_subset(edges_set, subset):
                    is_q3_free = False
                    break
            
            if is_q3_free:
                if len(edges_set) > best_count:
                    best_count = len(edges_set)
                    best_edges = list(edges_set)
                break
            else:
                # Restore e
                edges_set.add(e)
                removed.discard(e)
    
    if best_edges is not None:
        return [list(e) for e in best_edges]
    
    # Fallback: return empty graph
    return []

BET: {mechanism: "Greedy random removal from K16 with multiple trials finds a sparse Q3-free subgraph; the random restarts explore different removal orders to maximize retained edges.", q: 0.6, delta: [0.0, 0.1]}
