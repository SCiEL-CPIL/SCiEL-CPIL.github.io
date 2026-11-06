import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s kill
    
    n = 16
    
    # Precompute all 8-vertex subsets
    subsets_8 = list(itertools.combinations(range(n), 8))
    
    # For fast Q3 detection: Q3 is the 3-dimensional hypercube.
    # Two vertices in Q3 are adjacent iff their labels (3-bit) differ in exactly 1 bit.
    # We need to check if an 8-vertex induced subgraph contains Q3 as a subgraph.
    # Key insight: Q3 is bipartite with parts of size 4 (even/odd parity vertices).
    # For a candidate 8-set, we can try all 4! * 4! = 576 ways to assign the 
    # 8 vertices to the 8 positions of Q3, but that's too many.
    # Better: Precompute all 8! / 48 = 3360 distinct Q3 labelings up to automorphism.
    # Actually, let's use a different approach: 
    # Q3 has 12 edges. For a given 8-vertex set, if it contains Q3, then there exists
    # a bijection from the 8 vertices to {0,...,7} such that edges in the graph correspond
    # to pairs differing in exactly 1 bit.
    
    # Precompute: for each of the 8 positions in Q3, its neighbors (4 each)
    # pos 0 (000): neighbors 1(001), 2(010), 4(100)
    # pos 1 (001): neighbors 0(000), 3(011), 5(101)
    # pos 2 (010): neighbors 0(000), 3(011), 6(110)
    # pos 3 (011): neighbors 1(001), 2(010), 7(111)
    # pos 4 (100): neighbors 0(000), 5(101), 6(110)
    # pos 5 (101): neighbors 1(001), 4(100), 7(111)
    # pos 6 (110): neighbors 2(010), 4(100), 7(111)
    # pos 7 (111): neighbors 3(011), 5(101), 6(110)
    
    # Instead of checking all 40320 permutations, use the bipartite structure.
    # Q3 is bipartite: even parity {0,3,5,6} and odd parity {1,2,4,7}
    # For each 8-vertex set, try all ways to split into two 4-sets (70 ways),
    # then check if there's a perfect matching in the bipartite graph between them
    # that corresponds to Q3's bipartite structure.
    
    # Precompute: for each partition of {0..7} into two 4-sets, the required adjacency pattern
    # Actually, simpler: precompute all valid Q3 edge sets as frozensets of pairs of positions
    # There are only 12 edges in Q3. For a labeling, the edge set is determined.
    
    # Let's precompute all 8!/(48) = 3360 distinct Q3 subgraphs on positions 0-7
    # But that's a lot. Let's use a smarter check.
    
    # Fast check: For an 8-vertex set S, build the adjacency matrix.
    # Then for each of the 70 ways to split S into A (4 verts) and B (4 verts),
    # check if the bipartite graph between A and B contains Q3's bipartite edges.
    # Q3's bipartite edges: even->odd connections.
    # Even positions: {0,3,5,6}, Odd positions: {1,2,4,7}
    # Edges: 0-1, 0-2, 0-4, 3-1, 3-2, 3-7, 5-1, 5-4, 5-7, 6-2, 6-4, 6-7
    
    # For a given split (A,B), we need to check if there exists a bijection 
    # f: A -> {0,3,5,6} and g: B -> {1,2,4,7} such that for every Q3 edge (e,o),
    # the edge (f^{-1}(e), g^{-1}(o)) exists in the graph.
    
    # This is equivalent to: the bipartite adjacency matrix between A and B 
    # (after appropriate relabeling) contains the Q3 bipartite pattern.
    
    # Precompute: the 12 edges of Q3 as (even_idx, odd_idx) pairs where 
    # even_idx in {0,1,2,3} corresponds to {0,3,5,6} and odd_idx in {0,1,2,3} corresponds to {1,2,4,7}
    even_positions = [0, 3, 5, 6]  # indices 0,1,2,3 in even set
    odd_positions = [1, 2, 4, 7]   # indices 0,1,2,3 in odd set
    q3_edges = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,3), (2,0), (2,2), (2,3), (3,1), (3,2), (3,3)]
    q3_edges_set = set(q3_edges)
    
    # Precompute all 70 splits of {0..7} into two 4-sets
    splits = list(itertools.combinations(range(8), 4))
    
    # For each split, precompute the permutations needed
    # For a split (even_set_indices, odd_set_indices), we need to check if there exist
    # permutations pe of {0,1,2,3} and po of {0,1,2,3} such that for all (e,o) in q3_edges,
    # adj[even_set[pe[e]]][odd_set[po[o]]] is True
    
    # Precompute for each split the list of (pe, po) pairs to check - but that's 24*24=576 per split
    # Total: 70 * 576 = 40320 checks per 8-vertex set. Still too many for 12870 sets.
    
    # Better: Use Hall's theorem or bitmask matching.
    # For a given split and permutation pe of the even side, the required edges from 
    # even vertex pe[i] are to odd vertices po[0], po[1], ... according to q3_edges.
    # This is getting complex. Let's just use a fast bitset-based check with early termination.
    
    # Alternative fast approach: 
    # 1. Count edges in the 8-vertex induced subgraph. If < 12, no Q3.
    # 2. If >= 12, do a more detailed check.
    # In practice, most 8-vertex subsets in a sparse graph will have < 12 edges.
    
    # Precompute permutation lists for speed
    perms_4 = list(itertools.permutations(range(4)))
    
    def has_q3(adj_matrix):
        """
        adj_matrix: 16x16 list of lists of bools
        Returns True if the graph contains Q3 as a subgraph.
        """
        for s in subsets_8:
            # Quick edge count with early termination
            ec = 0
            for i in range(8):
                vi = s[i]
                for j in range(i+1, 8):
                    if adj_matrix[vi][s[j]]:
                        ec += 1
                        if ec >= 12:
                            break
                if ec >= 12:
                    break
            if ec < 12:
                continue
            
            # Detailed check: try all 70 splits
            for split in splits:
                even_set = [s[i] for i in split]
                # Build odd_set efficiently
                even_set_set = set(split)
                odd_set = [s[i] for i in range(8) if i not in even_set_set]
                
                # Build 4x4 adjacency matrix between even_set and odd_set
                bipartite = [[adj_matrix[even_set[i]][odd_set[j]] for j in range(4)] for i in range(4)]
                
                # Try all 24 permutations of the odd side (pe = identity)
                for po in perms_4:
                    ok = True
                    for e in range(4):
                        for o in range(4):
                            if (e, o) in q3_edges_set:
                                if not bipartite[e][po[o]]:
                                    ok = False
                                    break
                        if not ok:
                            break
                    if ok:
                        return True
                
                # Try all 24 permutations of the even side (po = identity)
                for pe in perms_4:
                    ok = True
                    for e in range(4):
                        for o in range(4):
                            if (e, o) in q3_edges_set:
                                if not bipartite[pe[e]][o]:
                                    ok = False
                                    break
                        if not ok:
                            break
                    if ok:
                        return True
        return False
    
    # Initial graph: Try a known Q3-free construction
    # Start with complete graph minus some edges, or use a known dense Q3-free graph
    # For n=16, a good starting point might be the complete bipartite K_{8,8} which has 64 edges
    # but K_{8,8} contains Q3? Let's think... Q3 is bipartite with parts of 4, so K_{8,8} 
    # contains many Q3 subgraphs. So K_{8,8} is NOT Q3-free.
    
    # Better starting point: Turan-type or other constructions.
    # Let's start with a random graph and use local search.
    
    adj = [[False]*n for _ in range(n)]
    current_edges = set()
    
    # Start with a moderately dense random graph
    random.seed(42)
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.5:
                adj[i][j] = True
                adj[j][i] = True
                current_edges.add((i, j))
    
    # Remove any Q3 subgraphs by removing edges
    changed = True
    while changed and time.time() < deadline - 30:
        changed = False
        if has_q3(adj):
            # Find a Q3 and remove an edge from it
            # For simplicity, just remove a random edge and recheck
            edge_list = list(current_edges)
            if edge_list:
                rem_u, rem_v = random.choice(edge_list)
                adj[rem_u][rem_v] = False
                adj[rem_v][rem_u] = False
                current_edges.remove((rem_u, rem_v))
                changed = True
    
    best_edges = set(current_edges)
    best_count = len(current_edges)
    
    # Local search: try adding edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    for restart in range(5):
        if time.time() > deadline - 15:
            break
        
        # Work from current best
        adj = [[False]*n for _ in range(n)]
        current_edges = set(best_edges)
        for u, v in best_edges:
            adj[u][v] = True
            adj[v][u] = True
        
        non_edges = [e for e in all_edges if e not in current_edges]
        random.shuffle(non_edges)
        
        # Greedy addition
        for u, v in non_edges:
            if time.time() > deadline - 5:
                break
            if (u, v) in current_edges:
                continue
            adj[u][v] = True
            adj[v][u] = True
            current_edges.add((u, v))
            
            if has_q3(adj):
                adj[u][v] = False
                adj[v][u] = False
                current_edges.remove((u, v))
            else:
                if len(current_edges) > best_count:
                    best_count = len(current_edges)
                    best_edges = set(current_edges)
        
        # 2-opt: remove one edge, add up to two
        if time.time() < deadline - 10:
            rem_u, rem_v = random.choice(list(current_edges))
            adj[rem_u][rem_v] = False
            adj[rem_v][rem_u] = False
            current_edges.remove((rem_u, rem_v))
            
            non_edges_2 = [e for e in all_edges if e not in current_edges]
            random.shuffle(non_edges_2)
            added = 0
            for u, v in non_edges_2:
                if added >= 2 or time.time() > deadline - 5:
                    break
                adj[u][v] = True
                adj[v][u] = True
                current_edges.add((u, v))
                if has_q3(adj):
                    adj[u][v] = False
                    adj[v][u] = False
                    current_edges.remove((u, v))
                else:
                    added += 1
                    if len(current_edges) > best_count:
                        best_count = len(current_edges)
                        best_edges = set(current_edges)
        
        random.shuffle(non_edges)
    
    result = [[u, v] for u, v in best_edges]
    return result

if __name__ == "__main__":
    start = time.time()
    edges = construct()
    print(f"Found {len(edges)} edges in {time.time() - start:.2f}s")
