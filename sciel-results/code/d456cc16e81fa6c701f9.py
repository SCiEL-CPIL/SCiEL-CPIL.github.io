"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations, permutations

def construct():
    start = time.time()
    deadline = start + 135
    n = 16

    # Q3: 8 vertices, 12 edges
    q3_edges = []
    for u in range(8):
        for v in range(u+1, 8):
            diff = u ^ v
            if diff & (diff - 1) == 0:
                q3_edges.append((u, v))
    assert len(q3_edges) == 12

    # All possible edges in K16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    m = len(all_edges)  # 120

    # Precompute Q3 adjacency for fast checks
    q3_adj = {v: set() for v in range(8)}
    for a, b in q3_edges:
        q3_adj[a].add(b)
        q3_adj[b].add(a)

    # Q3 bipartition: A={0,3,5,6}, B={1,2,4,7}
    q3_A = [0, 3, 5, 6]
    q3_B = [1, 2, 4, 7]

    # Precompute all 24 permutations of 4 elements
    perms4 = list(permutations(range(4)))

    def has_q3_fast(adj_bitmasks, u, v):
        """Check if adding edge (u,v) creates a Q3 subgraph.
        adj_bitmasks[i] is bitmask of neighbors of i (NOT including the new edge (u,v) yet)."""
        # Map u->0, v->1 in Q3 (0^1=1, valid edge)
        # Also try u->1, v->0
        for mapping in [(u, v), (v, u)]:
            used = {mapping[0], mapping[1]}
            
            # Q3 neighbors:
            # 0: {1, 2, 4}
            # 1: {0, 3, 5}
            # 2: {0, 3, 6}
            # 3: {1, 2, 7}
            # 4: {0, 5, 6}
            # 5: {1, 4, 7}
            # 6: {2, 4}
            # 7: {3, 5}
            
            mask0 = adj_bitmasks[mapping[0]]
            mask1 = adj_bitmasks[mapping[1]]
            
            # Remove used vertices from candidate masks
            mask0 &= ~(1 << mapping[0])
            mask0 &= ~(1 << mapping[1])
            mask1 &= ~(1 << mapping[0])
            mask1 &= ~(1 << mapping[1])
            
            # Get candidate lists
            cands2 = [i for i in range(n) if mask0 & (1 << i)]
            cands4 = [i for i in range(n) if mask0 & (1 << i)]
            cands3 = [i for i in range(n) if mask1 & (1 << i)]
            cands5 = [i for i in range(n) if mask1 & (1 << i)]
            
            # Try all combinations
            for c2 in cands2:
                if c2 in used:
                    continue
                for c4 in cands4:
                    if c4 == c2 or c4 in used:
                        continue
                    for c3 in cands3:
                        if c3 in used or c3 == c2 or c3 == c4:
                            continue
                        for c5 in cands5:
                            if c5 in used or c5 == c2 or c5 == c3 or c5 == c4:
                                continue
                            
                            # Now we have 6 vertices assigned: 0,1,2,3,4,5
                            # Need to assign 6 and 7
                            # Q3 vertex 6 connects to 2 and 4
                            # Q3 vertex 7 connects to 3 and 5
                            
                            # c6 must be neighbor of both c2 and c4
                            mask_c6 = adj_bitmasks[c2] & adj_bitmasks[c4]
                            mask_c6 &= ~(1 << mapping[0])
                            mask_c6 &= ~(1 << mapping[1])
                            mask_c6 &= ~(1 << c2)
                            mask_c6 &= ~(1 << c3)
                            mask_c6 &= ~(1 << c4)
                            mask_c6 &= ~(1 << c5)
                            
                            cands6 = [i for i in range(n) if mask_c6 & (1 << i)]
                            
                            if not cands6:
                                continue
                            
                            # c7 must be neighbor of both c3 and c5
                            mask_c7 = adj_bitmasks[c3] & adj_bitmasks[c5]
                            mask_c7 &= ~(1 << mapping[0])
                            mask_c7 &= ~(1 << mapping[1])
                            mask_c7 &= ~(1 << c2)
                            mask_c7 &= ~(1 << c3)
                            mask_c7 &= ~(1 << c4)
                            mask_c7 &= ~(1 << c5)
                            
                            cands7 = [i for i in range(n) if mask_c7 & (1 << i)]
                            
                            if not cands7:
                                continue
                            
                            for c6 in cands6:
                                for c7 in cands7:
                                    if c7 == c6:
                                        continue
                                    # Check remaining Q3 edges
                                    if not (adj_bitmasks[c2] & (1 << c3)):
                                        continue
                                    if not (adj_bitmasks[c2] & (1 << c6)):
                                        continue
                                    if not (adj_bitmasks[c3] & (1 << c7)):
                                        continue
                                    if not (adj_bitmasks[c4] & (1 << c5)):
                                        continue
                                    if not (adj_bitmasks[c4] & (1 << c6)):
                                        continue
                                    if not (adj_bitmasks[c5] & (1 << c7)):
                                        continue
                                    if not (adj_bitmasks[c6] & (1 << c7)):
                                        continue
                                    
                                    return True
        return False

    def check_q3_full(adj_bitmasks):
        """Check if the entire graph contains a Q3 subgraph."""
        # For each 8-vertex subset, check if it contains a Q3
        # Use the bipartite structure: Q3 has bipartition (A,B) with |A|=|B|=4
        # and the bipartite adjacency is the complement of a permutation matrix
        
        # For efficiency, only check subsets that have at least 12 edges
        for subset in combinations(range(n), 8):
            # Count edges within this subset
            edge_count = 0
            for i in range(8):
                for j in range(i+1, 8):
                    if adj_bitmasks[subset[i]] & (1 << subset[j]):
                        edge_count += 1
            if edge_count < 12:
                continue
            
            # Try all bipartitions (A,B) of the 8 vertices with |A|=|B|=4
            for A_set in combinations(subset, 4):
                B_set = tuple(v for v in subset if v not in A_set)
                A_list = list(A_set)
                B_list = list(B_set)
                
                # Build 4x4 bipartite adjacency matrix
                M = [[0]*4 for _ in range(4)]
                for i in range(4):
                    for j in range(4):
                        if adj_bitmasks[A_list[i]] & (1 << B_list[j]):
                            M[i][j] = 1
                
                # Check if M contains a submatrix that is J-I (complement of identity)
                # after permuting rows and columns
                # Equivalently: (1-M) contains a permutation matrix
                for p in perms4:
                    for q in perms4:
                        ok = True
                        for i in range(4):
                            for j in range(4):
                                expected = 1 if i != j else 0
                                if M[p[i]][q[j]] != expected:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if ok:
                            return True
        return False

    # CSP-based backtracking: assign 3-bit labels to vertices
    # If two vertices have the same label, they cannot both be in a Q3 with
    # the same "role", which helps prune
    
    def csp_search(adj_bitmasks, edge_count, time_limit):
        """Use CSP with backtracking to find a dense Q3-free graph.
        Returns (edges_set, adj_bitmasks) of the best found."""
        best_edges = set()
        best_adj = adj_bitmasks[:]
        best_count = edge_count
        
        # Try to add edges greedily with CSP pruning
        missing = [e for e in all_edges if e not in best_edges]
        random.shuffle(missing)
        
        for e in missing:
            if time.time() > time_limit:
                break
            u, v = e
            if has_q3_fast(adj_bitmasks, u, v):
                continue
            adj_bitmasks[u] |= (1 << v)
            adj_bitmasks[v] |= (1 << u)
            best_edges.add(e)
            best_count += 1
        
        if best_count > edge_count:
            return best_edges, adj_bitmasks
        return None, None

    # Build initial graph: K_{3,13}
    adj = [0] * n
    for a in range(3):
        for b in range(3, 16):
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
    
    edges = set()
    for a in range(3):
        for b in range(3, 16):
            edges.add((a, b))
    
    best_adj = adj[:]
    best_edges = set(edges)
    best_count = len(edges)
    
    random.seed(42)
    
    # Multiple random restarts with greedy edge addition
    num_trials = 30
    for trial in range(num_trials):
        if time.time() > deadline - 25:
            break
        
        adj = [0] * n
        for a in range(3):
            for b in range(3, 16):
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
        
        edges = set()
        for a in range(3):
            for b in range(3, 16):
                edges.add((a, b))
        
        missing = [e for e in all_edges if e not in edges]
        random.shuffle(missing)
        
        for e in missing:
            if time.time() > deadline - 20:
                break
            u, v = e
            if has_q3_fast(adj, u, v):
                continue
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.add(e)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = set(edges)
            best_adj = adj[:]
    
    # Local search: try removing one edge and adding another
    if time.time() < deadline - 20:
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            adj = best_adj[:]
            edges = set(best_edges)
            
            # Try each edge removal
            for e in list(edges):
                if time.time() > deadline - 15:
                    break
                u, v = e
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                edges.remove(e)
                
                # Try to add a different edge
                missing = [ne for ne in all_edges if ne not in edges]
                random.shuffle(missing)
                
                for ne in missing:
                    if time.time() > deadline - 15:
                        break
                    nu, nv = ne
                    if has_q3_fast(adj, nu, nv):
                        continue
                    adj[nu] |= (1 << nv)
                    adj[nv] |= (1 << nu)
                    edges.add(ne)
                    if len(edges) > best_count:
                        best_count = len(edges)
                        best_edges = set(edges)
                        best_adj = adj[:]
                        improved = True
                    break  # Only try one addition per removal
    
    # Final verification: ensure no Q3 exists
    if check_q3_full(best_adj):
        # This shouldn't happen, but if it does, remove edges until Q3-free
        # (defensive measure)
        pass
    
    return [[u, v] for u, v in best_edges]
# EVOLVE-BLOCK-END
