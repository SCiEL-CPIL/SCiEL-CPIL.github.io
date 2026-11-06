import time
import sys
from itertools import combinations

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # We'll use a backtracking approach with symmetry breaking.
    # Vertices are processed in order 0, 1, 2, ..., n-1.
    # For vertex i, we decide which of vertices 0..i-1 it connects to.
    # Symmetry breaking: enforce that the adjacency bitmask of vertex i is 
    # lexicographically >= the bitmask of vertex i-1 (when restricted to vertices < i).
    # Actually, a simpler symmetry breaking: 
    # We require that vertex 0 is connected to the maximum possible set first,
    # but that's not quite right.
    
    # Better symmetry breaking for K2,3-free graphs:
    # Process vertices in order. For vertex i, its neighbors among {0,...,i-1} 
    # must form a set that is "larger" in some sense than vertex i-1's neighbors.
    
    # Actually, let's use a different approach: 
    # Since n=20 is small, we can use a smart backtracking with strong pruning.
    # Key insight: In a K2,3-free graph, for any pair of vertices (u,v), 
    # |N(u) ∩ N(v)| <= 2.
    
    # We'll build the graph vertex by vertex.
    # State: adj[i] for i < current_vertex, and we decide adj[current_vertex] 
    # as a subset of {0, ..., current_vertex-1}.
    
    # To break symmetry, we enforce:
    # adj[i] (as a bitmask of neighbors in {0,...,i-1}) must be >= adj[i-1] 
    # in lexicographic order? No, that's not standard.
    
    # Standard symmetry breaking for graph generation: 
    # The adjacency matrix is upper triangular. We enforce that row i 
    # (restricted to columns 0..i-1) is lexicographically >= row i-1.
    
    # Let's implement this.
    
    best_edges = []
    best_count = 0
    
    def can_add_vertex(adj_masks, new_mask, i):
        """Check if adding vertex i with neighbor mask new_mask (subset of {0..i-1}) 
        maintains K2,3-free property."""
        # For each existing vertex j < i:
        # If j is in new_mask (i.e., edge (i,j) exists), then for each k < j:
        #   if k is in new_mask and k is in adj_masks[j], 
        #   then |N(i) ∩ N(k)| must be <= 2 and |N(j) ∩ N(k)| must be <= 2.
        # But |N(j) ∩ N(k)| is already determined by adj_masks.
        # We only need to check that adding i doesn't create a K2,3.
        
        # Adding vertex i with neighbors new_mask:
        # For any pair (a, b) where a, b < i:
        #   If both a and b are in new_mask, then |N(a) ∩ N(b)| increases by 1 (because i is now a common neighbor).
        #   So we need: for all a, b in new_mask, a < b, 
        #     (adj_masks[a] & adj_masks[b]).bit_count() < 3  (was already true)
        #     and after adding i: (adj_masks[a] & adj_masks[b] | (1<<i if i in both... wait no)
        
        # Actually, let me think more carefully.
        # N(a) = adj_masks[a] (neighbors among vertices < a, plus possibly some > a, but we haven't added those yet)
        # Wait, adj_masks[a] only contains neighbors among vertices < a.
        # When we add vertex i, its neighbors are in new_mask (subset of {0..i-1}).
        # For existing vertex a < i:
        #   N(a) now includes i if a is in new_mask.
        #   So N(a) = adj_masks[a] ∪ ({i} if a in new_mask else ∅)
        
        # For a pair (a, b) with a < b < i:
        #   |N(a) ∩ N(b)| = |adj_masks[a] ∩ adj_masks[b]| + [a in new_mask and b in new_mask]
        #   (because i is in N(a) iff a in new_mask, and i is in N(b) iff b in new_mask)
        #   We need this to be <= 2.
        
        # For a pair (a, i) where a < i:
        #   |N(a) ∩ N(i)| = |adj_masks[a] ∩ new_mask|
        #   (N(a) = adj_masks[a] ∪ ({i} if a in new_mask), N(i) = new_mask)
        #   Intersection: vertices in adj_masks[a] ∩ new_mask, plus i if i in adj_masks[a] (no, i > a so i not in adj_masks[a])
        #   Wait, i is not in adj_masks[a] because adj_masks[a] only has vertices < a.
        #   So |N(a) ∩ N(i)| = |adj_masks[a] ∩ new_mask|
        #   We need this to be <= 2.
        
        # So the checks are:
        # 1. For all a in new_mask: |adj_masks[a] ∩ new_mask| <= 2
        # 2. For all a < b, both in new_mask: (adj_masks[a] & adj_masks[b]).bit_count() + 1 <= 2
        #    i.e., (adj_masks[a] & adj_masks[b]).bit_count() <= 1
        
        # Check 1:
        for a in range(i):
            if new_mask & (1 << a):
                if (adj_masks[a] & new_mask).bit_count() > 2:
                    return False
        
        # Check 2:
        # For all pairs (a, b) in new_mask with a < b:
        #   (adj_masks[a] & adj_masks[b]).bit_count() <= 1
        neighbors = []
        for a in range(i):
            if new_mask & (1 << a):
                neighbors.append(a)
        
        for idx_a in range(len(neighbors)):
            for idx_b in range(idx_a + 1, len(neighbors)):
                a = neighbors[idx_a]
                b = neighbors[idx_b]
                if (adj_masks[a] & adj_masks[b]).bit_count() > 1:
                    return False
        
        return True
    
    def backtrack(adj_masks, i, edge_count):
        """Backtrack to build vertices 0..i-1. Decide mask for vertex i."""
        nonlocal best_edges, best_count
        
        if time.time() > deadline:
            return
        
        if i == n:
            if edge_count > best_count:
                best_count = edge_count
                # Reconstruct edges from adj_masks
                edges = []
                for v in range(n):
                    mask = adj_masks[v]
                    for u in range(v):
                        if mask & (1 << u):
                            edges.append([u, v])
                best_edges = edges
            return
        
        # Determine the set of possible masks for vertex i.
        # Mask is a subset of {0, ..., i-1}.
        # Symmetry breaking: mask must be >= adj_masks[i-1] in some sense.
        # For now, let's try all subsets but prune aggressively.
        
        # To break symmetry, we can enforce that the mask for vertex i 
        # is lexicographically >= the mask for vertex i-1 (when both are viewed 
        # as bit vectors of length i-1, with vertex i-1's mask padded).
        # Actually, a simpler approach: just try all masks but use the pruning.
        
        # Generate all subsets of {0, ..., i-1} in decreasing order of bit count 
        # to find high-edge solutions early.
        
        full_mask = (1 << i) - 1
        
        # Try masks from full to empty, but skip those that fail symmetry breaking.
        # For symmetry breaking: if i > 0, require that the mask for vertex i 
        # is >= the mask for vertex i-1 when both are projected to {0, ..., i-2}.
        # This is a standard symmetry breaking for graph generation.
        
        if i == 0:
            # Vertex 0 has no neighbors among {0..-1}, so mask must be 0.
            new_adj = adj_masks + [0]
            backtrack(new_adj, i + 1, edge_count)
            return
        
        # For vertex i, mask is subset of {0, ..., i-1}.
        # Symmetry breaking: mask & (1 << (i-1)) must be 1 if adj_masks[i-1] has any bits set? 
        # Actually, the standard symmetry breaking is:
        # The i-th row of the adjacency matrix (restricted to columns 0..i-1) 
        # must be lexicographically >= the (i-1)-th row (restricted to columns 0..i-2).
        
        # Let's implement a simpler symmetry breaking:
        # Require that if vertex i-1 has degree d_{i-1} among {0..i-2}, 
        # then vertex i must have degree >= d_{i-1} among {0..i-1}.
        # This is not quite right either.
        
        # Let's just try all masks but in an order that finds good solutions early.
        # Start with dense masks and go to sparse.
        
        # Get all subsets of {0, ..., i-1} sorted by bit count descending.
        # For i up to 20, this could be 2^20 = 1M subsets, which is too many to enumerate.
        # We need a smarter approach.
        
        # Alternative: use a greedy heuristic to find a good initial solution, 
        # then use local search.
        
        # Given the complexity, let's switch to a local search approach with 
        # multiple random restarts.
        
        pass  # We'll replace this with local search below
    
    # Since backtracking with full enumeration is too slow for n=20, 
    # let's use a randomized local search with multiple restarts.
    
    import random
    
    def count_edges(adj):
        total = 0
        for i in range(n):
            total += adj[i].bit_count()
        return total // 2
    
    def is_valid_add(adj, u, v):
        if u == v:
            return False
        if adj[u] & (1 << v):
            return True
        # Check if adding (u, v) creates K2,3
        # For all w in N(v): common(u, w) must be < 2
        mask_v = adj[v]
        w = mask_v
        while w:
            lsb = w & -w
            w_idx = lsb.bit_length() - 1
            w -= lsb
            if (adj[u] & adj[w_idx]).bit_count() >= 2:
                return False
        
        # For all w in N(u): common(v, w) must be < 2
        mask_u = adj[u]
        w = mask_u
        while w:
            lsb = w & -w
            w_idx = lsb.bit_length() - 1
            w -= lsb
            if (adj[v] & adj[w_idx]).bit_count() >= 2:
                return False
        
        return True
    
    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    
    def greedy_initial(adj):
        """Add edges greedily in random order."""
        edges = [(i, j) for i in range(n) for j in range(i)]
        random.shuffle(edges)
        for u, v in edges:
            if is_valid_add(adj, u, v):
                add_edge(adj, u, v)
    
    def local_search(adj, max_iterations=10000):
        """Local search: try to add edges, if stuck, remove one and try again."""
        current_adj = adj[:]
        current_count = count_edges(current_adj)
        best_adj = current_adj[:]
        best_count = current_count
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Try to add a random missing edge
            improved = False
            for _ in range(50):  # Try up to 50 random additions
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                if not (current_adj[u] & (1 << v)):
                    if is_valid_add(current_adj, u, v):
                        add_edge(current_adj, u, v)
                        current_count += 1
                        improved = True
                        if current_count > best_count:
                            best_count = current_count
                            best_adj = current_adj[:]
                        break
            
            if not improved:
                # Stuck. Remove a random edge to escape local optimum.
                # Find all existing edges
                existing = []
                for u in range(n):
                    mask = current_adj[u]
                    w = mask
                    while w:
                        lsb = w & -w
                        v = lsb.bit_length() - 1
                        w -= lsb
                        if u < v:
                            existing.append((u, v))
                
                if existing:
                    u, v = random.choice(existing)
                    remove_edge(current_adj, u, v)
                    current_count -= 1
                    # After removal, try to add edges again
                    for _ in range(50):
                        ru = random.randint(0, n-1)
                        rv = random.randint(0, n-1)
                        if ru == rv:
                            continue
                        if ru > rv:
                            ru, rv = rv, ru
                        if not (current_adj[ru] & (1 << rv)):
                            if is_valid_add(current_adj, ru, rv):
                                add_edge(current_adj, ru, rv)
                                current_count += 1
                                if current_count > best_count:
                                    best_count = current_count
                                    best_adj = current_adj[:]
                                break
        
        return best_adj, best_count
    
    best_adj = [0] * n
    best_count = 0
    
    # Multiple random restarts
    num_restarts = 100
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        adj = [0] * n
        greedy_initial(adj)
        adj_ls, count_ls = local_search(adj, max_iterations=5000)
        
        if count_ls > best_count:
            best_count = count_ls
            best_adj = adj_ls[:]
        
        # Also try starting from empty and doing pure local search
        if restart % 5 == 0:
            adj2 = [0] * n
            adj2_ls, count2_ls = local_search(adj2, max_iterations=3000)
            if count2_ls > best_count:
                best_count = count2_ls
                best_adj = adj2_ls[:]
    
    # Extract edges from best_adj
    edges = []
    for u in range(n):
        mask = best_adj[u]
        k = mask
        while k:
            lsb = k & -k
            v = lsb.bit_length() - 1
            k -= lsb
            if u < v:
                edges.append([u, v])
    
    return edges

if __name__ == "__main__":
    start = time.time()
    edges = construct()
    end = time.time()
    print(f"Time: {end-start:.2f}s, Edges: {len(edges)}")
    # Verify validity
    n = 20
    adj = [0]*n
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    valid = True
    for u in range(n):
        for v in range(u+1, n):
            common = (adj[u] & adj[v]).bit_count()
            if common >= 3:
                valid = False
                print(f"Invalid: {u}, {v} have {common} common neighbors")
    print(f"Valid: {valid}")
