import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin
    
    n = 40
    target_edges = 401
    
    # Helper to compute max book
    def compute_max_book(A):
        A2 = A @ A
        mx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j] == 1:
                    b = int(A2[i, j]) - 1
                    if b > mx:
                        mx = b
        return mx

    # Helper to get book sizes for all edges
    def get_books(A):
        A2 = A @ A
        books = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j] == 1:
                    books[i, j] = int(A2[i, j]) - 1
                    books[j, i] = books[i, j]
        return books

    # Generate a random 20-regular graph
    # We can use the configuration model or just random swaps from a base.
    # Base: Complete bipartite K20,20 is 20-regular.
    # But K20,20 has book size 0 for cross edges?
    # N(u) for u in L is R. N(v) for v in R is L.
    # Intersection is empty. Book 0.
    # Intra edges? None.
    # So K20,20 has max book 0?
    # Wait. If we add 1 edge (u,v) in L, book of (u,v) is |N(u) \cap N(v)| = |R| = 20.
    # So K20,20 + 1 intra edge has max book 20.
    # We need to distribute the "intra-ness" to reduce this.
    
    # Better initialization:
    # Start with K20,20.
    # We have 400 edges.
    # We need 1 more.
    # If we add 1 intra edge, max book is 20.
    # To reduce this, we must remove some cross edges and add intra edges.
    # But we must keep total edges 401.
    # So: Remove k cross edges, add k intra edges, then add 1 more intra edge?
    # Total change: -k + k + 1 = +1.
    # So we end up with 401 edges.
    # The max book will be determined by the intra edges.
    # If we have m_intra intra edges, distributed evenly, the book size will be lower.
    
    # Let's try to construct a graph with ~10 intra edges in each partition?
    # If we have 10 intra edges in L, and 10 in R?
    # Total intra = 20.
    # We removed 20 cross edges.
    # Remaining cross edges: 400 - 20 = 380.
    # Total edges: 380 + 20 = 400.
    # Need 1 more. Add 1 more intra edge.
    # Total intra: 21.
    # Total edges: 401.
    
    # How to minimize max book?
    # The book of an intra edge (u,v) in L is the number of common neighbors in R.
    # In K20,20, every u in L is connected to all 20 in R.
    # If we remove cross edges, we reduce the common neighbors.
    # If we remove a cross edge (u, w) where w in R, then u is no longer connected to w.
    # So for any intra edge (u, v) in L, if w is a common neighbor (connected to both u and v), and we remove (u,w), then w is no longer a common neighbor.
    # So we want to remove cross edges incident to vertices involved in intra edges.
    
    # Strategy:
    # 1. Start with K20,20.
    # 2. Select a set of vertices in L and R to form "cliques" or dense subgraphs?
    #    No, we want to minimize common neighbors.
    #    We want the intra edges to have few common neighbors.
    #    This implies we should disconnect the intra vertices from the same set of R vertices.
    
    # Let's use a greedy construction + local search.
    
    best_A = None
    best_max = 999
    
    for trial in range(10):
        if time.time() - start_time > DEADLINE - 30:
            break
            
        A = np.zeros((n, n), dtype=np.int32)
        
        # Start with K20,20
        for i in range(20):
            for j in range(20, 40):
                A[i, j] = 1
                A[j, i] = 1
        
        # We will perform swaps to create intra edges.
        # Let's target ~20 intra edges.
        # Randomly pick pairs in L and R to become intra, and remove corresponding cross edges.
        
        # To ensure we don't create a huge book, we should remove cross edges
        # that are "shared" by the intra edges.
        
        # Simple approach: Random swaps.
        # 1. Pick random u, v in L (u!=v).
        # 2. If A[u,v]==0, add it.
        # 3. To keep edge count, remove a random cross edge (u, w) or (v, w) or (x, y).
        #    Removing (u, w) reduces the book of (u,v) by 1 (if w was common).
        
        # Let's do 20 such swaps.
        for _ in range(20):
            # Pick intra edge
            while True:
                u = random.randint(0, 19)
                v = random.randint(0, 19)
                if u != v and A[u, v] == 0:
                    break
            
            # Add intra edge
            A[u, v] = 1
            A[v, u] = 1
            
            # Remove a cross edge to keep count 400
            # Prefer removing edges incident to u or v to reduce their degree and potential book size
            # But we need to be careful not to disconnect.
            
            # Pick a random cross edge to remove
            # To reduce book of (u,v), we should remove (u,w) or (v,w) where w is in N(u) \cap N(v)
            common = np.where((A[u] == 1) & (A[v] == 1))[0]
            common = [w for w in common if w >= 20] # Only cross neighbors
            
            if common:
                w = random.choice(common)
                # Remove (u, w) or (v, w)
                if random.random() < 0.5:
                    A[u, w] = 0
                    A[w, u] = 0
                else:
                    A[v, w] = 0
                    A[w, v] = 0
            else:
                # No common neighbors? Then book is 0. Just remove any cross edge.
                # Find any cross edge
                cross_edges = [(i, j) for i in range(20) for j in range(20, 40) if A[i, j] == 1]
                if cross_edges:
                    ru, rv = random.choice(cross_edges)
                    A[ru, rv] = 0
                    A[rv, ru] = 0
                else:
                    # Should not happen
                    A[u, v] = 0
                    A[v, u] = 0
        
        # Now we have 400 edges. Add 1 more.
        # Find the non-edge with the smallest potential book increase.
        # We want to add an edge (x,y) such that |N(x) \cap N(y)| is small.
        
        best_add_score = 999
        best_add = None
        
        # Sample non-edges
        for _ in range(500):
            x = random.randint(0, n - 1)
            y = random.randint(0, n - 1)
            if x == y or A[x, y] == 1:
                continue
            
            # Calculate book of (x,y)
            common = np.where((A[x] == 1) & (A[y] == 1))[0]
            b = len(common)
            
            if b < best_add_score:
                best_add_score = b
                best_add = (x, y)
        
        if best_add:
            x, y = best_add
            A[x, y] = 1
            A[y, x] = 1
        
        # Verify edge count
        edge_count = sum(1 for i in range(n) for j in range(i+1, n) if A[i,j] == 1)
        if edge_count != 401:
            # Fix if necessary
            if edge_count < 401:
                for i in range(n):
                    for j in range(i+1, n):
                        if A[i,j] == 0:
                            A[i,j] = 1
                            A[j,i] = 1
                            edge_count += 1
                            if edge_count == 401: break
                    if edge_count == 401: break
            else:
                # Remove extra
                edges = [(i, j) for i in range(n) for j in range(i+1, n) if A[i,j] == 1]
                while len(edges) > 401:
                    e = edges.pop()
                    A[e[0], e[1]] = 0
                    A[e[1], e[0]] = 0
        
        # Local Search
        A2 = A @ A
        cur_max = compute_max_book(A)
        
        for ls_iter in range(5000):
            if time.time() - start_time > DEADLINE - 5:
                break
                
            if cur_max <= 7:
                break
            
            # Find worst edge
            A2 = A @ A
            worst_b = 0
            worst_u = -1
            worst_v = -1
            for i in range(n):
                for j in range(i + 1, n):
                    if A[i, j] == 1:
                        b = int(A2[i, j]) - 1
                        if b > worst_b:
                            worst_b = b
                            worst_u = i
                            worst_v = j
            
            if worst_b <= 7:
                break
            
            # Try to reduce book of (worst_u, worst_v)
            common = np.where((A[worst_u] == 1) & (A[worst_v] == 1))[0]
            common = [w for w in common if w != worst_u and w != worst_v]
            
            if not common:
                continue
            
            improved = False
            for w in random.sample(common, min(len(common), 10)):
                # Remove (worst_u, w) or (worst_v, w)
                for ru, rv in [(worst_u, w), (worst_v, w)]:
                    # Find best replacement
                    best_rep = None
                    best_rep_max = worst_b
                    
                    # Sample non-edges
                    for _ in range(50):
                        a = random.randint(0, n - 1)
                        b = random.randint(0, n - 1)
                        if a == b or A[a, b] == 1:
                            continue
                        
                        # Test swap
                        A[ru, rv] = 0
                        A[rv, ru] = 0
                        A[a, b] = 1
                        A[b, a] = 1
                        
                        # Calculate new max book
                        # Only need to check edges incident to ru, rv, a, b
                        # But for simplicity and safety, check all?
                        # n=40, check all is fast.
                        new_max = compute_max_book(A)
                        
                        # Revert
                        A[ru, rv] = 1
                        A[rv, ru] = 1
                        A[a, b] = 0
                        A[b, a] = 0
                        
                        if new_max < best_rep_max:
                            best_rep_max = new_max
                            best_rep = (a, b)
                    
                    if best_rep and best_rep_max <= worst_b:
                        a, b = best_rep
                        A[ru, rv] = 0
                        A[rv, ru] = 0
                        A[a, b] = 1
                        A[b, a] = 1
                        cur_max = best_rep_max
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                # Stuck? Try a random move that doesn't increase max
                # Or break
                break
        
        if cur_max < best_max:
            best_max = cur_max
            best_A = A.copy()
        
        if best_max <= 7:
            break
    
    if best_A is None:
        # Fallback
        A = np.zeros((n, n), dtype=np.int32)
        for i in range(20):
            for j in range(20, 40):
                A[i, j] = 1
                A[j, i] = 1
        A[0, 1] = 1
        A[1, 0] = 1
        best_A = A
    
    final_A = best_A
    edge_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if final_A[i, j] == 1:
                edge_list.append([i, j])
    
    if len(edge_list) != target_edges:
        # Fix
        if len(edge_list) < target_edges:
            for i in range(n):
                for j in range(i+1, n):
                    if final_A[i,j] == 0:
                        final_A[i,j] = 1
                        final_A[j,i] = 1
                        edge_list.append([i, j])
                        if len(edge_list) == target_edges: break
                if len(edge_list) == target_edges: break
        else:
            edge_list = edge_list[:target_edges]
            final_A = np.zeros((n, n), dtype=np.int32)
            for u, v in edge_list:
                final_A[u, v] = 1
                final_A[v, u] = 1
    
    return edge_list
