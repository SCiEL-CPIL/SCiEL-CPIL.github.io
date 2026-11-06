import time
import random
import numpy as np
from itertools import combinations

def construct():
    N = 40
    M = 401
    deadline = time.time() + 110
    
    # All possible offsets for undirected edges in Z_40 are 1..19
    # (Offset 20 is self-pairing, handled separately if needed, but here we stick to 1..19 for 20-regular graphs)
    # A set S of size 10 from 1..19 creates a 20-regular graph with 400 edges.
    
    all_offsets = list(range(1, 20))
    
    best_edges = None
    best_score = float('inf')
    
    # We will iterate over all subsets S of size 10
    # To speed up, we can prune if the theoretical max book of the circulant part is already high
    # But 92k iterations is small.
    
    # Precompute common neighbor counts for each S?
    # For a given S, the book size of an edge with offset d is |S \cap (S-d)|.
    
    for S in combinations(all_offsets, 10):
        if time.time() > deadline - 5:
            break
            
        S_set = set(S)
        
        # Calculate max book size for the circulant graph
        max_circ_book = 0
        book_sizes = {}
        for d in S:
            # Common neighbors of 0 and d
            # Neighbors of 0: S
            # Neighbors of d: d + S = { (x+d)%40 for x in S }
            # Intersection size
            count = 0
            for s in S:
                # Check if s is in d+S
                # s = d + s' => s' = s - d
                if (s - d) % 40 in S_set:
                    count += 1
            # This count is the book size for edges with offset d
            book_sizes[d] = count
            if count > max_circ_book:
                max_circ_book = count
                
        # If the circulant part already has a book size >= best_score, 
        # adding an edge can only increase or maintain it. 
        # Actually, adding an edge might not increase the MAX if the max is elsewhere, 
        # but it definitely cannot decrease it.
        # So if max_circ_book >= best_score, we can skip?
        # Wait, if best_score is 7, and max_circ_book is 7, adding an edge might keep it 7 or make it 8.
        # We want to find if we can keep it 7.
        # So we should NOT skip if max_circ_book == best_score.
        # We should skip if max_circ_book > best_score.
        
        if max_circ_book > best_score:
            continue
            
        # Build the adjacency matrix for this S
        adj = np.zeros((N, N), dtype=np.int8)
        for i in range(N):
            for s in S:
                j1 = (i + s) % N
                j2 = (i - s) % N
                adj[i, j1] = 1
                adj[i, j2] = 1
        # Remove self loops (shouldn't happen as s!=0)
        np.fill_diagonal(adj, 0)
        
        # Get list of edges in this graph
        edges = []
        for i in range(N):
            for j in range(i+1, N):
                if adj[i, j]:
                    edges.append((i, j))
                
        # There are 400 edges. We need to add 1.
        # Non-edges are the 380 pairs not in edges.
        edge_set = set(edges)
        all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
        non_edges = [p for p in all_pairs if p not in edge_set]
        
        # Compute book sizes for all edges in the base graph
        # b(i,j) = dot(adj[i], adj[j])
        base_books = {}
        max_base_book = 0
        for i, j in edges:
            b = int(np.dot(adj[i], adj[j]))
            base_books[(i,j)] = b
            if b > max_base_book:
                max_base_book = b
                
        # For each non-edge, evaluate the new max book
        for u, v in non_edges:
            # New edge (u,v)
            # Common neighbors of u and v in base graph
            common = np.logical_and(adj[u], adj[v])
            common_indices = np.where(common)[0]
            b_new = len(common_indices)
            
            # Edges (u, a) and (v, a) for a in common_indices increase by 1
            # Other edges remain same.
            
            # The new max book is max of:
            # 1. b_new
            # 2. max( base_books[(u,a)] + 1, base_books[(v,a)] + 1 ) for a in common_indices
            # 3. max( base_books[e] for e in edges if e not incident to u or v )
            
            # Calculate part 2
            max_inc = 0
            for a in common_indices:
                # Edge (u,a)
                # Note: (u,a) is an edge in base graph because a is neighbor of u
                # (v,a) is an edge in base graph because a is neighbor of v
                bua = base_books.get((min(u,a), max(u,a)), 0)
                bva = base_books.get((min(v,a), max(v,a)), 0)
                if bua + 1 > max_inc: max_inc = bua + 1
                if bva + 1 > max_inc: max_inc = bva + 1
                
            # Calculate part 3
            # We need the max book size of edges NOT incident to u or v.
            # To do this fast, we can precompute for each vertex the max book of edges not incident to it?
            # Or just iterate? 400 edges * 380 non-edges * 10k S is too slow.
            # But we only do this for S that pass the filter.
            # Let's optimize by precomputing "max_book_excluding_vertex"
            
            # Actually, let's just check if the global max of base_books is incident to u or v.
            # If not, part 3 is just max_base_book.
            # If yes, we need the second highest, etc.
            
            # Simpler: Just compute the max over all edges, updating the ones that changed.
            # Since we only change edges incident to u or v that are in triangles,
            # and add (u,v), the rest are unchanged.
            
            # Let's find the max book size among all edges in the NEW graph.
            # We can start with max_base_book.
            # If the edge that had max_base_book is incident to u or v, and it increased, we are fine.
            # If it decreased? No, books only increase.
            # So the new max is at least max_base_book.
            # We just need to check if any edge exceeded max_base_book.
            
            current_max = max_base_book
            
            if b_new > current_max:
                current_max = b_new
                
            if max_inc > current_max:
                current_max = max_inc
                
            # Is it possible that an edge NOT incident to u or v had a book size > max_base_book?
            # No, max_base_book is the max of all base edges.
            
            # So current_max is the new max book size.
            
            if current_max < best_score:
                best_score = current_max
                # Construct the final edge list
                final_edges = [list(e) for e in edges]
                final_edges.append([u, v])
                # Ensure distinct and sorted? The evaluator just checks validity.
                best_edges = final_edges
                
                # If we hit the lower bound 7, we can stop early?
                # The lower bound is 7.
                if best_score == 7:
                    # We found an optimal solution.
                    # But we should ensure it's valid.
                    return best_edges

    # If we didn't find 7, return the best we found
    if best_edges:
        return best_edges
    
    # Fallback: Random graph
    random_edges = []
    pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    random.shuffle(pairs)
    for u, v in pairs[:M]:
        random_edges.append([u, v])
    return random_edges
