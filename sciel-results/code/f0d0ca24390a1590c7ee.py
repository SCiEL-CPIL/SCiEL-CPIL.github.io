import time
import random
import sys

def construct():
    """
    Construct a graph with 40 vertices and 401 edges minimizing the largest book size.
    The seed program (K20,20 + 1 edge) has a book size of 6 (score 1.0).
    We aim to achieve a book size of 5 (score 1.2) or 4 (score 1.5).
    
    Strategy:
    1. Start with a Turan-like graph T_2(40) (K20,20) plus 1 edge inside a part.
    2. Use a local search that specifically targets the edges with the highest book size.
    3. Perform edge swaps that reduce the maximum book size.
    """
    
    deadline = time.time() + 145.0  # 5 seconds safety margin
    
    n = 40
    m = 401
    
    # Initial graph: K20,20 + edge (0,1)
    # Vertices 0..19 in part A, 20..39 in part B
    # Edges: all (a, b) for a in A, b in B, plus (0, 1)
    
    # Represent graph as a set of frozensets for O(1) lookup
    edges = set()
    for a in range(20):
        for b in range(20, 40):
            edges.add(frozenset((a, b)))
    edges.add(frozenset((0, 1)))
    
    # Verify we have exactly 401 edges
    assert len(edges) == 401, f"Initial graph has {len(edges)} edges, expected 401"
    
    # Precompute adjacency sets for fast triangle counting
    adj = [set() for _ in range(n)]
    for e in edges:
        u, v = list(e)
        adj[u].add(v)
        adj[v].add(u)
    
    def compute_books():
        """Compute the book size for each edge. Returns dict: edge -> book_size."""
        books = {}
        for e in edges:
            u, v = list(e)
            # Common neighbors of u and v form triangles containing edge (u,v)
            common = adj[u] & adj[v]
            books[e] = len(common)
        return books
    
    def max_book(books):
        """Return the maximum book size."""
        return max(books.values()) if books else 0
    
    # Initial evaluation
    books = compute_books()
    best_book = max_book(books)
    best_edges = set(edges)
    best_books = books
    
    # Local search: repeatedly try to improve the worst edge
    # An edge swap: remove edge (u,v), add edge (x,y)
    # We focus on edges with high book size
    
    while time.time() < deadline:
        # Find edges with maximum book size
        max_b = max_book(books)
        if max_b <= 4:  # Target achieved (or better)
            break
        
        # Get the set of edges with max book size
        worst_edges = [e for e in edges if books[e] == max_b]
        
        if not worst_edges:
            break
        
        # Try to fix the worst edges
        improved = False
        random.shuffle(worst_edges)
        
        for worst_e in worst_edges:
            if time.time() > deadline:
                break
            
            u, v = list(worst_e)
            
            # Try removing (u,v) and adding a new edge (x,y) that doesn't create triangles
            # Strategy: find x,y such that (x,y) is not in edges, and the new edge has small book
            # Also, removing (u,v) should reduce books for edges adjacent to u or v
            
            # Candidate new edges: non-edges that are "safe"
            # To be efficient, only consider a few candidates per iteration
            
            # Find non-edges
            # A non-edge (x,y) where x != y and (x,y) not in edges
            # We want (x,y) to have few common neighbors
            
            # Heuristic: try to add an edge within a part that has low degree
            # Or between vertices that have few common neighbors
            
            # For speed, sample a limited number of candidate swaps
            num_candidates = 50
            found_swap = False
            
            for _ in range(num_candidates):
                # Pick two random vertices x, y
                x = random.randint(0, n-1)
                y = random.randint(0, n-1)
                if x == y:
                    continue
                new_e = frozenset((x, y))
                if new_e in edges:
                    continue
                
                # Check if adding (x,y) creates too many triangles
                # Book size of new edge = |adj[x] & adj[y]|
                # But we're also removing (u,v), which changes adj sets
                
                # Simplified: estimate book size of new edge in current graph
                new_book = len(adj[x] & adj[y])
                
                # If new_book is small, try the swap
                if new_book <= 2:
                    # Perform swap: remove (u,v), add (x,y)
                    # Check validity: (x,y) not in edges (already checked)
                    # Also need to ensure we don't create duplicate edges
                    
                    # Update graph
                    edges.remove(worst_e)
                    edges.add(new_e)
                    
                    # Update adjacency
                    adj[u].discard(v)
                    adj[v].discard(u)
                    adj[x].add(y)
                    adj[y].add(x)
                    
                    # Recompute books (expensive but necessary for accuracy)
                    books_new = compute_books()
                    max_b_new = max_book(books_new)
                    
                    if max_b_new < best_book:
                        # Accept improvement
                        best_book = max_b_new
                        best_edges = set(edges)
                        best_books = books_new
                        books = books_new
                        improved = True
                        found_swap = True
                        break
                    else:
                        # Revert
                        edges.remove(new_e)
                        edges.add(worst_e)
                        adj[u].add(v)
                        adj[v].add(u)
                        adj[x].discard(y)
                        adj[y].discard(x)
            
            if found_swap:
                continue
            
            if time.time() > deadline:
                break
        
        if not improved and max_book(books) == best_book:
            # No improvement in this pass; try a random restart-like perturbation
            # Or just break if we've been stuck for too long
            # For now, continue with smaller steps
            pass
    
    # Return the best result found
    result = []
    for e in best_edges:
        u, v = list(e)
        result.append([u, v])
    
    # Ensure exactly 401 distinct edges
    assert len(result) == 401, f"Result has {len(result)} edges, expected 401"
    # Ensure all edges are distinct
    assert len(set(frozenset(e) for e in result)) == 401, "Edges are not distinct"
    # Ensure vertices are in range
    for e in result:
        assert 0 <= e[0] <= 39 and 0 <= e[1] <= 39, f"Vertex out of range: {e}"
        assert e[0] != e[1], f"Self-loop: {e}"
    
    return result

if __name__ == "__main__":
    edges = construct()
    # Quick verification
    print(f"Number of edges: {len(edges)}")
    # Compute max book
    adj = [set() for _ in range(40)]
    for e in edges:
        adj[e[0]].add(e[1])
        adj[e[1]].add(e[0])
    max_b = 0
    for e in edges:
        b = len(adj[e[0]] & adj[e[1]])
        max_b = max(max_b, b)
    print(f"Max book size: {max_b}")
