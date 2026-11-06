import time
import random
import sys

def construct():
    """
    Construct a graph with 40 vertices and 401 edges minimizing the largest book size.
    Uses a faster local search with incremental book updates.
    """
    
    deadline = time.time() + 110.0  # Safety margin
    
    n = 40
    m = 401
    
    # Initial graph: K20,20 with 9 edges removed and 10 intra-A edges added (star)
    # Vertices 0..19 in part A, 20..39 in part B
    edges = set()
    adj = [set() for _ in range(n)]
    
    # Add all bipartite edges
    for a in range(20):
        for b in range(20, 40):
            edges.add(frozenset((a, b)))
            adj[a].add(b)
            adj[b].add(a)
    
    # Remove 9 random bipartite edges
    bipartite_edges = [e for e in edges if min(list(e)) < 20 and max(list(e)) >= 20]
    random.shuffle(bipartite_edges)
    to_remove = bipartite_edges[:9]
    for e in to_remove:
        u, v = list(e)
        edges.remove(e)
        adj[u].remove(v)
        adj[v].remove(u)
        
    # Add 10 intra-A edges: Star centered at 0 (0-1, 0-2, ..., 0-10)
    for i in range(1, 11):
        e = frozenset((0, i))
        edges.add(e)
        adj[0].add(i)
        adj[i].add(0)
        
    # Verify count
    assert len(edges) == 401, f"Initial edges: {len(edges)}"
    
    # Compute initial books
    books = {}
    for e in edges:
        u, v = list(e)
        books[e] = len(adj[u] & adj[v])
        
    best_book = max(books.values())
    best_edges = set(edges)
    
    # Helper to get max book
    def get_max_book(bks):
        return max(bks.values()) if bks else 0
        
    # Local search
    while time.time() < deadline:
        max_b = get_max_book(books)
        if max_b <= 3: # Try to go below 4
            break
            
        # Find worst edges
        worst_edges = [e for e in edges if books[e] == max_b]
        random.shuffle(worst_edges)
        
        improved = False
        
        for worst_e in worst_edges[:5]: # Limit work per iteration
            if time.time() > deadline:
                break
                
            u, v = list(worst_e)
            
            # Try to swap worst_e with a random non-edge
            # We want the new edge to have low book size
            # And the removal to reduce books of neighbors
            
            # Sample candidates
            for _ in range(20):
                x = random.randint(0, n-1)
                y = random.randint(0, n-1)
                if x == y:
                    continue
                new_e = frozenset((x, y))
                if new_e in edges:
                    continue
                    
                # Estimate new book size
                # Note: adj sets are current. Removing (u,v) doesn't affect common neighbors of x,y 
                # unless x,y are u,v or neighbors of u,v in a way that changes intersection.
                # Actually, adj[x] and adj[y] don't change due to removing (u,v) unless x or y is u or v.
                # If x==u or x==v, the book size calculation for new_e is tricky.
                # Let's just compute it exactly after swap if it looks promising.
                
                # Quick check: if x,y are u,v, skip (it's the same edge)
                if (x == u and y == v) or (x == v and y == u):
                    continue
                
                # Heuristic: prefer non-edges within A or B or between A,B with low common neighbors
                common = adj[x] & adj[y]
                # If common is large, skip
                if len(common) > max_b:
                    continue
                
                # Perform swap
                edges.remove(worst_e)
                edges.add(new_e)
                adj[u].remove(v)
                adj[v].remove(u)
                adj[x].add(y)
                adj[y].add(x)
                
                # Incremental update of books
                # Edges affected:
                # 1. new_e itself
                books[new_e] = len(adj[x] & adj[y])
                
                # 2. Edges incident to u or v (their books might decrease)
                #    Edges incident to x or y (their books might increase)
                
                # List of edges to update
                affected = set()
                # Neighbors of u (excluding v which is removed)
                for w in list(adj[u]):
                    affected.add(frozenset((u, w)))
                # Neighbors of v (excluding u which is removed)
                for w in list(adj[v]):
                    affected.add(frozenset((v, w)))
                # Neighbors of x (including y)
                for w in list(adj[x]):
                    affected.add(frozenset((x, w)))
                # Neighbors of y (including x)
                for w in list(adj[y]):
                    affected.add(frozenset((y, w)))
                
                # Update books for affected edges
                for e in affected:
                    if e in edges:
                        eu, ev = list(e)
                        books[e] = len(adj[eu] & adj[ev])
                    else:
                        # Edge removed? No, we only removed worst_e.
                        # If e is worst_e, it's not in edges anymore.
                        pass
                        
                # Remove worst_e from books
                del books[worst_e]
                
                # Check improvement
                new_max = get_max_book(books)
                
                if new_max < best_book:
                    best_book = new_max
                    best_edges = set(edges)
                    improved = True
                    break
                else:
                    # Revert
                    edges.remove(new_e)
                    edges.add(worst_e)
                    adj[u].add(v)
                    adj[v].add(u)
                    adj[x].remove(y)
                    adj[y].remove(x)
                    
                    # Revert books (simplest is to recompute affected or store old values)
                    # To save time, we just recompute the affected books
                    # But we lost the old values. Let's just recompute all books? No, too slow.
                    # Let's recompute affected books based on current adj (which is reverted)
                    for e in affected:
                        if e in edges:
                            eu, ev = list(e)
                            books[e] = len(adj[eu] & adj[ev])
                    books[worst_e] = len(adj[u] & adj[v])
                    
            if improved:
                break
                
    # Return best result
    result = []
    for e in best_edges:
        u, v = list(e)
        result.append([u, v])
        
    assert len(result) == 401
    return result

if __name__ == "__main__":
    edges = construct()
    adj = [set() for _ in range(40)]
    for e in edges:
        adj[e[0]].add(e[1])
        adj[e[1]].add(e[0])
    max_b = 0
    for e in edges:
        b = len(adj[e[0]] & adj[e[1]])
        max_b = max(max_b, b)
    print(f"Max book size: {max_b}")
