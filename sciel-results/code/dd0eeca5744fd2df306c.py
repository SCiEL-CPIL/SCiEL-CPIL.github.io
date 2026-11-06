"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a graph with 40 vertices and 401 edges minimizing the largest book size.
    Uses Simulated Annealing with a strict time limit.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    n = 40
    m = 401
    
    # Initialize with a random graph
    def init_graph():
        adj = [set() for _ in range(n)]
        edges = set()
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_possible)
        for u, v in all_possible[:m]:
            edges.add((u, v))
            adj[u].add(v)
            adj[v].add(u)
        return edges, adj

    def calc_books(adj, edges):
        books = {}
        for u, v in edges:
            books[(u, v)] = len(adj[u] & adj[v])
        return books

    def get_max_book(books):
        if not books:
            return 0
        return max(books.values())

    best_edges = None
    best_score = float('inf')

    # Run multiple restarts if time permits
    while time.time() < deadline:
        edges, adj = init_graph()
        books = calc_books(adj, edges)
        current_score = get_max_book(books)
        
        if current_score < best_score:
            best_score = current_score
            best_edges = set(edges)

        # Simulated Annealing
        T = 1.0
        T_min = 0.01
        alpha = 0.999
        no_improve = 0
        step_count = 0
        
        while time.time() < deadline - 5:
            step_count += 1
            
            # Pick a random edge to remove
            u, v = random.choice(list(edges))
            
            # Pick a random non-edge to add
            # We can pick from all possible edges and check if it's in edges
            # Or maintain a list of non-edges. For n=40, 780 edges total.
            # Checking membership is O(1).
            x, y = random.randint(0, n-1), random.randint(0, n-1)
            if x == y:
                continue
            if x > y:
                x, y = y, x
            
            new_e = (x, y)
            if new_e in edges:
                continue
            
            # Perform swap
            edges.remove((u, v))
            edges.add(new_e)
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].add(y)
            adj[y].add(x)
            
            # Update books for affected edges
            # Affected edges are those incident to u, v, x, y
            affected = set()
            for w in list(adj[u]):
                affected.add((min(u,w), max(u,w)))
            for w in list(adj[v]):
                affected.add((min(v,w), max(v,w)))
            for w in list(adj[x]):
                affected.add((min(x,w), max(x,w)))
            for w in list(adj[y]):
                affected.add((min(y,w), max(y,w)))
            
            # Recalculate book sizes for affected edges
            for e in affected:
                if e in edges:
                    eu, ev = e
                    books[e] = len(adj[eu] & adj[ev])
                else:
                    # If an edge was removed, remove it from books
                    # Note: (u,v) is removed. Is it in affected?
                    # Yes, if u is connected to v? No, we just removed it.
                    # But other edges incident to u/v are in affected.
                    # If an edge is NOT in edges, it shouldn't be in books.
                    # However, the removal of (u,v) means books should not have (u,v).
                    # The addition of (x,y) means books should have (x,y).
                    # The loop above updates books for all edges in `affected` that are in `edges`.
                    # It does NOT clean up books for edges that are in `affected` but NOT in `edges`.
                    # This is a problem if an edge was removed and is in `affected` but not `edges`.
                    # But we only update `books` for `e in edges`.
                    # We need to ensure `books` doesn't contain stale entries.
                    # It's safer to just rebuild `books` or carefully delete.
                    # Given the small size, let's just recalculate ALL books?
                    # 401 edges. `len(adj[u] & adj[v])` is fast.
                    # Rebuilding 401 books takes ~401 set intersections.
                    # This is very fast (<1ms).
                    # Let's just rebuild books to avoid state corruption.
                    pass 
            
            # Rebuilding books is safer and likely fast enough for N=40
            books = {}
            for eu, ev in edges:
                books[(eu, ev)] = len(adj[eu] & adj[ev])
            
            new_score = get_max_book(books)
            
            # Acceptance criteria
            if new_score < current_score or (random.random() < __import__('math').exp(-(new_score - current_score) / T) and new_score > current_score):
                current_score = new_score
                no_improve = 0
                if current_score < best_score:
                    best_score = current_score
                    best_edges = set(edges)
            else:
                # Revert
                edges.remove(new_e)
                edges.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
                adj[x].remove(y)
                adj[y].remove(x)
                
                # Revert books (rebuild)
                books = {}
                for eu, ev in edges:
                    books[(eu, ev)] = len(adj[eu] & adj[ev])
                
                no_improve += 1
                if no_improve > 5000:
                    break
            
            T *= alpha
            if T < T_min:
                break

    # Ensure valid return
    if best_edges is None:
        # Fallback
        adj = [set() for _ in range(n)]
        edges = set()
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_possible)
        for u, v in all_possible[:m]:
            edges.add((u, v))
        best_edges = edges
        
    edge_list = list(best_edges)
    if len(edge_list) != m:
        # Fallback: just return a random valid graph if something went wrong
        adj = [set() for _ in range(n)]
        edges = set()
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_possible)
        for u, v in all_possible[:m]:
            edges.add((u, v))
        edge_list = list(edges)
        
    return [[u, v] for u, v in edge_list]
# EVOLVE-BLOCK-END
