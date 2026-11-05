import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n = 40
    m = 401

    def build_bitmasks(edge_set):
        adj = [0] * n
        for (u, v) in edge_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def build_adj_matrix(edge_set):
        A = np.zeros((n, n), dtype=np.int64)
        for (u, v) in edge_set:
            A[u, v] = 1
            A[v, u] = 1
        return A

    def book(u, v, adj):
        return bin(adj[u] & adj[v]).count('1')

    def max_book(books):
        return max(books.values()) if books else 0

    def all_books(edge_set, adj):
        return {(u, v): book(u, v, adj) for (u, v) in edge_set}

    def spectral_init():
        """
        Initialize with a random regular-like graph using spectral/spreading approach.
        We start with a random graph and then adjust to make it more regular.
        """
        edge_set = set()
        
        # Start with a random regular-like structure
        # Target degree is 2*m/n = 802/40 = 20.05, so aim for degree ~20
        target_deg = 20
        
        # Create a random 20-regular graph approximation
        # Use configuration model or just random edges and adjust
        deg = [0] * n
        all_edges = []
        for i in range(n):
            for j in range(i+1, n):
                all_edges.append((i, j))
        random.shuffle(all_edges)
        
        # Greedily add edges to get close to regular degree 20
        for e in all_edges:
            u, v = e
            if deg[u] < target_deg and deg[v] < target_deg:
                edge_set.add(e)
                deg[u] += 1
                deg[v] += 1
            if len(edge_set) >= m:
                break
        
        # If we have too few edges, add more randomly
        while len(edge_set) < m:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u < v and (u, v) not in edge_set:
                edge_set.add((u, v))
                deg[u] += 1
                deg[v] += 1
        
        # If we have too many edges (shouldn't happen), remove some
        if len(edge_set) > m:
            edges = list(edge_set)
            random.shuffle(edges)
            for e in edges:
                if len(edge_set) <= m:
                    break
                edge_set.remove(e)
        
        return edge_set

    def spectral_local_search(edge_set, adj, books, current_score, max_steps=2000, stuck_limit=40):
        """
        Local search that uses spectral information (A^2 diagonal = degree, 
        off-diagonal = common neighbors) to guide edge swaps.
        We try to flatten the triangle distribution by targeting edges with 
        high book size and replacing them with edges that have low impact 
        on the spectral properties.
        """
        stuck = 0
        for step in range(max_steps):
            if time.time() > deadline - 2:
                return books, current_score

            # Find worst edge
            worst_edge = None
            worst_val = -1
            for e, val in books.items():
                if val > worst_val:
                    worst_val = val
                    worst_edge = e

            if worst_val <= 7:
                break

            u, v = worst_edge
            swapped = False

            # Build adjacency matrix for spectral analysis
            A = build_adj_matrix(edge_set)
            
            # A^2 gives number of paths of length 2
            A2 = A @ A
            # Diagonal of A^2 is degree
            # Off-diagonal A2[i,j] is number of common neighbors of i and j
            
            # Score for a potential new edge (i,j): 
            # We want to minimize the increase in max book.
            # Book(i,j) = A2[i,j] if (i,j) is an edge.
            # But adding edge (i,j) changes A2 for all edges incident to i or j.
            
            # Candidate edges: prefer those with low current A2 value (low potential book)
            # and that help balance the degree sequence
            cands = []
            seen = set()
            deg = A.sum(axis=1)
            
            for _ in range(300):
                ru = random.randint(0, n - 1)
                rv = random.randint(0, n - 1)
                if ru == rv:
                    continue
                if ru > rv:
                    ru, rv = rv, ru
                if (ru, rv) in edge_set or (ru, rv) == (u, v):
                    continue
                if (ru, rv) in seen:
                    continue
                seen.add((ru, rv))
                
                # Current common neighbors (would be book size if edge added)
                tc = A2[ru, rv]
                
                # Spectral score: prefer edges that connect vertices with 
                # similar degrees (to maintain regularity) and low common neighbors
                deg_diff = abs(deg[ru] - deg[rv])
                spectral_score = tc * 2 + deg_diff
                
                cands.append((spectral_score, tc, ru, rv))

            cands.sort(key=lambda x: x[0])

            for spectral_score, tc, ru, rv in cands[:30]:
                # Apply swap
                edge_set.remove((u, v))
                edge_set.add((ru, rv))
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                adj[ru] |= (1 << rv)
                adj[rv] |= (1 << ru)

                affected = {u, v, ru, rv}
                new_books = {}
                for e, val in books.items():
                    eu, ev = e
                    if eu in affected or ev in affected:
                        new_books[e] = book(eu, ev, adj)
                    else:
                        new_books[e] = val
                new_books[(ru, rv)] = book(ru, rv, adj)

                new_max = max_book(new_books)
                if new_max < current_score:
                    current_score = new_max
                    books = new_books
                    swapped = True
                    stuck = 0
                    break
                else:
                    # Revert
                    edge_set.remove((ru, rv))
                    edge_set.add((u, v))
                    adj[ru] &= ~(1 << rv)
                    adj[rv] &= ~(1 << ru)
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)

            if not swapped:
                stuck += 1
            else:
                stuck = 0

            if stuck > stuck_limit:
                break

        return books, current_score

    best_score = 999
    best_edge_set = None

    # Use spectral initialization
    for restart in range(30):
        if time.time() > deadline - 5:
            break
            
        edge_set = spectral_init()
        adj = build_bitmasks(edge_set)
        books = all_books(edge_set, adj)
        current_score = max_book(books)

        if current_score < best_score:
            best_score = current_score
            best_edge_set = set(edge_set)

        if current_score <= 7:
            break

        books, current_score = spectral_local_search(edge_set, adj, books, current_score)

        if current_score < best_score:
            best_score = current_score
            best_edge_set = set(edge_set)

        if current_score <= 7:
            break

    if best_edge_set is None:
        best_edge_set = spectral_init()

    return [[u, v] for (u, v) in best_edge_set]
