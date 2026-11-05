"""Rewritten program: random init + local search to minimize largest book."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    n = 40
    m = 401
    deadline = time.time() + 140  # safety margin before 150s kill

    def compute_books(adj, edge_list):
        """Return dict edge -> book size (common neighbors)."""
        books = {}
        for u, v in edge_list:
            common = adj[u] & adj[v]
            books[(u, v)] = len(common)
        return books

    def max_book(books):
        return max(books.values()) if books else 0

    def random_graph():
        edges = set()
        while len(edges) < m:
            u = random.randrange(n)
            v = random.randrange(n)
            if u != v:
                edges.add((min(u, v), max(u, v)))
        return list(edges)

    def build_adj(edge_list):
        adj = [set() for _ in range(n)]
        for u, v in edge_list:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    # Start with a random graph
    edge_list = random_graph()
    adj = build_adj(edge_list)
    books = compute_books(adj, edge_list)
    best_max = max_book(books)

    # Local search: try edge swaps to reduce max book
    # A swap: remove edge (a,b), add edge (c,d) where (c,d) not in graph
    # Accept if it doesn't increase max book, or if it reduces it
    
    # For efficiency, we focus on edges with high book sizes
    # and try to replace them with low-book non-edges
    
    # Precompute: for each vertex pair, whether it's an edge
    edge_set = set(edge_list)
    
    # Count degrees
    deg = [len(adj[i]) for i in range(n)]
    
    # Local search loop
    improvement = True
    while improvement and time.time() < deadline:
        improvement = False
        
        # Get edges sorted by book size descending
        sorted_edges = sorted(books.items(), key=lambda x: -x[1])
        
        # Try to improve high-book edges
        for (u, v), bk in sorted_edges:
            if time.time() >= deadline:
                break
            if bk < best_max:
                break  # no more edges with book >= best_max to improve
            
            # Try removing (u,v) and adding a random non-edge
            # Focus on non-edges that would create low book
            # Candidates: pairs (c,d) not in edge_set, c!=d
            # To keep it fast, try a limited number of random candidates
            
            # We want to add an edge (c,d) such that common(c,d) is small
            # common(c,d) = |adj[c] & adj[d]|
            # After removing (u,v): adj'[u] = adj[u] - {v}, adj'[v] = adj[v] - {u}
            # For new edge (c,d): common' = |adj'[c] & adj'[d]|
            # If c,d are not u,v, this is just |adj[c] & adj[d]|
            
            # Try random non-edges
            for _ in range(50):
                if time.time() >= deadline:
                    break
                c = random.randrange(n)
                d = random.randrange(n)
                if c == d:
                    continue
                c, d = min(c, d), max(c, d)
                if (c, d) in edge_set:
                    continue
                if (c, d) == (u, v):
                    continue
                
                # Compute common neighbors for (c,d) after removing (u,v)
                # adj'[c] = adj[c] - {u if u==c else nothing} ... actually:
                # removing edge (u,v) only affects adj[u] and adj[v]
                # so if c not in {u,v} and d not in {u,v}, common is unchanged
                # if c==u or d==u or c==v or d==v, need to adjust
                
                ac = adj[c]
                ad = adj[d]
                if c == u or c == v:
                    ac = ac - {u, v}  # remove both u and v from c's neighbors? No.
                    # Actually: adj[u] loses v, adj[v] loses u.
                    # If c == u: adj'[u] = adj[u] - {v}
                    # If c == v: adj'[v] = adj[v] - {u}
                    # If c not in {u,v}: adj'[c] = adj[c]
                if c == u:
                    ac = adj[u] - {v}
                elif c == v:
                    ac = adj[v] - {u}
                else:
                    ac = adj[c]
                
                if d == u:
                    ad = adj[u] - {v}
                elif d == v:
                    ad = adj[v] - {u}
                else:
                    ad = adj[d]
                
                common_cd = len(ac & ad)
                
                # Also need to check: after removing (u,v), the books of other edges change
                # Edges (u,x) for x in adj[u]-{v}: their book decreases by 1 if v was common
                # Edges (v,x) for x in adj[v]-{u}: their book decreases by 1 if u was common
                # This is complex. For simplicity, only accept if common_cd < bk
                # and hope that the overall max doesn't increase.
                
                if common_cd < bk:
                    # Tentatively accept: check if max book decreases or stays
                    # This is an approximation; we'll recompute after swap
                    # Do the swap
                    edge_set.discard((u, v))
                    edge_set.add((c, d))
                    adj[u].discard(v)
                    adj[v].discard(u)
                    adj[c].add(d)
                    adj[d].add(c)
                    
                    # Recompute books (expensive but necessary for correctness)
                    edge_list = list(edge_set)
                    books = compute_books(adj, edge_list)
                    new_max = max_book(books)
                    
                    if new_max <= best_max:
                        best_max = new_max
                        improvement = True
                        # Keep the swap
                    else:
                        # Revert
                        edge_set.discard((c, d))
                        edge_set.add((u, v))
                        adj[u].add(v)
                        adj[v].add(u)
                        adj[c].discard(d)
                        adj[d].discard(c)
                        edge_list = list(edge_set)
                        books = compute_books(adj, edge_list)
                    break  # try next edge
    
    # Final: return the best edge list found
    edge_list = list(edge_set)
    return [[u, v] for u, v in edge_list]
# EVOLVE-BLOCK-END
