"""Rewritten program: Random Circulant init + Simulated Annealing to minimize largest book."""
import time
import random
import math
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

    def get_metrics(books):
        """Return max book and sum of squared books."""
        if not books:
            return 0, 0
        max_bk = max(books.values())
        sum_sq = sum(b*b for b in books.values())
        return max_bk, sum_sq

    def random_circulant_init():
        """Generate a random circulant graph with n*10 edges, plus 1 random edge."""
        # 10 offsets for 400 edges
        offsets = random.sample(range(1, n//2 + 1), 10)
        edges = set()
        for i in range(n):
            for off in offsets:
                u = i
                v = (i + off) % n
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                edges.add((u, v))
        
        # Add 1 random edge to reach 401
        while len(edges) < m:
            u = random.randrange(n)
            v = random.randrange(n)
            if u != v:
                if u > v:
                    u, v = v, u
                edges.add((u, v))
        
        return list(edges)

    def build_adj(edge_list):
        adj = [set() for _ in range(n)]
        for u, v in edge_list:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    # Start with Random Circulant graph
    edge_list = random_circulant_init()
    edge_set = set(edge_list)
    adj = build_adj(edge_list)
    books = compute_books(adj, edge_list)
    current_max, current_sum_sq = get_metrics(books)
    
    best_max = current_max
    best_sum_sq = current_sum_sq
    best_edge_set = set(edge_set)

    # Simulated Annealing Parameters
    T = 10.0
    cooling = 0.999
    min_T = 0.1
    
    # Precompute non-edges for faster sampling? 
    # Total pairs = 40*39/2 = 780. Edges = 401. Non-edges = 379.
    # We can maintain a list of non-edges or just sample and check.
    # Sampling and checking is O(1) average if we just pick random pairs.
    
    start_time = time.time()
    
    while time.time() < deadline:
        if T < min_T:
            break
            
        # Generate a random move: remove one edge, add one non-edge
        # Pick a random edge to remove
        u, v = random.choice(list(edge_set))
        
        # Pick a random non-edge to add
        # Try a few times to find a valid non-edge
        added = False
        for _ in range(10):
            c = random.randrange(n)
            d = random.randrange(n)
            if c == d:
                continue
            if c > d:
                c, d = d, c
            if (c, d) in edge_set:
                continue
            if (c, d) == (u, v):
                continue
            
            # Tentative swap
            # Remove (u,v)
            edge_set.discard((u, v))
            adj[u].discard(v)
            adj[v].discard(u)
            
            # Add (c,d)
            edge_set.add((c, d))
            adj[c].add(d)
            adj[d].add(c)
            
            # Recompute metrics
            # Only edges incident to u, v, c, d change?
            # Actually, book size of edge (x,y) is |adj[x] & adj[y]|.
            # Removing (u,v) changes adj[u] and adj[v].
            # So books of edges incident to u or v change.
            # Adding (c,d) changes adj[c] and adj[d].
            # So books of edges incident to c or d change.
            # Edges not incident to {u,v,c,d} are unchanged.
            
            # However, for simplicity and safety, and since n=40 is small,
            # we can just recompute all books. 
            # 401 edges * intersection of sets.
            # This is fast enough for SA iterations if we limit them.
            
            edge_list_temp = list(edge_set)
            books_temp = compute_books(adj, edge_list_temp)
            new_max, new_sum_sq = get_metrics(books_temp)
            
            # Acceptance criteria
            delta_max = new_max - current_max
            delta_sum = new_sum_sq - current_sum_sq
            
            # We primarily care about max. Secondarily about sum of squares.
            # If max decreases, accept.
            # If max increases, accept with prob exp(-delta_max / T).
            # If max same, accept with prob exp(-delta_sum / (10*T)) to break ties.
            
            accept = False
            if delta_max < 0:
                accept = True
            elif delta_max == 0:
                if delta_sum < 0:
                    accept = True
                else:
                    # Probabilistic accept for tie-breaking
                    # Scale delta_sum appropriately
                    accept_prob = math.exp(-delta_sum / (100.0 * T))
                    accept = (random.random() < accept_prob)
            else:
                accept_prob = math.exp(-delta_max / T)
                accept = (random.random() < accept_prob)
            
            if accept:
                current_max = new_max
                current_sum_sq = new_sum_sq
                books = books_temp
                if current_max < best_max or (current_max == best_max and current_sum_sq < best_sum_sq):
                    best_max = current_max
                    best_sum_sq = current_sum_sq
                    best_edge_set = set(edge_set)
            else:
                # Revert
                edge_set.discard((c, d))
                edge_set.add((u, v))
                adj[c].discard(d)
                adj[d].discard(c)
                adj[u].add(v)
                adj[v].add(u)
            
            added = True
            break
        
        if not added:
            continue
            
        T *= cooling

    # Return the best solution found
    edge_list = list(best_edge_set)
    return [[u, v] for u, v in edge_list]
# EVOLVE-BLOCK-END
