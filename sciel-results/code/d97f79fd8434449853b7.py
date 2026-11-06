"""K3,3-free graph on 20 vertices: maximize edges via local search."""
import time
import random
import itertools
from collections import defaultdict

N = 20
DEADLINE = time.time() + 110  # safety margin before 120s limit

def count_common_neighbors(adj, u, v, w):
    """Count common neighbors of vertices u, v, w using adjacency bitmasks."""
    # adj is a list of ints (bitmasks)
    common = adj[u] & adj[v] & adj[w]
    return bin(common).count('1')

def has_k33(adj):
    """Check if graph contains K3,3.
    K3,3 exists iff there exist three vertices with at least 3 common neighbors.
    """
    for u, v, w in itertools.combinations(range(N), 3):
        common = adj[u] & adj[v] & adj[w]
        if bin(common).count('1') >= 3:
            return True
    return False

def edge_count(adj):
    """Count edges from adjacency bitmasks."""
    total = 0
    for i in range(N):
        total += bin(adj[i]).count('1')
    return total // 2

def try_add_edge(adj, u, v):
    """Try to add edge (u,v). Returns True if added without creating K3,3."""
    if u == v or (adj[u] >> v) & 1:
        return False
    # Tentatively add
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)
    # Check if this creates a K3,3: need to check all triples involving u or v
    # A K3,3 is created if there exist 3 vertices with >= 3 common neighbors.
    # Adding edge (u,v) can only affect triples that include u or v.
    # Actually, it can affect any triple because the common neighbor set changes.
    # But the only new common neighbors are u and v themselves for triples not containing them.
    # Let's just do a full check for correctness, but optimize:
    # A K3,3 with partition {a,b,c} and {x,y,z} exists iff |N(a)∩N(b)∩N(c)| >= 3
    # where N is the open neighborhood.
    # When we add edge (u,v), the neighborhoods of u and v gain each other.
    # For a triple (a,b,c):
    #   - If none of a,b,c is u or v: N(a)∩N(b)∩N(c) gains u if u is adjacent to all three, 
    #     and gains v if v is adjacent to all three.
    #   - If one of a,b,c is u: N(u) gains v, so the intersection gains v if v is adjacent to the other two.
    #   - Similarly for v.
    #   - If two of a,b,c are u and v: N(u)∩N(v)∩N(w) - N(u) gains v, N(v) gains u. 
    #     The intersection gains u if u is adjacent to w (yes, we just added it) and v is adjacent to w.
    #     And gains v if v is adjacent to w and u is adjacent to w.
    #     So it gains both u and v if w is adjacent to both u and v.
    
    # For efficiency, let's just check all triples but with early termination.
    # With N=20, there are C(20,3)=1140 triples. This is fast enough.
    for a, b, c in itertools.combinations(range(N), 3):
        common = adj[a] & adj[b] & adj[c]
        if bin(common).count('1') >= 3:
            # Rollback
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            return False
    return True

def greedy_construct():
    """Build a dense K3,3-free graph using greedy edge addition."""
    adj = [0] * N
    # Start with no edges, add all possible edges greedily
    # Order: randomize to get different solutions
    edges = [(u, v) for u in range(N) for v in range(u+1, N)]
    random.shuffle(edges)
    for u, v in edges:
        if not ((adj[u] >> v) & 1):
            try_add_edge(adj, u, v)
    return adj

def local_search(adj, max_iters=5000):
    """Improve the graph by trying edge additions and removals."""
    best_adj = adj[:]
    best_edges = edge_count(adj)
    
    for it in range(max_iters):
        if time.time() > DEADLINE:
            break
        improved = False
        
        # Try adding edges
        for u in range(N):
            for v in range(u+1, N):
                if not ((adj[u] >> v) & 1):
                    if try_add_edge(adj, u, v):
                        improved = True
                        break
            if improved:
                break
        
        if improved:
            new_edges = edge_count(adj)
            if new_edges > best_edges:
                best_adj = adj[:]
                best_edges = new_edges
            continue
        
        # Try removing edges to enable more additions
        # Find an edge whose removal allows at least 2 new edges
        for u in range(N):
            for v in range(u+1, N):
                if (adj[u] >> v) & 1:
                    # Tentatively remove
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    
                    # Count how many new edges we can add
                    new_edges = 0
                    for x in range(N):
                        for y in range(x+1, N):
                            if not ((adj[x] >> y) & 1):
                                if try_add_edge(adj, x, y):
                                    new_edges += 1
                                else:
                                    # Rollback this specific addition
                                    adj[x] &= ~(1 << y)
                                    adj[y] &= ~(1 << x)
                    
                    # If removing one edge allows >= 2 new edges, it's beneficial
                    if new_edges >= 2:
                        improved = True
                        break
                    else:
                        # Rollback the removal
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        # Also rollback any additions that were made
                        # (they were already rolled back in the loop)
            
            if improved:
                break
        
        if improved:
            new_edges = edge_count(adj)
            if new_edges > best_edges:
                best_adj = adj[:]
                best_edges = new_edges
            continue
        
        # No improvement possible with simple moves
        break
    
    return best_adj, best_edges

def construct():
    start_time = time.time()
    
    best_adj = None
    best_edges = 0
    
    # Multiple restarts with different random seeds
    for restart in range(50):
        if time.time() > DEADLINE - 10:
            break
        
        # Random seed for this restart
        seed = restart * 12345 + 678
        random.seed(seed)
        
        # Greedy construction
        adj = greedy_construct()
        edges = edge_count(adj)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
        
        # Local search improvement
        adj_ls, edges_ls = local_search(adj, max_iters=3000)
        
        if edges_ls > best_edges:
            best_edges = edges_ls
            best_adj = adj_ls
        
        if time.time() > DEADLINE - 5:
            break
    
    # Final check: ensure no K3,3
    if best_adj is not None:
        if has_k33(best_adj):
            # Fallback to seed
            best_adj = [0] * N
            for a in range(2):
                for b in range(2, 20):
                    best_adj[a] |= (1 << b)
                    best_adj[b] |= (1 << a)
    
    # Convert to edge list
    edges_list = []
    for u in range(N):
        for v in range(u+1, N):
            if (best_adj[u] >> v) & 1:
                edges_list.append([u, v])
    
    return edges_list
