import random
import time
import itertools

def construct():
    """
    Constructs a graph with no K5 and no independent set of size 5.
    Uses Simulated Annealing to minimize the number of violations.
    """
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    # Try to find a graph with as many vertices as possible
    # Best known is 42 (Exoo). We aim for 42, then fall back.
    for n in range(42, 37, -1):
        if time.time() > deadline:
            break
        result = find_graph(n, deadline)
        if result is not None:
            return result
            
    # Fallback to the known valid Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    return q, [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]

def find_graph(n, deadline):
    """
    Attempts to find a (5,5)-free graph on n vertices using Simulated Annealing.
    Returns (n, edges) if successful, else None.
    """
    # Initialize random graph
    # adj[i][j] = 1 if edge exists, 0 otherwise
    adj = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.5:
                adj[i][j] = 1
                adj[j][i] = 1
                
    # Precompute all 5-sets? No, too many. 
    # We need to calculate the cost C = #K5 + #independent_5_sets
    # And be able to update it efficiently when an edge is flipped.
    
    # For efficiency in delta calculation, we need to know the status of 
    # subsets involving the flipped edge.
    
    # Let's implement a function to calculate the total cost from scratch.
    # This is O(n^5). For n=42, ~130M ops? No, binom(42,5) = 850,668.
    # 850k * 10 checks = 8.5M ops. ~0.1-0.2s.
    # We can afford to recalculate cost every few steps or just use delta.
    
    # Let's use the delta approach for speed.
    # We need to track the current number of K5 and independent sets.
    
    def get_cost(adj):
        count_k5 = 0
        count_ind = 0
        # Iterate all 5-combinations
        # Optimization: 
        # A set S is a clique if all pairs are connected.
        # A set S is independent if all pairs are NOT connected.
        
        # To speed up, we can check 4-sets first?
        # If a 4-set is a clique, any 5th vertex connected to all 4 makes a K5.
        # This is complex to maintain incrementally.
        
        # Let's just iterate 5-sets.
        for combo in itertools.combinations(range(n), 5):
            is_clique = True
            is_ind = True
            # Check all pairs in combo
            for i in range(5):
                for j in range(i+1, 5):
                    u = combo[i]
                    v = combo[j]
                    if adj[u][v]:
                        is_ind = False
                    else:
                        is_clique = False
            if is_clique:
                count_k5 += 1
            if is_ind:
                count_ind += 1
        return count_k5 + count_ind

    # Initial cost
    current_cost = get_cost(adj)
    
    if current_cost == 0:
        return extract_edges(n, adj)

    # Simulated Annealing
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    # We will iterate. 
    # To make it faster, we can restrict the search to edges that are part of violations?
    # But finding those requires knowing the violations.
    # Let's just pick random edges.
    
    while time.time() < deadline - 10: # Keep 10s for fallback/extraction
        if current_cost == 0:
            return extract_edges(n, adj)
            
        # Pick random edge
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
        # Calculate delta cost
        # Flipping (u, v) affects all 5-sets containing u and v.
        # These sets are {u, v, a, b, c} where a,b,c are distinct from u,v.
        
        delta = 0
        # Iterate over combinations of 3 vertices from the remaining n-2
        # This is binom(n-2, 3). For n=42, binom(40,3) = 9880.
        # This is fast.
        
        # We need to check the status of the set BEFORE and AFTER flip.
        # Current state: adj[u][v] is val.
        # New state: adj[u][v] is 1-val.
        
        val = adj[u][v]
        
        for a, b, c in itertools.combinations([x for x in range(n) if x != u and x != v], 3):
            S = (u, v, a, b, c)
            
            # Check if S is K5 or Ind in current graph
            # We know the edge (u,v) is 'val'.
            # We need to check edges among (a,b,c) and edges from (u,v) to (a,b,c).
            
            # This check is O(1) if we have adj.
            
            # Let's define a helper to check if a 5-set is clique/ind given the status of one edge?
            # No, just check all 10 edges. It's small.
            
            # Current status
            is_clique_now = True
            is_ind_now = True
            # Pairs: (u,v), (u,a), (u,b), (u,c), (v,a), (v,b), (v,c), (a,b), (a,c), (b,c)
            
            # (u,v)
            if val:
                is_ind_now = False
            else:
                is_clique_now = False
                
            # (u,a)
            if adj[u][a]:
                is_ind_now = False
            else:
                is_clique_now = False
            # (u,b)
            if adj[u][b]:
                is_ind_now = False
            else:
                is_clique_now = False
            # (u,c)
            if adj[u][c]:
                is_ind_now = False
            else:
                is_clique_now = False
            # (v,a)
            if adj[v][a]:
                is_ind_now = False
            else:
                is_clique_now = False
            # (v,b)
            if adj[v][b]:
                is_ind_now = False
            else:
                is_clique_now = False
            # (v,c)
            if adj[v][c]:
                is_ind_now = False
            else:
                is_clique_now = False
            # (a,b)
            if adj[a][b]:
                is_ind_now = False
            else:
                is_clique_now = False
            # (a,c)
            if adj[a][c]:
                is_ind_now = False
            else:
                is_clique_now = False
            # (b,c)
            if adj[b][c]:
                is_ind_now = False
            else:
                is_clique_now = False
            
            cost_now = (1 if is_clique_now else 0) + (1 if is_ind_now else 0)
            
            # New status (flip u,v)
            # Only (u,v) changes.
            # If val was 1, now 0. If val was 0, now 1.
            
            # We can derive new status from old?
            # If is_clique_now was True, then all other edges were 1.
            # If we flip (u,v) to 0, it's no longer a clique.
            # If is_ind_now was True, then all other edges were 0.
            # If we flip (u,v) to 1, it's no longer independent.
            
            # So:
            # If val == 1 (edge exists):
            #   If is_clique_now: becomes not clique. (cost -1)
            #   If is_ind_now: impossible (since u,v edge exists, not independent).
            #   Could it become independent? No, u,v edge exists.
            #   Could it become clique? No, u,v edge removed.
            
            # If val == 0 (edge absent):
            #   If is_ind_now: becomes not independent. (cost -1)
            #   If is_clique_now: impossible.
            #   Could it become clique? No, u,v edge added? Wait.
            #   If val==0, and is_clique_now is False.
            #   If we add edge (u,v), it might become a clique IF all other edges were 1.
            
            # Let's just recalculate the two flags for the new state to be safe.
            # It's only 9 other edges.
            
            is_clique_new = True
            is_ind_new = True
            
            # (u,v) is flipped
            if not val:
                is_ind_new = False
            else:
                is_clique_new = False
                
            # (u,a)
            if adj[u][a]:
                is_ind_new = False
            else:
                is_clique_new = False
            # (u,b)
            if adj[u][b]:
                is_ind_new = False
            else:
                is_clique_new = False
            # (u,c)
            if adj[u][c]:
                is_ind_new = False
            else:
                is_clique_new = False
            # (v,a)
            if adj[v][a]:
                is_ind_new = False
            else:
                is_clique_new = False
            # (v,b)
            if adj[v][b]:
                is_ind_new = False
            else:
                is_clique_new = False
            # (v,c)
            if adj[v][c]:
                is_ind_new = False
            else:
                is_clique_new = False
            # (a,b)
            if adj[a][b]:
                is_ind_new = False
            else:
                is_clique_new = False
            # (a,c)
            if adj[a][c]:
                is_ind_new = False
            else:
                is_clique_new = False
            # (b,c)
            if adj[b][c]:
                is_ind_new = False
            else:
                is_clique_new = False
                
            cost_new = (1 if is_clique_new else 0) + (1 if is_ind_new else 0)
            
            delta += cost_new - cost_now
            
        # Acceptance criterion
        if delta < 0:
            # Accept
            adj[u][v] = 1 - adj[u][v]
            adj[v][u] = 1 - adj[v][u]
            current_cost += delta
        else:
            # Accept with probability
            if random.random() < pow(2.71828, -delta / T):
                adj[u][v] = 1 - adj[u][v]
                adj[v][u] = 1 - adj[v][u]
                current_cost += delta
                
        T *= cooling
        if T < min_T:
            T = min_T

    # If we exit the loop without cost 0, we might still have a good graph?
    # No, we need cost 0.
    # Check one last time
    if get_cost(adj) == 0:
        return extract_edges(n, adj)
        
    return None

def extract_edges(n, adj):
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                edges.append((i, j))
    return n, edges
