# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill
    
    n = 20
    # adjacency as bitmasks
    adj = [0] * n
    
    def has_edge(u, v):
        return (adj[u] >> v) & 1
    
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    
    def edge_count():
        cnt = 0
        for i in range(n):
            cnt += bin(adj[i]).count('1')
        return cnt // 2
    
    def is_valid_addition(u, v):
        """Check if adding edge (u,v) keeps the graph K3,3-free (per task definition).
        We need: for every triple containing u or v (or any triple), common neighbors <= 2.
        Efficient check: adding (u,v) can only increase common-neighbor counts for triples
        that include u or v. Specifically, for any pair (x,y) where x != u,v and y != u,v,
        the triple {x,y,z} for z in {u,v} might be affected.
        More precisely: for any w != u,v, the triple {u, v, w} now has common neighbors =
        N(u)∩N(v)∩N(w) (after adding the edge, u and v are neighbors of each other, but
        common neighbors are neighbors of ALL three, so u is not a neighbor of itself).
        Actually common neighbors of {a,b,c} = N(a)∩N(b)∩N(c). Adding (u,v) means:
        - For triple {u,v,w}: common neighbors = (N(u)∪{v}) ∩ (N(v)∪{u}) ∩ N(w)
          = (N(u)∩N(v)∩N(w)) ∪ ({v}∩N(v)∩N(w) if v in N(u)...) 
          This is getting complex. Let me just check all triples involving u or v.
        
        Simpler: after adding (u,v), check all triples that include u or v.
        """
        # For each triple containing u or v, check common neighbors <= 2
        # Triples with u: {u, x, y} for x < y, x,y != u
        # Triples with v: {v, x, y} for x < y, x,y != v
        # But to avoid double counting and be efficient, check:
        # For any pair (x, y) where x,y != u,v: triple {x, y, u} and {x, y, v}
        # Also triples {u, v, w} for w != u,v
        
        # Check triples {u, v, w} for all w != u, v
        for w in range(n):
            if w == u or w == v:
                continue
            # Common neighbors of u, v, w after adding (u,v)
            # N(u) now includes v, N(v) now includes u
            # common = (adj[u] | (1<<v)) & (adj[v] | (1<<u)) & adj[w]
            common = (adj[u] | (1 << v)) & (adj[v] | (1 << u)) & adj[w]
            if bin(common).count('1') > 2:
                return False
        
        # Check triples {u, x, y} for x < y, x,y != u
        for x in range(n):
            if x == u:
                continue
            for y in range(x + 1, n):
                if y == u or y == v:
                    continue
                if x == v:
                    continue
                # Common neighbors of u, x, y after adding (u,v)
                # N(u) includes v now
                common = (adj[u] | (1 << v)) & adj[x] & adj[y]
                if bin(common).count('1') > 2:
                    return False
        
        # Check triples {v, x, y} for x < y, x,y != v
        for x in range(n):
            if x == v:
                continue
            for y in range(x + 1, n):
                if y == v or y == u:
                    continue
                if x == u:
                    continue
                common = (adj[v] | (1 << u)) & adj[x] & adj[y]
                if bin(common).count('1') > 2:
                    return False
        
        return True
    
    # Initialize with a random dense graph
    random.seed(42)
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                add_edge(i, j)
    
    # Remove all violations
    def remove_violations():
        changed = True
        while changed:
            changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    if has_edge(i, j):
                        # Check if this edge is part of a violation
                        # Remove it if removing helps
                        # Actually, let's check if the graph is valid as-is
                        pass
        
        # Better: check all triples, if violation, remove an edge
        for a in range(n):
            for b in range(a + 1, n):
                for c in range(b + 1, n):
                    common = adj[a] & adj[b] & adj[c]
                    if bin(common).count('1') > 2:
                        # Find an edge to remove: remove (a, some common neighbor) or similar
                        # Remove edge (a, common_neighbor) where common_neighbor is in common
                        cn_list = [k for k in range(n) if (common >> k) & 1]
                        # Remove edge from a to one of the common neighbors
                        if cn_list:
                            target = cn_list[0]
                            if has_edge(a, target):
                                remove_edge(a, target)
                                changed = True
                                break
                        if changed:
                            break
                    if changed:
                        break
                if changed:
                    break
            if changed:
                break
    
    remove_violations()
    
    best_edges = edge_count()
    best_adj = adj[:]
    
    # Local search: try to add edges
    while time.time() < deadline:
        improved = False
        # Try adding each missing edge
        for i in range(n):
            for j in range(i + 1, n):
                if not has_edge(i, j):
                    if is_valid_addition(i, j):
                        add_edge(i, j)
                        improved = True
        if not improved:
            break
        
        # Try to remove and re-add to find better configurations
        # Simple: try removing each edge and see if we can add more
        for i in range(n):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n):
                if has_edge(i, j):
                    remove_edge(i, j)
                    ec_before = edge_count()
                    # Try to add edges
                    for x in range(n):
                        for y in range(x + 1, n):
                            if not has_edge(x, y):
                                if is_valid_addition(x, y):
                                    add_edge(x, y)
                    ec_after = edge_count()
                    if ec_after <= ec_before:
                        # Revert: remove added edges, re-add (i,j)
                        # This is complex; just re-add (i,j) and skip
                        add_edge(i, j)
                    else:
                        best_adj = adj[:]
                        best_edges = ec_after
    
    # Final: try adding all possible edges
    for i in range(n):
        for j in range(i + 1, n):
            if not has_edge(i, j):
                if is_valid_addition(i, j):
                    add_edge(i, j)
    
    # Convert to edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if has_edge(i, j):
                edges.append([i, j])
    
    return edges
# EVOLVE-BLOCK-END
