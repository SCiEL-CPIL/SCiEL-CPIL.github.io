"""Improved program: structured greedy + local search for K3,3-free graph on 20 vertices."""
import random
import time
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    n = 20
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Use bitmasks for fast common-neighbor computation
    # adj_mask[i] is a bitmask of neighbors of vertex i
    
    def is_k33_free(adj_mask):
        """Check if graph is K3,3-free using bitmask common neighbor counting."""
        for i in range(n):
            for j in range(i + 1, n):
                common_ij = adj_mask[i] & adj_mask[j]
                for k in range(j + 1, n):
                    if (common_ij & adj_mask[k]).bit_count() >= 3:
                        return False
        return True
    
    def edge_count(adj_mask):
        total = 0
        for i in range(n):
            total += adj_mask[i].bit_count()
        return total // 2
    
    def get_edges(adj_mask):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj_mask[i] & (1 << j):
                    edges.append([i, j])
        return edges
    
    def can_add_edge(adj_mask, u, v):
        """Check if adding edge (u,v) keeps graph K3,3-free."""
        # After adding (u,v), for any triple {a,b,c} containing u or v,
        # we need to check common neighbors.
        # Key insight: adding (u,v) only creates new K3,3 if u and v are on opposite sides.
        # So we need: exist a,b in N(v)\\{u} and c,d in N(u)\\{v} 
        # such that a,b are connected to c,d (i.e., a-c, a-d, b-c, b-d all exist)
        
        Nv = adj_mask[v] & ~(1 << u)
        Nu = adj_mask[u] & ~(1 << v)
        
        # Get lists of neighbors
        Nv_list = [x for x in range(n) if Nv & (1 << x)]
        Nu_list = [x for x in range(n) if Nu & (1 << x)]
        
        for a in Nv_list:
            for b in Nv_list:
                if a >= b:
                    continue
                # Check if a and b have >= 2 common neighbors in Nu_list
                common_ab_in_Nu = (adj_mask[a] & adj_mask[b] & Nu)
                if common_ab_in_Nu.bit_count() >= 2:
                    return False
        return True
    
    def greedy_build(adj_mask):
        """Try to add all possible edges that keep graph K3,3-free."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj_mask[i] & (1 << j)):
                        if can_add_edge(adj_mask, i, j):
                            adj_mask[i] |= (1 << j)
                            adj_mask[j] |= (1 << i)
                            improved = True
                            break
                if improved:
                    break
        return adj_mask
    
    def find_violating_triple(adj_mask):
        """Find a K3,3 violation, return (i,j,k) or None."""
        for i in range(n):
            for j in range(i + 1, n):
                common_ij = adj_mask[i] & adj_mask[j]
                for k in range(j + 1, n):
                    if (common_ij & adj_mask[k]).bit_count() >= 3:
                        return (i, j, k)
        return None
    
    def fix_violations(adj_mask):
        """Remove edges to make graph K3,3-free."""
        while time.time() < deadline:
            v = find_violating_triple(adj_mask)
            if v is None:
                break
            i, j, k = v
            # Find common neighbors
            common = adj_mask[i] & adj_mask[j] & adj_mask[k]
            # Remove one edge to break the violation
            # Try removing edge from i to a common neighbor
            for l in range(n):
                if common & (1 << l):
                    adj_mask[i] &= ~(1 << l)
                    adj_mask[l] &= ~(1 << i)
                    break
        return adj_mask
    
    def local_search(adj_mask):
        """Try edge swaps: remove an edge, then greedily add more."""
        current_count = edge_count(adj_mask)
        best_mask = list(adj_mask)
        best_count = current_count
        
        for i in range(n):
            for j in range(i + 1, n):
                if adj_mask[i] & (1 << j):
                    # Remove edge (i,j)
                    adj_mask[i] &= ~(1 << j)
                    adj_mask[j] &= ~(1 << i)
                    
                    # Try to greedily add edges
                    greedy_build(adj_mask)
                    
                    new_count = edge_count(adj_mask)
                    if new_count > best_count:
                        best_count = new_count
                        best_mask = list(adj_mask)
                    else:
                        # Restore
                        adj_mask[i] |= (1 << j)
                        adj_mask[j] |= (1 << i)
                    
                    if time.time() > deadline:
                        return best_mask, best_count
        
        return best_mask, best_count
    
    best_edges = None
    best_score = -1
    
    # Strategy 1: Start with empty graph, greedily add edges
    adj_mask = [0] * n
    greedy_build(adj_mask)
    if is_k33_free(adj_mask):
        score = edge_count(adj_mask)
        if score > best_score:
            best_score = score
            best_edges = get_edges(adj_mask)
    
    # Strategy 2: Start with random graphs and improve
    for attempt in range(80):
        if time.time() > deadline:
            break
        
        random.seed(attempt)
        adj_mask = [0] * n
        p = 0.3 + 0.4 * (attempt % 5) / 5  # Vary probability
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
        
        # If not K3,3-free, remove edges to fix
        adj_mask = fix_violations(adj_mask)
        
        if is_k33_free(adj_mask):
            greedy_build(adj_mask)
            if is_k33_free(adj_mask):
                # Local search
                adj_mask, score = local_search(adj_mask)
                if is_k33_free(adj_mask):
                    if score > best_score:
                        best_score = score
                        best_edges = get_edges(adj_mask)
    
    # Strategy 3: Start with known good constructions
    # Complete bipartite K_{2,18} has 36 edges
    adj_mask = [0] * n
    for i in range(2):
        for j in range(2, 20):
            adj_mask[i] |= (1 << j)
            adj_mask[j] |= (1 << i)
    greedy_build(adj_mask)
    if is_k33_free(adj_mask):
        score = edge_count(adj_mask)
        if score > best_score:
            best_score = score
            best_edges = get_edges(adj_mask)
    
    # Strategy 4: Random restarts with local search
    for attempt in range(40):
        if time.time() > deadline:
            break
        
        random.seed(1000 + attempt)
        adj_mask = [0] * n
        # Start with a random sparse graph
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.2:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
        
        # Make it K3,3-free
        adj_mask = fix_violations(adj_mask)
        
        if is_k33_free(adj_mask):
            greedy_build(adj_mask)
            if is_k33_free(adj_mask):
                # Local search
                adj_mask, score = local_search(adj_mask)
                if is_k33_free(adj_mask):
                    if score > best_score:
                        best_score = score
                        best_edges = get_edges(adj_mask)
    
    # Strategy 5: Start from K_{3,17} minus some edges, then optimize
    adj_mask = [0] * n
    for i in range(3):
        for j in range(3, 20):
            adj_mask[i] |= (1 << j)
            adj_mask[j] |= (1 << i)
    # K_{3,17} is NOT K3,3-free, so we need to remove edges
    adj_mask = fix_violations(adj_mask)
    if is_k33_free(adj_mask):
        greedy_build(adj_mask)
        if is_k33_free(adj_mask):
            adj_mask, score = local_search(adj_mask)
            if is_k33_free(adj_mask):
                if score > best_score:
                    best_score = score
                    best_edges = get_edges(adj_mask)
    
    if best_edges is not None:
        return best_edges
    
    # Fallback: K2,18
    return [[a, b] for a in range(2) for b in range(2, 20)]
# EVOLVE-BLOCK-END
