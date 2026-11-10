import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a K3,3-free graph on 20 vertices maximizing edges.
    Uses Simulated Annealing with bitmask adjacency for speed.
    """
    start_time = time.time()
    # Deadline guard: 120s limit, leave 10s margin
    deadline = start_time + 110
    
    n = 20
    # Precompute all triples (i, j, k) with i < j < k
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                triples.append((i, j, k))
    
    num_triples = len(triples)
    
    # Bitmask helper functions
    # adj is a list of integers, where bit v is 1 if edge (u, v) exists
    # adj[u] has bits for neighbors of u. 
    # Note: For simplicity in common neighbor calc, we store adj[u] as bitmask of neighbors.
    # common(i, j) = adj[i] & adj[j]
    # count_common(i, j, k) = popcount(adj[i] & adj[j] & adj[k])
    
    def popcount(x):
        # Python 3.10+ has int.bit_count(), but for compatibility/speed in loops, 
        # bit_count is very fast.
        return x.bit_count()
        
    def count_common(adj, i, j, k):
        return (adj[i] & adj[j] & adj[k]).bit_count()
        
    def get_violations(adj):
        """Returns list of (i,j,k) triples with >= 3 common neighbors."""
        viols = []
        for idx, (i, j, k) in enumerate(triples):
            c = (adj[i] & adj[j] & adj[k]).bit_count()
            if c >= 3:
                viols.append((i, j, k))
        return viols
        
    def is_valid(adj):
        return len(get_violations(adj)) == 0
        
    def edge_count(adj):
        total = 0
        for i in range(n):
            total += adj[i].bit_count()
        return total // 2

    # Initialize with a random dense graph
    # We want high degree. Let's start with a random graph G(n, 0.6) or similar
    # Actually, let's start with a random permutation-based construction or just random edges
    # Random edges with p=0.5 is a good start for SA.
    
    def random_graph(p=0.5):
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    # Initial state
    current_adj = random_graph(0.6)
    current_violations = get_violations(current_adj)
    current_edges = edge_count(current_adj)
    
    best_adj = current_adj[:]
    best_edges = current_edges
    best_violations = 0
    
    # If initial is valid, update best
    if not current_violations:
        best_adj = current_adj[:]
        best_edges = current_edges
        best_violations = 0
        
    # Simulated Annealing Parameters
    # Temperature schedule
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.999
    
    # We want to maximize edges, minimize violations.
    # Objective: Maximize (edges - penalty * violations)
    # Penalty needs to be large enough to prefer valid graphs, but we can also 
    # just reject invalid moves unless T is high.
    # Let's use a score: score = edges - 100 * num_violations
    # This encourages removing violations first, then adding edges.
    
    def score(adj, viols):
        return edge_count(adj) - 100 * len(viols)
        
    current_score = score(current_adj, current_violations)
    best_score = current_score
    
    # Main SA Loop
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Pick a random edge (u, v) to toggle
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
            
        # Ensure u < v for consistency if needed, but bitmask handles both
        # Toggle edge (u, v)
        bit_u = 1 << v
        bit_v = 1 << u
        
        # Check if edge exists
        if current_adj[u] & bit_u:
            # Remove edge
            current_adj[u] ^= bit_u
            current_adj[v] ^= bit_v
            delta_edges = -1
        else:
            # Add edge
            current_adj[u] ^= bit_u
            current_adj[v] ^= bit_v
            delta_edges = 1
            
        # Calculate change in violations
        # Only triples involving u or v can change their common neighbor count
        # A triple (i, j, k) is affected if u or v is in {i, j, k}
        # And the edge change affects adj[u] or adj[v].
        # Common neighbors of (i, j, k) is popcount(adj[i] & adj[j] & adj[k]).
        # If we change adj[u], then any triple containing u is affected.
        # Specifically, for triple (u, a, b), the common neighbors are adj[u] & adj[a] & adj[b].
        # Since adj[u] changed by toggling bit v, the intersection changes.
        
        # Efficiently update violations:
        # 1. Identify affected triples: those containing u or v.
        # 2. For each, calculate old and new common count.
        # 3. Update violation count.
        
        # To do this efficiently, we can precompute lists of triples containing each vertex.
        # But we are inside the loop. Let's generate them on the fly or precompute once.
        # Precompute triples_by_vertex
        if not hasattr(construct, '_triples_by_vertex'):
            triples_by_vertex = [[] for _ in range(n)]
            for idx, (i, j, k) in enumerate(triples):
                triples_by_vertex[i].append(idx)
                triples_by_vertex[j].append(idx)
                triples_by_vertex[k].append(idx)
            construct._triples_by_vertex = triples_by_vertex
            
        triples_by_vertex = construct._triples_by_vertex
        
        # Get affected triple indices
        # Union of triples containing u and triples containing v
        affected = set(triples_by_vertex[u]) | set(triples_by_vertex[v])
        
        # Calculate delta violations
        delta_violations = 0
        for idx in affected:
            i, j, k = triples[idx]
            # We need to compare before and after.
            # Since we already toggled the edge, we need to be careful.
            # Let's revert, calculate, then apply? No, that's slow.
            # Let's calculate the impact of toggling bit v in adj[u] (and bit u in adj[v]) on the intersection.
            
            # Case 1: u is in the triple.
            if u in (i, j, k):
                # The other two vertices are a, b
                others = [x for x in (i, j, k) if x != u]
                a, b = others
                # Old common neighbors: popcount(adj[u] & adj[a] & adj[b])
                # New common neighbors: popcount(new_adj[u] & adj[a] & adj[b])
                # new_adj[u] = old_adj[u] XOR bit_v
                # Let X = adj[a] & adj[b]
                # Old count = popcount(old_adj[u] & X)
                # New count = popcount((old_adj[u] ^ bit_v) & X)
                # This equals Old count +/- 1 depending on whether bit_v was in X and in old_adj[u]
                
                X = current_adj[a] & current_adj[b] # Note: current_adj is already updated!
                # Wait, if we updated current_adj, we can't easily get "old" without reverting.
                # Let's revert the change, calculate delta, then apply.
                # This is safer.
                
                # REVERT
                current_adj[u] ^= bit_u
                current_adj[v] ^= bit_v
                
                # Calculate old state for this triple
                if u in (i, j, k):
                    others = [x for x in (i, j, k) if x != u]
                    a, b = others
                    X = current_adj[a] & current_adj[b]
                    old_count = (current_adj[u] & X).bit_count()
                    # New adj[u] has bit_v toggled
                    # new_count = popcount((current_adj[u] ^ (1<<v)) & X)
                    # If bit v is in X:
                    if X & (1 << v):
                        # If bit v was in current_adj[u], it was counted in old. Now removed.
                        # If bit v was not in current_adj[u], it was not counted. Now added.
                        if current_adj[u] & (1 << v):
                            new_count = old_count - 1
                        else:
                            new_count = old_count + 1
                    else:
                        new_count = old_count
                else:
                    # u not in triple, so v must be in triple (since affected)
                    others = [x for x in (i, j, k) if x != v]
                    a, b = others
                    X = current_adj[a] & current_adj[b]
                    old_count = (current_adj[v] & X).bit_count()
                    if X & (1 << u):
                        if current_adj[v] & (1 << u):
                            new_count = old_count - 1
                        else:
                            new_count = old_count + 1
                    else:
                        new_count = old_count
                
                # Determine violation change
                old_viol = 1 if old_count >= 3 else 0
                new_viol = 1 if new_count >= 3 else 0
                delta_violations += (new_viol - old_viol)
                
                # RE-APPLY
                current_adj[u] ^= bit_u
                current_adj[v] ^= bit_v
                continue # This logic is getting messy with the revert/apply per triple.
        
        # The above per-triple revert/apply is too slow and error-prone.
        # Let's stick to a simpler but slightly slower method: 
        # Revert, calculate full delta for affected triples, Apply.
        
        # REVERT
        current_adj[u] ^= bit_u
        current_adj[v] ^= bit_v
        
        delta_violations = 0
        for idx in affected:
            i, j, k = triples[idx]
            # Calculate old common count
            old_c = (current_adj[i] & current_adj[j] & current_adj[k]).bit_count()
            
            # Simulate new state for this triple
            # The edge (u,v) is toggled.
            # If u in (i,j,k):
            if u in (i, j, k):
                others = [x for x in (i, j, k) if x != u]
                a, b = others
                # New adj[u] = current_adj[u] ^ (1<<v)
                new_adj_u = current_adj[u] ^ (1 << v)
                new_c = (new_adj_u & current_adj[a] & current_adj[b]).bit_count()
            elif v in (i, j, k):
                others = [x for x in (i, j, k) if x != v]
                a, b = others
                new_adj_v = current_adj[v] ^ (1 << u)
                new_c = (current_adj[a] & current_adj[b] & new_adj_v).bit_count()
            else:
                new_c = old_c # Should not happen for affected triples
            
            old_viol = 1 if old_c >= 3 else 0
            new_viol = 1 if new_c >= 3 else 0
            delta_violations += (new_viol - old_viol)
            
        # Calculate delta score
        delta_edges = 1 if (current_adj[u] & bit_u) == 0 else -1
        # Note: if bit_u is 0, we are adding, so delta_edges = +1.
        # If bit_u is 1, we are removing, so delta_edges = -1.
        
        delta_score = delta_edges - 100 * delta_violations
        
        # Acceptance probability
        if delta_score > 0:
            accept = True
        else:
            # Metropolis criterion
            # P = exp(delta_score / T)
            # delta_score is negative
            try:
                p = pow(2.718281828, delta_score / T)
                accept = random.random() < p
            except OverflowError:
                accept = False
                
        if accept:
            # Apply change
            current_adj[u] ^= bit_u
            current_adj[v] ^= bit_v
            current_edges += delta_edges
            current_violations_count = len(current_violations) + delta_violations
            # We don't maintain the list of violations easily, just the count for score.
            # But we need to check if it's valid for best tracking.
            # We can just calculate score.
            current_score += delta_score
            
            # Check if new state is better than best
            if current_score > best_score:
                # Need to check if it's actually valid? 
                # If score is high, it likely is. But to be safe, 
                # if current_score > best_score, we should verify validity if we want to store a valid best.
                # However, if we store an invalid best, we might return it.
                # The task requires returning a VALID graph.
                # So we should only update best if the graph is valid.
                # Checking validity takes O(n^3) which is 1330 * popcount. Fast enough?
                # 1330 ops per check. If we check every improvement, it might be okay.
                # But we can just check if delta_violations leads to 0 violations?
                # We don't track the exact violation list, just count.
                # Let's maintain violation count.
                # If current_violations_count == 0, it is valid.
                
                if current_violations_count == 0:
                    best_adj = current_adj[:]
                    best_edges = current_edges
                    best_score = current_score
                    best_violations = 0
            # If not valid, we still keep it as current state for SA, but don't update best.
            
        else:
            # Reject, state remains reverted (which it is)
            pass
            
        T *= cooling_rate
        
    # After SA, return the best valid graph found
    # Ensure best_adj is valid
    if best_violations != 0:
        # Fallback: if we didn't find a valid graph (unlikely with SA from random), 
        # or if best_adj is empty
        # Let's try a simple greedy repair on the best_adj or just return a known valid graph
        # K2,18 is a safe fallback
        return [[a, b] for a in range(2) for b in range(2, 20)]
        
    # Convert best_adj to list of edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
                
    return edges
# EVOLVE-BLOCK-END
