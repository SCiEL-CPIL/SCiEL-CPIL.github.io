import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

DEADLINE = time.time() + 265  # Safety margin

def _build_paley(q):
    """Return adjacency bitset list for Paley graph on q vertices (vertices 0..q-1)."""
    residues = {(x * x) % q for x in range(1, q)}
    adj = [0] * q
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _has_k_clique(adj, n, k):
    """Check if graph has a clique of size k. Returns True if found."""
    def search(current_clique, candidates):
        if len(current_clique) == k:
            return True
        if len(candidates) < k - len(current_clique):
            return False
        for i, v in enumerate(candidates):
            new_candidates = [u for u in candidates[i+1:] if (adj[v] & (1 << u))]
            if search(current_clique + [v], new_candidates):
                return True
        return False
    return search([], list(range(n)))

def _has_ind_clique(adj, n, k):
    """Check if graph has an independent set of size k. Returns True if found."""
    comp_adj = [0] * n
    for i in range(n):
        all_bits = (1 << n) - 1
        comp_adj[i] = all_bits & ~(adj[i] | (1 << i))
    return _has_k_clique(comp_adj, n, k)

def _is_valid(adj, n, k=5):
    """Check if graph is (k,k)-free."""
    if _has_k_clique(adj, n, k):
        return False
    if _has_ind_clique(adj, n, k):
        return False
    return True

def _find_all_ksets(adj, n, k, is_clique=True):
    """Find all sets of size k that are cliques (or independent sets if is_clique=False)."""
    if not is_clique:
        # Independent set = clique in complement
        comp_adj = [0] * n
        for i in range(n):
            all_bits = (1 << n) - 1
            comp_adj[i] = all_bits & ~(adj[i] | (1 << i))
        adj = comp_adj
    
    result = []
    def search(start, current):
        if len(current) == k:
            result.append(tuple(current))
            return
        remaining = k - len(current)
        for i in range(start, n - remaining + 1):
            v = i
            # Check if v is adjacent to all in current
            ok = True
            for u in current:
                if not (adj[v] & (1 << u)):
                    ok = False
                    break
            if ok:
                search(i + 1, current + [v])
    search(0, [])
    return result

def _try_extend(adj, n, max_time):
    """Try to extend graph by adding vertices one at a time using constraint-driven greedy."""
    start_time = time.time()
    k = 5
    
    # Precompute all (k-1)-cliques and (k-1)-independent sets
    k4_cliques = _find_all_ksets(adj, n, k-1, is_clique=True)
    k4_indeps = _find_all_ksets(adj, n, k-1, is_clique=False)
    
    best_n = n
    best_adj = adj[:]
    
    # Try to add up to 9 vertices (target n=46)
    for target_n in range(n + 1, min(n + 10, 47)):
        if time.time() - start_time > max_time * 0.8:
            break
        
        # Try multiple random restarts for this target size
        found = False
        for restart in range(50):
            if time.time() - start_time > max_time:
                break
            
            # Work with current best
            cur_adj = best_adj[:]
            cur_n = best_n
            
            # Try to add one vertex
            new_v = cur_n
            # Determine adjacency to existing vertices
            # Constraints:
            # 1. For each 4-clique C: new_v cannot be adjacent to all 4 vertices in C
            # 2. For each 4-independent set I: new_v cannot be non-adjacent to all 4 vertices in I
            
            # Build constraint sets
            must_not_all_connect = set()  # sets of 4 vertices where connecting to all would create K5
            must_not_all_disconnect = set()  # sets of 4 vertices where disconnecting from all would create ind-5
            
            for c4 in k4_cliques:
                must_not_all_connect.add(c4)
            for i4 in k4_indeps:
                must_not_all_disconnect.add(i4)
            
            # Greedy assignment with randomization
            # Start with a random initial assignment
            adj_new = [0] * (cur_n + 1)
            # Copy existing edges
            for i in range(cur_n):
                adj_new[i] = cur_adj[i]
            
            # Assign adjacency of new vertex to existing vertices
            # Use a randomized greedy approach
            unassigned = list(range(cur_n))
            random.shuffle(unassigned)
            
            # Initial random assignment
            new_adj_bits = 0
            for v in unassigned:
                if random.random() < 0.5:
                    new_adj_bits |= (1 << v)
            
            # Check if this assignment is valid
            def _check_valid_assignment(new_adj_bits, cur_n, must_not_all_connect, must_not_all_disconnect):
                for c4 in must_not_all_connect:
                    # Check if new vertex connects to all 4
                    all_connect = True
                    for v in c4:
                        if not (new_adj_bits & (1 << v)):
                            all_connect = False
                            break
                    if all_connect:
                        return False
                for i4 in must_not_all_disconnect:
                    # Check if new vertex disconnects from all 4
                    all_disconnect = True
                    for v in i4:
                        if (new_adj_bits & (1 << v)):
                            all_disconnect = False
                            break
                    if all_disconnect:
                        return False
                return True
            
            if _check_valid_assignment(new_adj_bits, cur_n, must_not_all_connect, must_not_all_disconnect):
                # Valid! Add the vertex
                for v in range(cur_n):
                    if new_adj_bits & (1 << v):
                        adj_new[v] |= (1 << new_v)
                        adj_new[new_v] |= (1 << v)
                best_n = cur_n + 1
                best_adj = adj_new
                found = True
                break
            else:
                # Try local search: flip bits to satisfy constraints
                for step in range(100):
                    if time.time() - start_time > max_time:
                        break
                    # Pick a random violated constraint and flip a bit
                    violated = None
                    for c4 in must_not_all_connect:
                        all_connect = all((new_adj_bits & (1 << v)) for v in c4)
                        if all_connect:
                            violated = c4
                            break
                    if violated is None:
                        for i4 in must_not_all_disconnect:
                            all_disconnect = all(not (new_adj_bits & (1 << v)) for v in i4)
                            if all_disconnect:
                                violated = i4
                                break
                    
                    if violated is None:
                        # Check if valid now
                        if _check_valid_assignment(new_adj_bits, cur_n, must_not_all_connect, must_not_all_disconnect):
                            for v in range(cur_n):
                                if new_adj_bits & (1 << v):
                                    adj_new[v] |= (1 << new_v)
                                    adj_new[new_v] |= (1 << v)
                            best_n = cur_n + 1
                            best_adj = adj_new
                            found = True
                            break
                        else:
                            # Random flip
                            v = random.choice(unassigned)
                            if new_adj_bits & (1 << v):
                                new_adj_bits &= ~(1 << v)
                            else:
                                new_adj_bits |= (1 << v)
                    else:
                        # Flip one of the vertices in the violated set
                        v = random.choice(list(violated))
                        if new_adj_bits & (1 << v):
                            new_adj_bits &= ~(1 << v)
                        else:
                            new_adj_bits |= (1 << v)
                
                if found:
                    break
        
        if not found:
            break
    
    return best_n, best_adj

def construct():
    # Try Paley(41) first
    q = 41
    adj41 = _build_paley(q)
    if _is_valid(adj41, q, 5):
        edges = []
        for i in range(q):
            for j in range(i + 1, q):
                if adj41[i] & (1 << j):
                    edges.append((i, j))
        return q, edges

    # Start with Paley(37)
    q = 37
    adj37 = _build_paley(q)
    
    # Try to extend
    remaining_time = DEADLINE - time.time()
    if remaining_time > 30:
        extended_n, extended_adj = _try_extend(adj37, q, remaining_time - 10)
        if extended_n > q:
            q = extended_n
            adj37 = extended_adj
    
    # Build edge list
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if adj37[i] & (1 << j):
                edges.append((i, j))
    return q, edges

# EVOLVE-BLOCK-END
