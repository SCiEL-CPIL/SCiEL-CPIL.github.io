import time
import random
import sys

DEADLINE = 330.0
SAFETY_MARGIN = 5.0

def _paley_graph(q):
    """Construct Paley graph on q vertices. Returns list of bitmasks."""
    residues = {(x * x) % q for x in range(1, q)}
    adj = [0] * q
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return q, adj

def _double_cover_paley(q, time_start=None):
    """
    Construct a double cover of the Paley graph on q vertices.
    Vertices are (i, parity) for i in 0..q-1, parity in {0,1}.
    Vertex index = i * 2 + parity.
    Edge between (i, p) and (j, p') if:
      - i != j
      - (j - i) mod q is a quadratic residue mod q (Paley adjacency)
      - p XOR p' = 1 (parity must differ)
    This creates a bipartite-like structure that may help avoid K5 and I5.
    """
    n = 2 * q
    residues = {(x * x) % q for x in range(1, q)}
    adj = [0] * n
    
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                # Connect (i,0)-(j,1) and (i,1)-(j,0)
                # Vertex indices: i*2+0, i*2+1, j*2+0, j*2+1
                v_i0 = i * 2
                v_i1 = i * 2 + 1
                v_j0 = j * 2
                v_j1 = j * 2 + 1
                
                # (i,0) connected to (j,1)
                adj[v_i0] |= (1 << v_j1)
                adj[v_j1] |= (1 << v_i0)
                
                # (i,1) connected to (j,0)
                adj[v_i1] |= (1 << v_j0)
                adj[v_j0] |= (1 << v_i1)
    
    # Also handle wrap-around: (q-1, p) to (0, p')
    # (j - i) mod q where j=0, i=q-1: (0 - (q-1)) % q = 1
    # So if 1 is a residue, connect (q-1,0)-(0,1) and (q-1,1)-(0,0)
    # But we already handled i<j pairs. For i=q-1, j=0: j-i = -(q-1) mod q = 1
    # We need to add these edges separately since j < i in our loop.
    for i in range(q):
        for j in range(q):
            if i == j:
                continue
            diff = (j - i) % q
            if diff in residues:
                # We've already added edges for i<j. 
                # For i>j, we add the complementary edge.
                if i > j:
                    v_i0 = i * 2
                    v_i1 = i * 2 + 1
                    v_j0 = j * 2
                    v_j1 = j * 2 + 1
                    
                    # (i,0) connected to (j,1)
                    adj[v_i0] |= (1 << v_j1)
                    adj[v_j1] |= (1 << v_i0)
                    
                    # (i,1) connected to (j,0)
                    adj[v_i1] |= (1 << v_j0)
                    adj[v_j0] |= (1 << v_i1)
    
    return n, adj

def _prune_to_55_free(adj, n, time_start):
    """
    Local search to prune edges from the double cover to make it (5,5)-free.
    Start with the full double cover, remove edges that create K5 or I5.
    """
    deadline = time_start + (DEADLINE - SAFETY_MARGIN)
    current_adj = adj[:]
    current_n = n
    
    # First, check if the full double cover is already (5,5)-free
    if _is_55_free(current_adj, current_n):
        return current_n, current_adj
    
    # Otherwise, we need to prune.
    # Strategy: iteratively find a K5 or I5, and remove an edge to break it.
    # This is slow but guarantees validity.
    
    # Better: use simulated annealing to find a large (5,5)-free subgraph.
    # Objective: maximize number of vertices while maintaining (5,5)-free.
    # But we're stuck with fixed n. So: maximize edges while (5,5)-free.
    # Actually, we want to keep as many vertices as possible.
    # The double cover has 74 vertices. We may need to remove some vertices.
    
    # Alternative: start with the full double cover, and use local search to
    # find a (5,5)-free subgraph. If we can't keep all 74, we'll end up with fewer.
    
    # Let's try a different approach: use the double cover structure to guide
    # which vertices to keep. The parity structure may help.
    
    # For now, return the original (5,5)-free check result.
    # The caller will fall back to the Paley graph if this doesn't work.
    
    return current_n, current_adj

def _has_k4_in_mask(mask, adj, n):
    """Check if the subgraph induced by vertices in 'mask' contains a K4."""
    # Extract vertices in mask
    verts = []
    m = mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        verts.append(v)
        m ^= lsb
    
    if len(verts) < 4:
        return False
        
    # Check for K4 among these vertices
    # For efficiency, only check if the induced subgraph is dense enough?
    # Simple O(k^4) check where k is popcount(mask)
    # If k is large, this is slow. But in the search, k might be around n/2.
    # For n=37, k~18, 18^4 = 100k ops. This is fast enough for a heuristic step.
    
    for i in range(len(verts)):
        vi = verts[i]
        for j in range(i+1, len(verts)):
            vj = verts[j]
            if not (adj[vi] & (1 << vj)):
                continue
            for k in range(j+1, len(verts)):
                vk = verts[k]
                if not (adj[vi] & (1 << vk)) or not (adj[vj] & (1 << vk)):
                    continue
                for l in range(k+1, len(verts)):
                    vl = verts[l]
                    if (adj[vi] & (1 << vl)) and (adj[vj] & (1 << vl)) and \
                       (adj[vk] & (1 << vl)):
                        return True
    return False

def _has_i4_in_mask(mask, adj, n):
    """Check if the subgraph induced by vertices in 'mask' contains an independent set of size 4 (I4)."""
    # This is equivalent to checking if the complement subgraph has a K4.
    # Construct complement adjacency for the vertices in mask?
    # Or just check directly: 4 vertices with no edges between them.
    
    verts = []
    m = mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        verts.append(v)
        m ^= lsb
        
    if len(verts) < 4:
        return False
        
    for i in range(len(verts)):
        vi = verts[i]
        for j in range(i+1, len(verts)):
            vj = verts[j]
            if (adj[vi] & (1 << vj)):
                continue
            for k in range(j+1, len(verts)):
                vk = verts[k]
                if (adj[vi] & (1 << vk)) or (adj[vj] & (1 << vk)):
                    continue
                for l in range(k+1, len(verts)):
                    vl = verts[l]
                    if (adj[vi] & (1 << vl)) or (adj[vj] & (1 << vl)) or (adj[vk] & (1 << vl)):
                        continue
                    return True
    return False

def _count_violations(S_mask, adj, n):
    """Count K4s in S and I4s in complement of S."""
    # We just need to know if there are ANY, for the search.
    # But for annealing, a count is better.
    # However, counting all K4s is expensive.
    # Let's just return 1 if violation exists, 0 otherwise.
    # This makes it a boolean search, which might get stuck in local minima.
    # Better: return the number of violating cliques?
    # Let's stick to boolean for now, but use a randomized restart strategy.
    
    if _has_k4_in_mask(S_mask, adj, n):
        return 1
    comp_mask = ((1 << n) - 1) ^ S_mask
    if _has_i4_in_mask(comp_mask, adj, n):
        return 1
    return 0

def _try_add_vertex(adj, n, time_start):
    """Try to find a valid neighborhood S for a new vertex."""
    full_mask = (1 << n) - 1
    
    # Heuristic initial S: 
    # Maybe start with a random subset of size n/2?
    # Or start with a small set and grow?
    # Let's try a few random initializations.
    
    best_S = None
    
    for init_attempt in range(20):
        if time.time() - time_start > DEADLINE - SAFETY_MARGIN:
            return None
            
        # Random initialization
        # Bias towards balanced sets?
        S_mask = 0
        for i in range(n):
            if random.random() < 0.5:
                S_mask |= (1 << i)
                
        # Local search / Simulated Annealing
        T = 1.0
        for step in range(1000):
            if time.time() - time_start > DEADLINE - SAFETY_MARGIN:
                return None
                
            current_violations = _count_violations(S_mask, adj, n)
            
            if current_violations == 0:
                return S_mask
            
            # Pick a random vertex to flip
            v = random.randint(0, n - 1)
            bit = 1 << v
            new_S_mask = S_mask ^ bit
            
            new_violations = _count_violations(new_S_mask, adj, n)
            
            # Accept if better, or with prob exp(-delta/T)
            delta = new_violations - current_violations
            if delta < 0 or random.random() < pow(2.71828, -delta / T):
                S_mask = new_S_mask
                
            T *= 0.995 # Cool down
    
    return None

def _is_55_free(adj, n):
    """Check if graph is (5,5)-free using bitsets. Optimized for larger n."""
    if n < 5:
        return True
    
    # Check K5
    for i in range(n):
        for j in range(i+1, n):
            if (adj[i] >> j) & 1:
                common = adj[i] & adj[j]
                # We need a K3 in common
                mask = common
                while mask:
                    lsb = mask & -mask
                    k = lsb.bit_length() - 1
                    mask ^= lsb
                    if k <= j: continue # Ensure i < j < k
                    C = adj[k] & common
                    if C & (C - 1):
                        mask_c = C
                        while mask_c:
                            lsb_c = mask_c & -mask_c
                            u = lsb_c.bit_length() - 1
                            mask_c ^= lsb_c
                            if (adj[u] & C):
                                return False # Found K5
    
    # Check I5 (K5 in complement)
    comp_adj = [0] * n
    full_mask = (1 << n) - 1
    for i in range(n):
        comp_adj[i] = full_mask ^ (1 << i) ^ adj[i]
        
    for i in range(n):
        for j in range(i+1, n):
            if (comp_adj[i] >> j) & 1:
                common = comp_adj[i] & comp_adj[j]
                mask = common
                while mask:
                    lsb = mask & -mask
                    k = lsb.bit_length() - 1
                    mask ^= lsb
                    if k <= j: continue
                    C = comp_adj[k] & common
                    if C & (C - 1):
                        mask_c = C
                        while mask_c:
                            lsb_c = mask_c & -mask_c
                            u = lsb_c.bit_length() - 1
                            mask_c ^= lsb_c
                            if (comp_adj[u] & C):
                                return False # Found I5
    return True

def _to_edges(adj, n):
    """Convert bitset adjacency to list of (i, j) edges."""
    edges = []
    for i in range(n):
        # Get bits set in adj[i] greater than i
        mask = adj[i] & ~((1 << (i+1)) - 1)
        while mask:
            lsb = mask & -mask
            j = lsb.bit_length() - 1
            edges.append((i, j))
            mask ^= lsb
    return edges

def _try_prune_double_cover(adj, n, time_start):
    """
    Try to prune the double cover to get a (5,5)-free graph with as many vertices as possible.
    Uses local search: remove edges or vertices to eliminate K5 and I5.
    """
    deadline = time_start + (DEADLINE - SAFETY_MARGIN)
    current_adj = adj[:]
    current_n = n
    
    # If already (5,5)-free, we're done
    if _is_55_free(current_adj, current_n):
        return current_n, current_adj
    
    # Otherwise, try to prune vertices.
    # Strategy: iteratively remove a vertex that is part of a K5 or I5.
    # This is slow, so we limit the number of iterations.
    
    max_iterations = 50
    for iter_num in range(max_iterations):
        if time.time() > deadline:
            break
        
        if _is_55_free(current_adj, current_n):
            break
        
        # Find a vertex to remove.
        # Heuristic: remove a vertex with high degree or high "violating" degree.
        # For simplicity, try removing each vertex and see if it helps.
        
        best_remove = -1
        best_score = float('inf')
        
        for v in range(current_n):
            if time.time() > deadline:
                break
            
            # Remove vertex v
            new_n = current_n - 1
            new_adj = []
            for i in range(current_n):
                if i == v:
                    continue
                # Remap indices
                new_mask = 0
                for j in range(current_n):
                    if i == j or j == v:
                        continue
                    if (current_adj[i] >> j) & 1:
                        # Map j to new index
                        if j < v:
                            new_j = j
                        else:
                            new_j = j - 1
                        new_mask |= (1 << new_j)
                new_adj.append(new_mask)
            
            # Check if (5,5)-free
            if _is_55_free(new_adj, new_n):
                # Score: prefer removing vertex with lower degree
                degree = bin(current_adj[v]).count('1')
                if degree < best_score:
                    best_score = degree
                    best_remove = v
        
        if best_remove >= 0:
            # Remove the best vertex
            new_n = current_n - 1
            new_adj = []
            for i in range(current_n):
                if i == best_remove:
                    continue
                new_mask = 0
                for j in range(current_n):
                    if i == j or j == best_remove:
                        continue
                    if (current_adj[i] >> j) & 1:
                        if j < best_remove:
                            new_j = j
                        else:
                            new_j = j - 1
                        new_mask |= (1 << new_j)
                new_adj.append(new_mask)
            
            current_adj = new_adj
            current_n = new_n
            print(f"Pruned to {current_n} vertices", file=sys.stderr)
        else:
            # Can't find a single vertex removal that works.
            # Try edge removals?
            break
    
    return current_n, current_adj

def construct():
    """Construct a (5,5)-free graph. Returns (n, edges)."""
    time_start = time.time()
    
    # Strategy 1: Start with Paley graph on 37 vertices
    q, adj_paley = _paley_graph(37)
    best_n = q
    best_adj = adj_paley[:]
    
    # Verify seed
    if not _is_55_free(adj_paley, q):
        raise ValueError("Seed graph is not (5,5)-free")
        
    random.seed(42)
    
    # Strategy 2: Try double cover approach
    # The double cover has 74 vertices. Prune to (5,5)-free.
    n_dc, adj_dc = _double_cover_paley(37, time_start)
    
    if n_dc > best_n:
        # Try to prune the double cover
        n_pruned, adj_pruned = _try_prune_double_cover(adj_dc, n_dc, time_start)
        
        if n_pruned > best_n and _is_55_free(adj_pruned, n_pruned):
            best_n = n_pruned
            best_adj = adj_pruned[:]
            print(f"Double cover pruned to {best_n} vertices", file=sys.stderr)
    
    # Strategy 3: Try to extend the best graph found so far
    current_adj = best_adj[:]
    current_n = best_n
    
    while time.time() - time_start < DEADLINE - SAFETY_MARGIN:
        # Try to add a vertex
        S_mask = _try_add_vertex(current_adj, current_n, time_start)
        
        if S_mask is not None:
            # Add new vertex
            new_n = current_n + 1
            new_adj = current_adj[:] + [0]
            
            for i in range(current_n):
                if (S_mask >> i) & 1:
                    new_adj[i] |= (1 << new_n)
                    new_adj[new_n] |= (1 << i)
            
            current_adj = new_adj
            current_n = new_n
            
            if current_n > best_n:
                best_n = current_n
                best_adj = current_adj[:]
                print(f"Extended to {current_n} vertices", file=sys.stderr)
        else:
            # Stuck. Try random edge flips to unblock
            improved = False
            for _ in range(50):
                if time.time() - time_start > DEADLINE - SAFETY_MARGIN:
                    break
                    
                i = random.randint(0, current_n - 1)
                j = random.randint(0, current_n - 1)
                if i == j: continue
                
                bit_i = 1 << j
                bit_j = 1 << i
                current_adj[i] ^= bit_i
                current_adj[j] ^= bit_j
                
                if _is_55_free(current_adj, current_n):
                    improved = True
                    break
                else:
                    current_adj[i] ^= bit_i
                    current_adj[j] ^= bit_j
            
            if not improved:
                break
                
    edges = _to_edges(best_adj, best_n)
    return best_n, edges

if __name__ == "__main__":
    n, edges = construct()
    print(f"Constructed graph with {n} vertices and {len(edges)} edges")
