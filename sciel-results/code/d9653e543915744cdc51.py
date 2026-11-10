import time
import random
import itertools

def construct():
    deadline = time.time() + 115
    n = 16
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Precompute Q3 structure
    # Q3 vertices 0..7, edges where popcount(i^j)==1
    q3_edges = []
    for i in range(8):
        for j in range(i+1, 8):
            if bin(i ^ j).count('1') == 1:
                q3_edges.append((i, j))
        
    # Precompute all 840 distinct Q3 edge masks in K8
    # An edge (a,b) in K8 (0<=a<b<=7) is mapped to index a*8+b
    # We use a set of masks. Each mask is a bitset of 64 bits (8x8 matrix flattened)
    # Actually, since we only care about a<b, we can use a 28-bit mask or just the 64-bit one.
    # Let's use the 64-bit one for simplicity, index = a*8+b.
    
    q3_masks = set()
    # We need to generate all embeddings of Q3 into K8.
    # There are 8! / 48 = 840 such embeddings.
    # Instead of iterating permutations, we can iterate over all subsets of 8 vertices?
    # No, the vertices are fixed 0..7 in the local coordinate system.
    # We iterate over all permutations of 0..7 to map Q3 vertices to local vertices.
    
    # Optimization: We can just generate the masks once.
    # To speed up generation, we can note that Q3 is the hypercube.
    # But 8! is 40320, which is small enough to do at startup.
    
    # Actually, we can be smarter.
    # A Q3 is defined by its 12 edges.
    # Let's just use the permutations.
    for perm in itertools.permutations(range(8)):
        mask = 0
        for u, v in q3_edges:
            a, b = perm[u], perm[v]
            if a > b:
                a, b = b, a
            # Map to index in 8x8 grid
            idx = a * 8 + b
            mask |= (1 << idx)
        q3_masks.add(mask)
        
    q3_masks_list = list(q3_masks)
    
    # Helper to check if a graph (represented by adjacency masks for 16 vertices)
    # contains a Q3.
    # We check all 8-subsets of {0..15}.
    # For each 8-subset S, we build the induced edge mask (in local 0..7 coords).
    # If any q3_mask is a subset of the induced mask, return True.
    
    all_8_subsets = list(itertools.combinations(range(n), 8))
    
    def has_q3(adj):
        # adj: list of 16 integers, adj[u] is bitmask of neighbors
        for subset in all_8_subsets:
            # Build local edge mask
            local_mask = 0
            # Map global vertex to local index 0..7
            # subset[i] is global vertex for local index i
            for i in range(8):
                for j in range(i+1, 8):
                    u, v = subset[i], subset[j]
                    if adj[u] & (1 << v):
                        local_mask |= (1 << (i * 8 + j))
            
            # Check if any Q3 mask is subset of local_mask
            # local_mask contains edges (i,j) where i<j.
            # q3_masks also contain edges (a,b) where a<b.
            # So we can directly check (local_mask & q3_mask) == q3_mask
            for qm in q3_masks_list:
                if (local_mask & qm) == qm:
                    return True
        return False

    # Greedy Construction
    # Start with empty graph
    adj = [0] * n
    edge_list = []
    
    # Shuffle edges to break symmetry and get varied results
    random.shuffle(all_edges)
    
    for u, v in all_edges:
        if time.time() > deadline - 10:
            break
        # Tentatively add edge
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        if has_q3(adj):
            # Remove it
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            edge_list.append([u, v])
            
    # Local Search Refinement
    # Try to improve the graph by swapping edges.
    # Or simply, try to add more edges if possible?
    # The greedy method produces a maximal Q3-free graph.
    # To improve, we can try to remove an edge that is "low impact" and add a different one.
    # Or, restart greedy with different shuffles.
    
    best_edges = edge_list
    best_count = len(edge_list)
    
    # Use remaining time for restarts or local search
    # Since has_q3 is expensive (12870 subsets * 840 masks), we can't do many iterations.
    # Let's estimate cost.
    # 12870 * 840 ~ 10.8M operations per has_q3 call.
    # In Python, 10M ops takes ~0.5-1s.
    # So we can afford ~100-200 has_q3 calls in 100s.
    # Greedy takes 120 has_q3 calls.
    # So we have time for ~1-2 more restarts or a small local search.
    
    # Let's do a few restarts with different seeds/shuffles
    start_time = time.time()
    while time.time() < deadline - 5:
        random.shuffle(all_edges)
        adj = [0] * n
        edge_list = []
        
        for u, v in all_edges:
            if time.time() > deadline - 5:
                break
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if has_q3(adj):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                edge_list.append([u, v])
                
        if len(edge_list) > best_count:
            best_count = len(edge_list)
            best_edges = edge_list[:]
            
    return best_edges

# Precomputation at module level to save time in construct?
# No, construct is called once. But precomputing q3_masks_list takes time.
# Let's move q3_masks generation to module level if possible, or just let it run.
# 40320 permutations * 12 edges = ~500k ops. Fast enough.

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(len(res))
