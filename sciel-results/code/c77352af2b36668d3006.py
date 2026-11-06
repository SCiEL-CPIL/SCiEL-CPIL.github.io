import random
import time
from itertools import combinations

# Precompute partitions of range(8) into two sets of 4
partitions_8 = []
verts_8 = list(range(8))
for A_idx in combinations(verts_8, 4):
    B_idx = tuple(v for v in verts_8 if v not in A_idx)
    partitions_8.append((list(A_idx), list(B_idx)))

# Precompute all 8-subsets of range(16)
subsets_8 = list(combinations(range(16), 8))

def find_q3_edge(adj):
    """
    Returns one edge (u, v) that is part of a Q3 subgraph, or None if Q3-free.
    adj: list of sets.
    """
    for S in subsets_8:
        for A_idx, B_idx in partitions_8:
            A = [S[i] for i in A_idx]
            B = [S[i] for i in B_idx]
            
            edge_count = 0
            missing_edges = []
            present_edges = []
            
            for a in A:
                for b in B:
                    if b in adj[a]:
                        edge_count += 1
                        present_edges.append((a, b))
                    else:
                        missing_edges.append((a, b))
            
            if edge_count < 12:
                continue
                
            if len(missing_edges) > 4:
                continue
                
            used = set()
            is_matching = True
            for u, v in missing_edges:
                if u in used or v in used:
                    is_matching = False
                    break
                used.add(u)
                used.add(v)
                
            if is_matching:
                # Return a random edge from present_edges
                return random.choice(present_edges)
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140 # 10s margin
    
    best_edge_count = 0
    best_adj = None
    
    # Try different densities
    densities = [0.8, 0.85, 0.75, 0.9, 0.7]
    
    while time.time() < deadline:
        p = random.choice(densities)
        # Initialize graph
        adj = [set() for _ in range(16)]
        for u in range(16):
            for v in range(u + 1, 16):
                if random.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        
        current_count = sum(len(s) for s in adj) // 2
        
        # Prune
        while time.time() < deadline:
            edge_to_remove = find_q3_edge(adj)
            if edge_to_remove is None:
                break
            u, v = edge_to_remove
            adj[u].remove(v)
            adj[v].remove(u)
            current_count -= 1
        
        if current_count > best_edge_count:
            best_edge_count = current_count
            best_adj = [set(s) for s in adj]
            
    # Convert best_adj to list of lists
    if best_adj is None:
        return []
        
    final_edges = []
    for u in range(16):
        for v in best_adj[u]:
            if u < v:
                final_edges.append([u, v])
                
    return final_edges
