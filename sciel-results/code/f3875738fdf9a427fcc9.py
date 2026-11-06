import itertools
import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 16
    vertices = list(range(n))
    
    # Precompute all 8-vertex subsets as bitmasks
    subsets_8 = list(itertools.combinations(vertices, 8))
    
    # Precompute for each 8-subset: list of 70 partitions (as two 4-vertex tuples)
    # Partition index: which 4 of the 8 positions go to X
    partition_indices = list(itertools.combinations(range(8), 4))
    
    # For each 8-subset, precompute which positions belong to X for each partition
    # partition_data[subset_idx] = list of (x_positions_tuple, y_positions_tuple)
    partition_data = []
    for s in subsets_8:
        parts = []
        for x_combo in partition_indices:
            x_set = set(x_combo)
            y_set = tuple(j for j in range(8) if j not in x_set)
            x_set_tuple = tuple(x_combo)
            parts.append((x_set_tuple, y_set))
        partition_data.append(parts)
    
    # Precompute subsets containing each pair of vertices
    # For fast incremental Q3 check: subsets_containing_edge[u][v] = list of subset indices
    subsets_containing_edge = {}
    for idx, s in enumerate(subsets_8):
        s_set = set(s)
        for i in range(8):
            for j in range(i+1, 8):
                u, v = s[i], s[j]
                key = (min(u,v), max(u,v))
                if key not in subsets_containing_edge:
                    subsets_containing_edge[key] = []
                subsets_containing_edge[key].append(idx)
    
    # For fast Q3 check: use bitmask adjacency
    # adj_mask[v] is a 16-bit integer where bit u is set if edge (v,u) exists
    
    def has_q3_full(adj_mask):
        """Check if graph (represented as list of 16-bit masks) contains Q3."""
        for idx, s in enumerate(subsets_8):
            # Count edges in s using bitmasks
            edge_count = 0
            for i in range(8):
                v = s[i]
                mask = adj_mask[v]
                for j in range(i+1, 8):
                    if mask & (1 << s[j]):
                        edge_count += 1
            if edge_count < 12:
                continue
            
            # For each partition into X, Y of size 4
            # Check if the bipartite graph between X and Y has min degree >= 3
            for x_set_tuple, y_set in partition_data[idx]:
                # Check min degree from X side
                valid = True
                for i in x_set_tuple:
                    v = s[i]
                    deg = 0
                    for j in y_set:
                        if adj_mask[v] & (1 << s[j]):
                            deg += 1
                    if deg < 3:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check min degree from Y side
                for j in y_set:
                    u = s[j]
                    deg = 0
                    for i in x_set_tuple:
                        if adj_mask[u] & (1 << s[i]):
                            deg += 1
                    if deg < 3:
                        valid = False
                        break
                
                if valid:
                    return True
        return False
    
    def has_q3_edge(adj_mask, u, v):
        """Check if adding edge (u,v) creates a Q3. Only check subsets containing both u and v."""
        key = (min(u,v), max(u,v))
        if key not in subsets_containing_edge:
            return False
        
        for idx in subsets_containing_edge[key]:
            s = subsets_8[idx]
            # Count edges in s using bitmasks (edge u-v is already in adj_mask)
            edge_count = 0
            for i in range(8):
                vi = s[i]
                mask = adj_mask[vi]
                for j in range(i+1, 8):
                    if mask & (1 << s[j]):
                        edge_count += 1
            if edge_count < 12:
                continue
            
            # For each partition into X, Y of size 4
            # Check if the bipartite graph between X and Y has min degree >= 3
            for x_set_tuple, y_set in partition_data[idx]:
                # Check min degree from X side
                valid = True
                for i in x_set_tuple:
                    vi = s[i]
                    deg = 0
                    for j in y_set:
                        if adj_mask[vi] & (1 << s[j]):
                            deg += 1
                    if deg < 3:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check min degree from Y side
                for j in y_set:
                    uj = s[j]
                    deg = 0
                    for i in x_set_tuple:
                        if adj_mask[uj] & (1 << s[i]):
                            deg += 1
                    if deg < 3:
                        valid = False
                        break
                
                if valid:
                    return True
        return False
    
    def build_adj_mask(edge_set):
        """Build adjacency mask list from edge set."""
        adj_mask = [0] * n
        for u, v in edge_set:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        return adj_mask
    
    # Try Shrikhande graph complement as initial candidate
    # Shrikhande graph: strongly regular (16, 6, 2, 2)
    # Complement has degree 16-1-6=9, so 16*9/2=72 edges
    # Actually let me construct the Shrikhande graph properly
    
    # Shrikhande graph vertices: Z_4 x Z_4
    # Two vertices (a,b) and (c,d) are adjacent if:
    # - They differ in exactly one coordinate by 1 mod 4, OR
    # - They differ in both coordinates by 2 mod 4 (i.e., (a+2,b+2) or (a-2,b-2))
    
    def shrikhande_adj(u, v):
        """Check if u,v are adjacent in Shrikhande graph. u,v in 0..15."""
        a, b = divmod(u, 4)
        c, d = divmod(v, 4)
        da = (a - c) % 4
        db = (b - d) % 4
        # Adjacent if:
        # 1. Same row, adjacent col: da==0 and (db==1 or db==3)
        # 2. Same col, adjacent row: db==0 and (da==1 or da==3)
        # 3. Diagonal: da==2 and db==2
        if da == 0 and (db == 1 or db == 3):
            return True
        if db == 0 and (da == 1 or da == 3):
            return True
        if da == 2 and db == 2:
            return True
        return False
    
    # Build Shrikhande graph edges
    shrikhande_edges = set()
    for u in range(16):
        for v in range(u+1, 16):
            if shrikhande_adj(u, v):
                shrikhande_edges.add((u, v))
    
    # Complement of Shrikhande
    all_edges = set((u, v) for u in range(16) for v in range(u+1, 16))
    complement_edges = all_edges - shrikhande_edges
    
    # Check if complement is Q3-free
    adj_mask = build_adj_mask(complement_edges)
    if not has_q3_full(adj_mask):
        best_edges = set(complement_edges)
        best_count = len(best_edges)
    else:
        # Try other structured graphs
        # Try complete bipartite K_8,8 minus some edges
        # Or try a random dense graph and prune
        best_edges = set()
        best_count = 0
        
        # Try several random dense graphs
        for attempt in range(10):
            if time.time() > deadline - 60:
                break
            random_graph = set()
            # Start with a random graph of ~70 edges
            all_edge_list = list(all_edges)
            random.shuffle(all_edge_list)
            # Try to build up from a moderate density
            current = set(all_edge_list[:60])
            adj_m = build_adj_mask(current)
            if not has_q3_full(adj_m):
                if len(current) > best_count:
                    best_count = len(current)
                    best_edges = set(current)
            
            # Greedy add
            for e in all_edge_list[60:]:
                if time.time() > deadline - 30:
                    break
                u, v = e
                if (u, v) in current:
                    continue
                adj_m[u] |= (1 << v)
                adj_m[v] |= (1 << u)
                if has_q3_edge(adj_m, u, v):
                    adj_m[u] &= ~(1 << v)
                    adj_m[v] &= ~(1 << u)
                else:
                    current.add((u, v))
            
            if len(current) > best_count:
                best_count = len(current)
                best_edges = set(current)
        
        # If still nothing good, use a known Q3-free construction
        if best_count < 50:
            # K_3,13 + K_3 (but verify it's actually Q3-free)
            candidate = set()
            for a in range(3):
                for b in range(3, 16):
                    candidate.add((a, b))
            for u, v in [(0,1), (1,2), (0,2)]:
                candidate.add((u, v))
            adj_m = build_adj_mask(candidate)
            if not has_q3_full(adj_m):
                if len(candidate) > best_count:
                    best_count = len(candidate)
                    best_edges = set(candidate)
            else:
                # Very conservative: sparse graph
                best_edges = set()
                best_count = 0
    
    # Local search: try to add edges to improve using fast edge-specific Q3 check
    adj_mask = build_adj_mask(best_edges)
    current_edges = set(best_edges)
    
    # Get all possible edges not in current graph
    candidate_edges = list(all_edges - current_edges)
    
    # Multiple passes of greedy addition with edge-specific Q3 check
    for restart in range(30):
        if time.time() > deadline - 10:
            break
        
        random.shuffle(candidate_edges)
        
        # Reset to best
        current_edges = set(best_edges)
        adj_mask = build_adj_mask(current_edges)
        
        for u, v in candidate_edges:
            if time.time() > deadline - 5:
                break
            if (u, v) in current_edges:
                continue
            
            # Try adding edge
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            
            if has_q3_edge(adj_mask, u, v):
                # Revert
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
            else:
                current_edges.add((u, v))
            
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = set(current_edges)
            # Update candidate list
            candidate_edges = list(all_edges - best_edges)
    
    # Final verification
    adj_mask = build_adj_mask(best_edges)
    if has_q3_full(adj_mask):
        # Fallback: remove edges until Q3-free
        edge_list = list(best_edges)
        random.shuffle(edge_list)
        current = set(best_edges)
        adj_m = build_adj_mask(current)
        for e in edge_list:
            if has_q3_full(adj_m):
                u, v = e
                adj_m[u] &= ~(1 << v)
                adj_m[v] &= ~(1 << u)
                current.discard(e)
        best_edges = current
        best_count = len(best_edges)
    
    result = [[u, v] for u, v in best_edges]
    return result

if __name__ == "__main__":
    start = time.time()
    edges = construct()
    print(f"Found {len(edges)} edges in {time.time() - start:.2f}s")
