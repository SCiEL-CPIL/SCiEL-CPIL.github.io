import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Precompute perms24
    perms24 = list(itertools.permutations(range(4)))
    
    # Precompute all 8-vertex subsets as tuples for faster lookup
    all_8subsets = list(itertools.combinations(range(16), 8))
    
    # Precompute for each 8-subset: the 70 A-splits (4 of 8) and for each split,
    # precompute which B-permutations need to be checked
    # Actually, let's precompute a lookup table: for each 8-subset, store the 70 splits
    # as (A_mask, B_indices) pairs
    
    # Better approach: precompute for each 8-subset the list of (A_idx_tuple, B_idx_tuple)
    # where A_idx and B_idx are tuples of positions (0-7) within the subset
    split_cache = {}
    for subset in all_8subsets:
        splits = []
        for A_positions in itertools.combinations(range(8), 4):
            A_mask = sum(1 << p for p in A_positions)
            B_positions = tuple(p for p in range(8) if not (A_mask >> p) & 1)
            splits.append((A_positions, B_positions))
        split_cache[subset] = splits
    
    def contains_q3_fast(vertices, edge_matrix):
        """Check if 8 vertices contain Q3 as subgraph. edge_matrix is 16x16 bool list."""
        if len(vertices) != 8:
            return False
        
        v_list = list(vertices)
        
        # Fast filter: count edges among the 8 vertices
        edge_count = 0
        for i in range(8):
            vi = v_list[i]
            row = edge_matrix[vi]
            for j in range(i+1, 8):
                if row[v_list[j]]:
                    edge_count += 1
        if edge_count < 12:
            return False
        
        # Build local adjacency: adj_local[i] is bitmask of which of the 8 vertices v_list[i] connects to
        adj_local = [0] * 8
        for i in range(8):
            vi = v_list[i]
            row = edge_matrix[vi]
            mask = 0
            for j in range(8):
                if i != j and row[v_list[j]]:
                    mask |= (1 << j)
            adj_local[i] = mask
            
        # Get precomputed splits for this vertex set
        key = tuple(sorted(v_list))
        if key not in split_cache:
            # Fallback: compute splits on the fly
            splits = []
            for A_positions in itertools.combinations(range(8), 4):
                A_mask = sum(1 << p for p in A_positions)
                B_positions = tuple(p for p in range(8) if not (A_mask >> p) & 1)
                splits.append((A_positions, B_positions))
        else:
            splits = split_cache[key]
        
        for A_positions, B_positions in splits:
            A_idx = list(A_positions)
            B_idx = list(B_positions)
            mask_A = sum(1 << p for p in A_positions)
            b_mask = sum(1 << p for p in B_positions)
            
            # Quick degree check: each vertex in A must have >=3 neighbors in B
            valid_split = True
            for a in A_idx:
                if (adj_local[a] & b_mask).bit_count() < 3:
                    valid_split = False
                    break
            if not valid_split:
                continue
            for b in B_idx:
                if (adj_local[b] & mask_A).bit_count() < 3:
                    valid_split = False
                    break
            if not valid_split:
                continue

            a0, a1, a2, a3 = A_idx
            
            for p in perms24:
                b0 = B_idx[p[0]]
                b1 = B_idx[p[1]]
                b2 = B_idx[p[2]]
                b3 = B_idx[p[3]]
                
                # Check the 12 required edges
                if not (adj_local[a0] >> b0) & 1: continue
                if not (adj_local[a0] >> b1) & 1: continue
                if not (adj_local[a0] >> b3) & 1: continue
                if not (adj_local[a1] >> b0) & 1: continue
                if not (adj_local[a1] >> b1) & 1: continue
                if not (adj_local[a1] >> b2) & 1: continue
                if not (adj_local[a2] >> b0) & 1: continue
                if not (adj_local[a2] >> b2) & 1: continue
                if not (adj_local[a2] >> b3) & 1: continue
                if not (adj_local[a3] >> b1) & 1: continue
                if not (adj_local[a3] >> b2) & 1: continue
                if not (adj_local[a3] >> b3) & 1: continue
                
                return True
        return False

    def run_greedy(initial_edges, seed, deadline):
        random.seed(seed)
        edges = set(initial_edges)
        
        # Build edge_matrix: 16x16 boolean list
        edge_matrix = [[False] * 16 for _ in range(16)]
        for (u, v) in edges:
            edge_matrix[u][v] = True
            edge_matrix[v][u] = True
            
        # Precompute degrees as a list for faster access
        def update_degrees():
            deg = [0] * 16
            for v in range(16):
                row = edge_matrix[v]
                deg[v] = sum(1 for x in range(16) if x != v and row[x])
            return deg
        
        degrees = update_degrees()
            
        all_pairs = list(itertools.combinations(range(16), 2))
        non_edges = [e for e in all_pairs if e not in edges]
        
        random.shuffle(non_edges)
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            random.shuffle(non_edges)
            
            # Filter to current non-edges and sort by degree sum
            current_non_edges = [e for e in non_edges if not edge_matrix[e[0]][e[1]]]
            current_non_edges.sort(key=lambda e: degrees[e[0]] + degrees[e[1]], reverse=True)
            
            for e in current_non_edges:
                if time.time() > deadline:
                    break
                u, v = e
                remaining = [x for x in range(16) if x != u and x != v]
                safe = True
                
                # Temporarily add edge
                edge_matrix[u][v] = True
                edge_matrix[v][u] = True
                edges.add((u, v))
                
                # Check for Q3: only need to check 8-sets containing both u and v
                for combo in itertools.combinations(remaining, 6):
                    verts = [u, v] + list(combo)
                    if contains_q3_fast(verts, edge_matrix):
                        safe = False
                        break
                
                if not safe:
                    edge_matrix[u][v] = False
                    edge_matrix[v][u] = False
                    edges.remove((u, v))
                else:
                    changed = True
                    # Update degrees
                    degrees[u] += 1
                    degrees[v] += 1
                    
        return edges

    # Initial seed: K_{3,13}
    initial_edges = set()
    for u in range(3):
        for v in range(3, 16):
            initial_edges.add((u, v))
            
    best_edges = set()
    best_count = -1
    
    seed = 42
    # First pass: standard greedy
    current_edges = run_greedy(initial_edges, seed, deadline)
    if len(current_edges) > best_count:
        best_count = len(current_edges)
        best_edges = set(current_edges)
    seed += 1
    
    # Second phase: Kicks - more aggressive with smaller kicks
    kick_seed = 100
    while time.time() < deadline - 5:
        # Alternate between removing 1, 2, and 3 edges
        kick_size = random.choice([1, 1, 2, 2, 3])
        
        perturbed_edges = set(best_edges)
        edges_list = list(perturbed_edges)
        random.shuffle(edges_list)
        for e in edges_list[:kick_size]:
            perturbed_edges.remove(e)
            
        current_edges = run_greedy(perturbed_edges, kick_seed, deadline)
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = set(current_edges)
        kick_seed += 1
    
    # Third phase: Final polish - try to add any remaining safe edges
    if time.time() < deadline - 2:
        edge_matrix = [[False] * 16 for _ in range(16)]
        for (u, v) in best_edges:
            edge_matrix[u][v] = True
            edge_matrix[v][u] = True
        
        all_pairs = list(itertools.combinations(range(16), 2))
        non_edges = [e for e in all_pairs if e not in best_edges]
        
        for e in non_edges:
            if time.time() > deadline - 2:
                break
            u, v = e
            if edge_matrix[u][v]:
                continue
            
            remaining = [x for x in range(16) if x != u and x != v]
            safe = True
            
            # Temporarily add edge
            edge_matrix[u][v] = True
            edge_matrix[v][u] = True
            best_edges.add((u, v))
            
            # Check for Q3
            for combo in itertools.combinations(remaining, 6):
                verts = [u, v] + list(combo)
                if contains_q3_fast(verts, edge_matrix):
                    safe = False
                    break
            
            if not safe:
                edge_matrix[u][v] = False
                edge_matrix[v][u] = False
                best_edges.remove((u, v))
    
    return [list(e) for e in best_edges]
