import random
import time
import math

# Precomputation
N_DIMS = 7
N_VERTS = 1 << N_DIMS
N_EDGES = N_DIMS * (N_VERTS // 2) # 7 * 64 = 448

def precompute():
    edges = []
    edge_map = {} # (u, v) with u<v -> index
    for u in range(N_VERTS):
        for i in range(N_DIMS):
            v = u ^ (1 << i)
            if u < v:
                edge_map[(u, v)] = len(edges)
                edges.append((u, v))
    
    edge_squares = [[] for _ in range(len(edges))]
    square_count = 0
    
    for i in range(N_DIMS):
        for j in range(i + 1, N_DIMS):
            other_bits = [k for k in range(N_DIMS) if k != i and k != j]
            for mask in range(1 << len(other_bits)):
                base = 0
                for k in range(len(other_bits)):
                    if (mask >> k) & 1:
                        base |= (1 << other_bits[k])
                
                v00 = base
                v10 = base | (1 << i)
                v01 = base | (1 << j)
                v11 = base | (1 << i) | (1 << j)
                
                e1 = (min(v00, v10), max(v00, v10))
                e2 = (min(v00, v01), max(v00, v01))
                e3 = (min(v10, v11), max(v10, v11))
                e4 = (min(v01, v11), max(v01, v11))
                
                idx1 = edge_map[e1]
                idx2 = edge_map[e2]
                idx3 = edge_map[e3]
                idx4 = edge_map[e4]
                
                edge_squares[idx1].append(square_count)
                edge_squares[idx2].append(square_count)
                edge_squares[idx3].append(square_count)
                edge_squares[idx4].append(square_count)
                
                square_count += 1
                
    return edges, edge_squares, square_count

EDGES, EDGE_SQUARES, NUM_SQUARES = precompute()

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_edges = []
    best_count = 0
    
    # Precompute for each square, the list of 4 edge indices
    square_edges = [[] for _ in range(NUM_SQUARES)]
    for e in range(len(EDGES)):
        for s in EDGE_SQUARES[e]:
            square_edges[s].append(e)
    
    def greedy_random():
        """Randomized greedy to get a starting solution."""
        indices = list(range(len(EDGES)))
        random.shuffle(indices)
        selected = [False] * len(EDGES)
        square_counts = [0] * NUM_SQUARES
        count = 0
        
        for orig_idx in indices:
            sqs = EDGE_SQUARES[orig_idx]
            valid = True
            for s in sqs:
                if square_counts[s] >= 3:
                    valid = False
                    break
            if valid:
                selected[orig_idx] = True
                count += 1
                for s in sqs:
                    square_counts[s] += 1
        return selected, count
    
    def greedy_degree():
        """Greedy with degree-based ordering: prefer edges that block fewer squares."""
        # Count how many squares each edge is in (always 2 for hypercube edges)
        # Instead, use a dynamic heuristic: prefer edges that are in squares with fewer already-selected edges
        selected = [False] * len(EDGES)
        square_counts = [0] * NUM_SQUARES
        count = 0
        
        # Precompute for each edge, the squares it's in
        # Use a priority queue approach: at each step, pick the edge that can be added with least "damage"
        remaining = set(range(len(EDGES)))
        
        while remaining and time.time() < deadline - 5.0:
            best_edge = None
            best_score = -1
            
            # Sample a subset to keep it fast
            if len(remaining) > 50:
                sample = random.sample(list(remaining), 50)
            else:
                sample = list(remaining)
            
            for e in sample:
                sqs = EDGE_SQUARES[e]
                # Score: prefer edges where squares have lower counts
                score = sum(square_counts[s] for s in sqs)
                # Check if we can add it
                can_add = True
                for s in sqs:
                    if square_counts[s] >= 3:
                        can_add = False
                        break
                if not can_add:
                    continue
                if score < best_score:
                    best_score = score
                    best_edge = e
            
            if best_edge is None:
                # Try any remaining edge
                for e in remaining:
                    sqs = EDGE_SQUARES[e]
                    can_add = True
                    for s in sqs:
                        if square_counts[s] >= 3:
                            can_add = False
                            break
                    if can_add:
                        best_edge = e
                        break
            
            if best_edge is None:
                break
            
            selected[best_edge] = True
            count += 1
            for s in EDGE_SQUARES[best_edge]:
                square_counts[s] += 1
            remaining.remove(best_edge)
        
        return selected, count
    
    def local_search(selected, count):
        """
        Max-SAT style local search with effective swaps including 2-for-2 swaps.
        """
        square_counts = [0] * NUM_SQUARES
        for e in range(len(EDGES)):
            if selected[e]:
                for s in EDGE_SQUARES[e]:
                    square_counts[s] += 1
        
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            
            # Try to add each unselected edge
            candidates = [e for e in range(len(EDGES)) if not selected[e]]
            random.shuffle(candidates)
            
            for e in candidates:
                if time.time() >= deadline - 3.0:
                    break
                
                sqs = EDGE_SQUARES[e]
                blocking_squares = []
                for s in sqs:
                    if square_counts[s] == 3:
                        blocking_squares.append(s)
                
                if not blocking_squares:
                    selected[e] = True
                    count += 1
                    for s in sqs:
                        square_counts[s] += 1
                    improved = True
                else:
                    # Try 1-for-1 swaps: remove one edge from a blocking square, add e
                    found_swap = False
                    for s in blocking_squares:
                        if time.time() >= deadline - 3.0:
                            break
                        
                        for other_e in square_edges[s]:
                            if other_e == e or not selected[other_e]:
                                continue
                            
                            other_sqs = EDGE_SQUARES[other_e]
                            valid = True
                            for es in sqs:
                                if es != s:
                                    if square_counts[es] >= 3:
                                        valid = False
                                        break
                            
                            if not valid:
                                continue
                            
                            # Perform swap
                            selected[other_e] = False
                            selected[e] = True
                            for os in other_sqs:
                                square_counts[os] -= 1
                            for es in sqs:
                                square_counts[es] += 1
                            improved = True
                            found_swap = True
                            break
                        
                        if found_swap:
                            break
                    
                    if found_swap:
                        # After a swap, try to add more edges that are now unblocked
                        for e2 in candidates:
                            if e2 == e or selected[e2]:
                                continue
                            sqs2 = EDGE_SQUARES[e2]
                            can_add = True
                            for s2 in sqs2:
                                if square_counts[s2] >= 3:
                                    can_add = False
                                    break
                            if can_add:
                                selected[e2] = True
                                count += 1
                                for s2 in sqs2:
                                    square_counts[s2] += 1
                        continue
                    
                    # Try 2-for-2 swaps: remove two edges, add e and another edge e2
                    for s1 in blocking_squares:
                        if time.time() >= deadline - 3.0:
                            break
                        for s2 in blocking_squares:
                            if s2 <= s1:
                                continue
                            for oe1 in square_edges[s1]:
                                if oe1 == e or not selected[oe1]:
                                    continue
                                for oe2 in square_edges[s2]:
                                    if oe2 == e or oe2 == oe1 or not selected[oe2]:
                                        continue
                                    
                                    # Check validity of removing oe1 and oe2, adding e
                                    oe1_sqs = EDGE_SQUARES[oe1]
                                    oe2_sqs = EDGE_SQUARES[oe2]
                                    
                                    # Build set of squares affected
                                    affected = {}
                                    for sq in sqs:
                                        affected[sq] = affected.get(sq, 0) + 1
                                    for sq in oe1_sqs:
                                        affected[sq] = affected.get(sq, 0) - 1
                                    for sq in oe2_sqs:
                                        affected[sq] = affected.get(sq, 0) - 1
                                    
                                    valid = True
                                    for sq, delta in affected.items():
                                        new_count = square_counts[sq] + delta
                                        if new_count < 0 or new_count > 3:
                                            valid = False
                                            break
                                    
                                    if not valid:
                                        continue
                                    
                                    # Also try to find an edge e2 that can be added along with e
                                    # For now, just do the 2-for-1 swap (remove 2, add 1)
                                    # This reduces count, so only do if it enables future gains
                                    # Actually, let's just do 1-for-1 swaps for simplicity
                                    # and rely on kicks for larger changes
                                    break
                            else:
                                continue
                            break
                        else:
                            continue
                        break
        
        return selected, count
    
    def kick_and_optimize(selected, count, kick_size=5):
        """Randomly remove some edges, then re-optimize with multiple passes."""
        selected = list(selected)
        
        # Remove kick_size random selected edges
        selected_indices = [i for i in range(len(EDGES)) if selected[i]]
        if len(selected_indices) > kick_size:
            to_remove = random.sample(selected_indices, kick_size)
            for i in to_remove:
                selected[i] = False
            count -= kick_size
        
        # Re-optimize
        square_counts = [0] * NUM_SQUARES
        for e in range(len(EDGES)):
            if selected[e]:
                for s in EDGE_SQUARES[e]:
                    square_counts[s] += 1
        
        # Try to add edges - multiple passes for better convergence
        for _pass in range(3):
            if time.time() >= deadline - 3.0:
                break
            candidates = [e for e in range(len(EDGES)) if not selected[e]]
            random.shuffle(candidates)
            
            for e in candidates:
                if time.time() >= deadline - 3.0:
                    break
                sqs = EDGE_SQUARES[e]
                can_add = True
                for s in sqs:
                    if square_counts[s] >= 3:
                        can_add = False
                        break
                if can_add:
                    selected[e] = True
                    count += 1
                    for s in sqs:
                        square_counts[s] += 1
        
        # Local search
        selected, count = local_search(selected, count)
        return selected, count
    
    # Phase 1: Multiple random greedy starts
    num_greedy = 0
    while time.time() < deadline - 30.0:
        if num_greedy % 3 == 0:
            selected, count = greedy_random()
        elif num_greedy % 3 == 1:
            selected, count = greedy_degree()
        else:
            selected, count = greedy_random()
        
        if count > best_count:
            best_count = count
            best_edges = [EDGES[i] for i in range(len(EDGES)) if selected[i]]
        num_greedy += 1
        if num_greedy >= 60:
            break
    
    # Build edge_map for later use
    edge_map = {}
    for idx, (u, v) in enumerate(EDGES):
        edge_map[(min(u, v), max(u, v))] = idx
    
    # Phase 2: Local search on best solution
    if best_count > 0 and time.time() < deadline - 10.0:
        selected = [False] * len(EDGES)
        for edge in best_edges:
            u, v = edge
            key = (min(u, v), max(u, v))
            selected[edge_map[key]] = True
        
        selected, count = local_search(selected, best_count)
        if count > best_count:
            best_count = count
            best_edges = [EDGES[i] for i in range(len(EDGES)) if selected[i]]
    
    # Phase 3: Kick and optimize to escape local optima
    consecutive_no_improve = 0
    while time.time() < deadline - 5.0 and consecutive_no_improve < 15:
        selected = [False] * len(EDGES)
        for edge in best_edges:
            u, v = edge
            key = (min(u, v), max(u, v))
            selected[edge_map[key]] = True
        
        # Try different kick sizes, biased toward smaller kicks for fine-tuning
        if consecutive_no_improve < 5:
            kick_size = random.choice([3, 4, 5])
        elif consecutive_no_improve < 10:
            kick_size = random.choice([5, 7, 8])
        else:
            kick_size = random.choice([7, 10, 12])
        
        selected, count = kick_and_optimize(selected, best_count, kick_size)
        
        if count > best_count:
            best_count = count
            best_edges = [EDGES[i] for i in range(len(EDGES)) if selected[i]]
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1
            # Try fresh greedy + local search
            if time.time() < deadline - 10.0:
                if random.random() < 0.5:
                    selected, count = greedy_random()
                else:
                    selected, count = greedy_degree()
                selected, count = local_search(selected, count)
                if count > best_count:
                    best_count = count
                    best_edges = [EDGES[i] for i in range(len(EDGES)) if selected[i]]
                    consecutive_no_improve = 0
    
    # Final polish: multiple passes to add any remaining edges
    if time.time() < deadline - 2.0:
        selected = [False] * len(EDGES)
        for edge in best_edges:
            u, v = edge
            key = (min(u, v), max(u, v))
            selected[edge_map[key]] = True
        
        square_counts = [0] * NUM_SQUARES
        for e in range(len(EDGES)):
            if selected[e]:
                for s in EDGE_SQUARES[e]:
                    square_counts[s] += 1
        
        # Multiple passes for thoroughness
        for _ in range(5):
            if time.time() >= deadline - 2.0:
                break
            candidates = [e for e in range(len(EDGES)) if not selected[e]]
            random.shuffle(candidates)
            added_any = False
            for e in candidates:
                sqs = EDGE_SQUARES[e]
                can_add = True
                for s in sqs:
                    if square_counts[s] >= 3:
                        can_add = False
                        break
                if can_add:
                    selected[e] = True
                    best_count += 1
                    for s in sqs:
                        square_counts[s] += 1
                    added_any = True
            if not added_any:
                break
        
        best_edges = [EDGES[i] for i in range(len(EDGES)) if selected[i]]
    
    return [[u, v] for u, v in best_edges]
