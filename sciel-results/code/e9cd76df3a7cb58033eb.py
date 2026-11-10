import random
import time
import math

# Precomputation
N_DIMS = 7
N_VERTS = 1 << N_DIMS
N_EDGES = N_DIMS * (N_VERTS // 2) # 7 * 64 = 448

# We need to map edges to indices 0..447
# And map squares to indices 0..671
# And for each edge, store the list of square indices it belongs to.

def precompute():
    edges = []
    edge_map = {} # (u, v) with u<v -> index
    for u in range(N_VERTS):
        for i in range(N_DIMS):
            v = u ^ (1 << i)
            if u < v:
                edge_map[(u, v)] = len(edges)
                edges.append((u, v))
    
    # Total squares
    # Defined by 2 bits (i, j) and 5 fixed bits.
    # Number of squares = C(7,2) * 2^5 = 21 * 32 = 672
    edge_squares = [[] for _ in range(len(edges))]
    square_count = 0
    
    for i in range(N_DIMS):
        for j in range(i + 1, N_DIMS):
            # Iterate over all assignments of the other 5 bits
            # The other bits are all k != i, k != j
            other_bits = [k for k in range(N_DIMS) if k != i and k != j]
            for mask in range(1 << len(other_bits)):
                # Determine the fixed value of the other bits
                # We need to construct the 4 vertices
                # The vertices differ in bits i and j.
                # Let's build the base vertex where bits i and j are 0, and other bits are set according to mask.
                base = 0
                for k in range(len(other_bits)):
                    if (mask >> k) & 1:
                        base |= (1 << other_bits[k])
                
                # The 4 vertices:
                v00 = base
                v10 = base | (1 << i)
                v01 = base | (1 << j)
                v11 = base | (1 << i) | (1 << j)
                
                # The 4 edges:
                # (v00, v10) - dim i
                # (v00, v01) - dim j
                # (v10, v11) - dim j
                # (v01, v11) - dim i
                
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
    deadline = start_time + 115.0 # Safety margin
    
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
    
    def local_search(selected, count):
        """
        Max-SAT style local search:
        - Objective: maximize number of selected edges
        - Constraints: no square has all 4 edges selected
        - Strategy: try to add unselected edges, if that creates a C4, 
          try to remove one edge from the C4 to make room
        """
        square_counts = [0] * NUM_SQUARES
        for e in range(len(EDGES)):
            if selected[e]:
                for s in EDGE_SQUARES[e]:
                    square_counts[s] += 1
        
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            
            # Try to add each unselected edge
            candidates = [e for e in range(len(EDGES)) if not selected[e]]
            random.shuffle(candidates)
            
            for e in candidates:
                if time.time() >= deadline - 2.0:
                    break
                
                sqs = EDGE_SQUARES[e]
                # Check if adding e would create a C4 (any square with count 3)
                blocking_squares = []
                for s in sqs:
                    if square_counts[s] == 3:
                        blocking_squares.append(s)
                
                if not blocking_squares:
                    # Can add freely
                    selected[e] = True
                    count += 1
                    for s in sqs:
                        square_counts[s] += 1
                    improved = True
                else:
                    # Try to free up a blocking square by removing one of its other edges
                    best_swap = None
                    best_gain = 0
                    
                    for s in blocking_squares:
                        # The 4 edges in this square
                        for other_e in square_edges[s]:
                            if other_e == e or not selected[other_e]:
                                continue
                            # If we remove other_e and add e, what's the net effect?
                            # We gain 1 (adding e) but lose 1 (removing other_e) = 0 net
                            # But we might be able to add more edges after this
                            # Let's check if removing other_e would unblock other squares
                            other_sqs = EDGE_SQUARES[other_e]
                            # Count how many squares other_e is blocking (has count 3 after removal)
                            # Actually, let's just try the swap and see if it leads to improvement
                            # For now, just do the swap if it doesn't create new C4s
                            
                            # Check if removing other_e and adding e keeps all squares valid
                            valid = True
                            for os in other_sqs:
                                if os != s and square_counts[os] == 3:
                                    # Removing other_e reduces count, so this is fine
                                    pass
                            
                            # After swap: square s goes from 3->3 (remove other, add e)
                            # Other squares of other_e: count decreases by 1
                            # Other squares of e: count increases by 1
                            # Need to check that no square exceeds 4
                            
                            # Squares of e that are not s: their count increases
                            for es in sqs:
                                if es != s:
                                    if square_counts[es] >= 3:
                                        valid = False
                                        break
                            
                            if valid:
                                # Perform the swap
                                selected[other_e] = False
                                selected[e] = True
                                # Update counts
                                for os in other_sqs:
                                    square_counts[os] -= 1
                                for es in sqs:
                                    square_counts[es] += 1
                                improved = True
                                best_swap = (other_e, e)
                                break
                        
                        if best_swap:
                            break
                    
                    if best_swap:
                        # After swap, try to add more edges that were previously blocked
                        # (this is handled by the outer loop continuing)
                        pass
        
        return selected, count
    
    # Phase 1: Multiple random greedy starts
    num_greedy = 0
    while time.time() < deadline - 30.0:
        selected, count = greedy_random()
        if count > best_count:
            best_count = count
            best_edges = [EDGES[i] for i in range(len(EDGES)) if selected[i]]
        num_greedy += 1
        if num_greedy >= 50:
            break
    
    # Phase 2: Local search on best solution
    if best_count > 0 and time.time() < deadline - 10.0:
        # Reconstruct the best selected array
        selected = [False] * len(EDGES)
        for edge in best_edges:
            u, v = edge
            e = None
            for i in range(N_DIMS):
                if (u ^ v) == (1 << i):
                    e = None  # need to find the index
                    break
            # Find edge index
            key = (min(u, v), max(u, v))
            # We need edge_map, let's rebuild it
            pass
        
        # Rebuild selected from best_edges
        edge_map = {}
        for idx, (u, v) in enumerate(EDGES):
            edge_map[(min(u, v), max(u, v))] = idx
        
        selected = [False] * len(EDGES)
        for edge in best_edges:
            u, v = edge
            key = (min(u, v), max(u, v))
            selected[edge_map[key]] = True
        
        selected, count = local_search(selected, best_count)
        if count > best_count:
            best_count = count
            best_edges = [EDGES[i] for i in range(len(EDGES)) if selected[i]]
    
    # Phase 3: More local search iterations with different randomizations
    while time.time() < deadline - 5.0:
        selected = [False] * len(EDGES)
        edge_map = {}
        for idx, (u, v) in enumerate(EDGES):
            edge_map[(min(u, v), max(u, v))] = idx
        for edge in best_edges:
            u, v = edge
            key = (min(u, v), max(u, v))
            selected[edge_map[key]] = True
        
        selected, count = local_search(selected, best_count)
        if count > best_count:
            best_count = count
            best_edges = [EDGES[i] for i in range(len(EDGES)) if selected[i]]
        else:
            # Try a fresh greedy + local search
            if time.time() < deadline - 10.0:
                selected, count = greedy_random()
                selected, count = local_search(selected, count)
                if count > best_count:
                    best_count = count
                    best_edges = [EDGES[i] for i in range(len(EDGES)) if selected[i]]
    
    return [[u, v] for u, v in best_edges]
