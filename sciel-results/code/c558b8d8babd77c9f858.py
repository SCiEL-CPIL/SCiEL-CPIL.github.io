import random
import time

# Precomputation
N_DIMS = 7
N_VERTS = 1 << N_DIMS

def precompute():
    edges = []
    edge_map = {}
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
                for k, bit_val in enumerate(mask):
                    if bit_val:
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
EDGE_COUNT = len(EDGES)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_indices = []
    best_count = 0
    
    # Local variables for speed
    edges = EDGES
    edge_squares = EDGE_SQUARES
    num_squares = NUM_SQUARES
    edge_count = EDGE_COUNT
    
    while time.time() < deadline:
        # Shuffle indices instead of edges to avoid swapping tuples?
        # Actually shuffling the list of tuples is fine.
        # But we need to map back to edges.
        # Let's shuffle a list of indices.
        order = list(range(edge_count))
        random.shuffle(order)
        
        selected = [False] * edge_count
        square_counts = [0] * num_squares
        count = 0
        
        for e_idx in order:
            sqs = edge_squares[e_idx]
            
            valid = True
            for s in sqs:
                if square_counts[s] >= 3:
                    valid = False
                    break
            
            if valid:
                selected[e_idx] = True
                count += 1
                for s in sqs:
                    square_counts[s] += 1
        
        if count > best_count:
            best_count = count
            best_indices = [i for i in range(edge_count) if selected[i]]
            
    return [[edges[i][0], edges[i][1]] for i in best_indices]
