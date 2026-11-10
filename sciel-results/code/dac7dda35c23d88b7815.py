import random
import time

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
    
    indices = list(range(len(EDGES)))
    
    while time.time() < deadline:
        random.shuffle(indices)
        
        selected = [False] * len(EDGES)
        square_counts = [0] * NUM_SQUARES
        count = 0
        
        for orig_idx in indices:
            # Get the squares this edge belongs to
            sqs = EDGE_SQUARES[orig_idx]
            
            # Check if any square is already full (3 edges selected)
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
        
        if count > best_count:
            best_count = count
            best_edges = [EDGES[i] for i in range(len(EDGES)) if selected[i]]
            
    return [[u, v] for u, v in best_edges]
