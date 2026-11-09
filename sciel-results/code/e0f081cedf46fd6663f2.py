import time
import random
import numpy as np
from itertools import combinations

def construct():
    """
    Tensor-slice approach for C4-free edge set in Q7.
    
    Represents edges as a 7x128 binary matrix M where M[axis][vertex] = 1
    means the edge along 'axis' at 'vertex' is selected. An edge (u,v) with
    u<v is represented by M[i][u]=1 where i is the differing bit.
    
    C4 constraint: For any pair of axes (a,b) and any fixed values of the
    other 5 bits, the 2x2 submatrix M[a][v0], M[a][v0^b], M[b][v0], M[b][v0^a]
    must not be all 1s. This is equivalent to: for each (a,b) and each
    "complement" c in the remaining 5 bits, the 2x2 slice has at most 3 ones.
    
    We use vectorized NumPy operations on these slices for fast constraint checking.
    """
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    d = 7
    n_vertices = 1 << d  # 128
    
    def time_left():
        return deadline - time.time()
    
    # Precompute edge representation: edge (u,v) with u<v, differing bit i -> M[i][u]
    # For each axis i and vertex v with bit i = 0, edge is (v, v^(1<<i))
    # Total edges: 7 * 64 = 448
    
    # Build mapping from (axis, vertex_with_bit0) to edge index
    edge_list = []  # list of (u, v)
    axis_vertex_to_eidx = {}  # (axis, v) -> eidx
    for axis in range(d):
        bit = 1 << axis
        for v in range(n_vertices):
            if v & bit:
                continue  # only represent each edge once, from lower vertex
            u = v
            w = v ^ bit
            eidx = len(edge_list)
            edge_list.append((u, w))
            axis_vertex_to_eidx[(axis, v)] = eidx
    
    n_edges = len(edge_list)
    assert n_edges == 448, f"Expected 448 edges, got {n_edges}"
    
    # For each axis pair (a, b), precompute the "complement" vertices
    # For axes a, b, the remaining bits are all others.
    # A 2x2 slice is determined by fixing all bits except a and b.
    # There are 2^5 = 32 such slices per axis pair.
    # For slice with fixed complement c (5 bits), the 4 vertices are:
    #   v00 = c, v10 = c | (1<<a), v01 = c | (1<<b), v11 = c | (1<<a) | (1<<b)
    # The 4 edges in the face are:
    #   M[a][v00] (edge from v00 along axis a)
    #   M[b][v00] (edge from v00 along axis b)
    #   M[a][v01] (edge from v01 along axis a) -- v01 has bit a=0, bit b=1
    #   M[b][v10] (edge from v10 along axis b) -- v10 has bit a=1, bit b=0
    
    # Precompute for each axis pair (a,b), the list of 32 faces as 4 (axis, vertex) tuples
    face_data = {}  # (a,b) -> list of 32 tuples, each tuple has 4 (axis, vertex) pairs
    for a in range(d):
        for b in range(a+1, d):
            other_bits = [i for i in range(d) if i != a and i != b]
            faces_for_pair = []
            for c_val in range(1 << 5):
                # Build the base vertex from complement bits
                base = 0
                for k, bit_idx in enumerate(other_bits):
                    if c_val & (1 << k):
                        base |= (1 << bit_idx)
                v00 = base
                v10 = base | (1 << a)
                v01 = base | (1 << b)
                v11 = base | (1 << a) | (1 << b)
                
                # The 4 edges: (axis, vertex_with_bit0)
                # Edge along axis a at v00: (a, v00)
                # Edge along axis b at v00: (b, v00)
                # Edge along axis a at v01: (a, v01) -- v01 has bit a = 0
                # Edge along axis b at v10: (b, v10) -- v10 has bit b = 0
                face_edges = [
                    (a, v00),
                    (b, v00),
                    (a, v01),
                    (b, v10)
                ]
                faces_for_pair.append(face_edges)
            face_data[(a, b)] = faces_for_pair
    
    # Convert to numpy arrays for fast lookup
    # For each (a,b), store as a (32, 4, 2) array: [face_idx][edge_pos][axis, vertex]
    face_arrays = {}
    for (a, b), faces in face_data.items():
        arr = np.zeros((32, 4, 2), dtype=np.int32)
        for fi, fe in enumerate(faces):
            for ei, (ax, vt) in enumerate(fe):
                arr[fi, ei, 0] = ax
                arr[fi, ei, 1] = vt
        face_arrays[(a, b)] = arr
    
    def check_valid(M):
        """Check if matrix M (7x128 bool) is C4-free."""
        for (a, b) in face_arrays:
            fa = face_arrays[(a, b)]
            # Extract the 4 values for each face
            # M[fa[:, :, 0], fa[:, :, 1]] gives (32, 4) array
            vals = M[fa[:, :, 0], fa[:, :, 1]]
            if np.any(vals.sum(axis=1) == 4):
                return False
        return True
    
    def count_edges(M):
        return int(np.sum(M))
    
    def greedy_tensor(seed=None):
        """Greedy construction using tensor representation."""
        rng = random.Random(seed)
        M = np.zeros((d, n_vertices), dtype=np.int8)
        
        # Create list of all (axis, vertex) positions
        positions = []
        for axis in range(d):
            bit = 1 << axis
            for v in range(n_vertices):
                if not (v & bit):
                    positions.append((axis, v))
        
        rng.shuffle(positions)
        
        for axis, v in positions:
            bit = 1 << axis
            if v & bit:
                continue
            # Check all faces containing this edge
            can_add = True
            for (a, b) in face_arrays:
                if axis not in (a, b):
                    continue
                fa = face_arrays[(a, b)]
                # Find which face(s) contain this edge
                # An edge (axis, v) is in a face if it matches one of the 4 positions
                for fi in range(32):
                    for ei in range(4):
                        if fa[fi, ei, 0] == axis and fa[fi, ei, 1] == v:
                            # Check if other 3 edges in this face are all selected
                            face_sum = 0
                            for ej in range(4):
                                if ej == ei:
                                    continue
                                ax2, vt2 = fa[fi, ej, 0], fa[fi, ej, 1]
                                if M[ax2, vt2]:
                                    face_sum += 1
                            if face_sum >= 3:
                                can_add = False
                                break
                    if not can_add:
                        break
                if not can_add:
                    break
            
            if can_add:
                M[axis, v] = 1
        
        return M
    
    def local_search_tensor(M, seed=None, max_iter=5000):
        """Local search on tensor representation."""
        rng = random.Random(seed)
        M = M.copy()
        
        improved = True
        iter_count = 0
        consecutive_no_improve = 0
        
        while improved and time_left() > 20 and iter_count < max_iter:
            improved = False
            iter_count += 1
            
            # First, try to add any removable edge
            positions = []
            for axis in range(d):
                bit = 1 << axis
                for v in range(n_vertices):
                    if not (v & bit) and not M[axis, v]:
                        positions.append((axis, v))
            
            if positions:
                rng.shuffle(positions)
                for axis, v in positions:
                    if time_left() < 20:
                        break
                    # Check if we can add
                    can_add = True
                    for (a, b) in face_arrays:
                        if axis not in (a, b):
                            continue
                        fa = face_arrays[(a, b)]
                        for fi in range(32):
                            for ei in range(4):
                                if fa[fi, ei, 0] == axis and fa[fi, ei, 1] == v:
                                    face_sum = 0
                                    for ej in range(4):
                                        if ej == ei:
                                            continue
                                        ax2, vt2 = fa[fi, ej, 0], fa[fi, ej, 1]
                                        if M[ax2, vt2]:
                                            face_sum += 1
                                    if face_sum >= 3:
                                        can_add = False
                                        break
                            if not can_add:
                                break
                        if not can_add:
                            break
                    
                    if can_add:
                        M[axis, v] = 1
                        improved = True
                        consecutive_no_improve = 0
                if improved:
                    continue
            
            # Try swap: remove one edge, add another
            if time_left() > 30:
                selected = []
                for axis in range(d):
                    bit = 1 << axis
                    for v in range(n_vertices):
                        if not (v & bit) and M[axis, v]:
                            selected.append((axis, v))
                
                unselected = []
                for axis in range(d):
                    bit = 1 << axis
                    for v in range(n_vertices):
                        if not (v & bit) and not M[axis, v]:
                            unselected.append((axis, v))
                
                if selected and unselected:
                    # Try random swaps
                    for _ in range(min(50, len(unselected))):
                        if time_left() < 20:
                            break
                        ax_add, v_add = rng.choice(unselected)
                        ax_rem, v_rem = rng.choice(selected)
                        
                        # Check if swap is valid
                        M_test = M.copy()
                        M_test[ax_rem, v_rem] = 0
                        M_test[ax_add, v_add] = 1
                        
                        if check_valid(M_test):
                            M = M_test
                            improved = True
                            consecutive_no_improve = 0
                            break
                    
                    if not improved:
                        # Perturbation
                        if len(selected) > 10:
                            n_rem = rng.randint(3, 7)
                            to_remove = rng.sample(selected, min(n_rem, len(selected) // 5))
                            for ax, v in to_remove:
                                M[ax, v] = 0
                            improved = True
                            consecutive_no_improve = 0
                        else:
                            consecutive_no_improve += 1
                            if consecutive_no_improve > 10:
                                break
        
        return M
    
    def polish_tensor(M, seed=None):
        """Polish: exhaustively try all additions."""
        M = M.copy()
        changed = True
        while changed and time_left() > 25:
            changed = False
            for axis in range(d):
                bit = 1 << axis
                for v in range(n_vertices):
                    if (v & bit) or M[axis, v]:
                        continue
                    # Check if can add
                    can_add = True
                    for (a, b) in face_arrays:
                        if axis not in (a, b):
                            continue
                        fa = face_arrays[(a, b)]
                        for fi in range(32):
                            for ei in range(4):
                                if fa[fi, ei, 0] == axis and fa[fi, ei, 1] == v:
                                    face_sum = 0
                                    for ej in range(4):
                                        if ej == ei:
                                            continue
                                        ax2, vt2 = fa[fi, ej, 0], fa[fi, ej, 1]
                                        if M[ax2, vt2]:
                                            face_sum += 1
                                    if face_sum >= 3:
                                        can_add = False
                                        break
                            if not can_add:
                                break
                        if not can_add:
                            break
                    if can_add:
                        M[axis, v] = 1
                        changed = True
        return M
    
    # Run multiple restarts
    best_count = 0
    best_M = None
    
    attempts = 0
    while time_left() > 30:
        attempts += 1
        seed = attempts * 104729 + 13
        
        if attempts % 4 == 1:
            M = greedy_tensor(seed)
        elif attempts % 4 == 2:
            M = greedy_tensor(seed + 99991)
        elif attempts % 4 == 3:
            if best_M is not None:
                M = best_M.copy()
                rng = random.Random(seed)
                # Perturb
                positions = []
                for axis in range(d):
                    bit = 1 << axis
                    for v in range(n_vertices):
                        if not (v & bit) and M[axis, v]:
                            positions.append((axis, v))
                if positions:
                    to_remove = rng.sample(positions, min(5, len(positions) // 6))
                    for ax, v in to_remove:
                        M[ax, v] = 0
            else:
                M = greedy_tensor(seed)
        else:
            if best_M is not None:
                M = best_M.copy()
                rng = random.Random(seed + 555)
                positions = []
                for axis in range(d):
                    bit = 1 << axis
                    for v in range(n_vertices):
                        if not (v & bit) and M[axis, v]:
                            positions.append((axis, v))
                if positions:
                    to_remove = rng.sample(positions, min(8, len(positions) // 4))
                    for ax, v in to_remove:
                        M[ax, v] = 0
            else:
                M = greedy_tensor(seed)
        
        M = local_search_tensor(M, seed)
        M = polish_tensor(M, seed + 7777)
        
        cnt = count_edges(M)
        if check_valid(M):
            if cnt > best_count:
                best_count = cnt
                best_M = M.copy()
        
        if time_left() < 45:
            break
    
    # Convert best_M to edge list
    if best_M is None or best_count < 250:
        # Fallback: simple greedy
        M = np.zeros((d, n_vertices), dtype=np.int8)
        for axis in range(d):
            bit = 1 << axis
            for v in range(n_vertices):
                if v & bit:
                    continue
                can_add = True
                for (a, b) in face_arrays:
                    if axis not in (a, b):
                        continue
                    fa = face_arrays[(a, b)]
                    for fi in range(32):
                        for ei in range(4):
                            if fa[fi, ei, 0] == axis and fa[fi, ei, 1] == v:
                                face_sum = 0
                                for ej in range(4):
                                    if ej == ei:
                                        continue
                                    ax2, vt2 = fa[fi, ej, 0], fa[fi, ej, 1]
                                    if M[ax2, vt2]:
                                        face_sum += 1
                                if face_sum >= 3:
                                    can_add = False
                                    break
                        if not can_add:
                            break
                    if not can_add:
                        break
                if can_add:
                    M[axis, v] = 1
        best_M = M
        best_count = count_edges(M)
    
    result = []
    for axis in range(d):
        bit = 1 << axis
        for v in range(n_vertices):
            if (v & bit) or not best_M[axis, v]:
                continue
            w = v ^ bit
            result.append([v, w])
    
    return result
