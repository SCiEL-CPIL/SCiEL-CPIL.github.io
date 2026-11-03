import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Strategy:
      1. Greedy + heuristic search for a maximum packing of disjoint 2x2x2 Strassen blocks.
      2. Fill remaining (i,j,k) triples with schoolbook rank-1 terms.
      3. Local refinement: detect pairs of terms whose sum tensor has rank 1
         (exact integer check via matricization minors), merge them to reduce rank.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 5, 7
    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    # Generate all possible 2x2x2 blocks
    all_blocks = []
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            indices = set()
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        indices.add((i, j, k))
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), frozenset(indices)))

    # Greedy search for maximum disjoint packing with multiple strategies
    best_num_blocks = 0
    best_packing = []

    def greedy_packing(blocks, used_indices, current_packing):
        for block_params, block_indices in blocks:
            if not (block_indices & used_indices):
                current_packing.append(block_params)
                used_indices = used_indices | block_indices
        return current_packing

    # Strategy 1: random shuffles
    random.seed(42)
    num_trials = 300
    for t in range(num_trials):
        if time.time() > deadline - 30.0:
            break
        shuffled = all_blocks[:]
        random.shuffle(shuffled)
        used_indices = set()
        current_packing = []
        current_packing = greedy_packing(shuffled, used_indices, current_packing)
        if len(current_packing) > best_num_blocks:
            best_num_blocks = len(current_packing)
            best_packing = current_packing[:]
            if best_num_blocks >= 13:
                break

    # Strategy 2: heuristic ordering - prefer blocks using "edge" indices
    # (edges have fewer conflicts since they share fewer neighbors)
    def edge_heuristic(block_params):
        i1, i2, j1, j2, k1, k2 = block_params
        # Prefer indices near the boundary
        score = 0
        for i in (i1, i2):
            score += min(i, n - 1 - i)
        for j in (j1, j2):
            score += min(j, m - 1 - j)
        for k in (k1, k2):
            score += min(k, p - 1 - k)
        return score

    sorted_blocks = sorted(all_blocks, key=lambda x: edge_heuristic(x[0]))
    used_indices = set()
    current_packing = []
    current_packing = greedy_packing(sorted_blocks, used_indices, current_packing)
    if len(current_packing) > best_num_blocks:
        best_num_blocks = len(current_packing)
        best_packing = current_packing[:]

    # Strategy 3: local search - swap blocks to improve
    if best_num_blocks > 0 and time.time() < deadline - 20.0:
        # Try removing one block and re-greedy the rest
        for remove_idx in range(len(best_packing)):
            if time.time() > deadline - 15.0:
                break
            remaining = [b for idx, b in enumerate(all_blocks) if b[0] != best_packing[remove_idx]]
            # Remove the specific block
            removed_block = best_packing[remove_idx]
            remaining = [b for b in remaining if b[0] != removed_block]
            # Also remove other selected blocks to re-pack
            selected_set = set(best_packing)
            remaining = [b for b in all_blocks if b[0] not in selected_set]
            random.shuffle(remaining)
            used_indices = set()
            current_packing = []
            current_packing = greedy_packing(remaining, used_indices, current_packing)
            if len(current_packing) > best_num_blocks:
                best_num_blocks = len(current_packing)
                best_packing = current_packing[:]

    # Construct U, V, W based on the best packing found
    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_entries, v_entries, w_entries):
        U_cols.append([0] * U_rows)
        V_cols.append([0] * V_rows)
        W_cols.append([0] * W_rows)
        for idx, val in u_entries.items():
            U_cols[-1][idx] = val
        for idx, val in v_entries.items():
            V_cols[-1][idx] = val
        for idx, val in w_entries.items():
            W_cols[-1][idx] = val

    def add_strassen_block(i1, i2, j1, j2, k1, k2):
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2

        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b21 = j2 * p + k1
        b22 = j2 * p + k2

        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2

        # M1 = (a11+a22)*(b11+b22)
        add_term({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1})
        # M2 = (a21+a22)*b11
        add_term({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1})
        # M3 = a11*(b12-b22)
        add_term({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1})
        # M4 = a22*(b21-b11)
        add_term({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1})
        # M5 = (a11+a12)*b22
        add_term({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1})
        # M6 = (a21-a11)*(b11+b12)
        add_term({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1})
        # M7 = (a12-a22)*(b21+b22)
        add_term({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1})

    # Add the Strassen blocks from the best packing
    covered = set()
    for block_params in best_packing:
        i1, i2, j1, j2, k1, k2 = block_params
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        for i in (i1, i2):
            for j in (j1, j2):
                for k in (k1, k2):
                    covered.add((i, j, k))

    # Add schoolbook terms for uncovered (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})

    rank = len(U_cols)

    # Local refinement: try to find pairs of terms whose sum is rank-1
    # A sum of two rank-1 tensors u1⊗v1⊗w1 + u2⊗v2⊗w2 has rank 1 iff
    # the tensor is separable. We check via matricization:
    # Reshape the 3D tensor into a 2D matrix and check if it has rank 1.
    # For exact integer arithmetic, we check if all 2x2 minors are zero.

    if time.time() < deadline - 10.0:
        def compute_tensor(u, v, w):
            """Compute the full 3D tensor from three vectors."""
            # Sparse representation: only non-zero entries
            nz_u = {i: val for i, val in enumerate(u) if val != 0}
            nz_v = {j: val for j, val in enumerate(v) if val != 0}
            nz_w = {k: val for k, val in enumerate(w) if val != 0}
            tensor = {}
            for i, ui in nz_u.items():
                for j, vj in nz_v.items():
                    for k, wk in nz_w.items():
                        tensor[(i, j, k)] = ui * vj * wk
            return tensor

        def tensors_sum(t1, t2):
            """Sum two sparse tensors."""
            result = dict(t1)
            for key, val in t2.items():
                result[key] = result.get(key, 0) + val
            # Remove zeros
            return {k: v for k, v in result.items() if v != 0}

        def is_rank_one(tensor):
            """Check if a sparse 3D tensor has rank 1 using exact arithmetic.
            A tensor has rank 1 iff it can be written as u⊗v⊗w.
            Check: for any four entries (i1,j1,k1), (i1,j1,k2), (i1,j2,k1), (i2,j1,k1),
            the "3x3" consistency condition must hold.
            More simply: matricize and check rank.
            """
            if not tensor:
                return True, ([0]*U_rows, [0]*V_rows, [0]*W_rows)

            # Get all distinct indices
            all_i = set()
            all_j = set()
            all_k = set()
            for (i, j, k) in tensor:
                all_i.add(i)
                all_j.add(j)
                all_k.add(k)

            # Pick a reference entry to normalize
            ref_key = list(tensor.keys())[0]
            ref_val = tensor[ref_key]
            ri, rj, rk = ref_key

            # Try to factor: tensor[i,j,k] = a_i * b_j * c_k
            # Set a_ri = 1, b_rj = 1, then c_rk = ref_val
            # For any (i, j, k): a_i * b_j * c_k = tensor[i,j,k]
            # => a_i * b_j = tensor[i,j,k] / c_k

            # Build candidate factorization
            # c_k = tensor[ri, rj, k] / tensor[ri, rj, rk] for k where (ri,rj,k) exists
            # a_i = tensor[i, rj, rk] / tensor[ri, rj, rk] for i where (i,rj,rk) exists
            # b_j = tensor[ri, j, rk] / tensor[ri, rj, rk] for j where (ri,j,rk) exists

            # Check consistency: for all (i,j,k) in tensor,
            # tensor[i,j,k] should equal a_i * b_j * c_k

            # Compute c_k values
            c_vals = {}
            c_vals[rk] = ref_val
            for k in all_k:
                key = (ri, rj, k)
                if key in tensor:
                    # c_k = tensor[ri,rj,k] / (a_ri * b_rj) = tensor[ri,rj,k] / 1
                    c_vals[k] = tensor[key]

            # Compute a_i values
            a_vals = {}
            a_vals[ri] = 1
            for i in all_i:
                key = (i, rj, rk)
                if key in tensor and rk in c_vals and c_vals[rk] != 0:
                    # a_i = tensor[i,rj,rk] / (b_rj * c_rk) = tensor[i,rj,rk] / c_rk
                    a_vals[i] = tensor[key] / c_vals[rk]

            # Compute b_j values
            b_vals = {}
            b_vals[rj] = 1
            for j in all_j:
                key = (ri, j, rk)
                if key in tensor and rk in c_vals and c_vals[rk] != 0:
                    # b_j = tensor[ri,j,rk] / (a_ri * c_rk) = tensor[ri,j,rk] / c_rk
                    b_vals[j] = tensor[key] / c_vals[rk]

            # Verify all entries
            for (i, j, k), val in tensor.items():
                if i in a_vals and j in b_vals and k in c_vals:
                    predicted = a_vals[i] * b_vals[j] * c_vals[k]
                    if abs(predicted - val) > 1e-9:
                        return False, None

            # Also verify that entries NOT in tensor are zero
            # (i.e., for all i,j,k not in tensor, a_i*b_j*c_k should be 0)
            # This is automatically satisfied if the factorization is correct
            # and the support matches.

            # Build the rank-1 vectors
            u = [0] * U_rows
            v = [0] * V_rows
            w = [0] * W_rows
            for i, val in a_vals.items():
                u[i] = val
            for j, val in b_vals.items():
                v[j] = val
            for k, val in c_vals.items():
                w[k] = val

            # Verify the full tensor matches (including zeros)
            for i in range(U_rows):
                for j in range(V_rows):
                    for k in range(W_rows):
                        predicted = u[i] * v[j] * w[k]
                        actual = tensor.get((i, j, k), 0)
                        if abs(predicted - actual) > 1e-9:
                            return False, None

            return True, (u, v, w)

        # Precompute tensors for all terms
        term_tensors = []
        for c in range(rank):
            term_tensors.append(compute_tensor(U_cols[c], V_cols[c], W_cols[c]))

        # Try to find mergeable pairs
        merged = [False] * rank
        new_U_cols = []
        new_V_cols = []
        new_W_cols = []

        for i in range(rank):
            if merged[i]:
                continue
            merged_i = False
            for j in range(i + 1, rank):
                if merged[j]:
                    continue
                # Check if terms i and j can be merged
                sum_tensor = tensors_sum(term_tensors[i], term_tensors[j])
                can_merge, merged_term = is_rank_one(sum_tensor)
                if can_merge:
                    u, v, w = merged_term
                    new_U_cols.append(u)
                    new_V_cols.append(v)
                    new_W_cols.append(w)
                    merged[i] = True
                    merged[j] = True
                    merged_i = True
                    break
            if not merged_i:
                new_U_cols.append(U_cols[i])
                new_V_cols.append(V_cols[i])
                new_W_cols.append(W_cols[i])

        if len(new_U_cols) < rank:
            U_cols = new_U_cols
            V_cols = new_V_cols
            W_cols = new_W_cols
            rank = len(U_cols)

    # Convert to matrix format
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]

    for c in range(rank):
        for r in range(U_rows):
            U[r][c] = U_cols[c][r]
        for r in range(V_rows):
            V[r][c] = V_cols[c][r]
        for r in range(W_rows):
            W[r][c] = W_cols[c][r]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
