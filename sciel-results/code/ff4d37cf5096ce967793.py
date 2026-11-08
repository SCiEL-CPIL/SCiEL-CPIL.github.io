import time
import random
import itertools

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses a Simulated Annealing approach to find a maximum packing of disjoint 2x2x2 Strassen blocks.
    The rank is 105 - (number of blocks).
    """
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    # Generate all possible 2x2x2 Strassen blocks
    # Each block is defined by (i1, i2, j1, j2, k1, k2) with i1<i2, j1<j2, k1<k2
    # It covers the set of cells {(i, j, k) for i in (i1,i2), j in (j1,j2), k in (k1,k2)}
    
    all_blocks = []
    block_indices = []
    
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
                            all_blocks.append((i1, i2, j1, j2, k1, k2))
                            block_indices.append(frozenset(indices))

    num_blocks = len(all_blocks)
    
    # Pre-compute conflict matrix (or adjacency list)
    # conflict[i] is a set of indices j such that block i and block j conflict (share a cell)
    conflicts = [set() for _ in range(num_blocks)]
    for i in range(num_blocks):
        for j in range(i + 1, num_blocks):
            if block_indices[i] & block_indices[j]:
                conflicts[i].add(j)
                conflicts[j].add(i)

    # Simulated Annealing to find max independent set
    best_size = 0
    best_set = set()
    
    # Initial solution: Greedy
    def greedy_init():
        s = set()
        remaining = list(range(num_blocks))
        random.shuffle(remaining)
        for b in remaining:
            if all((b not in conflicts[other]) or (other not in s) for other in s):
                # Check if b conflicts with any in s
                # More efficient: check if b is in conflicts[other] for any other in s
                # Or simply: if not any(b in conflicts[other] for other in s)
                if not any(b in conflicts[other] for other in s):
                    s.add(b)
        return s

    # Run SA for a few seconds
    current_set = greedy_init()
    current_size = len(current_set)
    if current_size > best_size:
        best_size = current_size
        best_set = current_set.copy()

    T = 1.0
    cooling = 0.999
    iterations = 50000
    
    for _ in range(iterations):
        if time.time() > deadline - 20.0:
            break
        if T < 0.01:
            T = 1.0
            current_set = greedy_init()
            current_size = len(current_set)
            if current_size > best_size:
                best_size = current_size
                best_set = current_set.copy()
            continue

        # Generate a neighbor
        # Move types: Add, Remove, Swap
        move_type = random.choice(['add', 'remove', 'swap'])
        
        new_set = current_set.copy()
        
        if move_type == 'add':
            # Try to add a random block not in set
            candidates = [b for b in range(num_blocks) if b not in current_set]
            if candidates:
                b = random.choice(candidates)
                # Check if b can be added (no conflict with current set)
                if not any(b in conflicts[other] for other in current_set):
                    new_set.add(b)
        elif move_type == 'remove':
            if current_set:
                b = random.choice(list(current_set))
                new_set.remove(b)
        elif move_type == 'swap':
            if current_set:
                b_out = random.choice(list(current_set))
                # Find a block to swap in that doesn't conflict with (current_set - {b_out})
                # and ideally conflicts with b_out (otherwise it's just add+remove)
                # But any valid block works.
                candidates = [b for b in range(num_blocks) if b not in current_set]
                if candidates:
                    b_in = random.choice(candidates)
                    # Check if b_in is compatible with current_set - {b_out}
                    if not any(b_in in conflicts[other] for other in current_set if other != b_out):
                        new_set.remove(b_out)
                        new_set.add(b_in)

        new_size = len(new_set)
        delta = new_size - current_size
        
        if delta >= 0 or random.random() < pow(2.71828, -delta / T):
            current_set = new_set
            current_size = new_size
            if current_size > best_size:
                best_size = current_size
                best_set = current_set.copy()
                if best_size == 13: # Theoretical max
                    break
        T *= cooling

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

    covered = set()
    for b_idx in best_set:
        i1, i2, j1, j2, k1, k2 = all_blocks[b_idx]
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        for i in (i1, i2):
            for j in (j1, j2):
                for k in (k1, k2):
                    covered.add((i, j, k))

    # Add schoolbook terms for uncovered (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    break
                if (i, j, k) not in covered:
                    add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})

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
