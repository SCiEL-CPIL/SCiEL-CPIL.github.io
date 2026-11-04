import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    D = 11

    # Type A: Norm 2 vectors (weight 1, value 2)
    vectors_A = []
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        vectors_A.append(v1)
        v2 = [0] * D
        v2[i] = -2
        vectors_A.append(v2)
    vectors_A = np.array(vectors_A, dtype=np.int64)

    # Type D: Norm 2 vectors (weight 4, value 1/-1)
    supports = list(itertools.combinations(range(D), 4))
    N_S = len(supports)
    
    # Conflict graph for supports (intersection >= 3)
    support_masks = [0] * N_S
    for i, s in enumerate(supports):
        mask = 0
        for j in s:
            mask |= (1 << j)
        support_masks[i] = mask
        
    conflict = np.zeros((N_S, N_S), dtype=bool)
    for i in range(N_S):
        for j in range(i + 1, N_S):
            if bin(support_masks[i] & support_masks[j]).count('1') >= 3:
                conflict[i, j] = True
                conflict[j, i] = True

    # Simulated Annealing for Independent Set
    best_support_set = set()
    best_size = 0
    rng = np.random.RandomState(42)
    py_rng = random.Random(42)
    
    def greedy_init(r):
        order = np.arange(N_S)
        rng.shuffle(order)
        x = np.zeros(N_S, dtype=bool)
        blocked = np.zeros(N_S, dtype=bool)
        for v in order:
            if not blocked[v]:
                x[v] = True
                blocked |= conflict[v]
        return x

    # SA Loop
    while time.time() < deadline - 30:
        x = greedy_init(1)
        current_size = np.sum(x)
        best_x = x.copy()
        best_current_size = current_size
        
        T = 1.0
        cooling = 0.999
        for it in range(10000): # Fixed iterations per restart, controlled by outer while
            if time.time() > deadline - 30:
                break
            # Moves: Add, Remove, Swap
            in_set = np.where(x)[0]
            not_in_set = np.where(~x)[0]
            
            if len(in_set) > 0:
                blocked = np.zeros(N_S, dtype=bool)
                for v in in_set:
                    blocked |= conflict[v]
            else:
                blocked = np.zeros(N_S, dtype=bool)
                
            add_candidates = [v for v in not_in_set if not blocked[v]]
            remove_candidates = list(in_set)
            
            if not add_candidates and not remove_candidates:
                break
                
            move_type = py_rng.random()
            if move_type < 0.5 and add_candidates:
                v = py_rng.choice(add_candidates)
                x[v] = True
                new_size = current_size + 1
                if new_size > best_current_size:
                    best_current_size = new_size
                    best_x = x.copy()
                current_size = new_size
            elif move_type < 0.8 and remove_candidates:
                v = py_rng.choice(remove_candidates)
                x[v] = False
                new_size = current_size - 1
                if py_rng.random() < np.exp(-1 / T):
                    current_size = new_size
                else:
                    x[v] = True
            else:
                if add_candidates and remove_candidates:
                    v_rem = py_rng.choice(remove_candidates)
                    v_add = py_rng.choice(add_candidates)
                    if v_rem != v_add:
                        # Check compatibility of v_add with in_set \ {v_rem}
                        in_set_no_rem = [v for v in in_set if v != v_rem]
                        if len(in_set_no_rem) > 0:
                            if np.any(conflict[v_add, in_set_no_rem]):
                                continue
                        x[v_rem] = False
                        x[v_add] = True
                        # Size unchanged
            T *= cooling
                        
        if best_current_size > best_size:
            best_size = best_current_size
            best_support_set = set(np.where(best_x)[0].tolist())

    # Construct Vectors from Best Supports
    vectors_D_core = []
    for s_idx in best_support_set:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            vectors_D_core.append(v)
    vectors_D_core = np.array(vectors_D_core, dtype=np.int64)
    
    current_vectors = np.vstack([vectors_A, vectors_D_core])
    current_set_tuples = set(map(tuple, current_vectors))

    # Greedy Fill: Add remaining Type D vectors
    all_vectors_D = []
    for s in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(s, signs):
                v[idx] = sign
            all_vectors_D.append(v)
    all_vectors_D = np.array(all_vectors_D, dtype=np.int64)
    
    # Shuffle for randomness
    indices = np.arange(len(all_vectors_D))
    rng.shuffle(indices)
    
    for i in indices:
        if time.time() > deadline - 10:
            break
        v = all_vectors_D[i]
        if tuple(v) in current_set_tuples:
            continue
        # Check compatibility: dot product <= 2
        # For Type D vs Type D: dot <= 2
        # For Type D vs Type A: dot <= 2 (always true as derived)
        # So we just check against current_vectors
        dots = current_vectors @ v
        if np.all(dots <= 2):
            current_vectors = np.vstack([current_vectors, v.reshape(1, -1)])
            current_set_tuples.add(tuple(v))

    # Greedy Fill: Add Type B (Weight 2) vectors
    # Condition: dot <= 1 with all existing vectors
    # Note: Existing vectors have norm 2. Type B has norm sqrt(2).
    # Max norm is 2. Min dist >= 2.
    # u (Type B) dot v (Existing) <= 1.
    
    vectors_B = []
    for i in range(D):
        for j in range(i+1, D):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    vectors_B.append(v)
    vectors_B = np.array(vectors_B, dtype=np.int64)
    
    indices_B = np.arange(len(vectors_B))
    rng.shuffle(indices_B)
    
    for i in indices_B:
        if time.time() > deadline - 5:
            break
        v = vectors_B[i]
        if tuple(v) in current_set_tuples:
            continue
        dots = current_vectors @ v
        if np.all(dots <= 1):
            current_vectors = np.vstack([current_vectors, v.reshape(1, -1)])
            current_set_tuples.add(tuple(v))

    return current_vectors
