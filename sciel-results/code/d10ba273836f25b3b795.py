"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 110
    random.seed(12345)
    np.random.seed(12345)
    
    N = 256
    aps = []
    for d in range(1, 86):
        max_start = N - 3 * d
        for a in range(1, max_start + 1):
            i0 = a - 1
            i1 = i0 + d
            i2 = i0 + 2 * d
            i3 = i0 + 3 * d
            aps.append((i0, i1, i2, i3))
    
    n_aps = len(aps)
    aps_arr = np.array(aps, dtype=np.int32)
    
    def count_mono(coloring):
        c = coloring[aps_arr]
        mono = (c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])
        return int(mono.sum())
    
    def compute_all_deltas(coloring):
        c = coloring
        c0 = c[aps_arr[:, 0]]
        c1 = c[aps_arr[:, 1]]
        c2 = c[aps_arr[:, 2]]
        c3 = c[aps_arr[:, 3]]
        
        mono = (c0 == c1) & (c1 == c2) & (c2 == c3)
        
        mono_flip0 = (c1 == c2) & (c2 == c3) & (c3 == 1 - c0)
        mono_flip1 = (c0 == c2) & (c2 == c3) & (c3 == 1 - c1)
        mono_flip2 = (c0 == c1) & (c1 == c3) & (c3 == 1 - c2)
        mono_flip3 = (c0 == c1) & (c1 == c2) & (c2 == 1 - c3)
        
        delta0 = mono_flip0.astype(np.int8) - mono.astype(np.int8)
        delta1 = mono_flip1.astype(np.int8) - mono.astype(np.int8)
        delta2 = mono_flip2.astype(np.int8) - mono.astype(np.int8)
        delta3 = mono_flip3.astype(np.int8) - mono.astype(np.int8)
        
        deltas = np.zeros(N, dtype=np.int32)
        np.add.at(deltas, aps_arr[:, 0], delta0)
        np.add.at(deltas, aps_arr[:, 1], delta1)
        np.add.at(deltas, aps_arr[:, 2], delta2)
        np.add.at(deltas, aps_arr[:, 3], delta3)
        
        return deltas
    
    best_coloring = None
    best_count = float('inf')
    
    initializations = []
    for _ in range(5):
        initializations.append(np.random.randint(0, 2, N))
    for block_size in [1, 2, 3, 4, 5, 8]:
        coloring = np.array([(i // block_size) % 2 for i in range(N)])
        initializations.append(coloring)
    for bit in [0, 1, 2, 3]:
        coloring = np.array([(i >> bit) & 1 for i in range(N)])
        initializations.append(coloring)
    for p in [0.3, 0.7]:
        initializations.append((np.random.random(N) < p).astype(np.int32))
    tm = np.array([bin(i).count('1') % 2 for i in range(N)], dtype=np.int32)
    initializations.append(tm)
    
    for init in initializations:
        if time.time() > deadline - 10:
            break
        
        coloring = init.copy()
        current_count = count_mono(coloring)
        
        if current_count < best_count:
            best_count = current_count
            best_coloring = coloring.copy()
        
        for sweep in range(300):
            if time.time() > deadline - 10:
                break
            
            deltas = compute_all_deltas(coloring)
            min_delta = deltas.min()
            
            if min_delta >= 0:
                if random.random() < 0.2:
                    idx = np.random.randint(N)
                    coloring[idx] = 1 - coloring[idx]
                    current_count = count_mono(coloring)
                    if current_count < best_count:
                        best_count = current_count
                        best_coloring = coloring.copy()
                else:
                    break
            
            best_idx = int(np.argmin(deltas))
            coloring[best_idx] = 1 - coloring[best_idx]
            current_count += min_delta
            if current_count < best_count:
                best_count = current_count
                best_coloring = coloring.copy()
        
        for _ in range(3):
            if time.time() > deadline - 10:
                break
            perturbed = coloring.copy()
            n_flips = random.randint(5, 15)
            flip_idx = np.random.choice(N, n_flips, replace=False)
            perturbed[flip_idx] = 1 - perturbed[flip_idx]
            current_count = count_mono(perturbed)
            
            for sweep in range(150):
                if time.time() > deadline - 10:
                    break
                deltas = compute_all_deltas(perturbed)
                min_delta = deltas.min()
                if min_delta >= 0:
                    break
                best_idx = int(np.argmin(deltas))
                perturbed[best_idx] = 1 - perturbed[best_idx]
                current_count += min_delta
                if current_count < best_count:
                    best_count = current_count
                    best_coloring = perturbed.copy()
    
    return best_coloring.tolist()
# EVOLVE-BLOCK-END
