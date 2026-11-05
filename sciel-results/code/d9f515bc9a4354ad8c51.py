"""Rewritten program: structural change to fast delta computation + parallel local search."""
import time
import numpy as np

def construct():
    deadline = time.time() + 115  # safety margin before 120s
    
    N = 256
    
    # Precompute all 4-term APs (0-indexed)
    aps = []
    for d in range(1, N // 3 + 1):
        max_start = N - 3 * d
        for a in range(1, max_start + 1):
            i0 = a - 1
            aps.append((i0, i0 + d, i0 + 2 * d, i0 + 3 * d))
    
    n_aps = len(aps)
    aps_arr = np.array(aps, dtype=np.int32)  # shape (n_aps, 4)
    
    # For each position, precompute the other 3 positions in each AP it belongs to
    # pos_ap_data[pos] = (ap_idxs, slots, others_array) where others_array is (n_aps_for_pos, 3)
    pos_ap_data = []
    for pos in range(N):
        ap_idxs = []
        slots = []
        others_list = []
        for ap_idx in range(n_aps):
            found = False
            for slot in range(4):
                if aps_arr[ap_idx, slot] == pos:
                    ap_idxs.append(ap_idx)
                    slots.append(slot)
                    others_list.append([int(aps_arr[ap_idx, s]) for s in range(4) if s != slot])
                    found = True
                    break
        if ap_idxs:
            pos_ap_data.append((np.array(ap_idxs, dtype=np.int32),
                               np.array(slots, dtype=np.int32),
                               np.array(others_list, dtype=np.int32)))
        else:
            pos_ap_data.append((np.array([], dtype=np.int32),
                               np.array([], dtype=np.int32),
                               np.array([], dtype=np.int32)))
    
    def count_mono(coloring):
        """Count monochromatic 4-APs."""
        c = coloring[aps_arr]
        mono = (c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])
        return int(mono.sum())
    
    def calc_all_deltas_fast(coloring):
        """
        Fast delta computation using precomputed other-positions.
        For position p in AP at slot s, with other positions (o1,o2,o3):
        - If o1==o2==o3: delta contribution is -1 if c_p==o1, +1 if c_p!=o1
        - If o1,o2,o3 not all equal: delta contribution is 0
        """
        deltas = np.zeros(N, dtype=np.int32)
        
        for idx in range(N):
            ap_idxs, slots, others = pos_ap_data[idx]
            if len(ap_idxs) == 0:
                continue
            
            c_others = coloring[others]  # shape (n_aps_for_pos, 3)
            
            # Check if the 3 others are all equal
            others_equal = (c_others[:, 0] == c_others[:, 1]) & (c_others[:, 1] == c_others[:, 2])
            
            if not others_equal.any():
                continue
            
            c_p = coloring[idx]
            # For APs where others are equal:
            # If c_p == c_others[0]: was mono, won't be after flip => -1
            # If c_p != c_others[0]: wasn't mono, will be after flip => +1
            delta = np.where(others_equal,
                            np.where(c_others[:, 0] == c_p, -1, 1),
                            0).sum()
            deltas[idx] = delta
        
        return deltas
    
    def greedy_local_search(coloring, current_count, max_sweeps=2000):
        """Greedy local search: flip best position until no improvement."""
        for _ in range(max_sweeps):
            if time.time() > deadline - 3:
                break
            deltas = calc_all_deltas_fast(coloring)
            min_delta = deltas.min()
            if min_delta >= 0:
                break
            best_idx = int(deltas.argmin())
            coloring[best_idx] = 1 - coloring[best_idx]
            current_count += int(min_delta)
        return coloring, current_count
    
    def parallel_local_search(coloring, current_count, max_sweeps=1000):
        """Flip all improving positions simultaneously per sweep."""
        for _ in range(max_sweeps):
            if time.time() > deadline - 3:
                break
            deltas = calc_all_deltas_fast(coloring)
            improving = np.where(deltas < 0)[0]
            if len(improving) == 0:
                break
            for p in improving:
                coloring[p] = 1 - coloring[p]
            current_count = count_mono(coloring)
        return coloring, current_count
    
    # Build pair interaction matrix for spectral init
    pair_count = np.zeros((N, N), dtype=np.int32)
    for ap in aps:
        for i in range(4):
            for j in range(i + 1, 4):
                pair_count[ap[i], ap[j]] += 1
                pair_count[ap[j], ap[i]] += 1
    
    # Spectral initialization
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(pair_count)
        v = eigenvectors[:, -2]
        median_val = np.median(v)
        coloring_spec = (v > median_val).astype(np.int32)
    except:
        coloring_spec = np.random.randint(0, 2, N)
    
    best_coloring = None
    best_count = float('inf')
    
    # Generate diverse initializations
    np.random.seed(42)
    initializations = []
    
    # Spectral
    initializations.append(coloring_spec.copy())
    
    # Random
    for _ in range(12):
        initializations.append(np.random.randint(0, 2, N))
    
    # Block patterns
    for block_size in [2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 32]:
        coloring = np.array([(i // block_size) % 2 for i in range(N)])
        initializations.append(coloring)
        initializations.append(1 - coloring)
    
    # Thue-Morse
    coloring = np.array([(bin(i).count('1') % 2) for i in range(N)])
    initializations.append(coloring)
    initializations.append(1 - coloring)
    
    # Striped
    for period in [4, 6, 8, 12, 16, 24, 32, 48, 64]:
        coloring = np.array([((i // period) % 2) for i in range(N)])
        initializations.append(coloring)
    
    # Run search from each initialization
    for init in initializations:
        if time.time() > deadline - 8:
            break
        
        coloring = init.copy()
        current_count = count_mono(coloring)
        
        if current_count < best_count:
            best_count = current_count
            best_coloring = coloring.copy()
        
        # Greedy local search
        coloring, current_count = greedy_local_search(coloring, current_count)
        
        if current_count < best_count:
            best_count = current_count
            best_coloring = coloring.copy()
        
        # Parallel local search
        if time.time() < deadline - 5:
            coloring_p = coloring.copy()
            count_p = current_count
            coloring_p, count_p = parallel_local_search(coloring_p, count_p)
            
            if count_p < best_count:
                best_count = count_p
                best_coloring = coloring_p.copy()
        
        # Perturbation + re-optimization
        if time.time() < deadline - 5 and best_coloring is not None:
            for _ in range(6):
                if time.time() > deadline - 5:
                    break
                perturbed = best_coloring.copy()
                n_flips = np.random.randint(3, 15)
                flip_idx = np.random.choice(N, n_flips, replace=False)
                perturbed[flip_idx] = 1 - perturbed[flip_idx]
                count_pt = count_mono(perturbed)
                
                perturbed, count_pt = greedy_local_search(perturbed, count_pt)
                
                if count_pt < best_count:
                    best_count = count_pt
                    best_coloring = perturbed.copy()
    
    # Final polish
    if best_coloring is not None and time.time() < deadline - 3:
        best_coloring, best_count = greedy_local_search(best_coloring.copy(), best_count)
    
    return best_coloring.tolist()
