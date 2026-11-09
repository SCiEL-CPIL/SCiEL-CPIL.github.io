"""Combined flow-shop permutation: fast NEH + Or-opt local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(order, P):
    """Compute makespan for a given job order. Fast list-based for small n_machines."""
    n_jobs = len(order)
    if n_jobs == 0:
        return 0.0
    n_machines = P.shape[1]
    if n_machines == 0:
        return 0.0
    if n_machines == 1:
        return float(P[order, 0].sum())
    
    # List-based for speed when n_machines is small
    first_job = order[0]
    row = P[first_job]
    curr = [0.0] * n_machines
    curr[0] = row[0]
    for m in range(1, n_machines):
        curr[m] = curr[m - 1] + row[m]
    
    for idx in range(1, n_jobs):
        j = order[idx]
        row = P[j]
        new = [0.0] * n_machines
        new[0] = curr[0] + row[0]
        for m in range(1, n_machines):
            a = curr[m]
            b = new[m - 1]
            new[m] = (a if a > b else b) + row[m]
        curr = new
    return curr[-1]

def _completion_matrix(order, P):
    """Return C[i,m] = completion time of order[i] on machine m."""
    n_jobs = len(order)
    n_machines = P.shape[1]
    if n_jobs == 0:
        return np.empty((0, n_machines), dtype=float)
    C = np.empty((n_jobs, n_machines), dtype=float)
    first = order[0]
    C[0, 0] = P[first, 0]
    for m in range(1, n_machines):
        C[0, m] = C[0, m - 1] + P[first, m]
    for i in range(1, n_jobs):
        j = order[i]
        C[i, 0] = C[i - 1, 0] + P[j, 0]
        for m in range(1, n_machines):
            a = C[i - 1, m]
            b = C[i, m - 1]
            C[i, m] = (a if a > b else b) + P[j, m]
    return C

def _neh_heuristic(P, sort_key='total'):
    """NEH heuristic with configurable sort key."""
    n_jobs, n_machines = P.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    if sort_key == 'total':
        key_vals = P.sum(axis=1)
    elif sort_key == 'first_last':
        key_vals = P[:, 0] + P[:, -1]
    elif sort_key == 'max':
        key_vals = P.max(axis=1)
    elif sort_key == 'mean':
        key_vals = P.mean(axis=1)
    else:
        key_vals = P.sum(axis=1)
    
    sorted_indices = np.argsort(-key_vals)
    current_seq = [int(sorted_indices[0])]
    
    for k in range(1, n_jobs):
        job = int(sorted_indices[k])
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(current_seq) + 1):
            new_seq = current_seq[:pos] + [job] + current_seq[pos:]
            ms = _makespan(new_seq, P)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        current_seq.insert(best_pos, job)
    
    return current_seq

def _johnson_rule(P):
    """Johnson's rule for 2-machine flow shop, extended heuristic for more machines."""
    n_jobs, n_machines = P.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    if n_machines <= 2:
        # Classic Johnson's rule
        group_a = []
        group_b = []
        for j in range(n_jobs):
            if P[j, 0] <= P[j, -1]:
                group_a.append(j)
            else:
                group_b.append(j)
        group_a.sort(key=lambda j: P[j, 0])
        group_b.sort(key=lambda j: P[j, -1], reverse=True)
        return group_a + group_b
    else:
        # For >2 machines: aggregate first half and second half
        half = n_machines // 2
        first_half = P[:, :half].sum(axis=1)
        second_half = P[:, half:].sum(axis=1)
        group_a = []
        group_b = []
        for j in range(n_jobs):
            if first_half[j] <= second_half[j]:
                group_a.append(j)
            else:
                group_b.append(j)
        group_a.sort(key=lambda j: first_half[j])
        group_b.sort(key=lambda j: second_half[j], reverse=True)
        return group_a + group_b

def _oropt_local_search(order, P, deadline):
    """Or-opt local search: remove blocks of size 1,2,3 and reinsert at best position."""
    n_jobs = len(order)
    if n_jobs < 2:
        return list(order)
    
    current_order = list(order)
    current_ms = _makespan(current_order, P)
    block_sizes = [1, 2, 3]
    
    while True:
        if time.time() > deadline:
            break
        found = False
        for i in range(n_jobs):
            if time.time() > deadline:
                break
            for k in block_sizes:
                if i + k > n_jobs:
                    continue
                block = current_order[i:i+k]
                rest = current_order[:i] + current_order[i+k:]
                best_pos = i
                best_ms = current_ms
                for j in range(len(rest) + 1):
                    if time.time() > deadline:
                        break
                    # Skip trivial re-insertion at same spot
                    if j == i and k == 1:
                        continue
                    new_order = rest[:j] + block + rest[j:]
                    ms = _makespan(new_order, P)
                    if ms < best_ms:
                        best_ms = ms
                        best_pos = j
                if best_ms < current_ms:
                    new_order = rest[:best_pos] + block + rest[best_pos:]
                    current_order = new_order
                    current_ms = best_ms
                    found = True
                    break
            if found:
                break
        if not found:
            break
    
    return current_order

def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on a flow-shop instance."""
    P = np.asarray(processing_times, dtype=float)
    if P.size == 0:
        return []
    n_jobs = P.shape[0]
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Generate diverse candidates
    candidates = []
    
    # 1. NEH with total processing time
    seq1 = _neh_heuristic(P, sort_key='total')
    candidates.append(seq1)
    
    # 2. NEH with first+last machine focus
    if time.time() < deadline - 30.0:
        seq2 = _neh_heuristic(P, sort_key='first_last')
        candidates.append(seq2)
    
    # 3. NEH with max processing time
    if time.time() < deadline - 20.0:
        seq3 = _neh_heuristic(P, sort_key='max')
        candidates.append(seq3)
    
    # 4. NEH with mean processing time
    if time.time() < deadline - 15.0:
        seq4 = _neh_heuristic(P, sort_key='mean')
        candidates.append(seq4)
    
    # 5. Johnson's rule
    if time.time() < deadline - 10.0:
        seq5 = _johnson_rule(P)
        candidates.append(seq5)
    
    # Select best candidate
    best_seq = candidates[0]
    best_ms = _makespan(best_seq, P)
    for seq in candidates[1:]:
        ms = _makespan(seq, P)
        if ms < best_ms:
            best_ms = ms
            best_seq = seq
    
    # Apply Or-opt local search to best candidate
    if time.time() < deadline - 3.0:
        improved = _oropt_local_search(best_seq, P, deadline)
        improved_ms = _makespan(improved, P)
        if improved_ms < best_ms:
            best_seq = improved
            best_ms = improved_ms
    
    return list(best_seq)

# EVOLVE-BLOCK-END
