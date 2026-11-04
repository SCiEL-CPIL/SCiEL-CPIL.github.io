import itertools
import time
import sys
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy import sparse

def construct(v, k, t):
    deadline = time.monotonic() + 115.0
    points = list(range(v))
    
    # Precompute
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)
    
    # Build sparse incidence matrix and block t-sub index lists
    rows = []
    cols = []
    for j, b in enumerate(all_kblocks):
        for ts in itertools.combinations(b, t):
            i = tsub_to_idx[ts]
            rows.append(i)
            cols.append(j)
    
    A = sparse.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(num_tsubs, num_kblocks))
    A_csc = A.tocsc()
    
    block_tsub_lists = [None] * num_kblocks
    for j in range(num_kblocks):
        start = A_csc.indptr[j]
        end = A_csc.indptr[j + 1]
        block_tsub_lists[j] = A_csc.indices[start:end].tolist()
    
    # Greedy warm-start using logarithmic gain (soft potential function)
    # log(1 + marginal_gain) approximates diminishing returns more smoothly
    uncovered_mask = (1 << num_tsubs) - 1
    selected = []
    
    # Precompute block sizes for quick filtering
    block_tsub_counts = np.array([len(l) for l in block_tsub_lists], dtype=np.int32)
    
    greedy_deadline = deadline - 30.0
    while uncovered_mask:
        if time.monotonic() > greedy_deadline:
            break
        best_gain = 0.0
        best_block = -1
        for j in range(num_kblocks):
            if block_tsub_counts[j] == 0:
                continue
            # Compute marginal gain as bit count
            common_bits = 0
            for ti in block_tsub_lists[j]:
                if (uncovered_mask >> ti) & 1:
                    common_bits += 1
            if common_bits > 0:
                # Logarithmic gain: log(1 + gain) to smooth the objective
                gain = np.log1p(common_bits)
                if gain > best_gain:
                    best_gain = gain
                    best_block = j
        if best_block == -1:
            break
        selected.append(best_block)
        for ti in block_tsub_lists[best_block]:
            uncovered_mask &= ~(1 << ti)
    
    # ILS (Iterated Local Search) refinement
    best_selected = list(selected)
    best_count = len(best_selected)
    
    ilp_deadline = deadline - 5.0
    remaining_time = ilp_deadline - time.monotonic()
    
    if remaining_time > 15.0 and best_selected:
        # ILS: perturbation + local search (pruning)
        current = list(best_selected)
        current_count = best_count
        no_improve = 0
        max_no_improve = 20
        
        while time.monotonic() < ilp_deadline and no_improve < max_no_improve:
            # Local search: iterative pruning
            current = _prune_set(current, block_tsub_lists, num_tsubs, deadline - 2.0)
            current_count = len(current)
            
            if current_count < best_count:
                best_count = current_count
                best_selected = list(current)
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve >= max_no_improve:
                break
            
            # Perturbation: remove a random block and re-greedy fill
            if len(current) > 1:
                idx = np.random.randint(0, len(current))
                current = list(current[:idx] + current[idx+1:])
                
                # Re-greedy to cover gaps
                current_set = set(current)
                # Compute uncovered
                cov = np.zeros(num_tsubs, dtype=np.int8)
                for j in current:
                    for ti in block_tsub_lists[j]:
                        cov[ti] += 1
                
                uncovered_indices = np.where(cov == 0)[0]
                if len(uncovered_indices) > 0:
                    # Greedy fill for uncovered
                    unc_mask = 0
                    for ti in uncovered_indices:
                        unc_mask |= (1 << ti)
                    
                    fill_deadline = time.monotonic() + 2.0
                    while unc_mask:
                        if time.monotonic() > fill_deadline:
                            break
                        best_gain = 0.0
                        best_block = -1
                        for j in range(num_kblocks):
                            if j in current_set:
                                continue
                            common_bits = 0
                            for ti in block_tsub_lists[j]:
                                if (unc_mask >> ti) & 1:
                                    common_bits += 1
                            if common_bits > 0:
                                gain = np.log1p(common_bits)
                                if gain > best_gain:
                                    best_gain = gain
                                    best_block = j
                        if best_block == -1:
                            break
                        current.append(best_block)
                        current_set.add(best_block)
                        for ti in block_tsub_lists[best_block]:
                            unc_mask &= ~(1 << ti)
                
                # Prune again after fill
                current = _prune_set(current, block_tsub_lists, num_tsubs, deadline - 2.0)
                current_count = len(current)
                
                if current_count < best_count:
                    best_count = current_count
                    best_selected = list(current)
                    no_improve = 0
    
    # Final pruning pass
    if best_selected:
        best_selected = _prune_set(best_selected, block_tsub_lists, num_tsubs, deadline - 1.0)
    
    return [list(all_kblocks[j]) for j in best_selected]


def _prune_set(selected, block_tsub_lists, num_tsubs, deadline):
    """Iteratively remove redundant blocks."""
    if not selected:
        return selected
    
    counts = np.zeros(num_tsubs, dtype=np.int32)
    for j in selected:
        for ti in block_tsub_lists[j]:
            counts[ti] += 1
    
    changed = True
    while changed:
        if time.monotonic() > deadline:
            break
        changed = False
        new_sel = []
        for j in selected:
            redundant = True
            for ti in block_tsub_lists[j]:
                if counts[ti] == 1:
                    redundant = False
                    break
            if redundant:
                for ti in block_tsub_lists[j]:
                    counts[ti] -= 1
                changed = True
            else:
                new_sel.append(j)
        selected = new_sel
    
    return selected
