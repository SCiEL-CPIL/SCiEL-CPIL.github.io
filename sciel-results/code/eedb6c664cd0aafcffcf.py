import time
import random
from itertools import combinations
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

START_TIME = time.time()
TIME_LIMIT = 110.0

def time_left():
    return TIME_LIMIT - (time.time() - START_TIME)

def construct():
    n = 11
    num_colorings = 1 << n
    all_edges = list(combinations(range(n), 5))
    num_edges = len(all_edges)
    
    # Precompute edge masks
    edge_masks = []
    for e in all_edges:
        v1, v2, v3, v4, v5 = e
        mask = 0
        for c in range(num_colorings):
            b1 = (c >> v1) & 1
            b2 = (c >> v2) & 1
            b3 = (c >> v3) & 1
            b4 = (c >> v4) & 1
            b5 = (c >> v5) & 1
            if b1 == b2 == b3 == b4 == b5:
                mask |= (1 << c)
        edge_masks.append(mask)
        
    full_mask = (1 << num_colorings) - 1
    best_edges = None
    best_len = float('inf')
    
    # 1. Fast Greedy with Randomization
    for trial in range(100):
        if time_left() < 10:
            break
        uncovered = full_mask
        selected_indices = []
        indices = list(range(num_edges))
        random.shuffle(indices)
        
        while uncovered:
            best_gain = -1
            best_idx = -1
            for idx in indices:
                em = edge_masks[idx]
                gain = (em & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
                elif gain == best_gain and gain > 0 and random.random() < 0.1:
                    best_idx = idx
                    
            if best_idx == -1:
                break
            selected_indices.append(best_idx)
            uncovered &= ~edge_masks[best_idx]
            
            if len(selected_indices) >= best_len:
                break
        
        if not uncovered:
            # Local Search: try to remove edges
            changed = True
            while changed and time_left() > 5:
                changed = False
                total_coverage = 0
                for idx in selected_indices:
                    total_coverage |= edge_masks[idx]
                
                for i in range(len(selected_indices)):
                    if time_left() < 5:
                        break
                    cov_without = 0
                    for j in range(len(selected_indices)):
                        if i != j:
                            cov_without |= edge_masks[selected_indices[j]]
                    if cov_without == total_coverage:
                        selected_indices.pop(i)
                        changed = True
                        break
            
            if len(selected_indices) < best_len:
                best_len = len(selected_indices)
                best_edges = [list(all_edges[idx]) for idx in selected_indices]
                if best_len <= 51:
                    break

    # 2. ILP to find optimal
    if time_left() > 20:
        try:
            rows = []
            cols = []
            data = []
            for e_idx, mask in enumerate(edge_masks):
                c = 0
                m = mask
                while m:
                    if m & 1:
                        rows.append(c)
                        cols.append(e_idx)
                        data.append(1)
                    c += 1
                    m >>= 1
            
            A = csr_matrix((data, (rows, cols)), shape=(num_colorings, num_edges))
            c_obj = [1] * num_edges
            constraint = LinearConstraint(A, lb=[1] * num_colorings, ub=[1] * num_colorings)
            bounds = Bounds(lb=[0] * num_edges, ub=[1] * num_edges)
            integrality = [1] * num_edges
            
            # Use time limit for ILP
            res = milp(c=c_obj, constraints=constraint, integrality=integrality, bounds=bounds, options={'time_limit': 30})
            
            if res.status == 0:
                x = res.x
                selected_indices = [i for i in range(num_edges) if x[i] > 0.5]
                covered = 0
                for idx in selected_indices:
                    covered |= edge_masks[idx]
                
                if covered == full_mask and len(selected_indices) < best_len:
                    best_len = len(selected_indices)
                    best_edges = [list(all_edges[idx]) for idx in selected_indices]
        except Exception:
            pass

    if best_edges is None:
        # Fallback: return a known valid set if we failed, or just the first 51?
        # Better to return the greedy result if we had one
        if best_len < float('inf'):
             return {'n': n, 'edges': best_edges}
        else:
             # Emergency: return a random set of 51? No, must be valid.
             # Just return the greedy result from trial 0 if it exists
             # Or construct a trivial one?
             # Let's return the result of the first greedy trial
             # We didn't store it separately, so we re-run one quick greedy
             uncovered = full_mask
             selected_indices = []
             indices = list(range(num_edges))
             while uncovered:
                 best_gain = -1
                 best_idx = -1
                 for idx in indices:
                     em = edge_masks[idx]
                     gain = (em & uncovered).bit_count()
                     if gain > best_gain:
                         best_gain = gain
                         best_idx = idx
                 if best_idx == -1: break
                 selected_indices.append(best_idx)
                 uncovered &= ~edge_masks[best_idx]
             return {'n': n, 'edges': [list(all_edges[idx]) for idx in selected_indices]}

    return {'n': n, 'edges': best_edges}

if __name__ == "__main__":
    res = construct()
    print(f"Found hypergraph with n={res['n']}, m={len(res['edges'])}")
