import time
import random
import numpy as np
import scipy.sparse as sp

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0

    total_space = q ** n

    # Precompute positional powers
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    def int_to_tuple(val):
        res = [0] * n
        for i in range(n):
            res[i] = (val // powers[i]) % q
        return res

    def run_heuristic():
        # Fallback: Greedy + Local Search
        # Re-using logic from current program but simplified
        
        # Precompute neighbor bitmasks
        masks = [0] * total_space
        for i in range(total_space):
            m = 1 << i
            for pos in range(n):
                digit_val = (i // powers[pos]) % q
                for new_val in range(q):
                    if new_val == digit_val:
                        continue
                    diff = new_val - digit_val
                    m |= (1 << (i + diff * powers[pos]))
            masks[i] = m

        full_mask = (1 << total_space) - 1
        
        # Greedy
        covered = 0
        selected = []
        remaining = full_mask
        
        while remaining and time.time() < deadline - 10:
            best_gain = -1
            best_word = -1
            # Optimization: iterate only over words that might add coverage?
            # For now, simple greedy
            for w in range(total_space):
                gain = (masks[w] & remaining).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_word = w
            
            if best_word == -1:
                break
                
            selected.append(best_word)
            covered |= masks[best_word]
            remaining = full_mask & ~covered
            
        # Polish
        if time.time() < deadline - 5:
            # Simple removal of redundant
            counts = np.zeros(total_space, dtype=np.int32)
            for w in selected:
                mm = masks[w]
                while mm:
                    lowbit = mm & -mm
                    idx = lowbit.bit_length() - 1
                    counts[idx] += 1
                    mm ^= lowbit
            
            changed = True
            while changed and time.time() < deadline - 5:
                changed = False
                indices = list(range(len(selected)))
                random.shuffle(indices)
                for idx in indices:
                    if idx >= len(selected):
                        continue
                    w = selected[idx]
                    m = masks[w]
                    removable = True
                    mm = m
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        if counts[pos] == 1:
                            removable = False
                            break
                        mm ^= lowbit
                    if removable:
                        mm = m
                        while mm:
                            lowbit = mm & -mm
                            pos = lowbit.bit_length() - 1
                            counts[pos] -= 1
                            mm ^= lowbit
                        selected.pop(idx)
                        changed = True
                        
        return [int_to_tuple(v) for v in selected]

    # Try ILP first
    try:
        import scipy.optimize as so
        
        rows = []
        cols = []
        data = []
        
        for i in range(total_space):
            # i is covered by i
            rows.append(i)
            cols.append(i)
            data.append(1)
            
            for pos in range(n):
                digit_val = (i // powers[pos]) % q
                for new_val in range(q):
                    if new_val == digit_val:
                        continue
                    diff = new_val - digit_val
                    j = i + diff * powers[pos]
                    rows.append(i)
                    cols.append(j)
                    data.append(1)
                    
        A = sp.csr_matrix((data, (rows, cols)), shape=(total_space, total_space))
        
        c = np.ones(total_space)
        
        lb = np.zeros(total_space)
        ub = np.ones(total_space)
        
        integrality = np.ones(total_space)
        
        constraints = so.LinearConstraint(A, lb=np.ones(total_space), ub=np.inf)
        
        res = so.milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=list(zip(lb, ub)),
            options={"time_limit": 130, "presolve": True, "mip_rel_gap": 0.0}
        )
        
        if res.status == 0:
            x = res.x
            selected = [i for i in range(total_space) if x[i] > 0.5]
            if len(selected) > 0:
                return [int_to_tuple(v) for v in selected]
                
    except Exception:
        pass

    return run_heuristic()
