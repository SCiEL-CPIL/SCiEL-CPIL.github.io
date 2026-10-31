import time
import random
import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import sys

# Global cache for prepared data
_CACHE = {}

def _prepare(v, k, t):
    key = (v, k, t)
    if key in _CACHE:
        return _CACHE[key]
    
    t_subs = [tuple(c) for c in itertools.combinations(range(v), t)]
    index = {sub: i for i, sub in enumerate(t_subs)}
    n_subs = len(t_subs)
    
    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)
    
    # Coverage matrix: n_blocks x n_subs (sparse)
    # We store as list of lists of indices for each block, or bitmasks?
    # Bitmasks are faster for bit operations but memory heavy if n_subs > 64?
    # n_subs = 3876. Python ints are arbitrary precision, so bitmasks are fine.
    
    cov_masks = [0] * n_blocks
    for i, blk in enumerate(blocks):
        m = 0
        for sub in itertools.combinations(blk, t):
            m |= (1 << index[sub])
        cov_masks[i] = m
            
    _CACHE[key] = (blocks, cov_masks, n_blocks, n_subs, index, t_subs)
    return _CACHE[key]

def _greedy_construct(blocks, cov_masks, n_subs, deadline, rng):
    """Greedy set cover with criticality."""
    uncovered = (1 << n_subs) - 1
    selected = []
    selected_set = set()
    counts = [0] * n_subs
    
    # Precompute list of blocks sorted by initial coverage? No, dynamic is better.
    # To speed up, we can just iterate.
    
    while uncovered and time.time() < deadline:
        best_block = -1
        best_score = -1.0
        
        # Optimization: Only consider blocks that cover at least one uncovered subset
        # We can maintain a list of candidate blocks, but for 50k blocks,
        # iterating is fast in Python if we use bit operations.
        
        for i in range(len(blocks)):
            if i in selected_set:
                continue
            
            m = cov_masks[i]
            new_cov_mask = m & uncovered
            if new_cov_mask == 0:
                continue
            
            # Criticality Score: Sum of 1/(count[s] + 1) for s in new_cov_mask
            # This favors covering subsets that are currently under-covered.
            # We approximate this by iterating bits.
            score = 0.0
            tmp = new_cov_mask
            while tmp:
                lb = tmp & (-tmp)
                idx = lb.bit_length() - 1
                # Using 1.0 / (counts[idx] + 1.0)
                score += 1.0 / (counts[idx] + 1.0)
                tmp &= tmp - 1
            
            # Add small random noise to break ties and allow exploration
            score += rng.random() * 0.01
            
            if score > best_score:
                best_score = score
                best_block = i
        
        if best_block == -1:
            break
            
        selected.append(best_block)
        selected_set.add(best_block)
        
        m = cov_masks[best_block]
        uncovered &= ~m
        # Update counts (only for bits that were uncovered? No, all bits in m)
        tmp = m
        while tmp:
            lb = tmp & (-tmp)
            idx = lb.bit_length() - 1
            counts[idx] += 1
            tmp &= tmp - 1
            
    return selected

def _local_search(selected, cov_masks, n_subs, deadline, rng):
    """Local search to reduce block count via 2-for-1 swaps."""
    sel = list(selected)
    sel_set = set(sel)
    counts = [0] * n_subs
    
    # Init counts
    for i in sel:
        m = cov_masks[i]
        while m:
            lb = m & (-m)
            counts[lb.bit_length() - 1] += 1
            m &= m - 1
            
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try dropping single blocks
        drop_idx = -1
        for pos, i in enumerate(sel):
            m = cov_masks[i]
            ok = True
            tmp = m
            while tmp:
                lb = tmp & (-tmp)
                if counts[lb.bit_length() - 1] == 1:
                    ok = False
                    break
                tmp &= tmp - 1
            if ok:
                drop_idx = pos
                break
        
        if drop_idx != -1:
            i = sel[drop_idx]
            m = cov_masks[i]
            while m:
                lb = m & (-m)
                counts[lb.bit_length() - 1] -= 1
                m &= m - 1
            sel.pop(drop_idx)
            sel_set.remove(i)
            improved = True
            continue
            
        # 2. Try 2-for-1 swaps
        # Sample pairs to avoid O(N^2) full scan if N is large
        # N is ~150. Pairs ~10k. This is fast enough to do fully or partially.
        pairs = [(p1, p2) for p1 in range(len(sel)) for p2 in range(p1+1, len(sel))]
        rng.shuffle(pairs)
        
        swap_done = False
        for p1, p2 in pairs:
            if time.time() > deadline:
                break
                
            i1 = sel[p1]
            i2 = sel[p2]
            m1 = cov_masks[i1]
            m2 = cov_masks[i2]
            
            # Find subsets that would be uncovered if we remove i1 and i2
            # A subset is critical if count <= 2 and it is covered by i1 or i2
            crit_mask = 0
            combined = m1 | m2
            tmp = combined
            while tmp:
                lb = tmp & (-tmp)
                idx = lb.bit_length() - 1
                c = counts[idx]
                if c <= 2:
                    # Check if i1 or i2 covers it
                    if (m1 & lb) or (m2 & lb):
                        if c == 1:
                            crit_mask |= lb
                        elif c == 2:
                            # Covered by both? Or one of them and another?
                            # If covered by i1 and i2, removing both uncovers it.
                            # If covered by i1 and X, removing i1 and i2 leaves X, so it's safe.
                            # We need to know if it's covered by BOTH i1 and i2.
                            if (m1 & lb) and (m2 & lb):
                                crit_mask |= lb
                tmp &= tmp - 1
                
            if crit_mask == 0:
                continue
                
            # Find a block j not in sel that covers crit_mask
            best_j = -1
            best_j_score = -1
            for j in range(len(cov_masks)):
                if j in sel_set:
                    continue
                mj = cov_masks[j]
                if (mj & crit_mask) == crit_mask:
                    # Score: how many extra subsets does j cover?
                    # We want to maximize future flexibility, so cover more is better.
                    extra = (mj & ~(m1 | m2)).bit_count()
                    score = extra + rng.random() * 0.01
                    if score > best_j_score:
                        best_j_score = score
                        best_j = j
                        
            if best_j != -1:
                # Perform swap
                j = best_j
                mj = cov_masks[j]
                
                # Remove i1
                while m1:
                    lb = m1 & (-m1)
                    counts[lb.bit_length() - 1] -= 1
                    m1 &= m1 - 1
                # Remove i2
                m2_copy = m2
                while m2_copy:
                    lb = m2_copy & (-m2_copy)
                    counts[lb.bit_length() - 1] -= 1
                    m2_copy &= m2_copy - 1
                # Add j
                while mj:
                    lb = mj & (-mj)
                    counts[lb.bit_length() - 1] += 1
                    mj &= mj - 1
                    
                # Update sel
                # Remove i1 and i2, add j
                # Careful with indices
                if p1 < p2:
                    sel.pop(p2)
                    sel.pop(p1)
                else:
                    sel.pop(p1)
                    sel.pop(p2)
                sel.append(j)
                
                sel_set.remove(i1)
                sel_set.remove(i2)
                sel_set.add(j)
                
                improved = True
                swap_done = True
                break
                
        if not swap_done:
            # If no swap found in this round, break to avoid infinite loop if no improvement possible
            # But we shuffled, so maybe try again?
            # For now, break if no improvement in a full pass.
            break
            
    return sel

def _solve_ilp(blocks, cov_masks, n_blocks, n_subs, deadline, initial_solution):
    """Solve the ILP formulation."""
    try:
        # Variables: x_b >= 0, x_b <= 1, integer
        # Objective: min sum x_b
        c = np.ones(n_blocks)
        
        # Constraints: For each subset s, sum_{b: s in b} x_b >= 1
        # A matrix: n_subs x n_blocks
        # This is large. We need to build it efficiently.
        
        # Instead of building a dense matrix, we can use the sparse structure.
        # SciPy's milp expects LinearConstraint(A, lb, ub).
        # We can build A as a sparse matrix using scipy.sparse.
        
        from scipy.sparse import lil_matrix
        
        A = lil_matrix((n_subs, n_blocks))
        
        # Fill A
        # For each block b, it covers certain subsets.
        # We need to set A[s, b] = 1 if s is in block b.
        
        # We have cov_masks. We can iterate through bits.
        # But 50k blocks * avg 35 subsets per block (C(7,4)=35) = 1.75M non-zeros.
        # This is fine.
        
        for b_idx in range(n_blocks):
            m = cov_masks[b_idx]
            while m:
                lb = m & (-m)
                s_idx = lb.bit_length() - 1
                A[s_idx, b_idx] = 1
                m &= m - 1
                
        A = A.tocsr()
        
        # Constraints: A @ x >= 1
        constraints = LinearConstraint(A, 1, np.inf)
        
        # Bounds: 0 <= x <= 1
        bounds = Bounds(0, 1)
        
        # Integrality: all integer
        integrality = np.ones(n_blocks)
        
        # Initial solution (warm start)
        # Convert initial_solution (list of block indices) to x vector
        x0 = np.zeros(n_blocks)
        for b_idx in initial_solution:
            x0[b_idx] = 1
            
        # Solve
        # Time limit is handled by the caller, but milp doesn't have a direct time limit param in older scipy.
        # However, we can check time in a loop? No, milp is a single call.
        # We rely on the fact that it's fast. If it's too slow, we catch the exception or timeout.
        # Actually, scipy.optimize.milp does not have a time limit argument directly.
        # We must run it in a thread? Or just hope it's fast.
        # For 50k vars, 4k constraints, HiGHS is very fast (usually < 1s for LP, < 10s for ILP).
        
        res = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality, x0=x0)
        
        if res.status == 0: # Optimal
            return res.x
        elif res.status == 1: # Feasible
            return res.x
        else:
            return None
            
    except Exception as e:
        print(f"ILP Error: {e}")
        return None

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 110.0
    rng = random.Random(12345)
    
    blocks, cov_masks, n_blocks, n_subs, index, t_subs = _prepare(v, k, t)
    
    # Phase 1: Greedy
    greedy_deadline = t0 + 15.0
    greedy_sel = _greedy_construct(blocks, cov_masks, n_subs, greedy_deadline, rng)
    
    # Verify Greedy
    covered = 0
    for i in greedy_sel:
        covered |= cov_masks[i]
    if covered != (1 << n_subs) - 1:
        # Fallback: add missing blocks
        # This shouldn't happen with correct greedy
        pass

    # Phase 2: ILP
    ilp_deadline = t0 + 90.0
    ilp_sel = None
    if time.time() < ilp_deadline:
        ilp_sel = _solve_ilp(blocks, cov_masks, n_blocks, n_subs, ilp_deadline, greedy_sel)
        
    # Phase 3: Local Search on the best of Greedy and ILP
    best_sel = greedy_sel
    if ilp_sel is not None:
        # Count blocks
        ilp_count = int(np.sum(ilp_sel > 0.5))
        greedy_count = len(greedy_sel)
        
        if ilp_count < greedy_count:
            best_sel = [i for i, x in enumerate(ilp_sel) if x > 0.5]
        else:
            # If ILP is not better, stick with Greedy (or try ILP if equal? No, Greedy is already there)
            best_sel = greedy_sel
            
    ls_deadline = deadline - 2.0
    if time.time() < ls_deadline:
        best_sel = _local_search(best_sel, cov_masks, n_subs, ls_deadline, rng)
        
    # Final Verification
    covered = 0
    for i in best_sel:
        covered |= cov_masks[i]
        
    if covered != (1 << n_subs) - 1:
        # Emergency fix
        uncovered = ((1 << n_subs) - 1) & ~covered
        for i in range(n_blocks):
            if i in best_sel:
                continue
            if cov_masks[i] & uncovered:
                best_sel.append(i)
                uncovered &= ~cov_masks[i]
                if not uncovered:
                    break
                    
    return [list(blocks[i]) for i in best_sel]
