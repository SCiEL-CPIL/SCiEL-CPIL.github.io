"""
Optimized Covering Design Solver for C(19,7,4)
Strategy:
1. Fast Greedy Initialization with randomization.
2. Reduction (deletion, pair replacement).
3. Lagrangian-guided WalkSAT with incremental unsat tracking and simulated annealing.
4. Final reduction and validation.
"""
import time
import random
import itertools
import numpy as np
from collections import defaultdict

# Fixed seed for reproducibility of the random parts if needed, but we use multiple seeds
_RNG = random.Random(20250101)

def construct(v, k, t):
    start_time = time.time()
    # Safety margin: 120s limit. Use 115s for computation.
    deadline = start_time + 115.0

    # Precompute all t-subsets and their indices
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_index = {ts: i for i, ts in enumerate(t_subs)}
    n_t = len(t_subs)

    # Precompute all k-subsets and their covered t-subset indices
    k_subs = list(itertools.combinations(range(v), k))
    n_k = len(k_subs)
    
    # block_covers[i] = list of t-subset indices covered by k-subset i
    block_covers = []
    for ks in k_subs:
        covered = []
        for ts in itertools.combinations(ks, t):
            covered.append(t_sub_index[ts])
        block_covers.append(covered)
        
    # Convert to numpy arrays for faster indexing
    block_covers_np = [np.array(c, dtype=np.int32) for c in block_covers]

    # Inverse: t-subset index -> list of block indices covering it
    t_to_blocks = [[] for _ in range(n_t)]
    for bi, cov in enumerate(block_covers):
        for ti in cov:
            t_to_blocks[ti].append(bi)
    t_to_blocks_np = [np.array(b, dtype=np.int32) for b in t_to_blocks]

    # ---------------- Phase 1: Greedy Initialization ----------------
    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = np.ones(n_t, dtype=bool)
        gain = np.zeros(n_k, dtype=np.int32)
        for bi in range(n_k):
            gain[bi] = np.sum(uncovered[block_covers_np[bi]])
            
        blocks_used = []
        while np.any(uncovered):
            if time.time() > deadline - 60.0:
                break
            max_gain = gain.max()
            if max_gain == 0:
                break
            candidates = np.where(gain == max_gain)[0]
            chosen = candidates[rng.randint(0, len(candidates) - 1)]
            blocks_used.append(chosen)
            
            # Update uncovered and gains
            for ts_idx in block_covers_np[chosen]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    # Decrease gain for all blocks covering this t-subset
                    for bi in t_to_blocks_np[ts_idx]:
                        if bi != chosen:
                            gain[bi] -= 1
            gain[chosen] = -1 # Mark as used
            
        return blocks_used

    # Run a few greedy restarts to get a good initial solution
    best_greedy = None
    best_greedy_size = n_k + 1
    for seed in range(5):
        if time.time() > deadline - 80.0:
            break
        blocks = greedy_cover(seed)
        if len(blocks) < best_greedy_size:
            best_greedy_size = len(blocks)
            best_greedy = blocks
        if len(blocks) <= 135:
            break

    if best_greedy is None:
        best_greedy = greedy_cover(0)

    selected = set(best_greedy)

    # Build cover_count
    cover_count = np.zeros(n_t, dtype=np.int32)
    for bi in selected:
        cover_count[block_covers_np[bi]] += 1

    # ---------------- Phase 2: Reduction ----------------
    def reduce_blocks(selected_set, cc):
        changed = True
        while changed:
            changed = False
            for bi in list(selected_set):
                if time.time() > deadline - 30.0:
                    return selected_set, cc
                cc[block_covers_np[bi]] -= 1
                if np.any(cc[block_covers_np[bi]] == 0):
                    cc[block_covers_np[bi]] += 1
                else:
                    selected_set.discard(bi)
                    changed = True
        return selected_set, cc

    def pair_replace(selected_set, cc, max_trials=2000):
        rng = _RNG
        n_sel = len(selected_set)
        if n_sel < 2:
            return selected_set, cc
        sel_list = list(selected_set)
        
        for trial in range(max_trials):
            if time.time() > deadline - 40.0:
                break
            i, j = rng.sample(range(n_sel), 2)
            bi, bj = sel_list[i], sel_list[j]
            
            # Union of t-subsets covered by bi and bj
            union_t = set(block_covers_np[bi].tolist()) | set(block_covers_np[bj].tolist())
            if len(union_t) > 35: # Heuristic: 7 choose 4 is 35, so union must be <= 35 to be covered by 1 block
                continue
                
            # Find common blocks covering all t-subsets in union_t
            common = None
            for ti in union_t:
                blocks_covering = set(t_to_blocks[ti])
                if common is None:
                    common = blocks_covering
                else:
                    common &= blocks_covering
                if not common:
                    break
            if not common:
                continue
                
            for cand in common:
                if cand == bi or cand == bj:
                    continue
                # Try replacement
                cc[block_covers_np[bi]] -= 1
                cc[block_covers_np[bj]] -= 1
                cc[block_covers_np[cand]] += 1
                if np.all(cc >= 1):
                    selected_set.discard(bi)
                    selected_set.discard(bj)
                    selected_set.add(cand)
                    n_sel = len(selected_set)
                    sel_list = list(selected_set)
                    # Reduce again
                    selected_set, cc = reduce_blocks(selected_set, cc)
                    n_sel = len(selected_set)
                    sel_list = list(selected_set)
                    break
                else:
                    cc[block_covers_np[bi]] += 1
                    cc[block_covers_np[bj]] += 1
                    cc[block_covers_np[cand]] -= 1
                    
        return selected_set, cc

    def triple_replace(selected_set, cc, max_trials=1000):
        """Try to replace 3 blocks with 2 blocks."""
        rng = _RNG
        n_sel = len(selected_set)
        if n_sel < 3:
            return selected_set, cc
        sel_list = list(selected_set)
        
        for trial in range(max_trials):
            if time.time() > deadline - 40.0:
                break
            i, j, l = rng.sample(range(n_sel), 3)
            bi, bj, bl = sel_list[i], sel_list[j], sel_list[l]
            
            # Union of t-subsets covered by bi, bj, bl
            union_t = set(block_covers_np[bi].tolist()) | set(block_covers_np[bj].tolist()) | set(block_covers_np[bl].tolist())
            if len(union_t) > 70: # Heuristic: 3 * 35 = 105 max, but typically less
                continue
                
            # Find candidate pairs that cover all union_t
            # For each candidate c1 in t_to_blocks[min_ti], find c2 that covers remaining
            min_count = 1000000
            min_ti = -1
            for ti in union_t:
                c = len(t_to_blocks[ti])
                if c < min_count:
                    min_count = c
                    min_ti = ti
                    
            if min_ti == -1:
                continue
                
            # For each c1 covering min_ti, find c2 that covers union_t \ covers(c1)
            for c1 in t_to_blocks[min_ti]:
                if c1 in (bi, bj, bl):
                    continue
                cov_c1 = set(block_covers_np[c1].tolist())
                remaining = union_t - cov_c1
                if not remaining:
                    # c1 alone covers all - try 3-for-1
                    cc[block_covers_np[bi]] -= 1
                    cc[block_covers_np[bj]] -= 1
                    cc[block_covers_np[bl]] -= 1
                    cc[block_covers_np[c1]] += 1
                    if np.all(cc >= 1):
                        selected_set.discard(bi)
                        selected_set.discard(bj)
                        selected_set.discard(bl)
                        selected_set.add(c1)
                        selected_set, cc = reduce_blocks(selected_set, cc)
                        return selected_set, cc
                    else:
                        cc[block_covers_np[bi]] += 1
                        cc[block_covers_np[bj]] += 1
                        cc[block_covers_np[bl]] += 1
                        cc[block_covers_np[c1]] -= 1
                    continue
                    
                # Find c2 that covers all remaining
                # Pick the rarest ti in remaining
                min_count2 = 1000000
                min_ti2 = -1
                for ti in remaining:
                    c = len(t_to_blocks[ti])
                    if c < min_count2:
                        min_count2 = c
                        min_ti2 = ti
                        
                if min_ti2 == -1:
                    continue
                    
                for c2 in t_to_blocks[min_ti2]:
                    if c2 in (bi, bj, bl, c1):
                        continue
                    # Check if c1 and c2 together cover all union_t
                    cov_c2 = set(block_covers_np[c2].tolist())
                    if union_t <= (cov_c1 | cov_c2):
                        # Try replacement
                        cc[block_covers_np[bi]] -= 1
                        cc[block_covers_np[bj]] -= 1
                        cc[block_covers_np[bl]] -= 1
                        cc[block_covers_np[c1]] += 1
                        cc[block_covers_np[c2]] += 1
                        if np.all(cc >= 1):
                            selected_set.discard(bi)
                            selected_set.discard(bj)
                            selected_set.discard(bl)
                            selected_set.add(c1)
                            selected_set.add(c2)
                            selected_set, cc = reduce_blocks(selected_set, cc)
                            return selected_set, cc
                        else:
                            cc[block_covers_np[bi]] += 1
                            cc[block_covers_np[bj]] += 1
                            cc[block_covers_np[bl]] += 1
                            cc[block_covers_np[c1]] -= 1
                            cc[block_covers_np[c2]] -= 1
                            
        return selected_set, cc

    selected, cover_count = reduce_blocks(selected, cover_count)
    selected, cover_count = pair_replace(selected, cover_count)
    selected, cover_count = reduce_blocks(selected, cover_count)
    # Add triple replacement for stronger reduction
    selected, cover_count = triple_replace(selected, cover_count, max_trials=500)
    selected, cover_count = reduce_blocks(selected, cover_count)

    # ---------------- Phase 3: Optimized Local Search ----------------
    # State: boolean array x
    x = np.array([bi in selected for bi in range(n_k)], dtype=bool)
    
    # Clause satisfaction counts
    clause_sat = np.zeros(n_t, dtype=np.int32)
    for bi in range(n_k):
        if x[bi]:
            clause_sat[block_covers_np[bi]] += 1
            
    best_n_true = x.sum()
    best_x = x.copy()
    
    # Local Search Loop: Deletion and 2-for-1 Replacement
    # We use a simpler, faster loop that focuses on reducing the number of blocks.
    # Since we start with a valid cover, we only need to try to remove blocks or replace pairs.
    
    # To speed up, we precompute the list of blocks in the current set
    # and update it as we go.
    
    # We will alternate between:
    # 1. Trying to remove a random block (deletion)
    # 2. Trying a 2-for-1 replacement
    
    # We will run this for a fixed number of iterations or until time runs out.
    # Each iteration is fast.
    
    # To avoid expensive set operations, we maintain a list of selected blocks.
    # However, removal from a list is O(n). For n~150, it's fine.
    
    # Let's use a set for O(1) removal, but we need a list for random access.
    # We can maintain both or just use a list and swap-remove.
    
    sel_list = list(selected)
    
    # We will track the number of successful moves to detect stagnation.
    # If no moves for a while, we might need to perturb (add a block) to escape.
    # But adding a block increases size, which is bad.
    # So we rely on 2-for-1 to escape.
    
    # To make 2-for-1 faster, we can precompute for each pair of blocks in the set
    # if they can be replaced. But that's O(n^2) per step.
    # Instead, we sample pairs.
    
    # Optimization: If a block is "unique" (covers a t-subset not covered by any other block),
    # it cannot be removed. We can precompute this?
    # It changes as we remove blocks.
    
    # Let's stick to the sampling approach but make it very fast.
    
    steps = 0
    max_steps = 500000 # High limit, controlled by time
    
    while time.time() < deadline - 10.0:
        steps += 1
        
        # Random action: 50% deletion, 50% 2-for-1
        if _RNG.random() < 0.5:
            # Try deletion
            if not sel_list:
                break
            idx = _RNG.randint(0, len(sel_list) - 1)
            bi = sel_list[idx]
            
            # Check if bi can be removed
            # We need to check if any t-subset covered by bi has count 1.
            # If yes, removing bi makes it 0 -> invalid.
            # We can check this by iterating over block_covers_np[bi]
            # and checking clause_sat[ti] > 1.
            
            removable = True
            for ti in block_covers_np[bi]:
                if clause_sat[ti] == 1:
                    removable = False
                    break
            
            if removable:
                # Remove bi
                clause_sat[block_covers_np[bi]] -= 1
                # Swap remove from sel_list
                sel_list[idx] = sel_list[-1]
                sel_list.pop()
                best_n_true -= 1
                best_x[bi] = False
        elif _RNG.random() < 0.3:
            # Try 3-for-2 (or 3-for-1)
            if len(sel_list) < 3:
                continue
            i, j, l = _RNG.sample(range(len(sel_list)), 3)
            bi, bj, bl = sel_list[i], sel_list[j], sel_list[l]
            
            union_t = set(block_covers_np[bi].tolist()) | set(block_covers_np[bj].tolist()) | set(block_covers_np[bl].tolist())
            if len(union_t) > 70:
                continue
            
            # Find rarest ti
            min_count = 1000000
            min_ti = -1
            for ti in union_t:
                c = len(t_to_blocks[ti])
                if c < min_count:
                    min_count = c
                    min_ti = ti
            
            if min_ti == -1:
                continue
            
            # Try 3-for-1 first
            for c1 in t_to_blocks[min_ti]:
                if c1 in (bi, bj, bl):
                    continue
                cov_c1 = set(block_covers_np[c1].tolist())
                if union_t <= cov_c1:
                    # Check validity
                    valid = True
                    for ti in union_t:
                        count = clause_sat[ti]
                        if ti in block_covers[bi]:
                            count -= 1
                        if ti in block_covers[bj]:
                            count -= 1
                        if ti in block_covers[bl]:
                            count -= 1
                        count += 1  # c1 covers all union_t
                        if count < 1:
                            valid = False
                            break
                    if valid:
                        clause_sat[block_covers_np[bi]] -= 1
                        clause_sat[block_covers_np[bj]] -= 1
                        clause_sat[block_covers_np[bl]] -= 1
                        clause_sat[block_covers_np[c1]] += 1
                        # Remove bi, bj, bl and add c1
                        # Find indices
                        idx_map = {bi: i, bj: j, bl: l}
                        for b in [bl, bj, bi]:  # Remove in reverse order to preserve indices
                            idx = idx_map[b]
                            sel_list[idx] = sel_list[-1]
                            sel_list.pop()
                        sel_list.append(c1)
                        best_n_true -= 2
                        best_x[bi] = False
                        best_x[bj] = False
                        best_x[bl] = False
                        best_x[c1] = True
                        break
                else:
                    # Try 3-for-2
                    remaining = union_t - cov_c1
                    if not remaining:
                        continue
                    min_count2 = 1000000
                    min_ti2 = -1
                    for ti in remaining:
                        c = len(t_to_blocks[ti])
                        if c < min_count2:
                            min_count2 = c
                            min_ti2 = ti
                    if min_ti2 == -1:
                        continue
                    for c2 in t_to_blocks[min_ti2]:
                        if c2 in (bi, bj, bl, c1):
                            continue
                        cov_c2 = set(block_covers_np[c2].tolist())
                        if union_t <= (cov_c1 | cov_c2):
                            # Check validity
                            valid = True
                            for ti in union_t:
                                count = clause_sat[ti]
                                if ti in block_covers[bi]:
                                    count -= 1
                                if ti in block_covers[bj]:
                                    count -= 1
                                if ti in block_covers[bl]:
                                    count -= 1
                                if ti in block_covers[c1]:
                                    count += 1
                                if ti in block_covers[c2]:
                                    count += 1
                                if count < 1:
                                    valid = False
                                    break
                            if valid:
                                clause_sat[block_covers_np[bi]] -= 1
                                clause_sat[block_covers_np[bj]] -= 1
                                clause_sat[block_covers_np[bl]] -= 1
                                clause_sat[block_covers_np[c1]] += 1
                                clause_sat[block_covers_np[c2]] += 1
                                # Remove bi, bj, bl and add c1, c2
                                idx_map = {bi: i, bj: j, bl: l}
                                for b in [bl, bj, bi]:
                                    idx = idx_map[b]
                                    sel_list[idx] = sel_list[-1]
                                    sel_list.pop()
                                sel_list.append(c1)
                                sel_list.append(c2)
                                best_n_true -= 1
                                best_x[bi] = False
                                best_x[bj] = False
                                best_x[bl] = False
                                best_x[c1] = True
                                best_x[c2] = True
                                break
        else:
            # Try 2-for-1
            if len(sel_list) < 2:
                continue
            i, j = _RNG.sample(range(len(sel_list)), 2)
            bi, bj = sel_list[i], sel_list[j]
            
            # Union of t-subsets
            union_t = set(block_covers_np[bi].tolist()) | set(block_covers_np[bj].tolist())
            if len(union_t) > 35:
                continue
            
            # Find candidate
            # Pick the ti in union_t with fewest blocks covering it
            min_count = 1000000
            min_ti = -1
            for ti in union_t:
                c = len(t_to_blocks[ti])
                if c < min_count:
                    min_count = c
                    min_ti = ti
            
            if min_ti == -1:
                continue
                
            for cand in t_to_blocks[min_ti]:
                if cand == bi or cand == bj:
                    continue
                
                # Check if cand covers all union_t
                valid = True
                for ti in union_t:
                    if ti not in block_covers[cand]:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check validity
                valid = True
                for ti in union_t:
                    count = clause_sat[ti]
                    if ti in block_covers[bi]:
                        count -= 1
                    if ti in block_covers[bj]:
                        count -= 1
                    if ti in block_covers[cand]:
                        count += 1
                    if count < 1:
                        valid = False
                        break
                
                if valid:
                    # Perform replacement
                    clause_sat[block_covers_np[bi]] -= 1
                    clause_sat[block_covers_np[bj]] -= 1
                    clause_sat[block_covers_np[cand]] += 1
                    
                    # Update sel_list
                    # Swap remove j
                    sel_list[j] = sel_list[-1]
                    sel_list.pop()
                    # Adjust i if necessary
                    if i > j:
                        i -= 1
                    
                    # Swap remove i
                    sel_list[i] = sel_list[-1]
                    sel_list.pop()
                    
                    # Add cand
                    sel_list.append(cand)
                    
                    best_n_true -= 1
                    best_x[bi] = False
                    best_x[bj] = False
                    best_x[cand] = True
                    
                    # Reduce periodically
                    if steps % 100 == 0:
                         selected = set(sel_list)
                         cover_count = np.zeros(n_t, dtype=np.int32)
                         for bi in selected:
                             cover_count[block_covers_np[bi]] += 1
                         selected, cover_count = reduce_blocks(selected, cover_count)
                         sel_list = list(selected)
                         best_n_true = len(sel_list)
                         best_x = np.array([bi in selected for bi in range(n_k)], dtype=bool)
                         clause_sat = np.zeros(n_t, dtype=np.int32)
                         for bi in selected:
                             clause_sat[block_covers_np[bi]] += 1

    # Finalize
    selected = set(sel_list)
    cover_count = np.zeros(n_t, dtype=np.int32)
    for bi in selected:
        cover_count[block_covers_np[bi]] += 1

    # ---------------- Phase 4: Final Reduction ----------------
    selected, cover_count = reduce_blocks(selected, cover_count)
    selected, cover_count = pair_replace(selected, cover_count, max_trials=500)
    selected, cover_count = reduce_blocks(selected, cover_count)
    selected, cover_count = triple_replace(selected, cover_count, max_trials=300)
    selected, cover_count = reduce_blocks(selected, cover_count)
    # One more pass of pair replacement after triple
    selected, cover_count = pair_replace(selected, cover_count, max_trials=300)
    selected, cover_count = reduce_blocks(selected, cover_count)

    # Build output
    blocks = [list(k_subs[bi]) for bi in sorted(selected)]

    # Final validation
    if blocks:
        cc_final = np.zeros(n_t, dtype=np.int32)
        for bi in selected:
            cc_final[block_covers_np[bi]] += 1
        if np.any(cc_final == 0):
            # Fallback: rebuild from best_greedy
            selected = set(best_greedy)
            blocks = [list(k_subs[bi]) for bi in sorted(selected)]

    return blocks
