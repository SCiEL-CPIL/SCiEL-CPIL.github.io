"""Covering design C(19,8,5): Max-SAT Formulation with WalkSAT Local Search.
Strategy:
1. Model the problem as a Max-SAT instance:
   - Variables: x_b for each k-block b (1 if selected, 0 otherwise).
   - Soft Clauses: For each k-block b, a clause (-x_b) to penalize selection.
   - Hard Clauses: For each t-subset s, a clause (x_{b1} OR ... OR x_{bm}) where b_i cover s.
   - Objective: Maximize satisfied clauses. Since hard clauses must be satisfied for validity,
     we prioritize them with high weight. However, for a valid covering, all hard clauses are satisfied.
     The number of selected blocks is the number of satisfied soft clauses of type x_b=1? 
     No, standard Max-SAT minimizes unsatisfied. 
     Let's flip it: Minimize sum(x_b) subject to coverage constraints.
     This is equivalent to Max-SAT with:
     - Hard constraints: Coverage clauses. If unsatisfied, huge penalty.
     - Soft constraints: (-x_b) for each block. We want to satisfy these (i.e., x_b=0).
     So, Score = (Number of satisfied soft clauses) - (Penalty * Number of unsatisfied hard clauses).
     Maximizing this score is equivalent to minimizing blocks while ensuring validity.
     
2. Initialize with a valid solution from Greedy Set Cover (to start in the feasible region).
3. Apply WalkSAT:
   - State: Assignment vector x (0/1 for each block).
   - Score: W_soft * (NumSoftSatisfied) - W_hard * (NumHardUnsatisfied).
     To ensure we stay valid or return to valid, W_hard must be large.
     Actually, if we start valid, we can use a "repair" strategy or ensure moves don't break validity if possible, 
     but WalkSAT allows breaking constraints temporarily.
     Better: Use a Lagrangian relaxation or simple penalty.
     Let W_hard = 1000. W_soft = 1.
     Score = (B - sum(x)) * 1 - (NumUnsatisfiedHard) * 1000.
     Wait, maximizing (B - sum(x)) is minimizing sum(x).
     So Score = -sum(x) - 1000 * NumUnsatisfiedHard.
     We want to maximize Score.
     
   - Move: Flip a variable x_b.
   - Variable Selection:
     - If unsatisfied hard clauses exist, pick a random unsatisfied clause, then a random variable in it.
     - With probability p_walk, flip that variable.
     - Else, pick the variable in that clause that maximizes the score delta (greedy).
     - If no unsatisfied hard clauses, pick a random variable to flip (to explore reducing sum(x)).
       Or pick a variable that is 1 and see if flipping to 0 increases score (i.e., doesn't break hard clauses).
       
   - Since the space is huge (75k vars), we need to be careful.
   - Start from Greedy. It is valid.
   - We want to reduce the number of 1s.
   - A "flip 1->0" is good if it doesn't create any unsatisfied hard clauses.
   - A "flip 0->1" is bad (increases cost) unless it allows other flips? 
     In WalkSAT, we usually just flip.
     
   Let's refine the WalkSAT for this specific minimization:
   - Current state: Assignment x.
   - Calculate Unsatisfied Hard Clauses (UHC).
   - If UHC is empty:
     - Try to flip a 1 to 0.
     - Pick a random block b where x_b=1.
     - Check if flipping it makes any hard clause unsatisfied.
     - If not, accept (score improves by 1).
     - If yes, reject (or try another).
     - To escape local minima where no single flip 1->0 works, we might need to flip 0->1? 
       No, flipping 0->1 increases cost. It only helps if it enables a sequence of 1->0 flips that net a reduction.
       Standard WalkSAT flips variables in unsatisfied clauses. Here, if valid, there are no unsatisfied clauses.
       So standard WalkSAT would stall.
       
   Modified WalkSAT for Valid State:
   - If valid:
     - Identify "critical" blocks: blocks that are the sole cover for some t-subset.
     - If we flip a critical block 1->0, we break validity.
     - So we can only freely flip non-critical blocks 1->0.
     - If no non-critical blocks exist, we are in a local minimum (set cover core).
     - To escape, we must perform a "swap": flip a critical block 1->0 AND flip some other block 0->1 to cover the lost subsets.
     - This is effectively the "Swap" move from the previous program, but framed as SAT.
     
   Let's implement a hybrid:
   1. Greedy Init.
   2. Prune non-critical blocks (1->0 flips that are safe).
   3. Local Search Loop:
      - Try to find a block b (x_b=1) such that removing it (x_b=0) breaks validity (creates UHC).
      - Let U be the set of t-subsets that become uncovered.
      - Find a block c (x_c=0) that covers all of U.
      - If such c exists, swap b->0, c->1. Cost is 0.
      - This doesn't reduce cost. We need to reduce cost.
      - To reduce cost, we need to remove a block and NOT add one.
      - This is only possible if the block is non-critical.
      - So, if we are stuck with only critical blocks, we cannot reduce the count further with single moves.
      - We need a 2-move or 3-move sequence that results in net reduction.
      - Example: Remove b1 (breaks U1), Remove b2 (breaks U2). Add c1 (covers U1), Add c2 (covers U2). Net change: -2 + 2 = 0. Still no reduction.
      - To get reduction: Remove b1, b2. Add c1. Where c1 covers U1 and U2. Net change: -2 + 1 = -1. Improvement!
      
   So the search should look for:
   - A set of blocks B_rem to remove (size r).
   - A set of blocks B_add to add (size a < r).
   - Such that U(B_rem) is covered by B_add.
   
   This is a complex combinatorial search.
   Given the time limit, let's stick to the "Swap" (cost 0) to rearrange the set, which might expose non-critical blocks for removal.
   The previous program did this. Why did it score 1.07?
   Maybe it didn't explore enough swaps.
   
   Let's implement the Max-SAT / WalkSAT idea more directly:
   Use a penalty function.
   Score = -sum(x) - Penalty * NumUnsatisfied.
   Start Valid.
   Move: Pick a variable to flip.
   - If valid: Pick a random 1. Flip to 0.
     - If becomes invalid, calculate penalty.
     - If Score improves (i.e., Penalty is small enough relative to 1?), accept.
     - But we want to stay valid.
     
   Let's use a "Repair" approach.
   1. Start Valid.
   2. Randomly flip a 1 to 0.
   3. If invalid, try to repair by flipping 0s to 1s (greedy repair) to minimize added blocks.
   4. If repaired solution has fewer blocks than before, accept.
   5. If repaired solution has same number of blocks, accept with some probability (to explore).
   6. If repaired solution has more blocks, reject.
   
   This is essentially a "Drop and Repair" local search.
   
   Implementation:
   - State: Set of selected blocks.
   - Move: 
     1. Pick a random block b from current set.
     2. Remove b.
     3. Let U be uncovered t-subsets.
     4. If U is empty: Accept (cost -1).
     5. Else: Find a minimum set of blocks (from unselected) to cover U.
        - Greedy: Pick block covering most of U, add it, repeat.
        - Let A be the set of added blocks.
        - New Cost = Old Cost - 1 + |A|.
        - If New Cost < Old Cost: Accept.
        - If New Cost == Old Cost: Accept with prob 0.5 (to allow rearrangement).
        - If New Cost > Old Cost: Reject.
        
   This is different from the previous SA which tried swaps (1 out, 1 in) or drops.
   This "Drop and Repair" allows dropping 1 and adding 2 (if it leads to a better state later? No, if it adds 2, cost increases by 1, so we reject immediately unless we expect future gains. But local search usually rejects increases).
   However, if we accept cost-increasing moves (Simulated Annealing style), we can escape.
   
   Let's implement "Drop and Repair" with Annealing.
   - T starts high.
   - Move: Drop random block. Repair greedily.
   - Delta = NewCost - OldCost.
   - Accept if Delta < 0 or rand() < exp(-Delta/T).
   
   This is a structural change from the previous "Swap/Drop" SA. It uses a repair heuristic which is more powerful for finding global structures.
"""

import time
import random
import itertools
import math
import numpy as np

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 115.0

    # Precompute
    all_t = list(itertools.combinations(range(v), t))
    N = len(all_t)
    idx = {s: i for i, s in enumerate(all_t)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    B = len(all_blocks)
    block_cover = []
    for b in all_blocks:
        covered = [idx[s] for s in itertools.combinations(b, t)]
        block_cover.append(covered)
        
    t_to_blocks = [[] for _ in range(N)]
    for bi, covered in enumerate(block_cover):
        for ti in covered:
            t_to_blocks[ti].append(bi)

    def greedy_cover():
        uncovered = set(range(N))
        blocks = []
        while uncovered:
            seed_idx = next(iter(uncovered))
            candidates = t_to_blocks[seed_idx]
            best_b = -1
            max_cov = -1
            for bi in candidates:
                cov = 0
                for ti in block_cover[bi]:
                    if ti in uncovered:
                        cov += 1
                if cov > max_cov:
                    max_cov = cov
                    best_b = bi
            if best_b == -1:
                break
            blocks.append(best_b)
            for ti in block_cover[best_b]:
                uncovered.discard(ti)
        return blocks

    def prune_redundant(blocks):
        cnt = np.zeros(N, dtype=np.int32)
        for bi in blocks:
            for ti in block_cover[bi]:
                cnt[ti] += 1
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for bi in blocks:
                essential = False
                for ti in block_cover[bi]:
                    if cnt[ti] == 1:
                        essential = True
                        break
                if not essential:
                    changed = True
                    for ti in block_cover[bi]:
                        cnt[ti] -= 1
                else:
                    new_blocks.append(bi)
            blocks = new_blocks
            if time.time() > deadline - 2.0:
                break
        return blocks

    init_blocks = greedy_cover()
    init_blocks = prune_redundant(init_blocks)
    
    best_blocks = list(init_blocks)
    best_score = len(best_blocks)
    
    rng = random.Random(42)
    
    # Drop and Repair Local Search
    T = 10.0
    cooling = 0.999
    
    current_blocks = list(best_blocks)
    current_cost = len(current_blocks)
    
    # Precompute counts for current state to speed up repair?
    # Repair needs to find blocks covering U.
    # We can use t_to_blocks.
    
    steps = 0
    max_steps = 50000
    
    while time.time() < deadline - 2.0 and steps < max_steps:
        steps += 1
        if len(current_blocks) == 0:
            break
            
        # 1. Drop a random block
        idx_drop = rng.randrange(len(current_blocks))
        block_drop = current_blocks[idx_drop]
        
        # Calculate uncovered subsets U
        # We need to know which subsets become uncovered.
        # We can maintain a count array, but for simplicity in the move,
        # let's compute U by checking if any other block covers it.
        # This is O(B * K) per move which is too slow.
        
        # Optimization: Maintain cnt array.
        # Let's initialize cnt for current_blocks.
        
    # Re-structure to maintain cnt globally.
    
    cnt = np.zeros(N, dtype=np.int32)
    for bi in current_blocks:
        for ti in block_cover[bi]:
            cnt[ti] += 1
            
    # Map block index to position in current_blocks list for quick removal?
    # Or just use list and remove by value. O(N).
    
    while time.time() < deadline - 2.0 and steps < max_steps:
        steps += 1
        
        if len(current_blocks) == 0:
            break
            
        idx_drop = rng.randrange(len(current_blocks))
        block_drop = current_blocks[idx_drop]
        
        # Tentatively remove
        U = []
        for ti in block_cover[block_drop]:
            cnt[ti] -= 1
            if cnt[ti] == 0:
                U.append(ti)
                
        if not U:
            # Valid drop
            current_blocks.pop(idx_drop)
            current_cost -= 1
            if current_cost < best_score:
                best_score = current_cost
                best_blocks = list(current_blocks)
            T *= cooling
            continue
            
        # Need to repair
        # Greedy repair: cover U with minimum number of new blocks
        uncovered = set(U)
        repair_blocks = []
        
        while uncovered:
            # Find block covering most of uncovered
            # Pick a random item from uncovered to anchor
            seed_item = next(iter(uncovered))
            candidates = t_to_blocks[seed_item]
            
            best_b = -1
            max_cov = -1
            for bi in candidates:
                if bi in current_blocks:
                    continue
                cov = 0
                for ti in block_cover[bi]:
                    if ti in uncovered:
                        cov += 1
                if cov > max_cov:
                    max_cov = cov
                    best_b = bi
                    
            if best_b == -1:
                # Cannot repair? Should not happen
                break
                
            repair_blocks.append(best_b)
            for ti in block_cover[best_b]:
                uncovered.discard(ti)
                if ti in U:
                    cnt[ti] += 1 # Restore count for items in U that are now covered
            
            # Note: We only restored counts for items in U.
            # What about items not in U that are in best_b?
            # They were covered by other blocks, so their count was >= 1.
            # Adding best_b increases their count.
            # We need to update cnt for ALL items in best_b.
            
            # Let's do it properly.
            # Undo the decrement for block_drop first? No, we already decremented.
            # We need to add counts for repair_blocks.
            
            # This incremental update is tricky.
            # Let's just recompute cnt for the affected items after repair is decided.
            # Or, simpler:
            # 1. Remove block_drop (update cnt).
            # 2. Find repair set R.
            # 3. Add R to cnt.
            # 4. Check if valid (all cnt >= 1).
            
            # Let's restart the repair logic to be clean.
            
            break # Break inner while, we'll redo logic outside
        
        # Cleaner Repair Logic
        # 1. We have U (items with cnt==0 after removing block_drop).
        # 2. We need to find min set of blocks from (All - Current - {block_drop}) to cover U.
        # 3. Greedy:
        uncovered = set(U)
        repair_blocks = []
        
        while uncovered:
            seed_item = next(iter(uncovered))
            candidates = t_to_blocks[seed_item]
            best_b = -1
            max_cov = -1
            for bi in candidates:
                if bi in current_blocks:
                    continue
                cov = 0
                for ti in block_cover[bi]:
                    if ti in uncovered:
                        cov += 1
                if cov > max_cov:
                    max_cov = cov
                    best_b = bi
                    
            if best_b == -1:
                # Fallback: pick any block covering seed_item
                best_b = candidates[0]
                
            repair_blocks.append(best_b)
            for ti in block_cover[best_b]:
                uncovered.discard(ti)
                
        # Apply repair
        for bi in repair_blocks:
            current_blocks.append(bi)
            for ti in block_cover[bi]:
                cnt[ti] += 1
                
        new_cost = len(current_blocks)
        delta = new_cost - current_cost
        
        # Acceptance
        if delta < 0 or (delta == 0 and rng.random() < 0.5) or (delta > 0 and rng.random() < math.exp(-delta / T)):
            current_cost = new_cost
            if new_cost < best_score:
                best_score = new_cost
                best_blocks = list(current_blocks)
        else:
            # Reject: Undo changes
            # Remove repair_blocks
            for bi in repair_blocks:
                current_blocks.remove(bi)
                for ti in block_cover[bi]:
                    cnt[ti] -= 1
            # Re-add block_drop
            current_blocks.insert(idx_drop, block_drop)
            for ti in block_cover[block_drop]:
                cnt[ti] += 1
                
        T *= cooling
        if T < 0.1:
            T = 0.1

    # Final Prune
    best_blocks = prune_redundant(best_blocks)
    
    # Verify
    cnt_final = np.zeros(N, dtype=np.int32)
    for bi in best_blocks:
        for ti in block_cover[bi]:
            cnt_final[ti] += 1
            
    if np.min(cnt_final) < 1:
        # Fallback
        return [list(all_blocks[bi]) for bi in greedy_cover()]
        
    return [list(all_blocks[bi]) for bi in best_blocks]
