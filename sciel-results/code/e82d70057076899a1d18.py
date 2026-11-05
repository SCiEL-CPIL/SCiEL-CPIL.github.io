import time
import random
import sys
from collections import defaultdict

N = 256
# Safety margin: Evaluator kills at 90s. We use 85s for computation.
DEADLINE = 85.0

# Precomputation
# Bitmask representation: bit i corresponds to element i+1.
# Element 1 is bit 0, Element 256 is bit 255.

# 1. Precompute 4-AP conflicts
# For each element x, store a list of bitmasks representing the other 3 elements
# of every 4-AP containing x.
conflict_masks_4 = [[] for _ in range(N + 1)]
# Also store the count of conflicts for ordering
conflict_degree = [0] * (N + 1)

for d in range(1, N // 3 + 1):
    for start in range(1, N - 3 * d + 1):
        a, b, c, e = start, start + d, start + 2 * d, start + 3 * d
        # For each element, the "other 3" mask
        m_a = (1 << (b - 1)) | (1 << (c - 1)) | (1 << (e - 1))
        m_b = (1 << (a - 1)) | (1 << (c - 1)) | (1 << (e - 1))
        m_c = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (e - 1))
        m_e = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
        
        conflict_masks_4[a].append(m_a)
        conflict_masks_4[b].append(m_b)
        conflict_masks_4[c].append(m_c)
        conflict_masks_4[e].append(m_e)
        
        conflict_degree[a] += 1
        conflict_degree[b] += 1
        conflict_degree[c] += 1
        conflict_degree[e] += 1

# 2. Precompute 3-AP partners
# For each element x, store a list of bitmasks representing the other 2 elements
# of every 3-AP containing x.
partner_masks_3 = [[] for _ in range(N + 1)]
# Also store the max possible contribution (degree) for bounding
partner_degree = [0] * (N + 1)

for d in range(1, N // 2 + 1):
    for start in range(1, N - 2 * d + 1):
        a, b, c = start, start + d, start + 2 * d
        m_a = (1 << (b - 1)) | (1 << (c - 1))
        m_b = (1 << (a - 1)) | (1 << (c - 1))
        m_c = (1 << (a - 1)) | (1 << (b - 1))
        
        partner_masks_3[a].append(m_a)
        partner_masks_3[b].append(m_b)
        partner_masks_3[c].append(m_c)
        
        partner_degree[a] += 1
        partner_degree[b] += 1
        partner_degree[c] += 1

def is_4ap_free_addition(mask, x):
    """Check if adding x to mask creates a 4-AP."""
    bit_x = 1 << (x - 1)
    if mask & bit_x:
        return False
    for cm in conflict_masks_4[x]:
        if (mask & cm) == cm:
            return False
    return True

def delta_3aps_addition(mask, x):
    """Calculate number of new 3-APs formed by adding x to mask."""
    delta = 0
    for pm in partner_masks_3[x]:
        if (mask & pm) == pm:
            delta += 1
    return delta

def count_3aps_full(mask):
    """Count total 3-APs in the mask for verification/final scoring."""
    total = 0
    m = mask
    while m:
        lsb = m & -m
        x = lsb.bit_length()
        base_mask = mask ^ lsb
        d = 0
        for pm in partner_masks_3[x]:
            if (base_mask & pm) == pm:
                d += 1
        total += d
        m ^= lsb
    return total // 3

def mask_to_list(mask):
    elements = []
    for i in range(N):
        if mask & (1 << i):
            elements.append(i + 1)
    return elements

def solve_cp(seed_val):
    """
    Constraint Programming solver with Branch and Bound.
    """
    random.seed(seed_val)
    
    # State
    mask = 0
    score = 0
    
    # Global best found so far
    best_score = -1
    best_mask = 0
    
    # Unassigned elements
    # We will pick them dynamically.
    # To make dynamic ordering efficient, we maintain a list of all elements 
    # and filter by 'assigned' status.
    # Assigned elements are those in mask (1) or in 'excluded' mask (0).
    excluded_mask = 0
    
    # We need a way to iterate unassigned elements.
    # Since N=256, we can just loop 1..N and check bits.
    
    # Depth limit to prevent infinite recursion? 
    # The tree has depth N. But we prune heavily.
    # We use iterative deepening or just depth-first with time check.
    
    def get_unassigned():
        unassigned = []
        for x in range(1, N + 1):
            bit = 1 << (x - 1)
            if not (mask & bit) and not (excluded_mask & bit):
                unassigned.append(x)
        return unassigned

    # To speed up dynamic ordering, we precompute a heuristic score for each element
    # that changes as mask/excluded_mask change.
    # Heuristic: 
    # 1. Conflict pressure: How many 4-APs are "threatened" or "blocked"?
    #    If we add x, we must ensure no 4-AP is completed.
    #    If we exclude x, we lose potential 3-APs.
    # 2. Potential gain: delta_3aps_addition(mask, x)
    
    # Simple dynamic ordering:
    # Sort unassigned by (potential_gain, -conflict_degree)
    # But calculating potential_gain for all unassigned at every node is O(N^2).
    # N=256, so 256 * 256 = 65k ops per node. 
    # If we have 1000 nodes, 65M ops. In Python, this might be ~10-30s.
    # We need to be careful.
    
    # Optimization: 
    # We don't need to sort ALL unassigned. We can just pick the "best" candidate.
    # Or we can use a simpler static ordering with backtracking?
    # The prompt asks for "dynamically determined by the element with the highest degree of 4-AP conflicts".
    
    # Let's implement a recursive solver.
    
    sys.setrecursionlimit(10000)
    
    def dfs(depth):
        nonlocal best_score, best_mask, mask, excluded_mask, score
        
        if time.time() > start_time + DEADLINE:
            return False # Timeout
        
        # Check if all assigned
        # Count assigned
        assigned_count = bin(mask).count('1') + bin(excluded_mask).count('1')
        if assigned_count == N:
            if score > best_score:
                best_score = score
                best_mask = mask
            return True
        
        # Pruning Bound:
        # Upper bound on remaining score.
        # Current score + sum of max possible gains for remaining elements?
        # This is an overestimate, but safe.
        # Max gain for x is partner_degree[x] (if all partners were present).
        # But partners might be excluded or unassigned.
        # A tighter bound: 
        # For each unassigned x, max_gain = delta_3aps_addition(mask, x) + 
        #   (number of unassigned partners of x) / 2 ? No, that's complex.
        # Let's use a simple bound: 
        # remaining_elements = N - assigned_count
        # max_additional_3aps = sum of partner_degree[x] for unassigned x // 3 ?
        # This is very loose.
        
        # Let's try a tighter bound:
        # For each unassigned x, the max new 3-APs it can form is at most partner_degree[x].
        # But many partners are already excluded or unassigned.
        # Actually, the number of new 3-APs formed by adding a set S is bounded by 
        # the number of pairs in S that are partners?
        
        # Let's stick to a simple bound for now to avoid overhead:
        # bound = score + sum(partner_degree[x] for x in unassigned) // 3
        # If bound <= best_score, prune.
        
        unassigned = []
        for x in range(1, N + 1):
            bit = 1 << (x - 1)
            if not (mask & bit) and not (excluded_mask & bit):
                unassigned.append(x)
                
        if not unassigned:
             if score > best_score:
                 best_score = score
                 best_mask = mask
             return True
             
        # Calculate bound
        # To make it fast, we can approximate.
        # sum_partner_deg = sum(partner_degree[x] for x in unassigned)
        # max_new_3aps = sum_partner_deg // 3
        # if score + max_new_3aps <= best_score: return True (pruned)
        
        # This calculation is O(N) per node.
        sum_partner_deg = 0
        for x in unassigned:
            sum_partner_deg += partner_degree[x]
        bound = score + sum_partner_deg // 3
        if bound <= best_score:
            return True
            
        # Dynamic Ordering:
        # Choose x with highest "priority".
        # Priority = delta_3aps_addition(mask, x) * 10 - conflict_degree[x]
        # We want high gain, low conflict.
        # Or, as per prompt: "highest degree of 4-AP conflicts".
        # Let's combine: 
        # We want to resolve conflicts early.
        # Let's pick x that has the most "active" conflicts with current mask.
        
        best_x = -1
        best_priority = -1
        
        for x in unassigned:
            # Calculate priority
            # Gain:
            gain = delta_3aps_addition(mask, x)
            # Conflict pressure:
            # How many 4-APs containing x are "almost complete" in mask?
            # If mask has 2 of the other 3, adding x completes a 4-AP -> Invalid.
            # If mask has 1 of the other 3, adding x is valid but blocks that AP.
            
            # Let's just use conflict_degree[x] as a proxy for constraint tightness.
            # Elements with high conflict_degree are more constrained.
            # But we also want high gain.
            
            # Heuristic: 
            # If gain is 0, we prefer high conflict (to prune early).
            # If gain is high, we prefer high gain.
            
            # Simple score: gain * 100 + conflict_degree[x]
            # We want to maximize this? 
            # If we pick high conflict first, we might prune branches where x is excluded.
            
            # Let's try: Pick x with max (gain, conflict_degree[x]) lexicographically?
            # Or just max gain?
            # The prompt says: "dynamically determined by the element with the highest degree of 4-AP conflicts".
            # So let's prioritize conflict_degree.
            
            # But if conflict_degree is static, the order is static.
            # "Dynamic" implies it changes with state.
            # Let's define dynamic conflict:
            # Number of 4-APs containing x where at least 2 other elements are in mask.
            # These are "critical" conflicts.
            
            critical_conflicts = 0
            for cm in conflict_masks_4[x]:
                # cm is mask of other 3.
                # Count how many are in mask.
                # If count >= 2, adding x is invalid.
                # If count == 2, it's a critical conflict (blocks x).
                # If count == 1, it's a soft conflict.
                
                # Efficient count:
                # cm has 3 bits.
                # (mask & cm).bit_count()
                c = (mask & cm).bit_count()
                if c >= 2:
                    critical_conflicts += 1
                    
            # If critical_conflicts > 0, x CANNOT be added.
            # So we should probably exclude x immediately? 
            # Or just skip "Add" branch?
            
            # For ordering, we want to pick elements that are "hard" to place.
            # High critical_conflicts means it's likely to be excluded.
            # High gain means it's valuable.
            
            # Let's use: priority = gain * 10 + critical_conflicts
            # We pick the max priority.
            
            priority = gain * 10 + critical_conflicts
            if priority > best_priority:
                best_priority = priority
                best_x = x
                
        if best_x == -1:
            return True
            
        # Branch 1: Add best_x
        # Check if valid
        if is_4ap_free_addition(mask, best_x):
            bit_x = 1 << (best_x - 1)
            old_mask = mask
            old_score = score
            old_excluded = excluded_mask
            
            mask |= bit_x
            score += delta_3aps_addition(old_mask, best_x)
            
            if dfs(depth + 1):
                # Restore
                mask = old_mask
                score = old_score
                excluded_mask = old_excluded
                return True
            else:
                # Restore
                mask = old_mask
                score = old_score
                excluded_mask = old_excluded
                
        # Branch 2: Exclude best_x
        bit_x = 1 << (best_x - 1)
        old_excluded = excluded_mask
        excluded_mask |= bit_x
        
        if dfs(depth + 1):
            # Restore
            excluded_mask = old_excluded
            return True
        else:
            # Restore
            excluded_mask = old_excluded
            
        return False

    start_time = time.time()
    
    # Initial greedy to get a good lower bound for pruning
    # This is crucial for Branch and Bound.
    # Let's run a fast greedy first.
    
    g_mask = 0
    g_score = 0
    # Simple greedy: add if valid and gain > 0
    # To make it fast, we just loop.
    for _ in range(N):
        best_g_x = -1
        best_g_gain = 0
        for x in range(1, N + 1):
            if g_mask & (1 << (x - 1)): continue
            if not is_4ap_free_addition(g_mask, x): continue
            g = delta_3aps_addition(g_mask, x)
            if g > best_g_gain:
                best_g_gain = g
                best_g_x = x
        if best_g_x == -1:
            break
        g_mask |= (1 << (best_g_x - 1))
        g_score += best_g_gain
        
    best_score = g_score
    best_mask = g_mask
    
    # Now run CP
    # We need to pass start_time to dfs.
    # We use a global or closure variable.
    
    # To allow multiple restarts with different seeds/orderings if time permits?
    # CP is deterministic if ordering is deterministic.
    # We can randomize the tie-breaking in priority.
    
    # Let's run one instance of CP with randomized tie-breaking.
    
    def dfs_randomized(depth):
        nonlocal best_score, best_mask, mask, excluded_mask, score
        
        if time.time() > start_time + DEADLINE:
            return False
        
        assigned_count = bin(mask).count('1') + bin(excluded_mask).count('1')
        if assigned_count == N:
            if score > best_score:
                best_score = score
                best_mask = mask
            return True
        
        unassigned = []
        for x in range(1, N + 1):
            bit = 1 << (x - 1)
            if not (mask & bit) and not (excluded_mask & bit):
                unassigned.append(x)
                
        if not unassigned:
             if score > best_score:
                 best_score = score
                 best_mask = mask
             return True
             
        # Bound
        sum_partner_deg = 0
        for x in unassigned:
            sum_partner_deg += partner_degree[x]
        bound = score + sum_partner_deg // 3
        if bound <= best_score:
            return True
            
        # Dynamic Ordering with Randomization
        # Calculate priority for all unassigned
        # This is O(N * avg_conflicts) ~ 256 * 10 = 2560 ops.
        # Plus O(N) for gain.
        # Total ~ 5000 ops per node.
        
        candidates = []
        for x in unassigned:
            gain = delta_3aps_addition(mask, x)
            
            critical_conflicts = 0
            for cm in conflict_masks_4[x]:
                c = (mask & cm).bit_count()
                if c >= 2:
                    critical_conflicts += 1
                    
            # Priority: High gain, High critical conflicts (to prune)
            # Add noise for randomization
            noise = random.random() * 0.1
            priority = gain * 10.0 + critical_conflicts * 5.0 + noise
            
            candidates.append((priority, x))
            
        # Sort by priority descending
        candidates.sort(key=lambda item: item[0], reverse=True)
        
        # Try top few candidates? 
        # Standard DFS tries one. 
        # Let's try the best one.
        # If it fails, backtrack.
        
        # To improve, we could try top 2 or 3? 
        # That increases branching factor.
        # Let's stick to top 1 for now to keep depth manageable.
        
        best_x = candidates[0][1]
        
        # Branch 1: Add best_x
        if is_4ap_free_addition(mask, best_x):
            bit_x = 1 << (best_x - 1)
            old_mask = mask
            old_score = score
            old_excluded = excluded_mask
            
            mask |= bit_x
            score += delta_3aps_addition(old_mask, best_x)
            
            if dfs_randomized(depth + 1):
                mask = old_mask
                score = old_score
                excluded_mask = old_excluded
                return True
            else:
                mask = old_mask
                score = old_score
                excluded_mask = old_excluded
                
        # Branch 2: Exclude best_x
        bit_x = 1 << (best_x - 1)
        old_excluded = excluded_mask
        excluded_mask |= bit_x
        
        if dfs_randomized(depth + 1):
            excluded_mask = old_excluded
            return True
        else:
            excluded_mask = old_excluded
            
        return False

    # Run CP
    # We might want to run multiple seeds if time allows?
    # But CP is deep. One run might take the whole time.
    # Let's allocate 80% time to CP, 20% to local search on result?
    
    cp_deadline = start_time + DEADLINE * 0.8
    # We need to pass cp_deadline to dfs_randomized.
    # Let's modify dfs_randomized to check a global deadline.
    
    global_deadline = start_time + DEADLINE
    
    def dfs_cp(depth):
        nonlocal best_score, best_mask, mask, excluded_mask, score
        
        if time.time() > global_deadline - 5: # 5s margin
            return False
        
        assigned_count = bin(mask).count('1') + bin(excluded_mask).count('1')
        if assigned_count == N:
            if score > best_score:
                best_score = score
                best_mask = mask
            return True
        
        unassigned = []
        for x in range(1, N + 1):
            bit = 1 << (x - 1)
            if not (mask & bit) and not (excluded_mask & bit):
                unassigned.append(x)
                
        if not unassigned:
             if score > best_score:
                 best_score = score
                 best_mask = mask
             return True
             
        # Bound
        sum_partner_deg = 0
        for x in unassigned:
            sum_partner_deg += partner_degree[x]
        bound = score + sum_partner_deg // 3
        if bound <= best_score:
            return True
            
        # Dynamic Ordering
        candidates = []
        for x in unassigned:
            gain = delta_3aps_addition(mask, x)
            
            critical_conflicts = 0
            for cm in conflict_masks_4[x]:
                c = (mask & cm).bit_count()
                if c >= 2:
                    critical_conflicts += 1
                    
            noise = random.random() * 0.5
            priority = gain * 10.0 + critical_conflicts * 5.0 + noise
            
            candidates.append((priority, x))
            
        candidates.sort(key=lambda item: item[0], reverse=True)
        
        best_x = candidates[0][1]
        
        # Branch 1: Add
        if is_4ap_free_addition(mask, best_x):
            bit_x = 1 << (best_x - 1)
            old_mask = mask
            old_score = score
            old_excluded = excluded_mask
            
            mask |= bit_x
            score += delta_3aps_addition(old_mask, best_x)
            
            if dfs_cp(depth + 1):
                mask = old_mask
                score = old_score
                excluded_mask = old_excluded
                return True
            else:
                mask = old_mask
                score = old_score
                excluded_mask = old_excluded
                
        # Branch 2: Exclude
        bit_x = 1 << (best_x - 1)
        old_excluded = excluded_mask
        excluded_mask |= bit_x
        
        if dfs_cp(depth + 1):
            excluded_mask = old_excluded
            return True
        else:
            excluded_mask = old_excluded
            
        return False

    # Start CP
    dfs_cp(0)
    
    # If time remains, do a local search on the best found
    if time.time() < global_deadline - 2:
        # Simple local search
        ls_mask = best_mask
        ls_score = best_score
        
        # Try adds and swaps
        improved = True
        while improved and time.time() < global_deadline - 1:
            improved = False
            
            # Adds
            best_add_x = -1
            best_add_delta = 0
            for x in range(1, N + 1):
                if ls_mask & (1 << (x - 1)): continue
                if not is_4ap_free_addition(ls_mask, x): continue
                d = delta_3aps_addition(ls_mask, x)
                if d > best_add_delta:
                    best_add_delta = d
                    best_add_x = x
                    
            if best_add_x != -1 and best_add_delta > 0:
                ls_mask |= (1 << (best_add_x - 1))
                ls_score += best_add_delta
                improved = True
                continue
                
            # Swaps (limited)
            # Remove x, Add y
            # Try random swaps
            for _ in range(50):
                x = random.randint(1, N)
                if not (ls_mask & (1 << (x - 1))): continue
                y = random.randint(1, N)
                if y == x or (ls_mask & (1 << (y - 1))): continue
                
                temp_mask = ls_mask & ~(1 << (x - 1))
                if not is_4ap_free_addition(temp_mask, y): continue
                
                d_add = delta_3aps_addition(temp_mask, y)
                d_rem = delta_3aps_addition(ls_mask & ~(1 << (x - 1)), x) # Cost of removing x
                
                if d_add > d_rem:
                    ls_mask = temp_mask | (1 << (y - 1))
                    ls_score += (d_add - d_rem)
                    improved = True
                    break
                    
        if ls_score > best_score:
            best_score = ls_score
            best_mask = ls_mask

    return best_mask

def construct():
    start_time = time.time()
    
    # Run CP with a few different seeds to diversify the random tie-breaking
    # But CP is expensive. 
    # Let's run 1 CP instance. 
    # If it finishes fast, run more?
    # Given the depth, it likely won't finish fast.
    
    best_mask = solve_cp(seed_val=42)
    
    # If time remains, try another seed?
    if time.time() < start_time + DEADLINE - 10:
        # Run another instance with different seed
        m2 = solve_cp(seed_val=123)
        s1 = count_3aps_full(best_mask)
        s2 = count_3aps_full(m2)
        if s2 > s1:
            best_mask = m2
            
    if time.time() < start_time + DEADLINE - 10:
        m3 = solve_cp(seed_val=999)
        s1 = count_3aps_full(best_mask)
        s3 = count_3aps_full(m3)
        if s3 > s1:
            best_mask = m3

    if best_mask == 0:
        return list(range(1, 11))
        
    return mask_to_list(best_mask)

# EVOLVE-BLOCK-END
