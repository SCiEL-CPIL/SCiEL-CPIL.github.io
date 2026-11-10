import time
import random
import math

N = 256
DEADLINE = 55.0

def _precompute():
    """
    Precompute bitmasks for 3-AP and 4-AP partners.
    For each element i, ap3_mask[i] has bits set for j such that (i, j, k) or (j, i, k) etc. form a 3-AP?
    Actually, it's better to store for each i, a mask of elements that form a 3-AP with i and one other element.
    Let's define ap3_partners_mask[i] as a bitmask where bit j is set if there exists k such that {i, j, k} is a 3-AP.
    This is not quite right for counting.
    
    Better approach for fast counting:
    For each i, let ap3_pairs[i] be a list of (j, k) such that {i, j, k} is a 3-AP.
    To count 3-APs in a set S represented by a bitmask M:
    Total 3-APs = 1/3 * sum_{i in S} (number of pairs (j, k) in ap3_pairs[i] such that j, k in S).
    
    For fast delta calculation in local search:
    If we add y, the gain is the number of (j, k) in ap3_pairs[y] such that j, k are in S.
    If we remove x, the loss is the number of (j, k) in ap3_pairs[x] such that j, k are in S.
    If we swap x (remove) and y (add), delta = Gain(y, S\{x}) - Loss(x, S).
    
    Using bitmasks:
    Let M be the bitmask of the current set S.
    For a given y, Gain(y, M) = popcount(M & ap3_partner_mask[y])?
    No, ap3_pairs[y] contains pairs (j, k). We need to check if BOTH j and k are in M.
    We can precompute for each y a list of bitmasks representing the pairs?
    Or, for each y, precompute a list of integers `pair_masks` where each integer has bits j and k set.
    Then Gain(y, M) = sum(1 for pm in pair_masks[y] if (M & pm) == pm).
    
    This is still O(degree). Degree is roughly N.
    However, N=256 is small.
    Let's stick to the list of pairs but optimize the inner loop.
    """
    ap3_pairs = [[] for _ in range(N + 1)]
    ap4_partners = [[] for _ in range(N + 1)]
    
    # 3-APs: (a, a+d, a+2d)
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            ap3_pairs[a].append((b, c))
            ap3_pairs[b].append((a, c))
            ap3_pairs[c].append((a, b))
            
    # 4-APs: (a, a+d, a+2d, a+3d)
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            ap4_partners[a].append((b, c, e))
            ap4_partners[b].append((a, c, e))
            ap4_partners[c].append((a, b, e))
            ap4_partners[e].append((a, b, c))
            
    return ap3_pairs, ap4_partners

def _count_3aps(in_set, ap3_pairs):
    """Count 3-APs in the set."""
    total = 0
    for i in range(1, N + 1):
        if in_set[i]:
            for j, k in ap3_pairs[i]:
                if in_set[j] and in_set[k]:
                    total += 1
    return total // 3

def _grasp_init(ap3_pairs, ap4_partners, k_factor=5, iterations=1):
    """
    GRASP Initialization.
    Repeatedly selects an element from the top K candidates that maximizes 3-AP gain without creating 4-APs.
    Returns the best set found in `iterations` runs.
    """
    best_set = [False] * (N + 1)
    best_score = -1
    
    for _ in range(iterations):
        in_set = [False] * (N + 1)
        candidates = list(range(1, N + 1))
        
        while candidates:
            valid_candidates = []
            gains = {}
            
            for y in candidates:
                # Check 4-AP validity
                valid = True
                for j, k, l in ap4_partners[y]:
                    if in_set[j] and in_set[k] and in_set[l]:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Calculate gain
                gain = 0
                for j, k in ap3_pairs[y]:
                    if in_set[j] and in_set[k]:
                        gain += 1
                
                valid_candidates.append(y)
                gains[y] = gain
            
            if not valid_candidates:
                break
            
            # Sort by gain descending
            valid_candidates.sort(key=lambda x: gains[x], reverse=True)
            
            # Select from top K
            K = min(k_factor, len(valid_candidates))
            top_k = valid_candidates[:K]
            
            # Random choice from top K
            y = random.choice(top_k)
            
            in_set[y] = True
            candidates.remove(y)
            
        score = _count_3aps(in_set, ap3_pairs)
        if score > best_score:
            best_score = score
            best_set = in_set[:]
            
    return best_score, best_set

def _tabu_search(initial_set, ap3_pairs, ap4_partners, deadline, tabu_tenure=15, max_no_improve=100):
    """
    Tabu Search local search.
    Moves: Add, Remove, Swap.
    Tabu list stores (type, arg1, arg2) tuples.
    Aspiration: Accept tabu move if it improves global best.
    """
    in_set = initial_set[:]
    present = [i for i in range(1, N + 1) if in_set[i]]
    absent = [i for i in range(1, N + 1) if not in_set[i]]
    
    score = _count_3aps(in_set, ap3_pairs)
    best_score = score
    best_set = in_set[:]
    
    tabu_list = []
    no_improve = 0
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            break
            
        best_delta = -10**9
        best_move = None
        best_move_type = None
        best_move_args = None
        
        # Generate candidate moves
        # 1. Swaps (most likely to improve)
        # Sample a subset of present and absent to keep it fast
        if present and absent:
            sample_size = min(20, len(present), len(absent))
            sample_present = random.sample(present, sample_size)
            sample_absent = random.sample(absent, sample_size)
            
            for x in sample_present:
                for y in sample_absent:
                    # Check 4-AP validity for swap: y added to S\{x}
                    valid = True
                    for j, k, l in ap4_partners[y]:
                        if (in_set[j] and j != x) and (in_set[k] and k != x) and (in_set[l] and l != x):
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Check Tabu
                    is_tabu = False
                    for t_type, t_a1, t_a2 in tabu_list:
                        if t_type == 2 and ((t_a1 == x and t_a2 == y) or (t_a1 == y and t_a2 == x)):
                            is_tabu = True
                            break
                    
                    # Calculate Delta
                    # Loss of x in S
                    loss_x = 0
                    for j, k in ap3_pairs[x]:
                        if in_set[j] and in_set[k]:
                            loss_x += 1
                            
                    # Gain of y in S\{x}
                    gain_y = 0
                    for j, k in ap3_pairs[y]:
                        if (in_set[j] and j != x) and (in_set[k] and k != x):
                            gain_y += 1
                            
                    delta = gain_y - loss_x
                    
                    # Aspiration: if not tabu, or if it improves best_score
                    if not is_tabu or (score + delta > best_score):
                        if delta > best_delta:
                            best_delta = delta
                            best_move = (x, y)
                            best_move_type = 2
        
        # 2. Adds
        if absent:
            sample_size = min(20, len(absent))
            sample_absent = random.sample(absent, sample_size)
            for y in sample_absent:
                valid = True
                for j, k, l in ap4_partners[y]:
                    if in_set[j] and in_set[k] and in_set[l]:
                        valid = False
                        break
                if not valid:
                    continue
                
                is_tabu = any(t_type == 0 and t_a1 == y for t_type, t_a1, t_a2 in tabu_list)
                
                gain_y = 0
                for j, k in ap3_pairs[y]:
                    if in_set[j] and in_set[k]:
                        gain_y += 1
                        
                delta = gain_y
                
                if not is_tabu or (score + delta > best_score):
                    if delta > best_delta:
                        best_delta = delta
                        best_move = (y,)
                        best_move_type = 0
                        
        # 3. Removes
        if present:
            sample_size = min(20, len(present))
            sample_present = random.sample(present, sample_size)
            for x in sample_present:
                is_tabu = any(t_type == 1 and t_a1 == x for t_type, t_a1, t_a2 in tabu_list)
                
                loss_x = 0
                for j, k in ap3_pairs[x]:
                    if in_set[j] and in_set[k]:
                        loss_x += 1
                            
                delta = -loss_x
                
                if not is_tabu or (score + delta > best_score):
                    if delta > best_delta:
                        best_delta = delta
                        best_move = (x,)
                        best_move_type = 1
        
        if best_move is None:
            # No valid non-tabu moves or all moves are terrible
            # If we have tabu moves, we might still take the best tabu one if we are stuck?
            # For simplicity, if no move found, restart or break.
            # Let's try to find the best move ignoring tabu if we are stuck for a while
            if no_improve > max_no_improve // 2:
                # Relax tabu
                tabu_list = []
                continue
            else:
                break
        
        # Apply move
        if best_move_type == 2:
            x, y = best_move
            in_set[x] = False
            in_set[y] = True
            # Update present/absent lists
            if x in present: present.remove(x)
            present.append(y)
            if y in absent: absent.remove(y)
            absent.append(x)
            tabu_list.append((2, x, y))
            
        elif best_move_type == 0:
            y = best_move[0]
            in_set[y] = True
            present.append(y)
            if y in absent: absent.remove(y)
            tabu_list.append((0, y, 0))
            
        elif best_move_type == 1:
            x = best_move[0]
            in_set[x] = False
            if x in present: present.remove(x)
            absent.append(x)
            tabu_list.append((1, x, 0))
        
        score += best_delta
        
        if score > best_score:
            best_score = score
            best_set = in_set[:]
            no_improve = 0
        else:
            no_improve += 1
            
        # Update Tabu List
        if len(tabu_list) >= tabu_tenure:
            tabu_list.pop(0)
            
    return best_score, best_set

def construct():
    start_time = time.time()
    ap3_pairs, ap4_partners = _precompute()
    
    best_score = -1
    best_set = [False] * (N + 1)
    
    # Phase 1: Multiple GRASP initializations and Tabu refinements
    # We will run this loop until time runs out
    while time.time() - start_time < DEADLINE - 2.0:
        # 1. Generate a new GRASP solution
        # Use a small number of iterations for GRASP to keep it fast
        grasp_score, grasp_set = _grasp_init(ap3_pairs, ap4_partners, k_factor=5, iterations=3)
        
        if grasp_score > best_score:
            best_score = grasp_score
            best_set = grasp_set[:]
            
        # 2. Refine with Tabu Search
        # Allocate remaining time for this refinement
        # We want to leave some time for the next iteration
        current_time = time.time()
        # Estimate time for one iteration: ~0.5s for GRASP, ~1-2s for Tabu
        # Let's give Tabu 2 seconds per iteration
        tabu_deadline = current_time + 2.0
        
        if time.time() < DEADLINE - 3.0:
            tabu_score, tabu_set = _tabu_search(grasp_set, ap3_pairs, ap4_partners, tabu_deadline)
            
            if tabu_score > best_score:
                best_score = tabu_score
                best_set = tabu_set[:]
                
        # 3. Occasionally, refine the current best_set directly with a fresh Tabu run
        # This helps if the best solution is found and GRASP is generating worse starts
        if time.time() - start_time < DEADLINE - 5.0 and best_score > 0:
            if random.random() < 0.2:
                tabu_deadline = time.time() + 2.0
                tabu_score, tabu_set = _tabu_search(best_set, ap3_pairs, ap4_partners, tabu_deadline)
                if tabu_score > best_score:
                    best_score = tabu_score
                    best_set = tabu_set[:]

    # Final check
    if not any(best_set):
        # Fallback: return a simple valid set if something went wrong
        # e.g., all odd numbers (no 4-AP? 1,5,9,13 is a 4-AP. 1,3,5,7 is 4-AP.
        # Actually, {1, 3, 5, ..., 255} has 4-APs.
        # A safe fallback is just {1}.
        best_set = [False] * (N + 1)
        best_set[1] = True
        best_score = 0

    result = [i for i in range(1, N + 1) if best_set[i]]
    return result

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Result size: {len(res)}")
