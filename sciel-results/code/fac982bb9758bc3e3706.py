"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    n = 16
    points = list(range(n))
    
    # Generate all C(16,3) = 560 triples
    all_triples = list(combinations(range(n), 3))
    num_triples = len(all_triples)
    
    # Map triple to index
    triple_index = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute constraints (K(2,2,2)s)
    # A constraint is a set of 8 triples forming a K(2,2,2)
    # We represent each constraint as a bitset (integer)
    # We also need a list of constraints for each triple to quickly check conflicts
    
    constraints = []
    triple_to_constraints = [[] for _ in range(num_triples)]
    
    # Hardcoding the 15 matchings for indices 0-5
    matchings = [
        [(0,1),(2,3),(4,5)],
        [(0,1),(2,4),(3,5)],
        [(0,1),(2,5),(3,4)],
        [(0,2),(1,3),(4,5)],
        [(0,2),(1,4),(3,5)],
        [(0,2),(1,5),(3,4)],
        [(0,3),(1,2),(4,5)],
        [(0,3),(1,4),(2,5)],
        [(0,3),(1,5),(2,4)],
        [(0,4),(1,2),(3,5)],
        [(0,4),(1,3),(2,5)],
        [(0,4),(1,5),(2,3)],
        [(0,5),(1,2),(3,4)],
        [(0,5),(1,3),(2,4)],
        [(0,5),(1,4),(2,3)]
    ]
    
    for combo in combinations(range(n), 6):
        for m in matchings:
            # Map local indices to actual points
            actual_pairs = []
            for idx1, idx2 in m:
                pt1 = combo[idx1]
                pt2 = combo[idx2]
                actual_pairs.append((pt1, pt2))
            
            # Now we have 3 pairs: (a1, a2), (b1, b2), (c1, c2)
            (a1, a2), (b1, b2), (c1, c2) = actual_pairs
            
            # The 8 transversal triples are:
            transversals = [
                (a1, b1, c1),
                (a1, b1, c2),
                (a1, b2, c1),
                (a1, b2, c2),
                (a2, b1, c1),
                (a2, b1, c2),
                (a2, b2, c1),
                (a2, b2, c2)
            ]
            
            # Convert to sorted tuples to match triple_index
            t_indices = []
            valid = True
            for t in transversals:
                st = tuple(sorted(t))
                if st in triple_index:
                    t_indices.append(triple_index[st])
                else:
                    valid = False
                    break
            
            if valid:
                # Create bitset
                mask = 0
                for idx in t_indices:
                    mask |= (1 << idx)
                
                constraints.append(mask)
                c_idx = len(constraints) - 1
                for idx in t_indices:
                    triple_to_constraints[idx].append(c_idx)
    
    # Count how many constraints each triple is in (for greedy ordering)
    triple_degree = [len(cts) for cts in triple_to_constraints]
    
    def is_valid_add(t_idx, selected_mask):
        # Check if adding t_idx completes any constraint
        for c_idx in triple_to_constraints[t_idx]:
            c_mask = constraints[c_idx]
            # If all other bits in c_mask are set in selected_mask, then adding t_idx completes it
            if (c_mask & ~selected_mask) == (1 << t_idx):
                return False
        return True

    # Multi-start Greedy
    best_mask = 0
    best_size = 0
    
    # Try a few random greedy orders
    for i in range(30):
        if time.time() > deadline - 60: break
        random.seed(i)
        order = list(range(num_triples))
        random.shuffle(order)
        
        curr_mask = 0
        curr_size = 0
        for t_idx in order:
            if is_valid_add(t_idx, curr_mask):
                curr_mask |= (1 << t_idx)
                curr_size += 1
        
        if curr_size > best_size:
            best_size = curr_size
            best_mask = curr_mask
    
    # Also try greedy by degree (ascending)
    order_deg = sorted(range(num_triples), key=lambda x: triple_degree[x])
    curr_mask = 0
    curr_size = 0
    for t_idx in order_deg:
        if is_valid_add(t_idx, curr_mask):
            curr_mask |= (1 << t_idx)
            curr_size += 1
    if curr_size > best_size:
        best_size = curr_size
        best_mask = curr_mask

    # Local Search: 2-for-2 exchange
    curr_mask = best_mask
    curr_size = best_size
    best_mask_ls = best_mask
    best_size_ls = best_size
    
    # Precompute list of selected triples for efficient removal
    selected_list = [i for i in range(num_triples) if curr_mask & (1 << i)]
    unselected_list = [i for i in range(num_triples) if not (curr_mask & (1 << i))]
    
    T = 5.0
    T_min = 0.1
    cooling = 0.995
    no_improve = 0
    
    while time.time() < deadline - 10 and T > T_min:
        if no_improve > 1000:
            break
        no_improve += 1
        
        # Choose move: 1-for-1, 2-for-2, or 1-for-2
        move_type = random.random()
        
        if move_type < 0.4:
            # 1-for-1 exchange: remove one, add one
            if not selected_list or not unselected_list:
                continue
            t_out = random.choice(selected_list)
            t_in = random.choice(unselected_list)
            
            # Remove t_out, check if t_in can be added
            temp_mask = curr_mask & ~(1 << t_out)
            if is_valid_add(t_in, temp_mask):
                new_mask = temp_mask | (1 << t_in)
                new_size = curr_size  # size unchanged
                # Accept if we can escape local optima (random walk)
                accept = True
                if accept:
                    curr_mask = new_mask
                    # Update lists
                    selected_list.remove(t_out)
                    selected_list.append(t_in)
                    unselected_list.remove(t_in)
                    unselected_list.append(t_out)
                    if curr_size > best_size_ls:
                        best_size_ls = curr_size
                        best_mask_ls = curr_mask
                        no_improve = 0
        
        elif move_type < 0.7:
            # 2-for-2 exchange: remove two, add two
            if len(selected_list) < 2 or len(unselected_list) < 2:
                continue
            t_out1, t_out2 = random.sample(selected_list, 2)
            t_in1, t_in2 = random.sample(unselected_list, 2)
            
            # Remove both, check if both can be added
            temp_mask = curr_mask & ~(1 << t_out1) & ~(1 << t_out2)
            if is_valid_add(t_in1, temp_mask) and is_valid_add(t_in2, temp_mask | (1 << t_in1)):
                new_mask = temp_mask | (1 << t_in1) | (1 << t_in2)
                new_size = curr_size  # size unchanged
                accept = True
                if accept:
                    curr_mask = new_mask
                    # Update lists
                    selected_list.remove(t_out1)
                    selected_list.remove(t_out2)
                    selected_list.append(t_in1)
                    selected_list.append(t_in2)
                    unselected_list.remove(t_in1)
                    unselected_list.remove(t_in2)
                    unselected_list.append(t_out1)
                    unselected_list.append(t_out2)
                    if curr_size > best_size_ls:
                        best_size_ls = curr_size
                        best_mask_ls = curr_mask
                        no_improve = 0
        
        else:
            # Try to add a triple (if possible)
            if not unselected_list:
                continue
            t_in = random.choice(unselected_list)
            if is_valid_add(t_in, curr_mask):
                curr_mask |= (1 << t_in)
                curr_size += 1
                selected_list.append(t_in)
                unselected_list.remove(t_in)
                if curr_size > best_size_ls:
                    best_size_ls = curr_size
                    best_mask_ls = curr_mask
                    no_improve = 0
            else:
                # Try to remove a triple to allow future additions
                if selected_list:
                    t_out = random.choice(selected_list)
                    # Accept removal with probability based on temperature
                    if random.random() < (1.0 / T):
                        curr_mask &= ~(1 << t_out)
                        curr_size -= 1
                        selected_list.remove(t_out)
                        unselected_list.append(t_out)
        
        T *= cooling
    
    # Final local search pass with 1-for-1 and 2-for-2 to polish
    curr_mask = best_mask_ls
    curr_size = best_size_ls
    best_mask_final = best_mask_ls
    best_size_final = best_size_ls
    
    selected_list = [i for i in range(num_triples) if curr_mask & (1 << i)]
    unselected_list = [i for i in range(num_triples) if not (curr_mask & (1 << i))]
    
    for _ in range(2000):
        if time.time() > deadline - 5:
            break
        if not selected_list or not unselected_list:
            break
        
        move_type = random.random()
        if move_type < 0.5:
            # 1-for-1
            t_out = random.choice(selected_list)
            t_in = random.choice(unselected_list)
            temp_mask = curr_mask & ~(1 << t_out)
            if is_valid_add(t_in, temp_mask):
                new_mask = temp_mask | (1 << t_in)
                curr_mask = new_mask
                selected_list.remove(t_out)
                selected_list.append(t_in)
                unselected_list.remove(t_in)
                unselected_list.append(t_out)
                if curr_size > best_size_final:
                    best_size_final = curr_size
                    best_mask_final = curr_mask
        else:
            # 2-for-2
            if len(selected_list) < 2 or len(unselected_list) < 2:
                continue
            t_out1, t_out2 = random.sample(selected_list, 2)
            t_in1, t_in2 = random.sample(unselected_list, 2)
            temp_mask = curr_mask & ~(1 << t_out1) & ~(1 << t_out2)
            if is_valid_add(t_in1, temp_mask) and is_valid_add(t_in2, temp_mask | (1 << t_in1)):
                new_mask = temp_mask | (1 << t_in1) | (1 << t_in2)
                curr_mask = new_mask
                selected_list.remove(t_out1)
                selected_list.remove(t_out2)
                selected_list.append(t_in1)
                selected_list.append(t_in2)
                unselected_list.remove(t_in1)
                unselected_list.remove(t_in2)
                unselected_list.append(t_out1)
                unselected_list.append(t_out2)
                if curr_size > best_size_final:
                    best_size_final = curr_size
                    best_mask_final = curr_mask
    
    # Convert best_mask_final to list of triples
    result = []
    for i in range(num_triples):
        if best_mask_final & (1 << i):
            result.append(list(all_triples[i]))
            
    return result
# EVOLVE-BLOCK-END
