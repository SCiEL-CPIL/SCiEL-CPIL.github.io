"""Combined program: greedy construction with local search refinement."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt
from itertools import combinations

def _can_add(S_list, x):
    """
    Check if x can be added to S.
    S_list is list(S). x is an int not in S.
    """
    n = len(S_list)
    
    # 1. x divides a*b for distinct a,b in S
    for i in range(n):
        a = S_list[i]
        for j in range(i+1, n):
            b = S_list[j]
            if (a * b) % x == 0:
                return False
    
    # 2. a divides x*b for distinct a,b in S
    for i in range(n):
        a = S_list[i]
        for j in range(n):
            if i == j:
                continue
            b = S_list[j]
            if (x * b) % a == 0:
                return False
                
    return True

def _is_valid_set(S_list):
    """
    Check if S is a valid set.
    """
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(n):
            if i == j: continue
            b = S_list[j]
            for k in range(n):
                if i == k or j == k: continue
                c = S_list[k]
                if (b * c) % a == 0:
                    return False
    return True

def _is_valid(S_list):
    """
    Check if S is a valid set.
    For every three distinct members a, b, c in S, a does not divide b*c.
    """
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(n):
            if i == j:
                continue
            b = S_list[j]
            for k in range(n):
                if i == k or j == k:
                    continue
                c = S_list[k]
                if (b * c) % a == 0:
                    return False
    return True

def _greedy_construct(order, deadline):
    """
    Greedily construct a set by iterating through 'order' and adding
    if the resulting set remains valid.
    """
    S = set()
    S_list = []
    
    for x in order:
        if time.time() > deadline:
            break
        if _can_add(S_list, x):
            S.add(x)
            S_list.append(x)
            
    return S, S_list

def _local_search_2for2(current_set, current_list, non_members, deadline):
    """
    Local search with 1-for-1 and 2-for-2 swaps.
    """
    current_size = len(current_set)
    best_set = current_set.copy()
    best_size = current_size
    no_improve = 0
    max_no_improve = 150
    
    while time.time() < deadline - 1.0:
        if not non_members:
            break
            
        improved = False
        move_type = random.random()
        
        if move_type < 0.6:
            # 1-for-1 swap or add
            if random.random() < 0.4 and current_size < 255:
                # Try systematic add: check all non-members, pick first that fits
                for idx in range(len(non_members)):
                    x = non_members[idx]
                    if _can_add(current_list, x):
                        current_set.add(x)
                        current_list.append(x)
                        non_members.pop(idx)
                        current_size += 1
                        improved = True
                        no_improve = 0
                        if current_size > best_size:
                            best_size = current_size
                            best_set = current_set.copy()
                        break
            else:
                # Try Swap: systematic search
                if len(current_set) > 0:
                    # Try each non-member with each current member
                    for idx in range(len(non_members)):
                        x = non_members[idx]
                        for y_idx in range(len(current_list)):
                            y = current_list[y_idx]
                            temp_list = [v for i, v in enumerate(current_list) if i != y_idx]
                            if _can_add(temp_list, x):
                                current_set.remove(y)
                                current_set.add(x)
                                current_list = temp_list + [x]
                                non_members[idx] = y
                                improved = True
                                no_improve = 0
                                break
                        if improved:
                            break
        else:
            # 2-for-2 swap: more systematic
            if len(current_set) >= 2 and len(non_members) >= 2:
                # Sample a few combinations
                for attempt in range(5):
                    # Pick 2 distinct to remove
                    y1_idx = random.randint(0, len(current_list) - 1)
                    y2_idx = random.randint(0, len(current_list) - 1)
                    if y1_idx == y2_idx:
                        y2_idx = (y2_idx + 1) % len(current_list)
                    y1 = current_list[y1_idx]
                    y2 = current_list[y2_idx]
                    
                    # Pick 2 distinct to add
                    x1_idx = random.randint(0, len(non_members) - 1)
                    x2_idx = random.randint(0, len(non_members) - 1)
                    if x1_idx == x2_idx:
                        x2_idx = (x2_idx + 1) % len(non_members)
                    x1 = non_members[x1_idx]
                    x2 = non_members[x2_idx]
                    
                    # Check if S' is valid
                    temp_list = [v for v in current_list if v != y1 and v != y2]
                    temp_list.extend([x1, x2])
                    
                    if _is_valid_set(temp_list):
                        # Apply swap
                        current_set.remove(y1)
                        current_set.remove(y2)
                        current_set.add(x1)
                        current_set.add(x2)
                        current_list = temp_list
                        # Update non_members
                        non_members[x1_idx] = y1
                        non_members[x2_idx] = y2
                        improved = True
                        no_improve = 0
                        break
        
        if not improved:
            no_improve += 1
            if no_improve > max_no_improve:
                # Restart or reset
                no_improve = 0
                if random.random() < 0.5:
                    current_set = best_set.copy()
                    current_list = list(current_set)
                    current_size = best_size
                    non_members = [x for x in range(2, 257) if x not in current_set]
                else:
                    # Kick: remove a random element to break symmetry
                    if current_list:
                        y = random.choice(current_list)
                        current_set.remove(y)
                        current_list.remove(y)
                        non_members.append(y)
                        current_size -= 1
        
    return best_set, best_size

def construct():
    deadline = time.time() + 55.0  # Safety margin for 60s limit
    start_time = time.time()
    
    all_nums = list(range(2, 257))
    
    best_set = set()
    best_size = 0
    
    # Phase 1: Multiple Greedy Constructions
    # Try specific orders first
    def is_prime(n):
        if n < 2: return False
        if n == 2: return True
        if n % 2 == 0: return False
        for i in range(3, int(n**0.5)+1, 2):
            if n % i == 0: return False
        return True
            
    primes = [x for x in all_nums if is_prime(x)]
    composites = [x for x in all_nums if not is_prime(x)]
    
    orders = [
        list(range(2, 257)),
        list(range(256, 1, -1)),
        primes + composites,
        composites + primes,
    ]
    
    # Add random orders
    for _ in range(30):
        random.shuffle(all_nums)
        orders.append(all_nums.copy())
        
    for order in orders:
        if time.time() > deadline:
            break
            
        S, S_list = _greedy_construct(order, deadline)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
    
    # Phase 2: Local Search with 2-for-2 swaps
    current_set = best_set.copy()
    current_list = list(current_set)
    current_size = best_size
    
    non_members = [x for x in all_nums if x not in current_set]
    
    # Run local search
    best_set, best_size = _local_search_2for2(current_set, current_list, non_members, deadline)
    
    # Phase 3: Final polish - try to add any remaining elements
    if time.time() < deadline - 0.5:
        current_set = best_set.copy()
        current_list = list(current_set)
        non_members = [x for x in all_nums if x not in current_set]
        
        # Repeatedly try to add elements until no more can be added
        changed = True
        while changed and time.time() < deadline - 0.5:
            changed = False
            for idx in range(len(non_members)):
                x = non_members[idx]
                if _can_add(current_list, x):
                    current_set.add(x)
                    current_list.append(x)
                    non_members.pop(idx)
                    changed = True
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = current_set.copy()
                    break
    
    # Final validation and cleanup
    result = list(best_set)
    # Ensure types are correct
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
