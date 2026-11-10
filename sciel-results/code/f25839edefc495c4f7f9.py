import time
import random
from fractions import Fraction

# Precompute 3-AP conflicts
# For each x in 1..256, store list of (y, z) such that {x, y, z} is a 3-AP
# y < z, and x, y, z are distinct.
# We need to check if adding x to S creates a 3-AP.
# This happens if there exist y, z in S such that {x, y, z} is a 3-AP.
# So for each x, we precompute all pairs (y, z) in [1, 256] such that {x, y, z} is a 3-AP.

def _precompute_conflicts(n=256):
    conflicts = {i: [] for i in range(1, n + 1)}
    for a in range(1, n + 1):
        for d in range(1, n):
            b = a + d
            c = a + 2 * d
            if c > n:
                break
            # Triple is (a, b, c)
            # For a, conflicts are (b, c)
            conflicts[a].append((b, c))
            # For b, conflicts are (a, c)
            conflicts[b].append((a, c))
            # For c, conflicts are (a, b)
            conflicts[c].append((a, b))
    return conflicts

CONFLICTS = _precompute_conflicts(256)
N = 256

def _can_add(S_bitmask, x):
    """Check if x can be added to S (represented as bitmask) without creating a 3-AP."""
    # x is not in S
    # Check all pairs (y, z) in CONFLICTS[x]
    # If y and z are both in S, then adding x creates a 3-AP.
    for y, z in CONFLICTS[x]:
        if (S_bitmask & (1 << (y - 1))) and (S_bitmask & (1 << (z - 1))):
            return False
    return True

def _score_float(S_list):
    return sum(1.0 / x for x in S_list)

def _bitmask_to_list(S_bitmask):
    return [i + 1 for i in range(N) if (S_bitmask & (1 << i))]

def _greedy_fill(S_bitmask, S_list, current_score, deadline):
    """Greedily add elements in order of reciprocal weight (smallest x first)."""
    for x in range(1, N + 1):
        if time.time() > deadline - 5:
            break
        if S_bitmask & (1 << (x - 1)):
            continue
        if _can_add(S_bitmask, x):
            S_bitmask |= (1 << (x - 1))
            S_list.append(x)
            current_score += 1.0 / x
    return S_bitmask, S_list, current_score

def _local_search(S_bitmask, S_list, current_score, best_S_bitmask, best_score, deadline, max_kick_restarts=10):
    """Local search with kicks to escape local optima."""
    no_improve_count = 0
    max_no_improve = 15  # Lower threshold for more kicks
    kick_count = 0
    max_kicks = max_kick_restarts * 3  # Allow multiple kicks per restart
    
    while time.time() < deadline - 2:
        improved = False
        
        # 1. Try Add
        best_add = None
        best_add_score = current_score
        for x in range(1, N + 1):
            if S_bitmask & (1 << (x - 1)):
                continue
            if _can_add(S_bitmask, x):
                new_score = current_score + 1.0 / x
                if new_score > best_add_score:
                    best_add_score = new_score
                    best_add = x
        
        if best_add is not None:
            S_bitmask |= (1 << (best_add - 1))
            S_list.append(best_add)
            current_score = best_add_score
            improved = True
            no_improve_count = 0
            if current_score > best_score:
                best_score = current_score
                best_S_bitmask = S_bitmask
            continue
        
        # 2. Try Swap
        best_swap_remove = None
        best_swap_add = None
        best_swap_score = current_score
        
        S_indices = [i for i in range(N) if (S_bitmask & (1 << i))]
        NotS_indices = [i for i in range(N) if not (S_bitmask & (1 << i))]
        
        for i in S_indices:
            x_remove = i + 1
            S_bitmask_no_x = S_bitmask & ~(1 << i)
            score_no_x = current_score - 1.0 / x_remove
            
            for j in NotS_indices:
                x_add = j + 1
                if _can_add(S_bitmask_no_x, x_add):
                    new_score = score_no_x + 1.0 / x_add
                    if new_score > best_swap_score:
                        best_swap_score = new_score
                        best_swap_remove = x_remove
                        best_swap_add = x_add
        
        if best_swap_remove is not None and best_swap_add is not None:
            S_bitmask &= ~(1 << (best_swap_remove - 1))
            S_bitmask |= (1 << (best_swap_add - 1))
            S_list.remove(best_swap_remove)
            S_list.append(best_swap_add)
            current_score = best_swap_score
            improved = True
            no_improve_count = 0
            if current_score > best_score:
                best_score = current_score
                best_S_bitmask = S_bitmask
            continue
        
        # 3. No improvement, try Kick
        no_improve_count += 1
        if no_improve_count >= max_no_improve:
            # Smart kick: Remove elements that block high-value additions
            # Find which elements in S block the most high-value additions
            S_set = set(S_list)
            block_count = {x: 0 for x in S_list}
            
            # For each element not in S, check which elements in S block it
            for x_add in range(1, N + 1):
                if x_add in S_set:
                    continue
                # Check if x_add is blocked
                for y, z in CONFLICTS[x_add]:
                    if y in S_set and z in S_set:
                        # Both y and z block x_add
                        block_count[y] += 1.0 / x_add
                        block_count[z] += 1.0 / x_add
                        break  # One blocking pair is enough
            
            # Sort by block count (descending) and remove top 2-3
            sorted_blocked = sorted(block_count.items(), key=lambda x: -x[1])
            num_remove = min(3, len(S_list))
            to_remove = [x for x, _ in sorted_blocked[:num_remove]]
            
            for x in to_remove:
                S_bitmask &= ~(1 << (x - 1))
                S_list.remove(x)
            current_score = _score_float(S_list)
            
            # Greedily add back
            S_bitmask, S_list, current_score = _greedy_fill(S_bitmask, S_list, current_score, deadline)
            
            no_improve_count = 0
            kick_count += 1
            if current_score > best_score:
                best_score = current_score
                best_S_bitmask = S_bitmask
            
            # If we've done many kicks, restart from scratch with different randomization
            if kick_count >= max_kicks:
                # Random restart: start fresh with random subset
                random.seed(int(time.time() * 1000) % 100000)
                S_bitmask = 0
                S_list = []
                current_score = 0.0
                
                # Start with a random subset of small numbers
                small_nums = list(range(1, 50))
                random.shuffle(small_nums)
                for x in small_nums[:20]:
                    if _can_add(S_bitmask, x):
                        S_bitmask |= (1 << (x - 1))
                        S_list.append(x)
                        current_score += 1.0 / x
                
                # Then greedy fill
                S_bitmask, S_list, current_score = _greedy_fill(S_bitmask, S_list, current_score, deadline)
                kick_count = 0
                no_improve_count = 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_S_bitmask = S_bitmask
    
    return S_bitmask, S_list, current_score, best_S_bitmask, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    # 1. Greedy Initialization
    S_bitmask = 0
    S_list = []
    current_score = 0.0
    
    S_bitmask, S_list, current_score = _greedy_fill(S_bitmask, S_list, current_score, deadline)
    
    if not S_list:
        return [1]
    
    best_S_bitmask = S_bitmask
    best_score = current_score
    
    # 2. Local Search with Kicks
    S_bitmask, S_list, current_score, best_S_bitmask, best_score = _local_search(
        S_bitmask, S_list, current_score, best_S_bitmask, best_score, deadline
    )
    
    # Final validation and return
    final_list = _bitmask_to_list(best_S_bitmask)
    if not final_list:
        final_list = [1]
        
    # Verify validity just in case
    s_set = set(final_list)
    valid = True
    for a in s_set:
        for b in s_set:
            if a >= b:
                continue
            c = 2 * b - a
            if c in s_set:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to simple greedy if something went wrong
        S_bitmask = 0
        S_list = []
        for x in range(1, N + 1):
            if _can_add(S_bitmask, x):
                S_bitmask |= (1 << (x - 1))
                S_list.append(x)
        final_list = S_list

    return sorted(final_list)
