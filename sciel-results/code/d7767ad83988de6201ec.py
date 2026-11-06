import time
import random
from itertools import combinations

def has_sidon_4(S):
    """
    Returns (True, subset) if any 4-element subset of S is Sidon.
    Returns (False, None) otherwise.
    """
    S = sorted(S)
    for combo in combinations(S, 4):
        sums = set()
        is_s = True
        for i in range(4):
            for j in range(i, 4):
                s = combo[i] + combo[j]
                if s in sums:
                    is_s = False
                    break
                sums.add(s)
        if is_s:
            return True, combo
    return False, None

def construct():
    start_time = time.time()
    # Safety margin: Evaluator kills at 90s, we aim for 50s to be safe.
    deadline = start_time + 50.0
    
    # Initial candidate: Random distinct integers
    # Using range(12) is a bad start because APs have many Sidon subsets?
    # Actually, APs have NO Sidon subsets of size > 2? 
    # Let's check: {0,1,2,3}. Sums: 0+0=0, 0+1=1, 0+2=2, 0+3=3, 1+1=2 (Collision with 0+2).
    # So APs are bad for Sidon. They have collisions.
    # We want to AVOID Sidon subsets.
    # So an AP is a GOOD starting point for minimizing Sidon size!
    # Wait, if APs have no 4-Sidon subsets, then max Sidon size is 3?
    # Let's check {0,1,2,3,4,5,6,7,8,9,10,11}.
    # Does it have a 4-Sidon subset?
    # A 4-Sidon subset requires all 10 sums to be unique.
    # In an AP, $a_i + a_j = i+j + 2a_0$.
    # Unordered pairs $(i,j)$ with $i \le j$.
    # Sums are determined by $i+j$.
    # If two pairs have the same sum $i+j$, they collide.
    # For size 4, indices 0,1,2,3.
    # Pairs: (0,0)->0, (0,1)->1, (0,2)->2, (0,3)->3, (1,1)->2 (Collision with 0,2), (1,2)->3 (Collision with 0,3), (1,3)->4, (2,2)->4 (Collision with 1,3), (2,3)->5, (3,3)->6.
    # Collisions exist. So {0,1,2,3} is NOT Sidon.
    # In fact, any 4 distinct elements from an AP will have collisions in pair sums?
    # Yes, because $i+j$ is not unique for all pairs in a 4-set?
    # Let's take {0, 1, 3, 4} from AP? No, that's not an AP itself, but a subset.
    # If S is an AP, any subset is not necessarily an AP, but the values are $k_i$.
    # $k_i + k_j$.
    # It is known that APs have small Sidon subsets.
    # Actually, the maximum Sidon subset of an AP of length $N$ is roughly $\sqrt{N}$.
    # For $N=12$, $\sqrt{12} \approx 3.4$. So max Sidon size is likely 3.
    # So `range(12)` might already be optimal (score 3)!
    
    # Let's verify if range(12) has a 4-Sidon subset.
    # If it doesn't, we can just return it immediately.
    
    current_set = list(range(12))
    has, _ = has_sidon_4(current_set)
    
    if not has:
        return current_set
        
    # If it does (unlikely for AP, but possible for random), we search.
    # But wait, the previous program failed. Why?
    # If range(12) is optimal, the previous program should have returned it quickly if it checked correctly.
    # The previous program used `count_sidon_4`.
    # If `count_sidon_4(range(12))` was 0, it would return.
    # Maybe `range(12)` DOES have a 4-Sidon subset?
    # Let's check {0, 1, 3, 7}? No, that's not in range(12) as a subset of *consecutive* integers?
    # The set is {0, 1, ..., 11}.
    # Subset {0, 1, 3, 4}: Sums 0,1,3,4,2,4,7,6,8. 4 repeats (1+3 and 0+4? No 1+3=4, 0+4=4). Collision.
    # Subset {0, 2, 3, 7}: 0,2,3,7,4,5,9,6,10,14. All unique?
    # 0+0=0, 0+2=2, 0+3=3, 0+7=7
    # 2+2=4, 2+3=5, 2+7=9
    # 3+3=6, 3+7=10
    # 7+7=14
    # Sums: 0,2,3,7,4,5,9,6,10,14.
    # Are they unique?
    # 0,2,3,4,5,6,7,9,10,14.
    # Yes! {0,2,3,7} is a Sidon subset of range(12).
    # So range(12) has a 4-Sidon subset. Score is at least 4.
    
    # We need to find a set with NO 4-Sidon subsets.
    
    best_set = current_set[:]
    best_has = True
    
    for i in range(5000): # Max iterations
        if time.time() > deadline:
            break
            
        has, subset = has_sidon_4(current_set)
        
        if not has:
            return current_set
            
        # We have a Sidon 4-subset. Try to break it.
        # Pick a random element from the subset to move.
        elem_to_move = random.choice(subset)
        
        # Try to find a new value that reduces the "badness"
        # Simple heuristic: Move to a random value and check if it eliminates this specific subset
        # and doesn't create too many new ones.
        # To be fast, we just try random moves and accept if `has_sidon_4` is still False? 
        # No, we want to reach False.
        # If we move, we might create new ones.
        
        # Strategy: Random Walk with Backtracking on "Count of Sidon 4-subsets" is too slow.
        # Strategy: Just try to find ANY set with 0.
        
        improved = False
        for _ in range(20): # Limit inner loop
            new_val = random.randint(0, 128)
            if new_val in current_set:
                continue
            
            new_set = [x for x in current_set if x != elem_to_move]
            new_set.append(new_val)
            
            # Check if new set is better.
            # "Better" means: If it has no Sidon 4, it's perfect.
            # If it still has, we just accept it with some probability or if it "looks" better?
            # Since we don't have a fast cost function, let's just do a random walk 
            # but bias towards values that create collisions.
            
            # Actually, let's just try the move. If it results in `has_sidon_4 == False`, we win.
            # If not, we keep it if we want to explore.
            # But we need to avoid cycling.
            
            # Let's just replace and check.
            current_set = new_set
            has, _ = has_sidon_4(current_set)
            
            if not has:
                return current_set
            
            # If we haven't won, we continue. 
            # To prevent getting stuck in a local minimum of "many Sidon subsets",
            # we occasionally restart.
            
            improved = True
            break # Only one move per outer iteration to keep it simple/fast
            
        if not improved:
            # If no valid random move found (collision in set), restart?
            # Just pick a random index and random value, force it.
            idx = random.randint(0, 11)
            while True:
                new_val = random.randint(0, 128)
                if new_val not in current_set:
                    current_set[idx] = new_val
                    break

    return current_set
