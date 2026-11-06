# EVOLVE-BLOCK-START
import time
from fractions import Fraction

N = 256

def construct():
    start_time = time.time()
    DEADLINE = start_time + 75  # Safety margin before 90s kill
    
    # 1. Greedy initialization
    selected = []
    in_S = [False] * (N + 1)
    
    for x in range(1, N + 1):
        valid = True
        for a in selected:
            for b in selected:
                if a < b:
                    if 2 * b - a == x:
                        valid = False
                        break
            if not valid:
                break
        if valid:
            selected.append(x)
            in_S[x] = True
            
    S = set(selected)
    
    # Helper: check if z can be added to a set represented by in_arr (boolean array)
    # O(|S|) check using the three AP cases
    def is_compatible(z, in_arr):
        for a in range(1, N + 1):
            if not in_arr[a]:
                continue
            # Case 1: z is middle. b = 2z - a
            b = 2 * z - a
            if b != a and 1 <= b <= N and in_arr[b]:
                return False
            # Case 2: z is largest or smallest with a as middle. b = (a+z)/2
            if (a + z) % 2 == 0:
                b = (a + z) // 2
                if b != a and b != z and 1 <= b <= N and in_arr[b]:
                    return False
            # Case 3: z is smallest. b = 2a - z
            b = 2 * a - z
            if b != a and b != z and 1 <= b <= N and in_arr[b]:
                return False
        return True
    
    # Fast compatibility check using only elements in S
    def is_compatible_fast(z, S_set, in_arr):
        for a in S_set:
            # Case 1: z is middle. b = 2z - a
            b = 2 * z - a
            if b != a and 1 <= b <= N and in_arr[b]:
                return False
            # Case 2: z is largest/smallest with a as middle. b = (a+z)/2
            if (a + z) % 2 == 0:
                b = (a + z) // 2
                if b != a and b != z and 1 <= b <= N and in_arr[b]:
                    return False
            # Case 3: z is smallest. b = 2a - z
            b = 2 * a - z
            if b != a and b != z and 1 <= b <= N and in_arr[b]:
                return False
        return True
    
    # Try to add elements greedily (free additions that don't require removals)
    def try_add_elements(S, in_arr):
        added = False
        for z in range(1, N + 1):
            if in_arr[z]:
                continue
            if is_compatible_fast(z, S, in_arr):
                S.add(z)
                in_arr[z] = True
                added = True
        return added
    
    # First, try to add any missing elements
    try_add_elements(S, in_S)
    
    # Local search: swap larger elements for smaller ones to increase reciprocal sum
    improved = True
    while improved and time.time() < DEADLINE:
        improved = False
        # Try swapping each y in S (descending) for best z < y not in S
        for y in sorted(S, reverse=True):
            if time.time() > DEADLINE:
                break
            
            # Temporarily remove y
            S.remove(y)
            in_S[y] = False
            
            # Find best z < y that can be added
            best_z = None
            for z in range(1, y):
                if in_S[z]:
                    continue
                if is_compatible_fast(z, S, in_S):
                    best_z = z
                    break  # Smallest z gives largest reciprocal gain
            
            if best_z is not None:
                S.add(best_z)
                in_S[best_z] = True
                improved = True
            else:
                # Put y back
                S.add(y)
                in_S[y] = True
        
        # After swaps, try to add more elements
        if time.time() < DEADLINE:
            try_add_elements(S, in_S)
    
    # Final pass: try to add any remaining elements
    if time.time() < DEADLINE:
        try_add_elements(S, in_S)
    
    return sorted(S)
# EVOLVE-BLOCK-END
