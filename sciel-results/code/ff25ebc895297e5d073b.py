import time
from fractions import Fraction

N = 256

def construct():
    start_time = time.time()
    DEADLINE = start_time + 85  # Safety margin

    # 1. Greedy Initialization
    # We want to maximize sum(1/x), so we prefer small x.
    # Greedy: add x if it doesn't form a 3-AP with existing elements.
    in_S = [False] * (N + 1)
    S = []
    
    for x in range(1, N + 1):
        valid = True
        # Check if adding x creates a 3-AP
        # A 3-AP involving x can be:
        # (a, x, c) -> c = 2x - a
        # (a, b, x) -> b = (a+x)/2
        # (x, b, c) -> b = (x+c)/2 (covered by above if we iterate a in S)
        # Specifically, for each a in S:
        # 1. x is middle: b = 2x - a. If b in S, invalid.
        # 2. x is end: b = (a+x)/2. If (a+x) even and b in S, invalid.
        for a in S:
            # Case 1: x is middle term. a, x, b
            b = 2 * x - a
            if 1 <= b <= N and in_S[b]:
                valid = False
                break
            
            # Case 2: x is largest term. a, b, x
            # b = (a + x) / 2
            if (a + x) % 2 == 0:
                b = (a + x) // 2
                if 1 <= b <= N and b != a and in_S[b]:
                    valid = False
                    break
                    
            # Case 3: x is smallest term. x, b, a
            # b = (x + a) / 2
            # This is the same calculation as Case 2.
            
        if valid:
            S.append(x)
            in_S[x] = True
            
    # 2. Local Search
    # Try to swap larger elements with smaller ones to increase reciprocal sum.
    # Gain from swapping y for z (z < y) is 1/z - 1/y > 0.
    # We want to find the smallest possible z that can replace y.
    
    improved = True
    while improved and time.time() < DEADLINE:
        improved = False
        # Sort S in descending order to try replacing the largest (least valuable) first
        # Actually, we should try to replace any y with a smaller z.
        # Iterating y in descending order allows us to find the smallest z quickly.
        
        # To be efficient, we can iterate y in descending order.
        # For each y, we try z from 1 upwards.
        # If we find a valid z, we swap and mark improved.
        
        # We need to be careful with the order of iteration if we modify S.
        # It's safer to collect swaps or do a full pass and then apply.
        # But since swaps are independent (mostly), we can do them one by one.
        
        # Let's iterate y in descending order.
        # If we swap y for z, z is smaller, so it might affect other checks?
        # Yes, but since we are increasing the weight, it's a good move.
        
        # To avoid missing opportunities, we can do multiple passes.
        
        # Optimization: Pre-compute conflicts?
        # Let's stick to the O(|S|) check for simplicity and speed.
        
        for y in sorted(S, reverse=True):
            if time.time() > DEADLINE:
                break
                
            # Temporarily remove y
            in_S[y] = False
            S.remove(y)
            
            # Find smallest z < y not in S that is compatible
            best_z = None
            for z in range(1, y):
                if in_S[z]:
                    continue
                # Check if z is compatible with current S
                compatible = True
                for a in S:
                    # z is middle: b = 2z - a
                    b = 2 * z - a
                    if 1 <= b <= N and in_S[b]:
                        compatible = False
                        break
                    # z is end: b = (a+z)/2
                    if (a + z) % 2 == 0:
                        b = (a + z) // 2
                        if 1 <= b <= N and b != a and in_S[b]:
                            compatible = False
                            break
                if compatible:
                    best_z = z
                    break # Smallest z is best
                    
            if best_z is not None:
                in_S[best_z] = True
                S.append(best_z)
                improved = True
            else:
                # Restore y
                in_S[y] = True
                S.append(y)
                
        # After a full pass of swaps, try to add any new small elements that became available?
        # The swap logic already tries to add z < y. 
        # But what if removing y allows adding a z > y? No, we want small z.
        # What if removing y allows adding a z that was previously blocked by y?
        # The swap logic handles this: it removes y, then checks z.
        
        # However, if no swap was made for a specific y, we restored it.
        # There might be elements z that can be added *without* removing anything?
        # The initial greedy did that. But after swaps, new holes might open up.
        # Let's add a "fill" pass.
        
        if time.time() < DEADLINE:
            fill_improved = True
            while fill_improved and time.time() < DEADLINE:
                fill_improved = False
                for z in range(1, N + 1):
                    if in_S[z]:
                        continue
                    compatible = True
                    for a in S:
                        b = 2 * z - a
                        if 1 <= b <= N and in_S[b]:
                            compatible = False
                            break
                        if (a + z) % 2 == 0:
                            b = (a + z) // 2
                            if 1 <= b <= N and b != a and in_S[b]:
                                compatible = False
                                break
                    if compatible:
                        in_S[z] = True
                        S.append(z)
                        fill_improved = True
                        break # Restart loop to check from 1 again? 
                        # Or just continue? If we add z, it might block smaller ones? 
                        # No, smaller ones are already checked or in S.
                        # Adding z might block other z's. 
                        # Restarting from 1 is safer for optimality.
                        
    return sorted(S)

# EVOLVE-BLOCK-END
