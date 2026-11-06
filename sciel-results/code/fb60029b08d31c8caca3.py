"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    
    # Greedy: try to build a large set by adding numbers that don't violate constraints
    # Start with a good initial set (e.g., large numbers which are less likely to divide sums of smaller ones)
    # But we need to maximize cardinality, so let's try a smart greedy order.
    # Numbers > 96 cannot divide a sum of 3 numbers <= 96 (max sum 96+95+94=285, 285/96 < 3, so only possible if sum is 0 or 96*1, 96*2).
    # Actually, if a > max_sum_of_3_others, then a cannot divide the sum (unless sum is 0, which is impossible for positive integers).
    # Max sum of 3 distinct others from [1,192] is 192+191+190 = 573.
    # So any number a > 573 is safe. But max is 192.
    # However, if we pick large numbers, their sums are large, making it harder for smaller numbers to divide them? No, smaller numbers dividing large sums is easier (more multiples).
    # Actually, if a is small, it divides many sums. If a is large, it divides fewer sums (only specific multiples).
    # So large numbers are "safer" to include.
    
    # Let's try to include as many large numbers as possible.
    # Sort candidates in descending order.
    candidates = list(range(192, 0, -1))
    
    selected = []
    
    for x in candidates:
        if time.time() > deadline:
            break
            
        # Check if adding x violates any constraint
        # Constraints:
        # 1. For any a in selected, x + (sum of 2 others from selected) must not be divisible by a? 
        #    No, the constraint is: "No selected integer may divide the sum of any k=3 DISTINCT OTHER selected integers."
        #    So if we add x, we need to check:
        #    - For every a in selected: does a divide sum(x, b, c) for any b,c in selected, b!=c, b!=x, c!=x?
        #      Since x is new, the "3 distinct others" for a can include x.
        #      Previously, a only checked sums of 3 from selected. Now a must also check sums involving x and 2 from selected.
        #    - For x itself: does x divide sum(a, b, c) for any a,b,c in selected?
        
        # Efficient check:
        # Precompute pair sums of selected?
        # Let S = selected.
        # Check 1: For each a in S, check if there exist b, c in S (b!=c) such that (x + b + c) % a == 0.
        # Check 2: Check if there exist a, b, c in S (distinct) such that (a + b + c) % x == 0.
        
        # Since |S| is small (max 48), and we iterate ~192 times, total checks are manageable.
        # But let's optimize slightly.
        
        # Check 2 first (often faster to fail if x is small and S has large sums? Or vice versa?)
        # Actually, if x is large, Check 2 is likely to pass (x > sum(a,b,c) for most triples if S has small numbers, but S has large numbers too).
        # If x is small, Check 2 is likely to fail (many multiples).
        
        # Let's just do the full check.
        
        valid = True
        
        # Check for existing elements a in selected
        for a in selected:
            # Need to check if x + b + c is divisible by a for any pair b,c in selected
            # This is equivalent to: does there exist b,c in selected such that (x + b + c) % a == 0?
            # We can precompute all pair sums mod a? Or just iterate.
            # |selected| <= 48, so pairs are ~1100. 192 * 48 * 1100 ~ 10M operations. Might be slow in Python.
            
            # Optimization: For a fixed a, we want to know if (x + b + c) % a == 0.
            # This means (b + c) % a == (-x) % a.
            # So we can precompute the set of pair sums modulo a.
            # But a changes.
            
            # Alternative: Iterate b, c.
            found = False
            for i, b in enumerate(selected):
                for c in selected[i+1:]:
                    if (x + b + c) % a == 0:
                        found = True
                        break
                if found:
                    break
            if found:
                valid = False
                break
        
        if not valid:
            continue
            
        # Check for x itself
        # Does x divide a + b + c for any triple a,b,c in selected?
        found = False
        # If x is large, a+b+c might be smaller than x, so only 0 is a multiple, but sums are positive.
        # So if x > max_sum_of_3_in_selected, it's automatically safe for Check 2.
        max_sum_3 = sum(sorted(selected, reverse=True)[:3]) if len(selected) >= 3 else 0
        if x <= max_sum_3:
            for i, a in enumerate(selected):
                for j, b in enumerate(selected):
                    if j == i: continue
                    for k, c in enumerate(selected):
                        if k == i or k == j: continue
                        if (a + b + c) % x == 0:
                            found = True
                            break
                        if found: break
                    if found: break
                if found: break
        
        if not found:
            selected.append(x)
            
        if len(selected) >= 48:
            break
            
    return selected
# EVOLVE-BLOCK-END
