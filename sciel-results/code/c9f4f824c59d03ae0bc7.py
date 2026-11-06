import numpy as np
from itertools import combinations
import random

def construct():
    # We aim for n=34
    n = 34
    
    # Initialize colors
    # Use a structured initialization for the first 32 vertices based on F_2^5
    # Vertices 0..31 map to 5-bit vectors
    # Color 0 (Red) if sum of bits of (u xor v xor w) is 0? 
    # Wait, the "sum=0 in F_2" means u+v+w=0.
    # Let's stick to the property: Red if u+v+w == 0 in F_2^5.
    
    colors = {}
    
    # Helper to get vector representation for v < 32
    def vec(v):
        return v # 0..31 are just 5-bit numbers
        
    # Initialize for n=32 part
    for a in range(32):
        for b in range(a+1, 32):
            for c in range(b+1, 32):
                # Check if a+b+c == 0 in F_2 (xor)
                # In F_2, sum is xor.
                if (a ^ b ^ c) == 0:
                    colors[(a,b,c)] = 0 # Red
                else:
                    colors[(a,b,c)] = 1 # Blue
                    
    # For vertices 32 and 33, we need to assign colors.
    # Let's initialize them randomly or with a heuristic.
    # To maximize chance, let's try to extend the algebraic structure if possible, 
    # but F_2^5 only has 32 elements. 
    # We can add 2 more elements to make a set of 34.
    # Let's just initialize the triples involving 32, 33 randomly to start, 
    # then repair.
    
    all_triples = list(combinations(range(n), 3))
    
    # Initialize remaining triples randomly
    for t in all_triples:
        if t not in colors:
            colors[t] = random.randint(0, 1)
            
    # Convert to list for efficiency
    # Map triple to index
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    color_list = [colors[t] for t in all_triples]
    
    # Helper to check if a set of 4 vertices is Red K4
    def is_red_k4(vertices):
        # vertices is a set of 4
        count = 0
        for a, b, c in combinations(vertices, 3):
            idx = triple_to_idx[(a,b,c)]
            if color_list[idx] == 0:
                count += 1
            else:
                return False
        return count == 4

    # Helper to check if a set of 5 vertices is Blue K5
    def is_blue_k5(vertices):
        count = 0
        for a, b, c in combinations(vertices, 3):
            idx = triple_to_idx[(a,b,c)]
            if color_list[idx] == 1:
                count += 1
            else:
                return False
        return count == 10

    # Repair Loop
    # We want to eliminate Blue K5s.
    # Strategy: Find a Blue K5. Pick a triple in it. Flip it to Red.
    # Check if this creates a Red K4. If yes, revert and try another triple.
    # If all triples in the K5 create Red K4s when flipped, we are stuck (for this K5).
    
    max_iterations = 5000
    for _ in range(max_iterations):
        # Find a Blue K5
        blue_k5_found = None
        # To speed up, we can sample random 5-sets or iterate.
        # Iterating all 5-sets is C(34,5) ~ 278k. Checking each is 10 lookups.
        # 2.7M lookups per pass. 5000 passes is too slow.
        # We need a faster check or fewer iterations.
        
        # Let's just do a few passes of random sampling or a full scan if fast enough.
        # 278k * 10 = 2.78M ops. In Python, this is ~1-2 seconds.
        # 120s limit allows ~50-100 full scans.
        
        found_violation = False
        for verts in combinations(range(n), 5):
            if is_blue_k5(verts):
                blue_k5_found = verts
                found_violation = True
                break
        
        if not found_violation:
            # No Blue K5s left? Check Red K4s just in case (though we started with 0 and only added Reds)
            # If no Blue K5s, we are done?
            # Wait, we might have introduced Red K4s.
            # But the task requires NO Red K4 and NO Blue K5.
            # So we must check both.
            
            # Check for Red K4
            red_k4_found = None
            for verts in combinations(range(n), 4):
                if is_red_k4(verts):
                    red_k4_found = verts
                    break
            
            if red_k4_found:
                # We have a Red K4. We need to flip one of its Red triples to Blue.
                # This might create a Blue K5.
                # This is a trade-off.
                # Given the seed has n=9, and we are aiming for 34, 
                # if we get stuck with Red K4s, we might fail.
                # However, the "Sum=0" init has 0 Red K4s.
                # Flipping Blue->Red creates Red K4s.
                # So we must be careful.
                
                # If we have a Red K4, we are in trouble if we can't fix it.
                # Let's try to flip a Red triple in the K4 to Blue.
                k4_triples = list(combinations(red_k4_found, 3))
                fixed = False
                for t in k4_triples:
                    idx = triple_to_idx[t]
                    # Flip to Blue
                    color_list[idx] = 1
                    
                    # Check if this creates a Blue K5
                    # We need to check all 5-sets containing t.
                    # This is expensive.
                    
                    # Heuristic: Just check if the current state has any Blue K5.
                    # If yes, revert.
                    # If no, keep.
                    
                    has_blue_k5 = False
                    # To speed up, only check 5-sets containing the vertices of t
                    t_verts = set(t)
                    for other in combinations(range(n), 2):
                        if other[0] in t_verts or other[1] in t_verts:
                            continue
                        # Make a 5-set: t_verts (3) + other (2)
                        # But we need to ensure distinctness.
                        # t is (a,b,c). other is (d,e).
                        # Set is {a,b,c,d,e}.
                        verts = tuple(sorted(t_verts | set(other)))
                        if len(verts) < 5: continue
                        if is_blue_k5(verts):
                            has_blue_k5 = True
                            break
                            
                    if not has_blue_k5:
                        fixed = True
                        break
                    else:
                        # Revert
                        color_list[idx] = 0
                        
                if not fixed:
                    # Stuck with Red K4 and can't fix without creating Blue K5
                    # Try to flip a different Red triple? Or give up?
                    # For now, break to avoid infinite loop.
                    break
            else:
                # No Blue K5, No Red K4. Success!
                break
                
        else:
            # Found a Blue K5
            if blue_k5_found:
                # Try to fix it by flipping a Blue triple to Red
                k5_triples = list(combinations(blue_k5_found, 3))
                fixed = False
                # Shuffle to vary which one we pick
                random.shuffle(k5_triples)
                for t in k5_triples:
                    idx = triple_to_idx[t]
                    # Flip to Red
                    color_list[idx] = 0
                    
                    # Check if this creates a Red K4
                    # Check all 4-sets containing t
                    t_verts = set(t)
                    has_red_k4 = False
                    for other in range(n):
                        if other in t_verts:
                            continue
                        # 4-set: t_verts (3) + other (1)
                        verts = tuple(sorted(t_verts | {other}))
                        if is_red_k4(verts):
                            has_red_k4 = True
                            break
                            
                    if not has_red_k4:
                        fixed = True
                        break
                    else:
                        # Revert
                        color_list[idx] = 1
                        
                if not fixed:
                    # Stuck with Blue K5
                    break

    # Final verification
    valid = True
    for verts in combinations(range(n), 4):
        if is_red_k4(verts):
            valid = False
            break
    if valid:
        for verts in combinations(range(n), 5):
            if is_blue_k5(verts):
                valid = False
                break

    if not valid:
        # If we failed to find a valid n=34, fall back to a smaller n
        # Try n=32 with the same logic? Or just return the seed?
        # Let's try to return a valid smaller n.
        # The "Sum=0" on 32 has no Red K4, but has Blue K5s.
        # So it's invalid.
        # We need a valid coloring.
        # The seed n=9 is valid.
        # Let's try to find the largest n that is valid using the repair.
        # If n=34 fails, try 33, 32...
        pass

    # For the purpose of this block, if we break out with a valid state, return it.
    # If not, we should return a known valid small n to ensure score > 0.
    # But the repair loop above breaks on failure.
    
    # Let's structure the return to ensure validity.
    # If the loop finished without finding a valid n=34, we might have an invalid state.
    # We must check validity before returning.
    
    # Re-check validity of the current color_list
    def check_validity(cl, n_val):
        # Rebuild map for this n? No, map is for n=34.
        # If we reduce n, we need to filter.
        # Let's just check the current n=34 state.
        for verts in combinations(range(n_val), 4):
            # Check Red K4
            all_red = True
            for a,b,c in combinations(verts, 3):
                if cl[triple_to_idx[(a,b,c)]] != 0:
                    all_red = False
                    break
            if all_red:
                return False
        for verts in combinations(range(n_val), 5):
            # Check Blue K5
            all_blue = True
            for a,b,c in combinations(verts, 3):
                if cl[triple_to_idx[(a,b,c)]] != 1:
                    all_blue = False
                    break
            if all_blue:
                return False
        return True

    if check_validity(color_list, n):
        return {'n': n, 'colors': color_list}
    else:
        # Fallback: Return the seed solution to guarantee a valid score
        # Or try a smaller n quickly?
        # Given the complexity, returning the seed is safest if the heuristic fails.
        # But we can try n=16 with the same algebraic init?
        # F_2^4 (n=16) Sum=0 has no Red K4. Does it have Blue K5?
        # In F_2^4, max size of cap set (no 3 sum to 0) is small?
        # Actually, we can just return the seed.
        from itertools import combinations as comb
        n_seed = 9
        return {'n': n_seed, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in comb(range(n_seed), 3)]}
