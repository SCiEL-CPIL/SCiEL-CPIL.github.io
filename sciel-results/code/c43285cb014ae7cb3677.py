"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Greedy construction with bitwise mask optimization.
    
    We build the coloring position by position (1..128). For each new position x,
    we maintain a bitmask of which colors are valid. A color c is valid for position x
    if:
      1. For every d such that x-3d >= 1: the set {c, color[x-d], color[x-2d], color[x-3d]} has size >= 3.
         This means NOT (all three previous are equal to c) AND NOT (exactly two of the three previous are c and the third is some other value such that total distinct < 3).
         Actually, distinct count < 3 means: the set has size 1 or 2.
         Size 1: all four are c -> color[x-d]=color[x-2d]=color[x-3d]=c
         Size 2: exactly two distinct values among the four. Since c is one of them, the other three must be from {c, y} for some y != c.
         So size 2 happens when: among {color[x-d], color[x-2d], color[x-3d]}, at most one value different from c, and that value is the same.
         i.e., the multiset of the three previous colors, when combined with c, has at most 2 distinct values.
         This is equivalent to: the three previous colors are all c (size 1 case), OR the three previous colors are all the same value y != c (then {c,y} has size 2), OR two are c and one is y (then {c,y} size 2), OR one is c and two are y (size 2), OR all three are different but one is c and the other two are the same y (size 2).
         
         Let me re-derive: We need |{c, a, b, c}| >= 3 where a=col[x-d], b=col[x-2d], c_prev=col[x-3d]. Wait, let's use different names.
         Let p1 = colors[x-1-d], p2 = colors[x-1-2d], p3 = colors[x-1-3d]. New color is c.
         We need |{c, p1, p2, p3}| >= 3.
         This fails iff |{c, p1, p2, p3}| <= 2.
         Case 1: |{c, p1, p2, p3}| = 1 -> p1=p2=p3=c.
         Case 2: |{c, p1, p2, p3}| = 2 -> The set {p1, p2, p3} is a subset of {c, y} for some y != c, and not all equal to c.
           Subcases for {p1, p2, p3}:
           - All three are y (y != c): set is {c, y}, size 2. FAIL.
           - Two are c, one is y: set is {c, y}, size 2. FAIL.
           - One is c, two are y: set is {c, y}, size 2. FAIL.
           - All three are c: already covered in Case 1.
           
         So a color c is INVALID for progression (d) if:
           (p1 == c and p2 == c and p3 == c) OR
           (p1 == p2 == p3 and p1 != c) OR
           (exactly two of p1,p2,p3 are c and the third is some y != c) OR
           (exactly one of p1,p2,p3 is c and the other two are equal to some y != c)
           
         Wait, let's simplify. |{c, p1, p2, p3}| <= 2 iff the number of distinct values in {p1, p2, p3} is at most 2, AND c is in that set OR c is not in that set but the set has size 1.
         Actually, easier: |{c, p1, p2, p3}| = |{p1, p2, p3} ∪ {c}|.
         Let S = {p1, p2, p3}. If c ∈ S, then |S ∪ {c}| = |S|. So we need |S| >= 3. But S has at most 3 elements, so |S| must be 3.
         If c ∉ S, then |S ∪ {c}| = |S| + 1. So we need |S| + 1 >= 3 => |S| >= 2.
         
         So:
         - If c is in {p1, p2, p3}: valid iff p1, p2, p3 are all distinct.
         - If c is not in {p1, p2, p3}: valid iff p1, p2, p3 are not all the same (i.e., at least two distinct).
         
         Let's verify:
         - c in S, p1=p2=p3: S={p1}, c=p1, |S∪{c}|=1. Invalid. Correct.
         - c in S, p1=p2!=p3: S={p1,p3}, c=p1 (say), |S∪{c}|=2. Invalid. Correct.
         - c in S, p1,p2,p3 distinct: S has 3 elements, c is one of them, |S∪{c}|=3. Valid. Correct.
         - c not in S, p1=p2=p3: S={p1}, c!=p1, |S∪{c}|=2. Invalid. Correct.
         - c not in S, p1=p2!=p3: S={p1,p3}, c not in S, |S∪{c}|=3. Valid. Correct.
         - c not in S, p1,p2,p3 distinct: S has 3 elements, c not in S, |S∪{c}|=4. Valid. Correct.
         
         So the condition for c to be valid for a given d is:
         Let S = {p1, p2, p3}.
         If c in S: require p1, p2, p3 all distinct.
         If c not in S: require not (p1==p2==p3).
         
         Now, for the bitwise approach:
         For each position x, we have a list of already assigned colors for positions 1..x-1.
         For each d, we look at p1, p2, p3.
         We can precompute for each d whether the "all distinct" condition holds for S, and whether "not all same" holds.
         
         Let:
         all_distinct_d = (p1 != p2) and (p1 != p3) and (p2 != p3)
         not_all_same_d = not (p1 == p2 == p3)
         
         For a candidate color c:
         If all_distinct_d is True: then c is valid for this d regardless of whether c is in S or not? 
           Wait. If all_distinct_d is True, then |S|=3.
           If c in S: valid (needs |S|>=3, yes).
           If c not in S: valid (needs |S|>=2, yes).
           So if all_distinct_d is True, c is always valid for this d.
         If all_distinct_d is False:
           If c in S: invalid (needs |S|>=3, but |S|<3).
           If c not in S: valid iff not_all_same_d is True.
           So if all_distinct_d is False and not_all_same_d is True (i.e., S has exactly 2 distinct values):
             c is valid iff c not in S.
           If all_distinct_d is False and not_all_same_d is False (i.e., S has 1 distinct value, all same):
             c is valid iff c not in S. (Since c not in S => |S∪{c}|=2, which is <3. Wait.)
             Let's recheck: c not in S, S={p}. |S∪{c}| = 2. We need >=3. So INVALID.
             So if S has 1 distinct value, c is invalid whether in S or not? 
             c in S: |S|=1, invalid.
             c not in S: |S∪{c}|=2, invalid.
             So if p1==p2==p3, NO color c is valid for this d? That can't be right.
             
         Let me re-read the problem: "Every nonconstant four-term progression of positions must use AT LEAST three distinct colors."
         A four-term progression is x, x+d, x+2d, x+3d. Nonconstant means d != 0.
         The colors are color[x], color[x+d], color[x+2d], color[x+3d].
         We are constructing color[x+3d] given color[x], color[x+d], color[x+2d].
         Let a=color[x], b=color[x+d], c=color[x+2d], and new color = z = color[x+3d].
         We need |{a, b, c, z}| >= 3.
         
         My previous notation was p1=colors[x-1-d], p2=colors[x-1-2d], p3=colors[x-1-3d].
         So x_new = x-1. x_new - d = x-1-d, x_new - 2d = x-1-2d, x_new - 3d = x-1-3d.
         So the progression is (x-1-3d, x-1-2d, x-1-d, x-1).
         So a = colors[x-1-3d], b = colors[x-1-2d], c = colors[x-1-d], z = colors[x-1].
         S = {a, b, c} = {p3, p2, p1}. Same set.
         
         So my analysis holds. If a=b=c, then |{a,b,c,z}| = |{a,z}| which is 1 or 2, always < 3.
         So if the first three are the same, it's IMPOSSIBLE to satisfy the condition?
         That means we must ensure that we never have three consecutive terms in a progression all the same color.
         
         So the constraint is: for every d, we must NOT have color[x-1-3d] == color[x-1-2d] == color[x-1-d].
         This is a hard constraint on the previous three colors, independent of z.
         If this happens, no valid z exists. So the greedy algorithm must avoid creating such a situation.
         
         Given that the previous three are not all the same, what colors z are valid?
         Let S = {a, b, c}.
         If |S| = 3 (all distinct): any z is valid. |{a,b,c,z}| >= 3 always.
         If |S| = 2 (say {u, v}):
           If z is u or v: |{u,v,z}| = 2. Invalid.
           If z is w not in {u,v}: |{u,v,w}| = 3. Valid.
         If |S| = 1 (all same, say {u}):
           We already established this is impossible to fix, so this state should never occur.
           
         So the valid colors for z are:
         - If |S| = 3: all 128 colors.
         - If |S| = 2: all colors NOT in S.
         - If |S| = 1: no colors (should not happen).
         
         So for each d, the invalid colors are:
         - If |S| = 2: the two colors in S.
         - If |S| = 3: none.
         
         So the set of valid colors for position x is the intersection over all d of:
         - All colors, if |S_d| = 3.
         - All colors except S_d, if |S_d| = 2.
         
         We can compute a bitmask of valid colors. Start with all 128 bits set.
         For each d, if |S_d| == 2, clear the bits corresponding to the two colors in S_d.
         Then pick the lowest set bit.
         
         This is much faster!
         
         Let's implement this.
    """
    colors = [0] * 128  # 1-indexed, colors[0] unused
    # We'll use a list of integers for colors
    
    for x in range(1, 129):
        # For the first 3 positions, no d is possible (x-1-3d >= 1 => 3d <= x-2 => d <= (x-2)//3)
        # For x=1,2,3: no d. So any color is valid. Pick 0.
        if x <= 3:
            colors[x] = 0
            continue
        
        # Start with all 128 colors valid
        valid_mask = (1 << 128) - 1  # 128 bits set
        
        max_d = (x - 2) // 3
        for d in range(1, max_d + 1):
            # p1 = colors[x-1-d], p2 = colors[x-1-2d], p3 = colors[x-1-3d]
            p1 = colors[x - 1 - d]
            p2 = colors[x - 1 - 2 * d]
            p3 = colors[x - 1 - 3 * d]
            
            # Check if p1, p2, p3 are all the same
            if p1 == p2 == p3:
                # This should not happen if we've been careful, but if it does, no valid color
                # For now, we'll handle it by picking any color, but this will likely fail later
                # Actually, this state is invalid, so we should avoid it.
                # But in greedy, we might hit it. Let's just pick 0 for now.
                pass
            else:
                # Count distinct values in {p1, p2, p3}
                s = {p1, p2, p3}
                if len(s) == 2:
                    # Clear the two colors in s from valid_mask
                    for c in s:
                        valid_mask &= ~(1 << c)
                # If len(s) == 3, no restriction
        
        # Pick the lowest set bit
        if valid_mask == 0:
            # No valid color, pick 0 as fallback (will likely lead to failure, but avoid crash)
            colors[x] = 0
        else:
            # Find index of lowest set bit
            colors[x] = (valid_mask & -valid_mask).bit_length() - 1
    
    # Return 0-indexed list of colors for positions 1..128
    return colors[1:]
# EVOLVE-BLOCK-END
