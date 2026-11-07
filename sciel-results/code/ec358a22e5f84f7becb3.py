"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt

N_MAX = 256
CANDIDATES = list(range(2, N_MAX + 1))
N_CAND = len(CANDIDATES)

def _build_conflict_graph():
    """
    Build a 2-uniform conflict graph.
    Edge (x, y) exists if there exists z in [2, 256] such that x | y*z.
    This means: for every triple (a,b,c) in our set, we need:
      - a does NOT divide b*c
      - b does NOT divide a*c  
      - c does NOT divide a*b
    
    So x and y are incompatible if there exists z in [2,256] with x | y*z.
    But wait - z must also be in our set. So the conflict depends on the set.
    
    Actually, re-reading: "For every THREE DISTINCT members a,b,c, require that a does not divide b*c."
    
    So the constraint is: for all distinct a,b,c in S: a ∤ b*c.
    
    This is a 3-uniform constraint. The directive says to model as 2-uniform where
    edge (x,y) exists if there exists ANY z in [2,256] such that x | y*z.
    
    This is a SUFFICIENT condition: if x and y are connected by an edge (i.e., ∃z: x|yz),
    then we cannot have both x and y in S (because if we also have any z in S, the constraint
    x ∤ y*z would be violated... wait no, z is the specific z that witnesses the divisibility).
    
    Actually: if x | y*z for some specific z, and if z is in S, then the triple (x, y, z) violates
    the constraint. But if z is NOT in S, then this particular witness doesn't matter.
    
    However, the directive says to use the "any z in [2,256]" version. This makes the conflict
    graph denser (more edges), so the maximum independent set in this graph is a valid set
    for the original problem (it's a sufficient condition for validity), but may not be optimal.
    
    Wait, let me re-read: "an edge connects x and y if there exists any z in [2, 256] such that x | yz"
    
    So the MIS of this graph is guaranteed to be valid for the original problem, because:
    If a,b,c are in the MIS, then for the triple (a,b,c): we need a ∤ b*c.
    If a | b*c, then there exists z = c such that a | b*z, so (a,b) would be an edge in the
    conflict graph, meaning a and b can't both be in the MIS. Contradiction.
    
    So yes, MIS of this 2-uniform graph is a valid solution. It may not be optimal, but it's valid.
    """
    # conflict[i] is a bitset (Python int) where bit j is set if candidates[i] and candidates[j] conflict
    # candidates[i] conflicts with candidates[j] if ∃z in [2,256]: candidates[i] | candidates[j] * z
    
    conflict = [0] * N_CAND
    
    for i in range(N_CAND):
        x = CANDIDATES[i]
        for j in range(i + 1, N_CAND):
            y = CANDIDATES[j]
            # Check if ∃z in [2, 256] such that x | y*z
            # y*z mod x == 0 for some z in [2, 256]
            # This is equivalent to: gcd(y, x) * z ≡ 0 (mod x) for some z in [2,256]
            # Let g = gcd(y, x). Then y = g*y', x = g*x', gcd(y', x') = 1
            # We need g*y'*z ≡ 0 (mod g*x') => y'*z ≡ 0 (mod x')
            # Since gcd(y', x') = 1, this means z ≡ 0 (mod x')
            # So we need z = k*x' for some k, with 2 ≤ z ≤ 256
            # i.e., x' ≤ 256 and there exists k ≥ 1 such that k*x' ≥ 2
            # k*x' ≥ 2 means k ≥ ceil(2/x')
            # If x' = 1: any z works (z ≥ 2)
            # If x' = 2: z ≥ 2, so z=2 works (k=1)
            # If x' > 1: need k*x' ≥ 2, so k ≥ 1, z = x'. Need x' ≥ 2 and x' ≤ 256.
            #   Also need x' ≥ 2 (already true since x' > 1) and x' ≤ 256.
            #   But also z = x' must be ≤ 256, which is true.
            #   And z ≥ 2, so x' ≥ 2.
            # 
            # So: conflict exists iff x' ≤ 256, where x' = x / gcd(x, y)
            # i.e., x / gcd(x,y) ≤ 256
            # i.e., x ≤ 256 * gcd(x,y)
            # 
            # Wait, that's always true since x ≤ 256 and gcd(x,y) ≥ 1, so x ≤ 256*1 = 256.
            # Hmm, that means EVERY pair conflicts? Let me recheck.
            #
            # x' = x / gcd(x,y). We need x' ≤ 256 AND x' ≥ 2 (for z = x' to be in [2,256]).
            # Actually z = k*x' with k ≥ 1. Minimum z is x' (when k=1).
            # We need x' ≤ 256 (so z=x' is in range) and x' ≥ 2 (so z=x' ≥ 2).
            # If x' = 1, then z = k*1 = k, and we need k ≥ 2, so z=2 works.
            # So x' = 1 also works!
            # 
            # So conflict exists iff x' ≤ 256, i.e., x/gcd(x,y) ≤ 256.
            # Since x ≤ 256 and gcd(x,y) ≥ 1, x/gcd(x,y) ≤ x ≤ 256. Always true!
            # 
            # Wait, that means every pair (x,y) with x,y in [2,256] has a conflict?
            # Let me verify with a small example. x=6, y=4. gcd=2, x'=3. z=3: 6|4*3=12? Yes.
            # x=5, y=3. gcd=1, x'=5. z=5: 5|3*5=15? Yes.
            # x=2, y=3. gcd=1, x'=2. z=2: 2|3*2=6? Yes.
            # 
            # So indeed, for ANY x,y in [2,256], there always exists z in [2,256] such that x|y*z.
            # This means the conflict graph is a complete graph, and the MIS has size 1!
            # 
            # That can't be right. Let me re-read the problem.
            #
            # "For every THREE DISTINCT members a,b,c, require that a does not divide b*c."
            #
            # So the constraint is on triples. The 2-uniform reduction should be:
            # x and y are incompatible if for ALL z in [2,256], x | y*z.
            # No wait, that's not right either.
            #
            # Let me think again. The directive says: "an edge connects x and y if there exists
            # any z in [2, 256] such that x | yz"
            #
            # But as I showed, this is always true. So the directive's approach as stated
            # would give a trivial answer. Let me re-read more carefully.
            #
            # Hmm, maybe I'm misunderstanding. Perhaps the intent is different. Let me re-read:
            # "Model the problem as a maximum independent set in a standard 2-uniform conflict
            # graph where an edge connects x and y if there exists any z in [2, 256] such that
            # x | yz, thereby reducing the 3-uniform hypergraph constraint to a pairwise
            # incompatibility"
            #
            # I think the issue is that the "any z" should be interpreted as: x and y cannot
            # coexist in a valid set because no matter what z we pick from the remaining candidates,
            # the constraint would be violated. But that's not what "there exists any z" means.
            #
            # Actually, I think the correct interpretation for a valid reduction is:
            # x and y are incompatible if for EVERY z in [2,256]\{x,y}, x | y*z.
            # Because if x,y are in S, and there's any z in S\{x,y}, the constraint must hold.
            # So if x|y*z for ALL z, then x and y can't both be in S (unless S has only {x,y},
            # but then there's no triple).
            #
            # Wait, but if S = {x, y}, there are no three distinct members, so the constraint is
            # vacuously satisfied. So x and y CAN coexist if they're the only two elements.
            #
            # I think the directive's approach is flawed as stated. Let me instead use the
            # original approach but optimize it. The current program gets 128 (upper half).
            # Let me try to improve the local search.
            
            # For now, let me just check if x/gcd(x,y) <= 256 (which is always true)
            # This confirms the directive's approach as literally stated doesn't work.
            # Let me use a different approach: keep the original design but improve it.
            pass
    
    # Since the 2-uniform reduction as stated in the directive is degenerate (complete graph),
    # I'll keep the original design family and improve the search.
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    n = N_MAX
    all_candidates = list(range(2, n + 1))
    
    # Start with upper half [129, 256] - antichain, no element divides another
    S = set(range(129, n + 1))
    
    # Try to add elements from lower half in descending order
    for x in range(128, 1, -1):
        if time.time() > deadline:
            break
        if x not in S:
            if _can_add(S, x):
                S.add(x)
    
    best_S = S.copy()
    best_size = len(S)
    
    # Local search: try to add elements, then try swaps
    improved = True
    iteration = 0
    while improved and time.time() < deadline:
        improved = False
        iteration += 1
        
        # Try to add any element
        added = False
        for x in all_candidates:
            if time.time() > deadline:
                break
            if x not in S:
                if _can_add(S, x):
                    S.add(x)
                    added = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = S.copy()
        if added:
            continue
        
        # If no add possible, try swaps
        s_list = list(S)
        c_list = [x for x in all_candidates if x not in S]
        
        if not s_list or not c_list:
            break
        
        random.shuffle(s_list)
        random.shuffle(c_list)
        
        swaps_tried = 0
        max_swaps = 50 + iteration * 10
        
        for y in s_list:
            if swaps_tried >= max_swaps or time.time() > deadline:
                break
            for x in c_list:
                if swaps_tried >= max_swaps or time.time() > deadline:
                    break
                if x == y:
                    continue
                
                S_trial = S - {y}
                if _can_add(S_trial, x):
                    S = S_trial | {x}
                    improved = True
                    swaps_tried += 1
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = S.copy()
                    break
            if improved:
                break
    
    # Final verification
    if not _is_valid_set(best_S):
        best_S = set(range(129, n + 1))
    
    return [int(x) for x in best_S]

def _can_add(S, x):
    """Check if x can be added to set S without violating constraints."""
    s_list = list(S)
    n = len(s_list)
    for i in range(n):
        a = s_list[i]
        for j in range(i + 1, n):
            b = s_list[j]
            if (a * b) % x == 0:
                return False
            if (x * b) % a == 0:
                return False
            if (x * a) % b == 0:
                return False
    return True

def _is_valid_set(S):
    """Verify the full 3-wise constraint for set S."""
    s_list = list(S)
    n = len(s_list)
    for i in range(n):
        a = s_list[i]
        for j in range(i + 1, n):
            b = s_list[j]
            for k in range(j + 1, n):
                c = s_list[k]
                if (b * c) % a == 0:
                    return False
                if (a * c) % b == 0:
                    return False
                if (a * b) % c == 0:
                    return False
    return True

# EVOLVE-BLOCK-END
