"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations_with_replacement

def construct():
    n = 256
    max_size = 64
    deadline = time.time() + 75  # safety margin before 90s limit
    
    best_total = 0
    best_A = [1]
    best_B = [2]
    
    def validate(A, B):
        """Check Sidon for both sets and disjoint nonzero differences."""
        if not A or not B:
            return False
        if len(A) > max_size or len(B) > max_size:
            return False
        # Check Sidon for A
        sums_A = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                s = A[i] + A[j]
                if s in sums_A:
                    return False
                sums_A.add(s)
        
        # Check Sidon for B
        sums_B = set()
        for i in range(len(B)):
            for j in range(i, len(B)):
                s = B[i] + B[j]
                if s in sums_B:
                    return False
                sums_B.add(s)
        
        # Check disjoint nonzero differences
        diffs_A = set()
        for i in range(len(A)):
            for j in range(len(A)):
                if i != j:
                    d = abs(A[i] - A[j])
                    if d > 0:
                        diffs_A.add(d)
        
        diffs_B = set()
        for i in range(len(B)):
            for j in range(len(B)):
                if i != j:
                    d = abs(B[i] - B[j])
                    if d > 0:
                        diffs_B.add(d)
        
        if diffs_A & diffs_B:
            return False
        
        return True
    
    # === Structured algebraic construction via Z_16 x Z_16 ===
    # Map [1,256] to Z_16 x Z_16: x -> ((x-1)//16, (x-1)%16)
    # Inverse: (a,b) -> a*16 + b + 1
    
    # Strategy: Pick two subgroups/cosets in Z_16 x Z_16 whose difference sets
    # are disjoint, then project to [1,256].
    
    # A simple approach: use arithmetic progressions with different step patterns
    # In Z_16 x Z_16, the set {(i, 0) : i in S} has differences {(d, 0) : d in diff(S)}
    # The set {(0, j) : j in T} has differences {(0, d) : d in diff(T)}
    # These are disjoint if both S and T are nonempty!
    
    # But we need Sidon property. A 1D Sidon set in Z_16 can have at most ~4 elements.
    # Let's use 2D constructions instead.
    
    # Better: Use a grid-based construction.
    # Let A be points on one "line" family and B on another.
    
    # Construction 1: A = {(i, 2*i mod 16) : i in S_A}, B = {(i, 3*i mod 16) : i in S_B}
    # Differences of A: {(d, 2d mod 16) : d in diff(S_A)}
    # Differences of B: {(d, 3d mod 16) : d in diff(S_B)}
    # These are disjoint if for all d1 in diff(S_A), d2 in diff(S_B):
    #   (d1, 2*d1%16) != (d2, 3*d2%16)
    # i.e., d1 != d2 OR 2*d1%16 != 3*d2%16
    # If d1 == d2, we need 2*d1%16 != 3*d1%16, i.e., d1%16 != 0 (which is true for nonzero diffs)
    # Wait, 2*d1 == 3*d1 mod 16 means d1 == 0 mod 16, so for nonzero d1, they differ.
    # But if d1 != d2, the first components differ, so they're automatically disjoint!
    # So differences are always disjoint! Great.
    
    # Now check Sidon: A = {(i, 2i mod 16)} is a Sidon set in Z_16 x Z_16 iff
    # the map i -> (i, 2i mod 16) is injective (it is, since i is determined by first coord)
    # and pair sums are unique. Sum of (i,2i)+(j,2j) = (i+j mod 16, 2(i+j) mod 16)
    # This is determined by i+j mod 16, so we need i+j mod 16 to be unique for all pairs (i,j) with i<=j.
    # This is exactly the Sidon condition in Z_16 for the set S_A!
    
    # So we need S_A and S_B to be Sidon sets in Z_16.
    # Max Sidon set in Z_16 has size about sqrt(16) = 4, but can be larger.
    # Actually, a Sidon set in Z_m can have size up to ~sqrt(m). For m=16, max is 4 or 5.
    
    # Let's find good Sidon sets in Z_16:
    # {0, 1, 4, 9} - check: sums: 0,1,4,9, 1+1=2, 1+4=5, 1+9=10, 4+4=8, 4+9=13, 9+9=18=2... 
    # 0+9=9, 4+4=8... wait let me be more careful.
    # Pairs (i,j) with i<=j: (0,0)=0, (0,1)=1, (0,4)=4, (0,9)=9, (1,1)=2, (1,4)=5, (1,9)=10, (4,4)=8, (4,9)=13, (9,9)=18%16=2
    # 2 appears twice! Not Sidon.
    
    # Try {0, 1, 4, 10}: (0,0)=0,(0,1)=1,(0,4)=4,(0,10)=10,(1,1)=2,(1,4)=5,(1,10)=11,(4,4)=8,(4,10)=14,(10,10)=20%16=4
    # 4 appears twice! Not Sidon.
    
    # Try {0, 1, 3, 8}: (0,0)=0,(0,1)=1,(0,3)=3,(0,8)=8,(1,1)=2,(1,3)=4,(1,8)=9,(3,3)=6,(3,8)=11,(8,8)=16%16=0
    # 0 appears twice! Not Sidon.
    
    # Try {0, 1, 4, 7}: (0,0)=0,(0,1)=1,(0,4)=4,(0,7)=7,(1,1)=2,(1,4)=5,(1,7)=8,(4,4)=8
    # 8 appears twice! Not Sidon.
    
    # Try {0, 2, 5, 11}: (0,0)=0,(0,2)=2,(0,5)=5,(0,11)=11,(2,2)=4,(2,5)=7,(2,11)=13,(5,5)=10,(5,11)=16%16=0
    # 0 appears twice!
    
    # Hmm, size 4 Sidon sets in Z_16 are tricky. Let me try size 3:
    # {0, 1, 4}: (0,0)=0,(0,1)=1,(0,4)=4,(1,1)=2,(1,4)=5,(4,4)=8 - all distinct! Sidon!
    # {0, 1, 5}: (0,0)=0,(0,1)=1,(0,5)=5,(1,1)=2,(1,5)=6,(5,5)=10 - all distinct! Sidon!
    # {0, 2, 5}: (0,0)=0,(0,2)=2,(0,5)=5,(2,2)=4,(2,5)=7,(5,5)=10 - all distinct! Sidon!
    
    # For larger sets, let's try a different approach.
    # Use a 2D grid: A = {(a, b) : a in SA, b in SB} where SA x SB is a Sidon set in Z_16 x Z_16.
    # This is Sidon if SA and SB are Sidon in Z_16 and |SA|*|SB| is small enough.
    # Actually, a rectangle SA x SB is Sidon in Z_m x Z_m if and only if both SA and SB are Sidon in Z_m.
    # Proof: (a1,b1)+(a2,b2) = (a3,b3)+(a4,b4) implies a1+a2=a3+a4 and b1+b2=b3+b4.
    # If SA is Sidon, {a1,a2}={a3,a4}. Similarly {b1,b2}={b3,b4}.
    # So the pair of pairs is the same. Yes!
    
    # So let's use: A = SA x SB, B = SC x SD in Z_16 x Z_16
    # where SA, SB, SC, SD are Sidon in Z_16.
    # Differences of A: (diff(SA) x {0}) union ({0} x diff(SB)) union (diff(SA) x diff(SB))
    # Wait no. If a1=(x1,y1), a2=(x2,y2) in A, diff = (x1-x2, y1-y2) where x1-x2 in diff(SA) and y1-y2 in diff(SB).
    # So diffs(A) = diff(SA) x diff(SB) (as a set of pairs, but not exactly - it's the set of all (d1,d2) where d1 in diff(SA), d2 in diff(SB)).
    # Actually yes, diffs(A) = diff(SA) × diff(SB) as Cartesian product of the difference sets.
    
    # For disjoint differences: diff(SA) × diff(SB) and diff(SC) × diff(SD) must be disjoint.
    # This is hard to ensure unless we make one difference set empty (impossible for |S|>1).
    
    # Alternative: Use the linear construction.
    # A = {(i, f(i) mod 16) : i in S} where f is a function.
    # Diffs of A: {(d, f(i+d)-f(i) mod 16) : d in diff(S), i in S, i+d in S}
    # If f(i) = c*i mod 16, then diffs are {(d, c*d mod 16) : d in diff(S)}.
    # This is a "line" in the difference space.
    # For B with slope c': diffs are {(d, c'*d mod 16) : d in diff(S')}.
    # These are disjoint if for all d1 in diff(S), d2 in diff(S'): (d1, c*d1%16) != (d2, c'*d2%16)
    # If d1 == d2, need c*d1%16 != c'*d1%16, i.e., (c-c')*d1%16 != 0.
    # If c-c' is coprime to 16 (i.e., odd), this holds for all nonzero d1.
    # If d1 != d2, first components differ, so automatically disjoint.
    
    # So: A = {(i, i mod 16) : i in SA}, B = {(i, 2*i mod 16) : i in SB} where SA, SB are Sidon in Z_16.
    # Wait, I need c and c' to differ by an odd number. c=1, c'=2: difference is 1 (odd). Good.
    # Or c=0, c'=1: difference is 1 (odd). Good.
    
    # Let's use c=0 for A: A = {(i, 0) : i in SA}
    # c=1 for B: B = {(i, i mod 16) : i in SB}
    # Diffs of A: {(d, 0) : d in diff(SA)}
    # Diffs of B: {(d, d mod 16) : d in diff(SB)}
    # Disjoint? (d1, 0) vs (d2, d2%16): if d1==d2, need 0 != d1%16, which is true for nonzero d1.
    # If d1 != d2, first components differ. So yes, disjoint!
    
    # Now find large Sidon sets in Z_16.
    # Let's brute force small ones.
    
    def is_sidon_z16(S):
        """Check if S is a Sidon set in Z_16."""
        S = sorted(S)
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                s = (S[i] + S[j]) % 16
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    # Find all Sidon sets of size 3 and 4 in Z_16
    sidon3 = []
    for c3 in combinations_with_replacement(range(16), 3):
        if len(set(c3)) == 3 and is_sidon_z16(c3):
            sidon3.append(c3)
    
    sidon4 = []
    for c4 in combinations(range(16), 4):
        if is_sidon_z16(c4):
            sidon4.append(c4)
    
    # Try to find size 5
    sidon5 = []
    for c5 in combinations(range(16), 5):
        if is_sidon_z16(c5):
            sidon5.append(c5)
    
    # Use the largest available Sidon sets
    # For A: use SA (Sidon in Z_16), A = {(i, 0) : i in SA} -> |A| = |SA|
    # For B: use SB (Sidon in Z_16), B = {(i, i%16) : i in SB} -> |B| = |SB|
    
    # But we can also use 2D constructions for larger sets.
    # Rectangle: SA x SB where SA, SB Sidon in Z_16. Size = |SA|*|SB|.
    # But then differences are diff(SA) x diff(SB), which is a 2D set.
    # For B, if we use a different structure, we need disjointness.
    
    # Let's stick with the 1D-in-2D construction for now and maximize |SA| + |SB|.
    
    # Map from Z_16 x Z_16 to [1,256]: (a, b) -> a*16 + b + 1
    
    def z16_to_int(a, b):
        return a * 16 + b + 1
    
    # Build initial solution using the best Sidon sets found
    if sidon5:
        SA = list(sidon5[0])
        SB = list(sidon5[0])  # Can reuse same set since differences will be disjoint by construction
    elif sidon4:
        SA = list(sidon4[0])
        SB = list(sidon4[0])
    elif sidon3:
        SA = list(sidon3[0])
        SB = list(sidon3[0])
    else:
        SA = [0, 1, 4]
        SB = [0, 1, 5]
    
    # Construct A and B in [1,256]
    A_struct = [z16_to_int(i, 0) for i in SA]
    B_struct = [z16_to_int(i, i % 16) for i in SB]
    
    # Verify
    if validate(A_struct, B_struct):
        total = len(A_struct) + len(B_struct)
        if total > best_total:
            best_total = total
            best_A = A_struct[:]
            best_B = B_struct[:]
    
    # === Greedy improvement ===
    def get_diffs(aset):
        diffs = set()
        for i in range(len(aset)):
            for j in range(len(aset)):
                if i != j:
                    d = abs(aset[i] - aset[j])
                    if d > 0:
                        diffs.add(d)
        return diffs
    
    def can_add_sidon(aset, x):
        """Check if x can be added to aset maintaining Sidon property."""
        for a in aset:
            s = a + x
            for b in aset:
                if a < b:  # avoid double counting, but need to be careful
                    pass
            # More efficient: check all existing pair sums
            for b in aset:
                if a + x == a + b and b != x:
                    return False
                if a + x == b + x and a != b:
                    return False
        # Check x+x
        for a in aset:
            if a + a == x + x:
                return False
        # Full check
        new_sums = set()
        for i in range(len(aset)):
            for j in range(i, len(aset)):
                s = aset[i] + aset[j]
                new_sums.add(s)
        for a in aset:
            s = a + x
            if s in new_sums:
                return False
        if x + x in new_sums:
            return False
        return True
    
    def greedy_extend(A, B, order, deadline):
        """Try to extend A and B using elements in given order."""
        A = A[:]
        B = B[:]
        diffs_A = get_diffs(A)
        diffs_B = get_diffs(B)
        
        for x in order:
            if time.time() > deadline:
                break
            if len(A) >= max_size and len(B) >= max_size:
                break
            
            # Try adding to A
            if len(A) < max_size:
                new_diffs = set(abs(a - x) for a in A if a != x)
                if not (new_diffs & diffs_B) and can_add_sidon(A, x):
                    A.append(x)
                    diffs_A |= new_diffs
                    continue
            
            # Try adding to B
            if len(B) < max_size:
                new_diffs = set(abs(b - x) for b in B if b != x)
                if not (new_diffs & diffs_A) and can_add_sidon(B, x):
                    B.append(x)
                    diffs_B |= new_diffs
        
        return A, B
    
    # Multiple randomized greedy passes starting from structural solution
    elements = list(range(1, n + 1))
    
    for trial in range(300):
        if time.time() > deadline:
            break
        
        random.shuffle(elements)
        A, B = greedy_extend(best_A, best_B, elements, deadline)
        
        total = len(A) + len(B)
        if total > best_total and validate(A, B):
            best_total = total
            best_A = A[:]
            best_B = B[:]
    
    # Also try greedy from scratch with different orderings
    for trial in range(100):
        if time.time() > deadline:
            break
        
        random.shuffle(elements)
        A = []
        B = []
        diffs_A = set()
        diffs_B = set()
        
        for x in elements:
            if len(A) >= max_size and len(B) >= max_size:
                break
            
            if len(A) < max_size:
                new_diffs = set(abs(a - x) for a in A if a != x)
                if not (new_diffs & diffs_B) and can_add_sidon(A, x):
                    A.append(x)
                    diffs_A |= new_diffs
                    continue
            
            if len(B) < max_size:
                new_diffs = set(abs(b - x) for b in B if b != x)
                if not (new_diffs & diffs_A) and can_add_sidon(B, x):
                    B.append(x)
                    diffs_B |= new_diffs
        
        total = len(A) + len(B)
        if total > best_total and validate(A, B):
            best_total = total
            best_A = A[:]
            best_B = B[:]
    
    # Ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
    
    # Final validation
    if not validate(best_A, best_B):
        # Fallback
        best_A = [1, 2, 4, 8, 16, 32, 64, 128]
        best_B = [3, 6, 12, 24, 48, 96]
        if not validate(best_A, best_B):
            best_A = [1]
            best_B = [2]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
