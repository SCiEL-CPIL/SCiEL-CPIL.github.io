import numpy as np
import itertools
import time
import random

N = 16
NUM_POINTS = N * N

# Precomputation
# Iso[P, A, B] is True if P, A, B form an isosceles triangle
# We use a 3D boolean array or a list of lists of lists.
# For speed, a flat numpy array or a list of sets might be better.
# Let's use a list of sets for QsForPair.

def _precompute():
    # Iso[P][A][B]
    # To save memory and time, we only store P < A < B? No, order doesn't matter for geometry.
    # But for "QsForPair[P, A]", we need to know which Q are isosceles with P and A.
    
    # Let's compute distances first
    # d2[i][j]
    
    # Efficient check for isosceles:
    # d2(P,A) == d2(P,B) or d2(P,A) == d2(A,B) or d2(P,B) == d2(A,B)
    
    # We will build:
    # 1. `valid_pairs_for_P`: For a new point P, we need to check against all pairs in S.
    #    Actually, we just need to check if P conflicts with S.
    #    Conflict(P, S) exists if there are A, B in S such that Iso(P, A, B).
    
    # 2. `QsForPair[P, A]`: List of Q such that Iso(Q, P, A).
    
    # Precomputation loop
    # 256^3 is 16 million iterations. In Python, this might take a few seconds.
    # Let's optimize.
    
    points = [(x, y) for y in range(N) for x in range(N)]
    
    # d2[i][j]
    d2 = [[0]*NUM_POINTS for _ in range(NUM_POINTS)]
    for i in range(NUM_POINTS):
        xi, yi = points[i]
        for j in range(i+1, NUM_POINTS):
            xj, yj = points[j]
            d2[i][j] = (xi-xj)**2 + (yi-yj)**2
            d2[j][i] = d2[i][j]
            
    # QsForPair: list of lists of lists
    # QsForPair[P][A] = list of Qs
    QsForPair = [[[] for _ in range(NUM_POINTS)] for __ in range(NUM_POINTS)]
    
    # Also, for independence check, we need to know if P can be added to S.
    # This requires checking all pairs in S.
    
    # Let's precompute `BadPairsForP[P]`?
    # If we add P, it is invalid if any pair (A,B) in S is a "bad pair" for P.
    # Bad pair for P is a pair (A,B) such that Iso(P, A, B).
    # So `BadPairsForP[P]` is a set of (min(A,B), max(A,B)).
    
    BadPairsForP = [set() for _ in range(NUM_POINTS)]
    
    for P in range(NUM_POINTS):
        for A in range(NUM_POINTS):
            if A == P: continue
            for B in range(A+1, NUM_POINTS):
                if B == P: continue
                # Check if P, A, B is isosceles
                dPA = d2[P][A]
                dPB = d2[P][B]
                dAB = d2[A][B]
                if dPA == dPB or dPA == dAB or dPB == dAB:
                    # P, A, B is isosceles
                    # Add (A, B) to BadPairsForP[P]
                    BadPairsForP[P].add((min(A, B), max(A, B)))
                    # Add P to QsForPair[A][B] and QsForPair[B][A]?
                    # QsForPair is symmetric in the pair?
                    # QsForPair[P][A] contains Q.
                    # So if Iso(Q, P, A), then Q is in QsForPair[P][A] and Q is in QsForPair[A][P].
                    QsForPair[A][P].append(P) # Wait, Q is P here? No.
                    # The triple is (Q, P, A).
                    # So for pair (P, A), the third point is Q.
                    # Here the third point is P? No, the points are P, A, B.
                    # So for pair (A, B), the third point is P.
                    # So P should be added to QsForPair[A][B] and QsForPair[B][A].
                    
    # Re-do the loop correctly for QsForPair
    QsForPair = [[[] for _ in range(NUM_POINTS)] for __ in range(NUM_POINTS)]
    for Q in range(NUM_POINTS):
        for P in range(NUM_POINTS):
            if P == Q: continue
            for A in range(P+1, NUM_POINTS):
                if A == Q: continue
                dQP = d2[Q][P]
                dQA = d2[Q][A]
                dPA = d2[P][A]
                if dQP == dQA or dQP == dPA or dQA == dPA:
                    # Q, P, A is isosceles
                    # Q is the "third point" for pair (P, A)
                    QsForPair[P][A].append(Q)
                    QsForPair[A][P].append(Q)
                    
    return points, d2, BadPairsForP, QsForPair

# Global variables
points, d2, BadPairsForP, QsForPair = _precompute()

def _is_valid_addition(P, S_mask):
    # S_mask is a set or list of points in S
    # Check if any pair in S is in BadPairsForP[P]
    # For speed, we can iterate over A in S and check if (A, B) is bad.
    # But BadPairsForP[P] is a set of tuples.
    # We need to check if ANY pair in S is in BadPairsForP[P].
    
    # Optimization: 
    # If |S| is small, iterate over pairs in S.
    # If |S| is large, iterate over BadPairsForP[P] and check if both are in S.
    # BadPairsForP[P] size: 
    # For a point P, how many isosceles triangles does it participate in?
    # Roughly O(N^2) or less.
    # |S| is likely < 50.
    # Pairs in S: |S|^2 / 2.
    
    s_list = list(S_mask)
    n_s = len(s_list)
    
    # Strategy: Iterate over BadPairsForP[P]
    for pair in BadPairsForP[P]:
        A, B = pair
        # Check if A in S and B in S
        # Using a set for S is O(1) lookup
        if A in S_mask and B in S_mask:
            return False
    return True

def _get_uncovered_count(S_mask, Counts):
    # Counts[Q] is the number of pairs in S that form isosceles with Q
    # Q is uncovered if Q not in S and Counts[Q] == 0
    unc = 0
    for Q in range(NUM_POINTS):
        if Q not in S_mask and Counts[Q] == 0:
            unc += 1
    return unc

def construct():
    start_time = time.time()
    deadline = start_time + 290 # 10s margin
    
    # Initial greedy construction
    # Randomize order to find different starting points
    random.seed(42)
    
    best_S = None
    best_cost = float('inf')
    
    # We will run multiple restarts
    num_restarts = 100
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        S = set()
        # Greedy add
        # Randomize point order
        pts = list(range(NUM_POINTS))
        random.shuffle(pts)
        
        for P in pts:
            if _is_valid_addition(P, S):
                S.add(P)
                
        # Initialize Counts
        Counts = [0] * NUM_POINTS
        # Compute initial Counts
        # For each Q not in S, count pairs (A, B) in S such that Iso(Q, A, B)
        # Using QsForPair:
        # For each pair (A, B) in S, for each Q in QsForPair[A][B], increment Counts[Q]
        # This is O(|S|^2 * avg_Qs)
        
        s_list = list(S)
        for i in range(len(s_list)):
            A = s_list[i]
            for j in range(i+1, len(s_list)):
                B = s_list[j]
                for Q in QsForPair[A][B]:
                    Counts[Q] += 1
                    
        # Simulated Annealing
        T = 10.0
        cooling = 0.99
        min_T = 0.01
        
        current_S = S
        current_Counts = Counts
        current_unc = _get_uncovered_count(current_S, current_Counts)
        current_cost = len(current_S) + 100 * current_unc
        
        best_local_S = set(current_S)
        best_local_cost = current_cost
        
        steps = 0
        while T > min_T:
            if time.time() > deadline:
                break
                
            steps += 1
            # Pick a random point
            P = random.randint(0, NUM_POINTS - 1)
            
            if P in current_S:
                # Try to remove
                new_S = set(current_S)
                new_S.remove(P)
                
                # Update Counts
                # For each A in new_S, for each Q in QsForPair[P][A], decrement Counts[Q]
                # Note: QsForPair[P][A] contains Q such that Iso(Q, P, A)
                # Removing P removes pairs (P, A) for all A in S.
                # So for each A in new_S, we look at QsForPair[P][A] and decrement.
                new_Counts = current_Counts[:]
                for A in new_S:
                    for Q in QsForPair[P][A]:
                        new_Counts[Q] -= 1
                        
                new_unc = _get_uncovered_count(new_S, new_Counts)
                new_cost = len(new_S) + 100 * new_unc
                
                delta = new_cost - current_cost
                if delta < 0 or random.random() < pow(2.718, -delta / T):
                    current_S = new_S
                    current_Counts = new_Counts
                    current_unc = new_unc
                    current_cost = new_cost
                    
                    if current_cost < best_local_cost:
                        best_local_cost = current_cost
                        best_local_S = set(current_S)
            else:
                # Try to add
                if _is_valid_addition(P, current_S):
                    new_S = set(current_S)
                    new_S.add(P)
                    
                    # Update Counts
                    # Adding P adds pairs (P, A) for all A in current_S.
                    # For each A in current_S, for each Q in QsForPair[P][A], increment Counts[Q]
                    new_Counts = current_Counts[:]
                    for A in current_S:
                        for Q in QsForPair[P][A]:
                            new_Counts[Q] += 1
                            
                    new_unc = _get_uncovered_count(new_S, new_Counts)
                    new_cost = len(new_S) + 100 * new_unc
                    
                    delta = new_cost - current_cost
                    if delta < 0 or random.random() < pow(2.718, -delta / T):
                        current_S = new_S
                        current_Counts = new_Counts
                        current_unc = new_unc
                        current_cost = new_cost
                        
                        if current_cost < best_local_cost:
                            best_local_cost = current_cost
                            best_local_S = set(current_S)
                            
            T *= cooling
            
        # End SA
        # Check if best_local_S is maximal
        # If not, we can try to repair it or just discard if it's worse than best_S
        # Actually, if best_local_cost < best_cost, we accept it.
        # But we must ensure the final result is maximal.
        # The SA might end with a non-maximal set if the penalty wasn't high enough or it got stuck.
        # We should enforce maximality in the final step.
        
        # Repair: Add points until maximal
        # This might increase size, but ensures validity.
        repaired_S = set(best_local_S)
        # Recompute counts for repaired_S
        r_Counts = [0] * NUM_POINTS
        r_list = list(repaired_S)
        for i in range(len(r_list)):
            A = r_list[i]
            for j in range(i+1, len(r_list)):
                B = r_list[j]
                for Q in QsForPair[A][B]:
                    r_Counts[Q] += 1
                    
        # Find uncovered points
        while True:
            uncovered = [Q for Q in range(NUM_POINTS) if Q not in repaired_S and r_Counts[Q] == 0]
            if not uncovered:
                break
            if time.time() > deadline:
                break
            # Pick a random uncovered Q
            Q = random.choice(uncovered)
            # We need to add points to cover Q.
            # Q is covered if there is A, B in S such that Iso(Q, A, B).
            # We can try to add a point P that helps cover Q.
            # Or just add a random valid point?
            # Better: Find a pair (A, B) such that Iso(Q, A, B).
            # If A, B are not in S, we can try to add them.
            # But adding A might not cover Q if B is not in S.
            # So we need to add a "witness" pair.
            
            # Find all pairs (A, B) such that Iso(Q, A, B).
            # These are in QsForPair? No, QsForPair[A][B] contains Q.
            # So we need to find A, B such that Q in QsForPair[A][B].
            # This is the reverse lookup.
            # We didn't precompute this.
            
            # Alternative: Just try to add random valid points until Q is covered.
            # This is slow.
            
            # Let's try a specific strategy:
            # Pick a random point P not in S.
            # If valid, add it.
            # Repeat until maximal.
            # This is the "Greedy Repair".
            
            added = False
            candidates = [P for P in range(NUM_POINTS) if P not in repaired_S]
            random.shuffle(candidates)
            for P in candidates:
                if _is_valid_addition(P, repaired_S):
                    repaired_S.add(P)
                    # Update counts
                    for A in repaired_S:
                        if A == P: continue
                        for Q2 in QsForPair[P][A]:
                            r_Counts[Q2] += 1
                    added = True
                    break
            if not added:
                # Should not happen if S is not maximal?
                # If no point can be added, S is maximal.
                # But we found uncovered Q. Contradiction?
                # If Q is uncovered, it means no pair in S covers it.
                # If we can't add any point, it means all points P conflict with S.
                # This is possible.
                # In that case, the set is maximal by definition?
                # No, maximal means NO point can be added.
                # If no point can be added, it IS maximal.
                # But wait, if Q is uncovered, can we add Q?
                # If we add Q, it must not form isosceles with any pair in S.
                # If Q is uncovered, it means NO pair in S forms isosceles with Q.
                # So Q can ALWAYS be added if it is not in S and is uncovered?
                # Wait.
                # Condition for adding Q: No pair (A, B) in S such that Iso(Q, A, B).
                # This is exactly the condition for Q being uncovered!
                # So if Q is uncovered, we CAN add Q.
                # So the loop "if not added" will never trigger if uncovered is not empty.
                # Because Q itself is a valid candidate to add.
                pass
                
        # Final check
        if len(repaired_S) < best_cost or (best_S is None and len(repaired_S) < 100):
             # We need to compare correctly
             # best_cost was len + 100*unc.
             # Here unc is 0.
             # So we compare len(repaired_S) with best_S size?
             # Let's keep the smallest maximal set found.
             if best_S is None or len(repaired_S) < len(best_S):
                 best_S = repaired_S
                 
    if best_S is None:
        # Fallback to initial greedy
        S = set()
        for P in range(NUM_POINTS):
            if _is_valid_addition(P, S):
                S.add(P)
        return list(S)
        
    return list(best_S)

# Test run
if __name__ == "__main__":
    res = construct()
    print(len(res))
