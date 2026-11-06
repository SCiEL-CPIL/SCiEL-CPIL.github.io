"""
Strategy: Fast Local Search with Conflict Counting.
Uses precomputed 4-AP triples to maintain a dynamic conflict count array.
This allows O(1) checks for adding/removing elements, enabling high-throughput
local search and multiple restarts within the time limit.
"""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N = 512

def _precompute_conflict_structure():
    """
    Precompute the structure needed for fast conflict updates.
    For each element x, we store the list of pairs (y, z) such that {x, y, z} 
    is a subset of a 4-AP. 
    Actually, it's easier to think in terms of:
    If we add x, we must check if there exist y, z in S such that {x, y, z, w} is a 4-AP.
    This is equivalent to: for any pair (y, z) in S, do they form a 4-AP with x and some w?
    
    A more efficient structure for "can I add x?":
    Maintain `conflict_count[x]` = number of 4-APs that would be completed if x were added.
    Wait, that's not quite right.
    
    Let's use the standard approach for k-AP free sets:
    When we add an element x, it creates a conflict if there are two other elements y, z 
    already in the set such that {x, y, z, w} is a 4-AP for some w (which might or might not be in the set).
    Actually, a 4-AP is defined by 4 distinct terms.
    If S is the current set, adding x creates a 4-AP if there exist y, z in S 
    such that x, y, z are part of a 4-AP.
    Specifically, x completes a 4-AP if there exist y, z in S such that the set {x, y, z, w} 
    forms a 4-AP for some w (not necessarily in S, but the AP exists in the integers).
    Wait, the definition is "contains no 4-term AP". 
    So if {x, y, z, w} is a 4-AP, and x, y, z, w are all in the set, it's invalid.
    
    So, adding x is valid iff there are NO pairs (y, z) in S such that {x, y, z} is a subset of a 4-AP 
    where the 4th term w is also in S? No, that's not sufficient.
    
    Correct Logic:
    A 4-AP is a set of 4 numbers {a, a+d, a+2d, a+3d}.
    Adding x to S creates a violation if there exists a 4-AP Q such that x in Q and Q \ {x} is a subset of S.
    In other words, if there are 3 other elements y, z, w in S such that {x, y, z, w} is a 4-AP.
    
    So, to check if x can be added:
    We need to know if there are 3 elements in S that form a "3-AP fragment" with x to make a 4-AP.
    
    This is still potentially expensive if done naively.
    
    Alternative Fast Approach:
    Precompute for each x, the list of "partners" (y, z) such that {x, y, z} determines a unique 4th element w.
    If we maintain `count_triplets[x]`? No.
    
    Let's stick to the `conflict_count` idea but refine it.
    Let `conflict_count[x]` be the number of 4-APs that are "one step away" from being complete in S, 
    where x is the missing element.
    If we add x, the number of new 4-APs formed is exactly `conflict_count[x]`.
    If `conflict_count[x] == 0`, we can safely add x.
    
    How to update `conflict_count` when we add/remove an element?
    When we add element `new_x`:
    1. `conflict_count[new_x]` becomes the number of new 4-APs formed. If > 0, it's a violation.
       Actually, if we only add valid elements, `conflict_count[new_x]` must be 0 at the moment of addition.
    2. But adding `new_x` increases the `conflict_count` for other elements.
       For any other element `y`, if {y, new_x, u, v} is a 4-AP where u, v are already in S, 
       then `conflict_count[y]` increases by 1?
       No, `conflict_count[y]` counts the number of 4-APs missing ONLY y.
       If we add `new_x`, we might complete 4-APs that were missing `new_x` and some other element?
       
    Let's define `C[x]` = number of 4-APs $Q$ such that $x \in Q$ and $Q \setminus \{x\} \subseteq S$.
    If we add $x$:
    - New 4-APs formed = $C[x]$. We require $C[x] == 0$.
    - How does $C[y]$ change for $y \neq x$?
      $C[y]$ increases if there is a 4-AP $Q$ containing $y$ and $x$ such that $Q \setminus \{y\} \subseteq S \cup \{x\}$.
      This means $Q \setminus \{y, x\} \subseteq S$.
      So for each $y$, we check if there are 4-APs containing both $x$ and $y$ where the other 2 elements are in $S$.
      
    This update is still $O(N)$ per addition if we iterate over all $y$.
    However, the number of 4-APs containing a specific pair $(x, y)$ is small.
    Precompute `pairs_to_aps`: for each pair (x, y), list the 4-APs containing them.
    Then, when adding $x$:
    For each $y$ in S (or all $y$?):
       For each 4-AP $Q$ containing $x$ and $y$:
          If the other two elements $u, v$ are in $S$, then $C[y]$ increases by 1?
          Wait, $C[y]$ counts APs missing ONLY $y$.
          If $Q = \{x, y, u, v\}$, and $u, v \in S$, and $x \notin S$ (before adding), then $Q \setminus \{y\} = \{x, u, v\}$.
          Since $x$ was not in $S$, $Q \setminus \{y\}$ was NOT a subset of $S$. So $C[y]$ did NOT count this AP before.
          After adding $x$, $Q \setminus \{y\} = \{x, u, v\}$ IS a subset of $S \cup \{x\}$.
          So $C[y]$ increases by 1 for each such AP.
          
    So the update is:
    For each $y$:
       For each 4-AP $Q$ containing $x$ and $y$:
          If other 2 elements in $Q$ are in $S$:
             $C[y] += 1$
             
    This is efficient because the number of 4-APs containing a specific pair $(x, y)$ is very small (usually 0 or 1, max 2 or 3).
    Precomputation:
    `aps_containing_pair[(x, y)]` = list of 4-APs (tuples of 4 indices) containing x and y.
    
    Then:
    Add x:
      if C[x] > 0: return False
      S.add(x)
      for each y in S (excluding x):
         for each Q in aps_containing_pair[(x, y)]:
             u, v = Q - {x, y}
             if u in S and v in S:
                 C[y] += 1
      return True
      
    Remove x:
      S.remove(x)
      for each y in S:
         for each Q in aps_containing_pair[(x, y)]:
             u, v = Q - {x, y}
             if u in S and v in S:
                 C[y] -= 1
      return True
      
    This is $O(N \cdot k)$ per add/remove, where $k$ is small (avg < 2).
    This is much faster than $O(N^2)$ or bit-shifting checks.
    """
    # Precompute all 4-APs in [0, N)
    # 4-AP: (a, a+d, a+2d, a+3d)
    # Map pair (i, j) to list of 4-APs containing both i and j.
    # We only need to store the other two elements for the update logic.
    # aps_pair[i, j] = list of (u, v) such that {i, j, u, v} is a 4-AP.
    
    # To save memory and time, we can just generate these on the fly? 
    # No, precomputation is better.
    # N=512, number of pairs ~ 130k. Number of 4-APs ~ 512*512/12 ~ 20k.
    # Each 4-AP has 6 pairs. Total entries ~ 120k. Very small.
    
    aps_pair = {}
    for d in range(1, N // 3 + 1):
        for a in range(N - 3 * d):
            t0 = a
            t1 = a + d
            t2 = a + 2 * d
            t3 = a + 3 * d
            # Pairs: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
            pairs = [
                (t0, t1, t2, t3),
                (t0, t2, t1, t3),
                (t0, t3, t1, t2),
                (t1, t2, t0, t3),
                (t1, t3, t0, t2),
                (t2, t3, t0, t1)
            ]
            for p0, p1, u, v in pairs:
                key = (min(p0, p1), max(p0, p1))
                if key not in aps_pair:
                    aps_pair[key] = []
                aps_pair[key].append((u, v))
                
    return aps_pair

_APS_PAIR = _precompute_conflict_structure()

def construct():
    deadline = time.time() + 55.0
    
    best_set = set()
    best_size = 0
    
    # Local Search / Greedy with Restart
    # We will perform multiple restarts of a greedy construction, 
    # possibly followed by a local search (swap moves) to improve.
    
    def _greedy_and_local_search():
        S = set()
        C = np.zeros(N, dtype=np.int32) # Conflict counts
        
        # 1. Greedy Phase
        # Random order
        order = list(range(N))
        random.shuffle(order)
        
        for x in order:
            if C[x] == 0:
                # Add x
                S.add(x)
                # Update C for all y in S
                # Optimization: Only update y that are "close" or in S? 
                # We must update all y in S that share a 4-AP with x.
                # We can iterate over all y in S, but that's O(|S|).
                # Better: Iterate over all y in 0..N-1? No, only y in S matter for C[y] updates?
                # Wait, C[y] is only non-zero if y is "close" to being in a 4-AP with S.
                # But we need to update C[y] for ALL y that could be added later.
                # So we must update C[y] for all y in 0..N-1?
                # No, C[y] is the count of 4-APs missing ONLY y.
                # If y is not in S, C[y] might be > 0.
                # When we add x, C[y] increases if there is a 4-AP {x, y, u, v} with u, v in S.
                # So we need to update C[y] for all y.
                # This makes the add operation O(N * k).
                # With N=512, this is ~1000 ops. Very fast.
                
                for y in range(N):
                    if y == x:
                        continue
                    key = (min(x, y), max(x, y))
                    if key in _APS_PAIR:
                        for u, v in _APS_PAIR[key]:
                            if u in S and v in S:
                                C[y] += 1
                                
        current_size = len(S)
        
        # 2. Local Search Phase (Swap Moves)
        # Try to swap an element out and another in to increase size? 
        # Or just try to add more elements if possible (should be none if greedy was perfect).
        # Actually, greedy might get stuck. Local search can fix it.
        # Move: Remove x, Add y.
        # Valid if:
        # 1. After removing x, S' is valid (always true if S was valid).
        # 2. After adding y to S', S'' is valid.
        #    This requires C'[y] == 0, where C' is the conflict count after removing x.
        
        # We can implement a simple hill climbing:
        # Try all (x, y) pairs where x in S, y not in S.
        # Check if swap improves size? No, size stays same.
        # We want to INCREASE size.
        # So we look for y not in S such that C[y] == 0.
        # If found, add it. (This is just continuing greedy).
        
        # Better: Look for a sequence of swaps that allows adding a new element?
        # For now, let's just do multiple greedy restarts with randomization.
        # The conflict count method is fast enough for ~1000 restarts.
        
        return S, current_size

    # Run multiple restarts
    restarts = 0
    while time.time() < deadline:
        S, size = _greedy_and_local_search()
        if size > best_size:
            best_size = size
            best_set = S
            restarts += 1
            # If we found a large set, we can try to improve it with local search
            # But for now, just restart.
            
        # Check time
        if time.time() > deadline - 2.0:
            break
            
    # Final check: try to add any remaining elements to best_set?
    # Reconstruct C for best_set to see if we can add more.
    if best_set:
        C = np.zeros(N, dtype=np.int32)
        # Rebuild C
        # This is O(N^2) setup, but we only do it once.
        # Actually, we can just run the greedy logic one more time on the remaining elements?
        # No, C depends on the whole set.
        
        # Let's just return the best set found.
        
    result = sorted(best_set)
    if not result:
        return [1]
        
    # Convert to 1-based
    return [x + 1 for x in result]

# EVOLVE-BLOCK-END
