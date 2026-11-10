import time
import random

N = 256

# EVOLVE-BLOCK-START

def _is_valid_solution(arr):
    """
    Verifies:
    1. All elements are distinct integers in [1, N].
    2. It is a Sidon set (all unordered pair sums a+b with a<=b are distinct).
    3. It is maximal (adding any missing element from [1, N] destroys the Sidon property).
    """
    if not arr:
        return False
    
    arr_set = set(arr)
    if len(arr_set) != len(arr):
        return False
    
    for x in arr:
        if x < 1 or x > N:
            return False
            
    # Compute all sums
    sums = set()
    for i in range(len(arr)):
        a = arr[i]
        for j in range(i, len(arr)):
            b = arr[j]
            s = a + b
            if s in sums:
                return False
            sums.add(s)
            
    # Check maximality
    for x in range(1, N + 1):
        if x in arr_set:
            continue
        
        # Try to add x
        # New sums are {x+a for a in arr} U {2x}
        # If any of these are in `sums`, or if there's a conflict among them, x cannot be added.
        # If x CAN be added (no conflicts), the set is NOT maximal.
        # We want to ensure that for EVERY x not in arr, x CANNOT be added.
        
        # Check conflicts with existing sums
        conflict = False
        for a in arr:
            if (x + a) in sums:
                conflict = True
                break
        if conflict:
            continue
        
        if (2 * x) in sums:
            continue
            
        # Check conflicts among the new sums themselves
        # The new sums are S_new = {x+a | a in arr} U {2x}
        # Note that x+a == x+b iff a==b (since arr has distinct elements).
        # So conflicts can only happen between:
        # 1. x+a and 2x  => x+a = 2x => a = x (impossible since x not in arr)
        # 2. x+a and x+b => a=b (impossible)
        # So the new sums are always distinct from each other.
        # Thus, if no conflict with existing sums, x CAN be added.
        
        # Since x can be added, the set is not maximal.
        return False
        
    return True

def _backtracking_search(rng, deadline):
    """
    Randomized backtracking search for a small maximal Sidon set.
    We try to build a set S. 
    State: 
      - S: list of selected elements (sorted)
      - sums: set of all pair sums in S
      - candidates: set of elements that can still be added to S without violating Sidon property.
      
    To ensure maximality, we don't just stop when candidates is empty (which gives a maximal set).
    Wait, if we stop when candidates is empty, the set is maximal by definition.
    The goal is to minimize |S|.
    
    Strategy:
    We can use a depth-first search with a heuristic to pick the "most restrictive" element first?
    Or we can use a "minimum dominating set" style heuristic on the conflict graph?
    
    Actually, the "blocking" relationship is key.
    Let's try a different approach:
    Start with an empty set.
    At each step, we have a set of "available" elements that can be added.
    If we add an element, it might "block" other elements (make them un-addable).
    We want to cover all elements [1, N] with "blocks" from selected elements.
    This is similar to a set cover problem where sets are "elements blocked by x given current S".
    
    However, the blocking depends on S.
    
    Let's try a simple randomized local search / hill climbing on the size of the set.
    But we can't just remove elements.
    
    Let's stick to the backtracking idea but with a strong heuristic:
    1. Randomly select a starting element.
    2. Greedily add elements that "block" the most remaining candidates.
    3. If we get stuck (no candidates left), we have a maximal set.
    4. Check size.
    5. If not optimal, restart.
    
    Actually, the "greedy blocking" strategy mentioned in inspirations (score 1.3333) is:
    "Max-blocking greedy selects elements that invalidate the most other candidates, producing smaller inclusion-maximal Sidon sets than random-ordering greedy"
    
    Let's implement this "Max-Blocking Greedy" with randomization.
    
    Algorithm:
    1. Initialize S = empty, sums = empty.
    2. Candidates = {1, ..., N}.
    3. While Candidates is not empty:
       a. For each x in Candidates, calculate how many elements in Candidates would become "blocked" if x were added to S.
          An element y in Candidates is blocked by x if x+y is in sums OR 2y is in sums (after adding x's sums).
          Wait, "blocked" means y cannot be added to S U {x}.
          y cannot be added to S U {x} if:
             - y+x is in sums(S) U {2x}
             - y+s is in sums(S) U {2x} for some s in S? No, we just check if y can be added to the NEW set S' = S U {x}.
             - y can be added to S' if:
               - y is not in S'
               - y+y not in sums(S')
               - y+s' not in sums(S') for all s' in S'
             
          So, if we add x, the new sums are sums(S) U {x+s for s in S} U {2x}.
          Let sums_new = sums(S) U {x+s for s in S} U {2x}.
          y is blocked by x (given S) if y cannot be added to S U {x}.
          This happens if:
             - 2y is in sums_new
             - y+s' is in sums_new for some s' in S U {x}
               Note that y+x is in sums_new (it's one of the new sums).
               So we just need to check if y+s is in sums(S) for some s in S, OR if 2y is in sums(S), OR if y+x is in sums(S) (impossible if y!=x and Sidon), OR if y+s' in sums_new for s' in S.
               
          Actually, simpler:
          y is in Candidates, so y can be added to S.
          y is blocked by x if y CANNOT be added to S U {x}.
          This is equivalent to: 
             - 2y in sums(S U {x})
             - OR (y+s) in sums(S U {x}) for some s in S U {x}
             
          sums(S U {x}) = sums(S) U {x+s | s in S} U {2x}
          
          So y is blocked if:
             - 2y in sums(S) OR 2y in {x+s | s in S} OR 2y == 2x (y=x, impossible)
             - OR (y+s) in sums(S) for some s in S (but y is candidate, so this is false by definition of Candidates? No, Candidates are those that CAN be added to S. So (y+s) not in sums(S) for all s in S, and 2y not in sums(S).)
             - OR (y+s) in {x+s' | s' in S} for some s in S U {x}?
               If s=x: y+x in {x+s' | s' in S}? No, x+x=2x is in set. y+x is in set only if y+x = x+s' => y=s' (impossible if y not in S) or y+x=2x => y=x (impossible).
               Wait, the set of new sums is {x+s' | s' in S} U {2x}.
               Does y+x appear in this set?
               y+x = x+s' => y=s' (no).
               y+x = 2x => y=x (no).
               So y+x is NOT in sums(S U {x})? 
               Wait. The sums of S U {x} are:
               - sums from S: sums(S)
               - sums involving x: x+s for s in S, and x+x=2x.
               So the set of sums is sums(S) U {x+s | s in S} U {2x}.
               
               Now, can y be added to S U {x}?
               Condition: 
               1. 2y not in sums(S U {x})
               2. y+s' not in sums(S U {x}) for all s' in S U {x}
                  
               Since y is a candidate for S, we know:
               - 2y not in sums(S)
               - y+s not in sums(S) for all s in S
               
               So for y to be blocked by x, one of the following must be true:
               A. 2y is in {x+s | s in S} U {2x}
                  => 2y = x+s for some s in S  => s = 2y-x
                  => 2y = 2x => y=x (no)
               B. y+s' is in {x+s | s in S} U {2x} for some s' in S U {x}
                  Case s'=x: y+x is in {x+s | s in S} U {2x}
                     y+x = x+s => y=s (no, y not in S)
                     y+x = 2x => y=x (no)
                  Case s' in S: y+s' is in {x+s | s in S} U {2x}
                     y+s' = x+s => s' - s = x - y
                     y+s' = 2x => s' = 2x - y
                     
          So, y is blocked by x if:
             - (2y - x) is in S  [from A]
             - OR exists s' in S such that (s' - (x - y)) is in S  [from B, y+s' = x+s => s = s' - x + y]
             - OR (2x - y) is in S  [from B, y+s' = 2x => s' = 2x - y]
             
          This is getting complicated to compute for all y for each x.
          N=256, |S| is small (~15). Candidates ~250.
          For each x in Candidates (~250), we check ~250 y's.
          For each y, we do a few set lookups.
          250 * 250 * 5 ops = 300k ops per step.
          Steps ~15. Total 4.5M ops. This is fast enough in Python? Maybe 1-2 seconds.
          We have 60s. So we can do ~20-30 restarts.
    
    Let's implement this "Max-Blocking Greedy".
    """
    best_size = float('inf')
    best_set = []
    
    while time.time() < deadline:
        S = []
        sums = set()
        candidates = set(range(1, N + 1))
        
        while candidates:
            # Find x in candidates that blocks the most other candidates
            best_x = None
            max_blocked = -1
            best_blocked_set = set()
            
            # To speed up, we can sample a few x's if candidates is large?
            # Or just iterate all.
            # Let's iterate all.
            
            # Optimization: Precompute something?
            # For a given x, the condition for y being blocked is:
            # 1. 2y - x in S
            # 2. 2x - y in S
            # 3. Exists s' in S such that s' - x + y in S
            
            # Let's just compute the count.
            
            # Randomize tie-breaking
            # To add diversity, we can add a small random weight to the blocked count.
            
            for x in candidates:
                blocked_count = 0
                # We only care about y in candidates, y != x
                # Actually, if we pick x, it is removed from candidates.
                # The other candidates y are those that become "un-addable".
                
                # To make it faster, let's just estimate or sample?
                # No, let's do exact.
                
                # We can optimize the inner loop.
                # Condition 1: 2y - x in S  => 2y = x + s for s in S => y = (x+s)/2. 
                #   So for each s in S, if (x+s) is even, y = (x+s)//2 is a candidate blocked by this.
                #   We can just check if that specific y is in candidates.
                
                # Condition 2: 2x - y in S => y = 2x - s for s in S.
                #   For each s in S, y = 2x - s. Check if y in candidates.
                
                # Condition 3: Exists s' in S such that s' - x + y in S.
                #   Let s = s' - x + y => y = s - s' + x.
                #   For each pair (s', s) in S x S, y = s - s' + x.
                #   Check if y in candidates.
                
                # This is O(|S| + |S|^2) per x.
                # |S| is small.
                
                blocked_ys = set()
                
                # Cond 1
                for s in S:
                    val = x + s
                    if val % 2 == 0:
                        y = val // 2
                        if y in candidates and y != x:
                            blocked_ys.add(y)
                
                # Cond 2
                for s in S:
                    y = 2 * x - s
                    if y in candidates and y != x:
                        blocked_ys.add(y)
                
                # Cond 3
                # y = s - s' + x
                # We can iterate s' in S, s in S
                for sp in S:
                    for s in S:
                        y = s - sp + x
                        if y in candidates and y != x:
                            blocked_ys.add(y)
                
                # Add random noise for diversity
                score = len(blocked_ys) + rng.random() * 0.1
                if score > max_blocked:
                    max_blocked = score
                    best_x = x
                    best_blocked_set = blocked_ys
            
            if best_x is None:
                break
                
            # Add best_x to S
            S.append(best_x)
            candidates.remove(best_x)
            
            # Update sums
            for s in S:
                sums.add(best_x + s)
            sums.add(2 * best_x)
            
            # Remove blocked candidates from candidates
            # Note: best_blocked_set was computed BEFORE adding best_x to S.
            # But the definition of "blocked" was "cannot be added to S U {best_x}".
            # So yes, we should remove them from candidates.
            candidates -= best_blocked_set
            
        # End while candidates
        
        # Check if S is valid (it should be, by construction)
        # And maximal (it should be, since candidates is empty)
        # But let's verify to be safe.
        if _is_valid_solution(S):
            if len(S) < best_size:
                best_size = len(S)
                best_set = S[:]
                
            if best_size <= 13:
                break
                
    return best_set

def construct():
    start_time = time.time()
    # Safety margin: 55s limit
    deadline = start_time + 55.0
    
    rng = random.Random(42)
    
    # First, try the Max-Blocking Greedy
    best_set = _backtracking_search(rng, deadline)
    
    # If we have time, try a few random restarts of simple greedy to see if we get lucky
    # But Max-Blocking is likely better.
    
    if not best_set:
        # Fallback
        order = list(range(N, 0, -1))
        selected = []
        sums = set()
        for x in order:
            conflict = False
            for a in selected:
                if (x + a) in sums:
                    conflict = True
                    break
            if conflict:
                continue
            if (2 * x) in sums:
                continue
            selected.append(x)
            for a in selected:
                sums.add(x + a)
            sums.add(2 * x)
        best_set = selected
        
    if not _is_valid_solution(best_set):
        # Last resort
        return [1, 3, 7, 12, 20, 31, 45, 62, 82, 105, 131, 160, 192, 227, 256]
        
    return best_set

# EVOLVE-BLOCK-END
