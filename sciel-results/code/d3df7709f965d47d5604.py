import random
import time
import sys

# Precomputation
N = 256
triples = []
quads = []

# Generate all 3-APs (a, a+d, a+2d)
for d in range(1, N // 2 + 1):
    for a in range(1, N - 2 * d + 1):
        b = a + d
        c = a + 2 * d
        triples.append((a, b, c))

# Generate all 4-APs (a, a+d, a+2d, a+3d)
for d in range(1, N // 3 + 1):
    for a in range(1, N - 3 * d + 1):
        b = a + d
        c = a + 2 * d
        e = a + 3 * d
        quads.append((a, b, c, e))

# Index structures for fast lookup
# For each element, which triples/quads does it participate in?
triples_with = [[] for _ in range(N + 1)]
quads_with = [[] for _ in range(N + 1)]

for i, t in enumerate(triples):
    for x in t:
        triples_with[x].append(i)

for i, q in enumerate(quads):
    for x in q:
        quads_with[x].append(i)

def count_3aps(in_A):
    count = 0
    for t in triples:
        if in_A[t[0]] and in_A[t[1]] and in_A[t[2]]:
            count += 1
    return count

def has_4ap(in_A):
    for q in quads:
        if in_A[q[0]] and in_A[q[1]] and in_A[q[2]] and in_A[q[3]]:
            return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s safety margin
    
    best_set = set()
    best_score = -1
    
    # Helper to check if adding y creates a 4-AP given current in_A
    def creates_4ap_if_added(y, in_A):
        for q_idx in quads_with[y]:
            q = quads[q_idx]
            # Check if other 3 are in A
            ok = True
            for x in q:
                if x == y:
                    continue
                if not in_A[x]:
                    ok = False
                    break
            if ok:
                return True
        return False

    # Helper to count new 3-APs if y is added
    def gain_3aps_if_added(y, in_A):
        gain = 0
        for t_idx in triples_with[y]:
            t = triples[t_idx]
            ok = True
            for x in t:
                if x == y:
                    continue
                if not in_A[x]:
                    ok = False
                    break
            if ok:
                gain += 1
        return gain

    # Helper to count lost 3-APs if x is removed
    def loss_3aps_if_removed(x, in_A):
        loss = 0
        for t_idx in triples_with[x]:
            t = triples[t_idx]
            ok = True
            for z in t:
                if z == x:
                    continue
                if not in_A[z]:
                    ok = False
                    break
            if ok:
                loss += 1
        return loss

    while time.time() < deadline:
        # 1. Initialize random set
        # Start with a moderately dense set
        size = random.randint(80, 120)
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        current_set = set(candidates[:size])
        in_A = [False] * (N + 1)
        for x in current_set:
            in_A[x] = True
        
        # 2. Clean: Remove elements to break 4-APs
        # Find a 4-AP and remove a random element from it
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Find first 4-AP
            found_quad = None
            for q in quads:
                if in_A[q[0]] and in_A[q[1]] and in_A[q[2]] and in_A[q[3]]:
                    found_quad = q
                    break
            if found_quad:
                # Remove a random element from the quad
                x_to_remove = random.choice(found_quad)
                in_A[x_to_remove] = False
                current_set.remove(x_to_remove)
                changed = True
        
        # Calculate initial score
        current_score = count_3aps(in_A)
        
        # 3. Local Search
        # Phase 1: Greedy Add
        # Try to add elements that don't create 4-APs
        added = True
        while added and time.time() < deadline:
            added = False
            # Shuffle candidates to avoid bias
            not_in_A = [x for x in range(1, N + 1) if not in_A[x]]
            random.shuffle(not_in_A)
            for y in not_in_A:
                if creates_4ap_if_added(y, in_A):
                    continue
                # If it doesn't create 4-AP, add it (it will increase or maintain 3-APs)
                in_A[y] = True
                current_set.add(y)
                current_score += gain_3aps_if_added(y, in_A) # Note: gain calculated BEFORE adding? 
                # Wait, gain_3aps_if_added checks if others are in A. Since we are adding y, 
                # the "others" were already in A. So the gain is correct.
                added = True
                # Break after one add? Or continue? 
                # Adding one might block others. Let's just do one at a time and re-evaluate.
                break
        
        # Phase 2: Swaps
        # Try to swap x in A with y not in A
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Pick random x in A
            if not current_set: break
            x = random.choice(list(current_set))
            # Pick random y not in A
            not_in_A = [z for z in range(1, N + 1) if not in_A[z]]
            if not not_in_A: break
            y = random.choice(not_in_A)
            
            if x == y: continue
            
            # Temporarily swap
            in_A[x] = False
            in_A[y] = True
            current_set.remove(x)
            current_set.add(y)
            
            # Check validity
            valid = True
            # Check if any 4-AP is formed involving y? 
            # Actually, removing x might break 4-APs, adding y might create them.
            # We need to check if the NEW set has ANY 4-AP.
            # Efficient check: 
            # 1. Did removing x break all 4-APs that y would complete? 
            # 2. Are there any 4-APs that y completes and x was NOT part of?
            
            # Simpler: Just check if y creates a 4-AP in the context of (A - {x} + {y})
            # But we also need to ensure no other 4-AP exists. 
            # Since previous set had no 4-APs, the only new 4-APs can involve y.
            # So we just need to check if y forms a 4-AP with elements in (A - {x} + {y}).
            
            if creates_4ap_if_added(y, in_A): # in_A currently has y=True, x=False
                # Invalid
                in_A[y] = False
                in_A[x] = True
                current_set.remove(y)
                current_set.add(x)
                # Revert score? We didn't update score yet.
                continue
            
            # Valid. Calculate score change.
            # Old score: current_score
            # Loss from removing x:
            loss = loss_3aps_if_removed(x, in_A) # Note: in_A has x=False. 
            # This function counts triples where x is present and others are present.
            # But x is NOT in in_A. So it will return 0.
            # We need to calculate loss based on the state BEFORE removal.
            
            # Let's re-calculate properly.
            # State Before: A_old. Score S_old.
            # State After: A_new = A_old - {x} + {y}.
            
            # Loss: Triples in A_old containing x.
            # Gain: Triples in A_new containing y.
            
            # We need to check triples against A_old for loss, and A_new for gain.
            
            # Restore x to A_old state for calculation?
            # Let's just calculate delta.
            
            # Loss:
            loss = 0
            for t_idx in triples_with[x]:
                t = triples[t_idx]
                # Check if other 2 are in A_old (which is A_new + {x} - {y})
                # A_new has x=False, y=True.
                # A_old has x=True, y=False.
                # We are in A_new state (x=False, y=True).
                # So we need to check if other 2 are in A_new OR (if y is one of them, it was in A_old? No, y was not in A_old).
                # Actually, simpler:
                # Loss = number of triples {x, u, v} such that u, v in A_old.
                # Since A_old = A_new - {y} + {x}, 
                # u, v in A_old <=> (u, v in A_new) AND (u != y, v != y).
                
                ok = True
                for z in t:
                    if z == x:
                        continue
                    if z == y:
                        # y was not in A_old
                        ok = False
                        break
                    if not in_A[z]: # z is in A_new
                        ok = False
                        break
                if ok:
                    loss += 1
            
            # Gain:
            gain = 0
            for t_idx in triples_with[y]:
                t = triples[t_idx]
                # Check if other 2 are in A_new.
                ok = True
                for z in t:
                    if z == y:
                        continue
                    if not in_A[z]:
                        ok = False
                        break
                if ok:
                    gain += 1
            
            delta = gain - loss
            
            if delta > 0:
                current_score += delta
                improved = True
                # Keep the swap
            else:
                # Revert
                in_A[y] = False
                in_A[x] = True
                current_set.remove(y)
                current_set.add(x)
        
        # Update global best
        if current_score > best_score:
            best_score = current_score
            best_set = set(current_set)
            
        # If we found a very good set, maybe break early?
        # No, keep trying to improve.
        
    return sorted(best_set)

# Run once to ensure it works? No, construct is the entry point.
# The evaluator calls construct().

# To ensure we don't timeout, we rely on the deadline inside construct.
# However, precomputation is fast.

# One issue: `count_3aps` is O(N^3) roughly? No, it's O(#triples) which is ~16k. Fast.
# `creates_4ap_if_added` is O(#quads_with[y]) which is small.
# `loss_3aps_if_removed` and gain logic inside swap is O(#triples_with[x]) + O(#triples_with[y]). Small.

# The bottleneck is the loop iterations.
# With 85s, we can do many restarts and local searches.

# Let's add a small optimization: 
# If best_score is very high, we might be done.
# But we don't know the max.

# Final check on logic:
# The `creates_4ap_if_added` function checks if adding y to the CURRENT in_A creates a 4-AP.
# In the swap logic, we set in_A[y]=True, in_A[x]=False.
# Then we call `creates_4ap_if_added(y, in_A)`.
# This checks if y completes any 4-AP with the CURRENT in_A (which includes y and excludes x).
# This is correct.

# The loss/gain calculation:
# Loss: Triples containing x that were valid in A_old.
# A_old = A_new - {y} + {x}.
# So a triple {x, u, v} was valid in A_old if u, v in A_old.
# u, v in A_old means (u, v in A_new) AND (u != y, v != y).
# My code checks:
# for z in t (z != x):
#   if z == y: ok=False (correct, y not in A_old)
#   if not in_A[z]: ok=False (z not in A_new, so not in A_old)
# This is correct.

# Gain: Triples containing y valid in A_new.
# My code checks:
# for z in t (z != y):
#   if not in_A[z]: ok=False
# This is correct.

# The code looks solid.

# One potential issue: `random.choice(list(current_set))` creates a list every time.
# For N=256, this is fine.

# Let's ensure `best_set` is not empty.
# If no valid set found, return a single element?
# The problem says "nonempty subset".
# A single element has 0 3-APs and 0 4-APs.
# We should initialize best_set with a valid small set if needed.
# But our random init + cleaning should always produce a valid set.
# If it fails, we can fallback to [1].

# Add fallback
if not best_set:
    best_set = {1}
    best_score = 0

return sorted(best_set)
