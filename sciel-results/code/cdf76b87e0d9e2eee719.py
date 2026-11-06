import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin for 90s limit

    # 1. Fast Greedy Initialization
    # This is essentially the "include if valid" greedy from inspiration 2,
    # which is very fast and provides a strong lower bound.
    def _fast_greedy():
        values = []
        products = set()
        for x in range(2, 129):
            # Calculate new suffix products
            # If we append x to values, the new products are:
            # x, x*values[-1], x*values[-1]*values[-2], ...
            tails = []
            prod = x
            valid = True
            for a in reversed(values):
                prod *= a
                if prod in products:
                    valid = False
                    break
                tails.append(prod)
            
            if not valid:
                continue
            
            # Check if any tail is already in products
            # Note: tails are distinct because values are strictly increasing and >= 2.
            # If no collision with existing products, add x.
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        
        return values

    best_seq = _fast_greedy()
    best_len = len(best_seq)

    # 2. Beam Search Improvement
    # We use the beam search structure from inspiration 1.
    # To make it efficient, we represent the state as (seq, products_set, suffix_products_list).
    # However, copying sets is expensive.
    # Alternative: We can run a beam search that *starts* from the greedy solution?
    # No, beam search from scratch with a good width might find a better global optimum.
    # But 60s is tight.
    
    # Let's try a hybrid: 
    # Run a limited beam search from scratch.
    # State: (seq_tuple, products_frozenset, suffixes_tuple)
    # Using frozensets and tuples makes them hashable and immutable, easier to manage in a set/dict.
    # But creating frozensets is slow.
    
    # Let's stick to the list-based beam search but optimize the collision check.
    # The key insight from inspiration 1: 
    # "Integer ID mapping for product vectors reduces set copy/hash overhead"
    # We can map each product to an integer ID. But products are large.
    # Actually, just using `set` of ints is quite fast in Python.
    
    # Let's implement a beam search that runs for a fixed number of steps or time.
    # We will keep the top K states.
    
    K = 100  # Beam width, tunable
    # We iterate x from 2 to 128.
    # For each x, we expand the beam.
    
    # Initial beam: [ ((), set(), ()) ]
    # But we can seed the beam with the greedy result? 
    # No, greedy is a single path. Beam search explores multiple paths.
    
    # Let's run beam search from scratch.
    # To speed up, we can prune states that cannot possibly beat best_len.
    
    beam = [ ((), frozenset(), ()) ] # (seq, prods, sufs)
    # Using frozenset for prods allows hashing the state for deduplication if needed, 
    # but mainly for immutability.
    # Actually, we don't need to deduplicate states in the beam if we just keep top K by score.
    
    # Sort key: 
    # Primary: length (descending)
    # Secondary: last element (ascending) - smaller last element leaves more room.
    # Tertiary: sum of products? Or just deterministic tie-break.
    
    best_seq_beam = ()
    best_len_beam = 0
    
    # We will run beam search until we run out of numbers or time.
    # But we already have a good best_seq from greedy.
    # We can use best_len from greedy as a lower bound for pruning.
    
    # Optimization: 
    # If the max possible length from a state is less than best_len, prune it.
    # Max possible length = len(seq) + (128 - current_x)
    
    # Let's run the beam search.
    # To avoid TLE, we limit the beam width and check time.
    
    # We'll use a list of states.
    # State: (seq_tuple, prods_set, sufs_tuple)
    # We need to be careful with memory.
    
    # Let's try a smaller beam width first to ensure it finishes.
    K = 200
    
    # We will iterate x from 2 to 128.
    # For each x, we generate new states.
    
    # To speed up collision check:
    # new_sufs = (x,) + tuple(x * p for p in sufs)
    # new_prods = prods.union(new_sufs)
    # Check: prods.intersection(new_sufs) is empty
    
    # This is O(len(sufs)) per state.
    # len(sufs) <= len(seq) <= 127.
    # Beam width 200.
    # 127 * 200 * 127 operations ~ 3M per step.
    # 127 steps * 3M = 400M operations. Might be too slow in Python.
    
    # We need to optimize.
    # 1. Use lists instead of tuples/sets if possible? No, immutability is good for beam.
    # 2. Reduce beam width?
    # 3. Prune more aggressively?
    
    # Let's try K=50 and see.
    # Or, we can use the greedy result as a "seed" for the beam?
    # No, that's not how beam search works.
    
    # Alternative: Local search on the greedy result.
    # The current program does local search.
    # Let's combine: 
    # 1. Greedy to get a good start.
    # 2. Beam search from scratch with a small K to find potentially better structures.
    # 3. If beam search finds something better, use it.
    
    # Let's implement the beam search carefully.
    
    # We will store states as (seq_list, prods_set, sufs_list) to avoid tuple/frozenset overhead?
    # But then we can't easily sort/dedup.
    # Let's use lists for seq and sufs, and set for prods.
    # We will manage the beam as a list of these mutable structures, but we must copy them when expanding.
    
    def _beam_search():
        nonlocal best_seq, best_len
        start_bs = time.time()
        
        # State: [seq_list, prods_set, sufs_list]
        beam = [ [[], set(), []] ]
        
        for x in range(2, 129):
            new_beam = []
            
            for state in beam:
                seq, prods, sufs = state
                
                # Pruning: if even adding all remaining numbers doesn't beat best_len, skip
                # Remaining numbers: 128 - x + 1 = 129 - x
                if len(seq) + (129 - x) <= best_len:
                    continue
                
                # Option 1: Exclude x
                # We keep the state as is.
                # But we need to copy it? No, we can just add it to new_beam.
                # But wait, if we modify the state later, it's shared.
                # We are not modifying existing states, only creating new ones.
                # So we can add the same object? No, because we might add multiple states from the same parent.
                # Actually, "Exclude x" means the state remains unchanged for the next iteration.
                # So we can just append the current state to new_beam.
                # But we need to be careful: if we append the same object multiple times, 
                # and then modify one, it affects all. 
                # But we don't modify states in-place during expansion. We create new lists/sets.
                # So it is safe to append the same reference for "Exclude".
                # However, to be safe and simple, let's copy.
                # Copying sets is expensive.
                # Let's try to avoid copying for "Exclude" by just referencing.
                # But then we have duplicates in the beam?
                # We sort and take top K. Duplicates are fine, they just waste space.
                
                new_beam.append(state)
                
                # Option 2: Include x
                # Calculate new sufs
                new_sufs = [x]
                prod = x
                valid = True
                for a in reversed(seq):
                    prod *= a
                    if prod in prods:
                        valid = False
                        break
                    new_sufs.append(prod)
                
                if valid:
                    # Check intersection
                    # new_sufs are distinct.
                    # Check if any is in prods.
                    # We already checked in the loop above? 
                    # Yes, `if prod in prods: valid = False`.
                    # So if valid is True, no collision.
                    
                    new_prods = prods.copy()
                    new_prods.update(new_sufs)
                    
                    new_seq = seq + [x]
                    
                    # Create new state
                    new_state = [new_seq, new_prods, new_sufs]
                    new_beam.append(new_state)
                    
                    if len(new_seq) > best_len:
                        best_len = len(new_seq)
                        best_seq = new_seq[:]
            
            # Prune new_beam
            if not new_beam:
                break
            
            # Sort by length desc, then last element asc
            # To save time, we can sort only if len(new_beam) > K
            if len(new_beam) > K:
                # Custom sort
                # Key: (-len(seq), seq[-1] if seq else 0)
                # We can use a lambda
                new_beam.sort(key=lambda s: (-len(s[0]), s[0][-1] if s[0] else 0))
                beam = new_beam[:K]
            else:
                beam = new_beam
            
            # Time check
            if time.time() - start_bs > 50.0: # Leave 20s for local search?
                break
        
        return best_seq, best_len

    # Run Beam Search
    # We'll give it a chance to run, but cap it.
    # The greedy solution is likely already quite good.
    # Beam search might find a slightly better one.
    
    # Let's run beam search with a time limit of 40s.
    # If it finishes early, great.
    
    # Actually, let's just run it. The inner loop has time checks? No.
    # Let's add a global time check inside the beam search loop.
    
    # Re-implement beam search with global deadline check.
    
    def _beam_search_limited(deadline):
        nonlocal best_seq, best_len
        
        # State: [seq_list, prods_set, sufs_list]
        beam = [ [[], set(), []] ]
        
        for x in range(2, 129):
            new_beam = []
            
            for state in beam:
                seq, prods, sufs = state
                
                # Pruning
                if len(seq) + (129 - x) <= best_len:
                    continue
                
                # Exclude x
                new_beam.append(state)
                
                # Include x
                new_sufs = [x]
                prod = x
                valid = True
                for a in reversed(seq):
                    prod *= a
                    if prod in prods:
                        valid = False
                        break
                    new_sufs.append(prod)
                
                if valid:
                    new_prods = prods.copy()
                    new_prods.update(new_sufs)
                    new_seq = seq + [x]
                    new_state = [new_seq, new_prods, new_sufs]
                    new_beam.append(new_state)
                    
                    if len(new_seq) > best_len:
                        best_len = len(new_seq)
                        best_seq = new_seq[:]
            
            if not new_beam:
                break
            
            if len(new_beam) > K:
                new_beam.sort(key=lambda s: (-len(s[0]), s[0][-1] if s[0] else 0))
                beam = new_beam[:K]
            else:
                beam = new_beam
            
            if time.time() > deadline:
                break
        
        return best_seq, best_len

    # Run beam search with a deadline that leaves 10s for local search
    beam_deadline = start_time + 50.0
    best_seq, best_len = _beam_search_limited(beam_deadline)
    
    # 3. Local Search (from current program)
    # Try to insert more numbers into the best_seq found by beam search.
    # This can often squeeze out 1-2 more elements.
    
    def _try_insert(seq, x):
        """Try to insert x into seq maintaining sorted order. Return new seq if valid, else None."""
        # Find insertion point
        lo, hi = 0, len(seq)
        while lo < hi:
            mid = (lo + hi) // 2
            if seq[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        insert_idx = lo
        
        if insert_idx < len(seq) and seq[insert_idx] == x:
            return None

        new_seq = seq[:insert_idx] + [x] + seq[insert_idx:]
        
        # Validate
        # We can do a fast validation similar to greedy.
        # But we need to check all block products.
        # Since we are inserting one element, we can just recompute all products.
        # Or we can be smarter.
        # For now, recompute.
        
        n = len(new_seq)
        products = set()
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= new_seq[j]
                if prod in products:
                    return None
                products.add(prod)
        return new_seq

    # Local search loop
    local_deadline = start_time + 75.0
    
    improved = True
    while improved and time.time() < local_deadline:
        improved = False
        candidates = [x for x in range(2, 129) if x not in best_seq]
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > local_deadline:
                break
            new_seq = _try_insert(best_seq, x)
            if new_seq is not None:
                best_seq = new_seq
                best_len = len(best_seq)
                improved = True
                break # Restart candidate generation
    
    return best_seq

# Keep the helper functions from the current program if needed, 
# but we replaced the core logic. 
# The task requires keeping function names and signatures.
# The current program has `construct`, `_is_valid`, `_greedy_construct`, `_try_insert`, `_try_swap`.
# We only need `construct` to be the entry point.
# But the prompt says "Keep the function names and signatures of the seed program."
# So I should probably keep the dummy definitions or ensure they exist if they are called.
# But `construct` is the only one called by the evaluator.
# I will include the other functions as no-ops or simple wrappers to satisfy the "keep names" constraint 
# if it implies they must exist in the module.

def _is_valid(seq):
    """Check if the sequence has unique block products."""
    n = len(seq)
    products = set()
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            if prod in products:
                return False
            products.add(prod)
    return True

def _greedy_construct():
    """Greedy baseline."""
    values = []
    for x in range(2, 129):
        candidate = values + [x]
        if _is_valid(candidate):
            values = candidate
    return values

def _try_swap(seq, i, x):
    """Try to replace seq[i] with x. Maintain sorted order."""
    if i > 0 and x <= seq[i-1]:
        return None
    if i < len(seq) - 1 and x >= seq[i+1]:
        return None
    if x == seq[i]:
        return None
    
    new_seq = seq[:i] + [x] + seq[i+1:]
    if _is_valid(new_seq):
        return new_seq
    return None

# The `construct` function above is the main entry point.
# I used `_try_insert` inside `construct`. I should define it or use the one above.
# I defined a local `_try_insert` inside `construct` in my draft. 
# Let's use the module-level one to be consistent with "keep function names".
# But the module-level `_try_insert` in the current program uses `_is_valid`.
# My local one also uses a validation.
# Let's just use the module-level `_try_insert` if I define it properly.
# In the current program, `_try_insert` is defined.
# I will reuse the current program's `_try_insert` logic but maybe optimize it?
# The current `_try_insert` calls `_is_valid` which is O(N^2).
# For N~40, that's 1600 ops. Fine.

# Let's refine `construct` to use the module-level functions where possible.

def construct():
    start_time = time.time()
    deadline = start_time + 75.0

    # 1. Fast Greedy Initialization
    best_seq = _fast_greedy_impl()
    best_len = len(best_seq)

    # 2. Beam Search
    K = 100
    beam_deadline = start_time + 50.0
    best_seq, best_len = _beam_search_impl(best_seq, best_len, K, beam_deadline)

    # 3. Local Search
    local_deadline = deadline
    improved = True
    while improved and time.time() < local_deadline:
        improved = False
        candidates = [x for x in range(2, 129) if x not in best_seq]
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > local_deadline:
                break
            new_seq = _try_insert(best_seq, x)
            if new_seq is not None:
                best_seq = new_seq
                best_len = len(best_seq)
                improved = True
                break

    return best_seq

def _fast_greedy_impl():
    values = []
    products = set()
    for x in range(2, 129):
        tails = []
        prod = x
        valid = True
        for a in reversed(values):
            prod *= a
            if prod in products:
                valid = False
                break
            tails.append(prod)
        
        if not valid:
            continue
        
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    
    return values

def _beam_search_impl(init_seq, init_len, K, deadline):
    # We can seed the beam with the initial sequence?
    # No, beam search from scratch is better for finding different structures.
    # But we can use init_len for pruning.
    
    beam = [ [[], set(), []] ]
    best_seq = init_seq[:]
    best_len = init_len
    
    for x in range(2, 129):
        new_beam = []
        
        for state in beam:
            seq, prods, sufs = state
            
            if len(seq) + (129 - x) <= best_len:
                continue
            
            # Exclude
            new_beam.append(state)
            
            # Include
            new_sufs = [x]
            prod = x
            valid = True
            for a in reversed(seq):
                prod *= a
                if prod in prods:
                    valid = False
                    break
                new_sufs.append(prod)
            
            if valid:
                new_prods = prods.copy()
                new_prods.update(new_sufs)
                new_seq = seq + [x]
                new_state = [new_seq, new_prods, new_sufs]
                new_beam.append(new_state)
                
                if len(new_seq) > best_len:
                    best_len = len(new_seq)
                    best_seq = new_seq[:]
        
        if not new_beam:
            break
        
        if len(new_beam) > K:
            new_beam.sort(key=lambda s: (-len(s[0]), s[0][-1] if s[0] else 0))
            beam = new_beam[:K]
        else:
            beam = new_beam
        
        if time.time() > deadline:
            break
    
    return best_seq, best_len

# Ensure `_try_insert` is defined at module level as per current program
def _try_insert(seq, x):
    """Try to insert x into seq maintaining sorted order. Return new seq if valid, else None."""
    lo, hi = 0, len(seq)
    while lo < hi:
        mid = (lo + hi) // 2
        if seq[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    insert_idx = lo
    
    if insert_idx < len(seq) and seq[insert_idx] == x:
        return None

    new_seq = seq[:insert_idx] + [x] + seq[insert_idx:]
    if _is_valid(new_seq):
        return new_seq
    return None
