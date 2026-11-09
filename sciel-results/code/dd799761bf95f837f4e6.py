"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    n = 128
    
    # Initial construction: Greedy with random restarts
    # We will use the fast bad_pairs method for local search
    
    def _greedy_init(order):
        S = set()
        bad_pairs = set()  # Set of tuples (a, b) with a < b, such that a+b in S
        
        for x in order:
            # Check if x can be added
            valid = True
            for a, b in bad_pairs:
                if (a + x) in S and (b + x) in S:
                    valid = False
                    break
            
            if valid:
                S.add(x)
                # Update bad_pairs
                # New pairs (x, y) where x+y in S
                for y in S:
                    if y == x: continue
                    if x + y in S:
                        if x < y:
                            bad_pairs.add((x, y))
                        else:
                            bad_pairs.add((y, x))
        
        return S, bad_pairs

    best_S = set()
    best_size = 0
    
    # Try a few random orders for initialization
    for _ in range(20):
        order = list(range(1, n + 1))
        random.shuffle(order)
        S, bp = _greedy_init(order)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
    
    # Also try sorted orders
    for order in [list(range(1, n+1)), list(range(n, 0, -1))]:
        S, bp = _greedy_init(order)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S

    # Local Search with Swap Moves
    # State: S, bad_pairs
    # Move: Remove a, Add b
    
    S = set(best_S)
    # Reconstruct bad_pairs for best_S
    bad_pairs = set()
    for a in S:
        for b in S:
            if a < b and a + b in S:
                bad_pairs.add((a, b))

    current_size = len(S)
    
    # Simulated Annealing / Hill Climbing
    # We'll do a mix of best-improvement and random walks
    
    def can_add(x, S, bad_pairs):
        for a, b in bad_pairs:
            if (a + x) in S and (b + x) in S:
                return False
        return True

    def add_element(x, S, bad_pairs):
        S.add(x)
        for y in S:
            if y == x: continue
            if x + y in S:
                if x < y:
                    bad_pairs.add((x, y))
                else:
                    bad_pairs.add((y, x))

    def remove_element(x, S, bad_pairs):
        # Remove pairs involving x
        to_remove = []
        for p in bad_pairs:
            if p[0] == x or p[1] == x:
                to_remove.append(p)
        for p in to_remove:
            bad_pairs.remove(p)
        S.remove(x)

    # Local Search Loop
    no_improve_count = 0
    temperature = 1.0
    
    while time.time() < deadline:
        # Try to find a better swap
        # Pick a random element to remove
        if not S:
            break
            
        a = random.choice(list(S))
        
        # Try to find a b to add such that len(S) increases or stays same but we escape local min?
        # For maximizing size, we primarily want size + 1.
        # If no size + 1, try size 0 change (swap) to escape.
        
        found_better = False
        found_swap = False
        
        # First pass: Try to add any element (size + 1)
        # This is rare in local search, but we check it.
        # Actually, if we are at a local max, we can't add any.
        
        # So we must swap.
        # Remove a, try to add b.
        # If len(S \ {a} U {b}) > len(S), accept. (Impossible since sizes are equal)
        # We want to escape local minima where no single add is possible, but a swap enables future adds?
        # Or just find a larger set via swaps.
        
        # Strategy: 
        # 1. Try all b not in S. If S\{a} U {b} is valid, and then we can add more?
        # Simpler: Just do random valid swaps. If we get stuck, restart or anneal.
        
        # Let's do a standard local search:
        # 1. Try to add any element. If found, accept and restart search.
        for x in range(1, n + 1):
            if x not in S:
                if can_add(x, S, bad_pairs):
                    add_element(x, S, bad_pairs)
                    current_size += 1
                    no_improve_count = 0
                    found_better = True
                    break
        
        if found_better:
            continue
            
        # 2. If no add possible, try a swap.
        # Remove a (randomly chosen above, or iterate all a?)
        # To be efficient, let's iterate a few a's.
        
        improved = False
        for _ in range(10): # Try 10 random swaps
            a = random.choice(list(S))
            # Temporarily remove a
            # We need to check if S\{a} U {b} is valid.
            # Validity of S\{a} U {b}:
            # 1. S\{a} is valid (it is).
            # 2. Adding b to S\{a} must not create forbidden triple.
            #    Forbidden triple in S\{a} U {b} must involve b.
            #    So check if there exist u,v in S\{a} such that:
            #    u+v in S\{a} U {b} AND u+b in S\{a} U {b} AND v+b in S\{a} U {b}.
            
            # Note: u+v in S\{a} U {b} means u+v in S\{a} OR u+v == b.
            # Since u,v >= 1, u+v >= 2. b is in 1..128.
            
            # Let's just use the bad_pairs logic on the modified set?
            # It's easier to just check validity of the new set directly if we maintain bad_pairs?
            # Updating bad_pairs for a swap is complex.
            
            # Alternative: Just check validity of S_new = S - {a} + {b} using the O(N^3) check?
            # No, too slow.
            
            # Let's stick to the bad_pairs incremental update.
            # To check if b can be added to S\{a}:
            # We need bad_pairs of S\{a}.
            # Let's compute bad_pairs_minus_a on the fly?
            # bad_pairs_minus_a = { (u,v) in bad_pairs : a not in (u,v) } 
            # PLUS new pairs? No, removing a only removes pairs involving a.
            # Wait, if a was part of a sum u+v=a, that pair (u,v) is still in S\{a}, but the sum a is NOT in S\{a}.
            # So (u,v) is NO LONGER a bad pair in S\{a}.
            # This makes incremental removal tricky.
            
            # Given the complexity, let's use a simpler, slower but robust check for swaps, 
            # relying on the fact that we only do this when stuck.
            
            S_test = S - {a}
            # Check if b can be added to S_test
            # Use the N_x method on S_test
            N_b = [u for u in S_test if u + b <= n and (u + b) in S_test]
            
            valid_b = True
            for i in range(len(N_b)):
                u = N_b[i]
                for j in range(i+1, len(N_b)):
                    v = N_b[j]
                    if u + v in S_test:
                        valid_b = False
                        break
                if not valid_b:
                    break
            
            if valid_b:
                # Accept swap
                remove_element(a, S, bad_pairs)
                add_element(b, S, bad_pairs)
                improved = True
                no_improve_count = 0
                break
        
        if not improved:
            no_improve_count += 1
            # If stuck for a long time, restart with a new random init?
            # Or perturb.
            if no_improve_count > 100:
                # Perturb: Remove 5 random elements, re-add greedily
                for _ in range(5):
                    if S:
                        a = random.choice(list(S))
                        remove_element(a, S, bad_pairs)
                # Greedy fill
                for x in range(1, n+1):
                    if x not in S:
                        if can_add(x, S, bad_pairs):
                            add_element(x, S, bad_pairs)
                no_improve_count = 0

    # Final verification
    S_final = list(S)
    if not _is_valid_set(S_final):
        S_final = list(range(65, 129))
        
    return [int(x) for x in S_final if 1 <= x <= 128]

def _is_valid_set(S):
    S_set = set(S)
    for i in range(len(S)):
        for j in range(i+1, len(S)):
            for k in range(j+1, len(S)):
                a, b, c = S[i], S[j], S[k]
                if (a+b) in S_set and (a+c) in S_set and (b+c) in S_set:
                    return False
    return True
# EVOLVE-BLOCK-END
