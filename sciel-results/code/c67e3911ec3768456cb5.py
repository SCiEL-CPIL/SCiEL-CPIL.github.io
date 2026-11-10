import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    TARGET = 200
    MASK = (1 << (TARGET + 1)) - 1  # bits 0..200
    
    def reachable_bitset(elements):
        """Compute subset sums <= TARGET as a bitset integer."""
        bits = 1  # bit 0 set, sum 0 reachable
        for x in elements:
            bits |= (bits << x) & MASK
        return bits
    
    def can_add(bits, x):
        """Check if adding x keeps TARGET unreachable."""
        return not (bits & (1 << (TARGET - x)))
    
    def add_to_bits(bits, x):
        """Add element x to reachable bitset."""
        return bits | ((bits << x) & MASK)
    
    # Strategy: try multiple initial configurations and greedy completions
    best_set = None
    best_size = 0
    
    def try_greedy(initial_selected, order):
        """Greedy add elements in given order if safe."""
        selected = set(initial_selected)
        bits = reachable_bitset(selected)
        for x in order:
            if x in selected:
                continue
            if can_add(bits, x):
                selected.add(x)
                bits = add_to_bits(bits, x)
        return selected, bits
    
    # Configuration 1: Start with {66..100}, add descending
    init1 = set(range(66, 101))
    order1 = list(range(65, 0, -1))
    sel1, bits1 = try_greedy(init1, order1)
    if len(sel1) > best_size:
        best_set, best_bits = sel1, bits1
        best_size = len(sel1)
    
    # Configuration 2: Start with {67..100}, add descending
    init2 = set(range(67, 101))
    order2 = list(range(66, 0, -1))
    sel2, bits2 = try_greedy(init2, order2)
    if len(sel2) > best_size:
        best_set, best_bits = sel2, bits2
        best_size = len(sel2)
    
    # Configuration 3: Start with {65..100} minus problematic elements
    # 65+67+68=200, so can't have all three. Try {65,66,68..100}
    init3 = {65, 66} | set(range(68, 101))
    order3 = list(range(64, 0, -1)) + [67]
    sel3, bits3 = try_greedy(init3, order3)
    if len(sel3) > best_size:
        best_set, best_bits = sel3, bits3
        best_size = len(sel3)
    
    # Configuration 4: Start with {64..100} minus problematic
    # 64+67+69=200, 64+68+68 not possible (unique), 64+66+70=200
    # Try {64,65,66,68,71..100} - skip 67,69,70
    init4 = {64, 65, 66, 68} | set(range(71, 101))
    order4 = list(range(63, 0, -1)) + [67, 69, 70]
    sel4, bits4 = try_greedy(init4, order4)
    if len(sel4) > best_size:
        best_set, best_bits = sel4, bits4
        best_size = len(sel4)
    
    # Local search: try to improve by removing one and adding two
    selected = best_set
    bits = best_bits
    improved = True
    max_iters = 50
    
    for iteration in range(max_iters):
        if time.time() > deadline - 5:
            break
        if not improved:
            break
        improved = False
        # Try removing one element and adding up to two others
        for rem in sorted(selected, reverse=True):
            if time.time() > deadline - 5:
                break
            if len(selected) <= 2:
                break
            # Remove rem
            test_selected = selected - {rem}
            test_bits = reachable_bitset(test_selected)
            
            # Try adding pairs of non-selected elements
            non_sel = [x for x in range(100, 0, -1) if x not in test_selected]
            found = False
            for i, a in enumerate(non_sel):
                if time.time() > deadline - 5:
                    break
                if found:
                    break
                if not can_add(test_bits, a):
                    continue
                temp_bits = add_to_bits(test_bits, a)
                for b in non_sel[i+1:]:
                    if time.time() > deadline - 5:
                        break
                    if not can_add(temp_bits, b):
                        continue
                    # Adding both a and b
                    final_bits = add_to_bits(temp_bits, b)
                    if not (final_bits & (1 << TARGET)):
                        # Valid improvement
                        selected = test_selected | {a, b}
                        bits = final_bits
                        improved = True
                        found = True
                        break
            if found:
                break
    
    # Also try single-element swaps (remove one, add one) - though this doesn't increase size
    # More useful: try to add one more element after the pair swaps
    # Try adding any remaining element
    for x in range(100, 0, -1):
        if time.time() > deadline - 5:
            break
        if x not in selected and can_add(bits, x):
            selected.add(x)
            bits = add_to_bits(bits, x)
    
    return list(selected)
