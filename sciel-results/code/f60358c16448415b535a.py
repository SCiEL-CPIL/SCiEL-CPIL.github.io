"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import math
    import random
    
    DEADLINE = time.time() + 80
    
    N = 512
    rng = random.Random(42)
    
    # Build conflict bitsets
    conflict_mask = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if i * j // math.gcd(i, j) > N:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    
    def is_valid(s):
        s_list = sorted(s)
        for idx, a in enumerate(s_list):
            for b in s_list[idx+1:]:
                if (conflict_mask[a] >> b) & 1:
                    return False
        return True
    
    def greedy(order):
        result = set()
        for x in order:
            ok = True
            for y in result:
                if (conflict_mask[x] >> y) & 1:
                    ok = False
                    break
            if ok:
                result.add(x)
        return result
    
    best = set()
    best = max(best, greedy(range(1, N + 1)), key=len)
    best = max(best, greedy(range(N, 0, -1)), key=len)
    
    order3 = sorted(range(1, N + 1), key=lambda x: bin(conflict_mask[x]).count('1'))
    best = max(best, greedy(order3), key=len)
    
    for _ in range(10):
        order5 = list(range(1, N + 1))
        rng.shuffle(order5)
        best = max(best, greedy(order5), key=len)
    
    def add_nonconflicting(s):
        s = set(s)
        changed = True
        while changed and time.time() < DEADLINE:
            changed = False
            for x in range(1, N + 1):
                if x in s:
                    continue
                ok = True
                for y in s:
                    if (conflict_mask[x] >> y) & 1:
                        ok = False
                        break
                if ok:
                    s.add(x)
                    changed = True
        return s
    
    best = add_nonconflicting(best)
    
    # Efficient swap search using bitsets
    def local_search_swaps(s, max_rounds=20):
        s = set(s)
        s_bits = 0
        for x in s:
            s_bits |= (1 << x)
        
        for _ in range(max_rounds):
            if time.time() >= DEADLINE:
                break
            improved = False
            for x in sorted(s):
                if time.time() >= DEADLINE:
                    break
                # Conflict mask for s \ {x}
                # Union of conflict_mask[z] for z in s, z != x
                # We can compute: total_conflict = OR of conflict_mask[z] for all z in s
                # Then conflict_for_others = total_conflict & ~(conflict_mask[x]) ... no that's not right
                # Actually: an element y conflicts with s\{x} if y is in conflict_mask[z] for some z in s, z!=x
                # = (OR_{z in s, z!=x} conflict_mask[z]) & ~(1<<x) & ~s_bits ... 
                # Simpler: compute union_conflict = OR of conflict_mask[z] for z in s
                # Then y is blocked by s\{x} if y is in (union_conflict - conflict_mask[x] contribution from x)
                # But OR is not easily invertible. Let's just compute directly for small s.
                
                # For efficiency, precompute union for s
                # Actually, let's just do it naively but with early termination
                s_without_x = s - {x}
                # Find all y not in s that conflict with s_without_x
                blocked = 0
                for z in s_without_x:
                    blocked |= conflict_mask[z]
                
                # Available y: not in s, not in blocked, y >= 1, y <= N
                available = (~blocked) & (~s_bits) & ((1 << (N+1)) - 1) & ~1  # exclude bit 0
                # Clear bits for elements in s
                available &= ~s_bits
                
                # Try to find any available y
                while available:
                    lsb = available & (-available)
                    y = lsb.bit_length() - 1
                    if y < 1 or y > N:
                        available &= available - 1
                        continue
                    # y is available, swap x with y
                    s.remove(x)
                    s.add(y)
                    s_bits = 0
                    for elem in s:
                        s_bits |= (1 << elem)
                    improved = True
                    break
                    available &= available - 1
                
                if improved:
                    break
            if not improved:
                break
        return s
    
    best = local_search_swaps(best, max_rounds=30)
    
    for _ in range(30):
        if time.time() >= DEADLINE:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        candidate = greedy(order)
        candidate = add_nonconflicting(candidate)
        candidate = local_search_swaps(candidate, max_rounds=10)
        if len(candidate) > len(best):
            best = candidate
    
    result = sorted(best)
    if not is_valid(set(result)):
        return list(range(1, math.isqrt(N) + 1))
    
    return result
# EVOLVE-BLOCK-END
