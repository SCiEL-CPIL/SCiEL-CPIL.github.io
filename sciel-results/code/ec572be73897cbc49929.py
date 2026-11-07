"""Optimized program for large set without triple and all pair sums."""
import time

N = 128

def construct():
    start = time.time()
    deadline = start + 80  # safety margin before 90s kill
    
    # Precompute: for each pair (a,b) with a<b, the forbidden values a+b, a+c, b+c
    # Actually, the constraint is: no three distinct a,b,c in set such that
    # a+b, a+c, b+c are ALL in the set.
    # So for a triple {a,b,c}, the forbidden condition is:
    #   (a+b) in set AND (a+c) in set AND (b+c) in set
    # We need to avoid any such triple.
    
    # Use bitmask: bit i represents whether value (i+1) is in the set
    # Values are 1..128, so bit 0 = value 1, bit 127 = value 128
    
    # Precompute all forbidden triples: for each triple (a,b,c) with a<b<c,
    # the constraint is violated if bits (a+b-1), (a+c-1), (b+c-1) are all set.
    # But a+b, a+c, b+c must be in 1..128 to matter.
    
    # Let's precompute for each triple (a,b,c) the three sum bits
    # sums_bits[a][b][c] = (sum_ab, sum_ac, sum_bc) if all in range
    
    # Better: precompute a list of constraints. Each constraint is a tuple of 3 bit positions.
    # A set is valid if for NO constraint are all 3 bits set.
    
    # Precompute all constraints
    constraints = []
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            for c in range(b+1, N+1):
                s1 = a + b
                s2 = a + c
                s3 = b + c
                if s1 <= N and s2 <= N and s3 <= N:
                    # bits are (s1-1), (s2-1), (s3-1)
                    constraints.append((s1-1, s2-1, s3-1))
    
    # Also need: the sums themselves might equal a, b, or c? No, since a,b,c >= 1
    # and sums are >= 3, so sums are always > each of a,b,c when all positive.
    # Wait, a+b could equal c? If a+b=c, then s1=c, so bit c-1 is one of the sum bits.
    # That's fine, the constraint still applies: if c is in set AND a+c in set AND b+c in set.
    
    # Initial solution: greedy from top
    mask = 0
    for v in range(N, 0, -1):
        bit = v - 1
        # Check if adding v creates a violation
        valid = True
        for (x, y, z) in constraints:
            if x == bit or y == bit or z == bit:
                # Check if the other two bits are set
                other1 = x if x != bit else (y if y != bit else z)
                other2 = y if y != bit else (z if z != bit else x)
                # Wait, this is getting complicated. Let me just check all constraints.
                pass
        # Simpler: just check all constraints
        for (x, y, z) in constraints:
            if (mask & (1 << x)) and (mask & (1 << y)) and (mask & (1 << z)):
                # Already violated? No, we're building up, so this shouldn't happen
                pass
            # Check if adding bit would violate: need bit to be one of x,y,z
            if x == bit:
                if (mask & (1 << y)) and (mask & (1 << z)):
                    valid = False
                    break
            elif y == bit:
                if (mask & (1 << x)) and (mask & (1 << z)):
                    valid = False
                    break
            elif z == bit:
                if (mask & (1 << x)) and (mask & (1 << y)):
                    valid = False
                    break
        if valid:
            mask |= (1 << bit)
    
    best_mask = mask
    best_size = bin(mask).count('1')
    
    # Local search with bit-parallel checks
    # For efficiency, precompute for each bit position which constraints involve it
    # and what the other two bits are
    bit_to_constraints = [[] for _ in range(N)]
    for (x, y, z) in constraints:
        bit_to_constraints[x].append((y, z))
        bit_to_constraints[y].append((x, z))
        bit_to_constraints[z].append((x, y))
    
    def is_valid(mask):
        """Check if mask is valid (no constraint has all 3 bits set)."""
        for (x, y, z) in constraints:
            if (mask & (1 << x)) and (mask & (1 << y)) and (mask & (1 << z)):
                return False
        return True
    
    def try_add(mask, bit):
        """Try to add bit to mask. Return new mask if valid, else None."""
        if mask & (1 << bit):
            return mask
        for (y, z) in bit_to_constraints[bit]:
            if (mask & (1 << y)) and (mask & (1 << z)):
                return None
        return mask | (1 << bit)
    
    def try_remove(mask, bit):
        """Remove bit from mask. Always valid."""
        return mask & ~(1 << bit)
    
    # Local search: try to add bits, remove bits to unblock additions
    import random
    rng = random.Random(42)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding each bit not in mask
        for bit in range(N):
            if not (mask & (1 << bit)):
                new_mask = try_add(mask, bit)
                if new_mask is not None:
                    mask = new_mask
                    improved = True
                    if bin(mask).count('1') > best_size:
                        best_size = bin(mask).count('1')
                        best_mask = mask
        
        # Try remove-one-add-many: remove a bit, then greedily add back as many as possible
        if time.time() < deadline:
            for bit in range(N):
                if (mask & (1 << bit)) and time.time() < deadline:
                    # Remove this bit
                    temp_mask = mask & ~(1 << bit)
                    # Now greedily add all possible bits
                    for add_bit in range(N):
                        if not (temp_mask & (1 << add_bit)):
                            new_mask = try_add(temp_mask, add_bit)
                            if new_mask is not None:
                                temp_mask = new_mask
                    if bin(temp_mask).count('1') > bin(mask).count('1'):
                        mask = temp_mask
                        improved = True
                        if bin(mask).count('1') > best_size:
                            best_size = bin(mask).count('1')
                            best_mask = mask
    
    # Final: use best found
    result = []
    for v in range(1, N+1):
        if best_mask & (1 << (v-1)):
            result.append(v)
    
    # Ensure we return Python ints in a list
    return [int(x) for x in result]
