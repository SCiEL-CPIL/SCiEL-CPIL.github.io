import sys

def construct():
    # Max size 48, values <= 512
    # We want to maximize cardinality.
    # Greedy: try to add smallest? largest? 
    # Usually, picking numbers that are "far" from each other helps.
    # But the bitset approach is fast enough to try all candidates 1..512.
    # To maximize count, we should try to add as many as possible.
    # Order: 1 to 512? Or random?
    # If we go 1..512, we might fill up with small numbers.
    # Small numbers have smaller differences, more collisions?
    # Let's try 1..512.
    
    MAX_N = 512
    TARGET_SIZE = 48
    OFFSET = 25000
    RANGE = 2 * OFFSET + 1
    MASK = (1 << RANGE) - 1
    
    selected = []
    # state[a] is a list of bitsets for each subset size k
    # We store it in a dictionary keyed by the value 'a'
    state = {}
    
    def update_state(bits, val):
        """
        Update bitsets 'bits' (list of ints) by adding element 'val'.
        bits[k] represents sums of subsets of size k.
        """
        n = len(bits) - 1 # current max size
        # Iterate backwards to avoid using the new element multiple times in one step
        # We are adding ONE element to the set of available elements.
        # So we update from high k to low k.
        # But 'bits' currently represents the set BEFORE adding 'val'.
        # We want to compute the new set.
        # new_bits[k] = old_bits[k] | (old_bits[k-1] shifted by val)
        # So we can do it in place if we go backwards?
        # No, because old_bits[k-1] is needed.
        # If we update bits[k] first, it doesn't affect bits[k-1].
        # So iterate k from n down to 0.
        
        # Shift val
        if val > 0:
            shift = val
        else:
            shift = -val
            
        for k in range(n, -1, -1):
            if bits[k]:
                if val > 0:
                    shifted = (bits[k] << shift) & MASK
                else:
                    shifted = (bits[k] >> shift) & MASK
                bits[k+1] |= shifted
        
        # Ensure bits list is long enough?
        # It is initialized with size MAX_N+1, so it's fine.

    def check_target(bits, target, min_size):
        """
        Check if 'target' sum is reachable with at least 'min_size' elements.
        """
        idx = target + OFFSET
        if idx < 0 or idx >= RANGE:
            return False
        for k in range(min_size, len(bits)):
            if bits[k] & (1 << idx):
                return True
        return False

    for x in range(1, MAX_N + 1):
        if len(selected) >= TARGET_SIZE:
            break
            
        # 1. Check if x is a mean of current selected set
        valid = True
        if len(selected) >= 2:
            # Values t - x
            # We need to check if 0 is a subset sum of size >= 2
            # Build temporary bitsets for x
            temp_bits = [0] * (len(selected) + 1)
            temp_bits[0] = 1 # sum 0 with 0 elements
            # Apply all t in selected
            for t in selected:
                val = t - x
                # Update temp_bits
                # We can't use update_state directly because it modifies in place and we need to be careful with the loop direction
                # But update_state does it correctly (backwards).
                # However, update_state expects 'bits' to be the state BEFORE adding.
                # So we can just call it.
                update_state(temp_bits, val)
            
            if check_target(temp_bits, 0, 2):
                valid = False
        
        # 2. Check if any existing a becomes a mean involving x
        if valid:
            for a in selected:
                # Target for a is a - x
                # We need to check if (a-x) is in state[a] with size >= 1
                # state[a] represents sums of subsets of selected \ {a}
                target = a - x
                if check_target(state[a], target, 1):
                    valid = False
                    break
        
        if valid:
            # Add x to selected
            selected.append(x)
            
            # Update state for existing a's
            for a in selected:
                if a == x:
                    continue
                val = x - a
                update_state(state[a], val)
            
            # Initialize state for x
            # Others are all t in selected \ {x}
            x_bits = [0] * (len(selected) + 1)
            x_bits[0] = 1
            for t in selected:
                if t == x:
                    continue
                val = t - x
                update_state(x_bits, val)
            state[x] = x_bits
            
            # Optional: cleanup state for elements that are no longer in selected?
            # We never remove, so no cleanup needed.
            
    return selected

if __name__ == "__main__":
    print(construct())
