"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = 58.0  # Safety margin (90s limit, leave buffer)
    
    N = 128
    
    def can_add(M, x):
        """Check if adding x to set M (bitmask) keeps it valid.
        Valid means no three distinct a,b,c in the set such that a+b, a+c, b+c are all in the set."""
        
        # Case 1: x is one of the distinct members a, b, c.
        # WLOG a = x.
        # Need b, c in M, b != c, b != x, c != x.
        # Such that x+b in M, x+c in M, b+c in M U {x}.
        
        # P = { b in M | x+b in M }
        # b is value. b_idx = b-1.
        # x+b in M => bit (x+b-1) in M.
        # b in M => bit (b-1) in M.
        # P_mask bits correspond to b-1.
        # If bit i is set in M, value is i+1.
        # x + (i+1) = x+i+1. Bit index x+i.
        # So P_mask = M & (M >> x)
        
        P_mask = M & (M >> x)
        
        # Iterate b in P_mask
        b_mask = P_mask
        while b_mask:
            b_bit = b_mask & -b_mask
            b_idx = b_bit.bit_length() - 1
            b_val = b_idx + 1
            
            # c must be in P_mask, c != b
            c_mask = P_mask & ~(1 << b_idx)
            
            # Condition 1: b+c == x
            # b_val + c_val = x
            # (b_idx+1) + (c_idx+1) = x
            # c_idx = x - 2 - b_idx
            c_idx_1 = x - 2 - b_idx
            if 0 <= c_idx_1 < N:
                if c_idx_1 != b_idx and (P_mask & (1 << c_idx_1)):
                    return False
            
            # Condition 2: b+c in M
            # b_val + c_val = v
            # v = b_idx + c_idx + 2
            # Bit index v-1 = b_idx + c_idx + 1
            # We need bit (b_idx + c_idx + 1) in M.
            # Let target_mask = M >> (b_idx + 1)
            # Then bit c_idx in target_mask must be set.
            # We need c_idx in c_mask AND c_idx in target_mask.
            
            target_mask = M >> (b_idx + 1)
            if (c_mask & target_mask) != 0:
                return False
            
            b_mask &= b_mask - 1
        
        # Case 2: x is one of the sums.
        # WLOG x = a+b.
        # Need a, b, c in M, distinct.
        # Such that a+c in M and b+c in M.
        
        # a+b = x
        # a_val + b_val = x
        # a_idx + 1 + b_idx + 1 = x
        # a_idx + b_idx = x - 2
        
        a_mask = M
        while a_mask:
            a_bit = a_mask & -a_mask
            a_idx = a_bit.bit_length() - 1
            a_val = a_idx + 1
            
            b_val = x - a_val
            if b_val <= 0 or b_val > N:
                a_mask &= a_mask - 1
                continue
            b_idx = b_val - 1
            if a_idx == b_idx:
                a_mask &= a_mask - 1
                continue
            if not (M & (1 << b_idx)):
                a_mask &= a_mask - 1
                continue
            
            # Found valid pair (a,b) in M summing to x.
            # Check if there exists c in M \ {a,b}
            # such that a+c in M and b+c in M.
            
            # a+c in M => bit (a_idx + c_idx + 1) in M.
            # b+c in M => bit (b_idx + c_idx + 1) in M.
            
            # Let M_a = M >> (a_idx + 1). Bit c_idx in M_a corresponds to a+c in M.
            # Let M_b = M >> (b_idx + 1). Bit c_idx in M_b corresponds to b+c in M.
            
            M_a = M >> (a_idx + 1)
            M_b = M >> (b_idx + 1)
            
            candidates = M_a & M_b
            
            # Exclude a and b from candidates
            candidates &= ~(1 << a_idx)
            candidates &= ~(1 << b_idx)
            
            if candidates != 0:
                return False
            
            a_mask &= a_mask - 1
        
        return True
    
    def greedy_fill(M, candidates_list, time_limit=None):
        """Try to add elements from candidates_list to M greedily.
        Returns (new_M, added_count)."""
        if time_limit is None:
            time_limit = deadline
        added = 0
        for x in candidates_list:
            if time.time() > time_limit:
                break
            if not (M >> (x - 1)) & 1:  # not already in M
                if can_add(M, x):
                    M |= (1 << (x - 1))
                    added += 1
        return M, added
    
    # Initial Set: start with upper half
    S = set(range(64, 129))
    M = 0
    for v in S:
        M |= (1 << (v - 1))
    
    best_M = M
    best_size = len(S)
    
    # Greedy Add: try lower numbers first (they're more likely to be addable)
    candidates = list(range(1, 64))
    random.shuffle(candidates)
    M, _ = greedy_fill(M, candidates)
    
    if bin(M).count('1') > best_size:
        best_size = bin(M).count('1')
        best_M = M
    
    # Local Search with multiple strategies
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        current_M = best_M
        current_size = best_size
        
        if current_size == 0:
            break
        
        # Get list of elements in current set
        items = [i+1 for i in range(128) if (current_M >> i) & 1]
        
        # Strategy 1: Single removal + greedy refill
        if iteration % 3 == 0 or random.random() < 0.5:
            r = random.choice(items)
            new_M = current_M & ~(1 << (r - 1))
            new_size = current_size - 1
            
            # Try to add back other elements
            add_candidates = [i for i in range(1, 129) if not (new_M >> (i-1)) & 1]
            random.shuffle(add_candidates)
            
            new_M, added = greedy_fill(new_M, add_candidates)
            new_size = bin(new_M).count('1')
            
            if new_size > best_size:
                best_size = new_size
                best_M = new_M
        # Strategy 2: Double removal + greedy refill
        elif iteration % 3 == 1:
            if len(items) < 2:
                continue
            r1, r2 = random.sample(items, 2)
            new_M = current_M & ~(1 << (r1 - 1)) & ~(1 << (r2 - 1))
            new_size = current_size - 2
            
            add_candidates = [i for i in range(1, 129) if not (new_M >> (i-1)) & 1]
            random.shuffle(add_candidates)
            
            new_M, added = greedy_fill(new_M, add_candidates)
            new_size = bin(new_M).count('1')
            
            if new_size > best_size:
                best_size = new_size
                best_M = new_M
        # Strategy 3: Triple removal + greedy refill
        else:
            if len(items) < 3:
                continue
            r1, r2, r3 = random.sample(items, 3)
            new_M = current_M & ~(1 << (r1 - 1)) & ~(1 << (r2 - 1)) & ~(1 << (r3 - 1))
            new_size = current_size - 3
            
            add_candidates = [i for i in range(1, 129) if not (new_M >> (i-1)) & 1]
            random.shuffle(add_candidates)
            
            new_M, added = greedy_fill(new_M, add_candidates)
            new_size = bin(new_M).count('1')
            
            if new_size > best_size:
                best_size = new_size
                best_M = new_M
    
    # Final polish: try to add any remaining elements
    add_candidates = [i for i in range(1, 129) if not (best_M >> (i-1)) & 1]
    random.shuffle(add_candidates)
    best_M, _ = greedy_fill(best_M, add_candidates)
    best_size = bin(best_M).count('1')
    
    # Convert best_M to list
    result = []
    for i in range(128):
        if (best_M >> i) & 1:
            result.append(i + 1)
            
    return result

# EVOLVE-BLOCK-END
