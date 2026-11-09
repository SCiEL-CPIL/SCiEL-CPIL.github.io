"""XOR circuit synthesis with basis-aware greedy cost minimization."""
import time
from collections import defaultdict

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    
    if m == 0 or n == 0:
        return {'gates': [], 'outputs': []}
    
    # Convert matrix rows to integer bitmasks for fast XOR
    rows = []
    for i in range(m):
        val = 0
        for j in range(n):
            if matrix[i][j]:
                val |= (1 << j)
        rows.append(val)
    
    # Identify basis rows using Gaussian Elimination
    # We want to process basis rows first to establish a strong set of available wires
    basis_indices = []
    non_basis_indices = []
    
    # Simple Gaussian elimination to find pivot rows
    # We work on the rows themselves
    processed_rows = []
    row_pivots = {} # index -> pivot bit
    
    for i in range(m):
        r = rows[i]
        if r == 0:
            non_basis_indices.append(i)
            continue
            
        # Check if r is independent of processed_rows
        # Reduce r by processed_rows
        reduced_r = r
        for pr, pivot_bit in processed_rows:
            if (reduced_r >> pivot_bit) & 1:
                reduced_r ^= pr
        
        if reduced_r != 0:
            # Find pivot
            pivot_bit = (reduced_r & (-reduced_r)).bit_length() - 1
            processed_rows.append((r, pivot_bit)) # Store original r and its pivot
            basis_indices.append(i)
        else:
            non_basis_indices.append(i)
            
    # Sort basis indices by popcount (sparse first)
    basis_indices.sort(key=lambda i: bin(rows[i]).count('1'))
    # Sort non-basis indices by popcount
    non_basis_indices.sort(key=lambda i: bin(rows[i]).count('1'))
    
    # Combined processing order
    processing_order = basis_indices + non_basis_indices
    
    # Map from bitmask value to wire index
    val_to_wires = {}
    
    # Initialize with input wires
    for j in range(n):
        val = 1 << j
        val_to_wires[val] = j
    
    # Constant zero
    val_to_wires[0] = n
    
    gates = []
    next_wire = n + 1
    
    # Track which rows have been computed
    row_to_wire = [None] * m
    
    def popcount(x):
        return bin(x).count('1')
        
    def build_chain(target):
        """Fallback: build target from inputs via chain."""
        nonlocal next_wire, gates, val_to_wires
        if target == 0:
            return n
        if target in val_to_wires:
            return val_to_wires[target]
            
        # Find lowest set bit
        lowest_bit = (target & (-target)).bit_length() - 1
        input_wire = lowest_bit
        remaining = target ^ (1 << lowest_bit)
        
        if remaining == 0:
            return input_wire
        
        # Build chain of XORs with inputs
        current_wire = input_wire
        current_val = 1 << lowest_bit
        rem = remaining
        while rem:
            if time.time() > deadline:
                break
            lb = (rem & (-rem)).bit_length() - 1
            gates.append([current_wire, lb])
            current_wire = next_wire
            next_wire += 1
            current_val ^= (1 << lb)
            val_to_wires[current_val] = current_wire
            rem ^= (1 << lb)
        return current_wire

    for idx in processing_order:
        if time.time() > deadline:
            break
        
        target = rows[idx]
        
        # Check if already available
        if target in val_to_wires:
            row_to_wire[idx] = val_to_wires[target]
            continue
        
        # Try to find two existing wires whose XOR equals target
        found = False
        
        # Optimization: iterate over a copy of keys or a list to avoid mutation issues if we added inside loop (we don't, but good practice)
        # Also, we can prioritize 'a' values that are 'close' to target to potentially find a solution faster?
        # Actually, any solution is 1 gate. So first found is fine.
        
        for a, a_wire in val_to_wires.items():
            b = target ^ a
            if b in val_to_wires:
                b_wire = val_to_wires[b]
                # Avoid using the same wire twice (unless it's zero, but target!=0 here)
                if a_wire == b_wire and target != 0:
                    continue
                # Create gate: a_wire XOR b_wire = new_wire
                gates.append([a_wire, b_wire])
                row_to_wire[idx] = next_wire
                val_to_wires[target] = next_wire
                next_wire += 1
                found = True
                break
        
        if not found:
            # Smart Fallback:
            # 1. Find 'a' in val_to_wires that minimizes popcount(target ^ a)
            # 2. Compute diff = target ^ a
            # 3. If diff is not in val_to_wires, build it from inputs
            # 4. XOR a and diff
            
            best_a = None
            best_diff = None
            min_pc = float('inf')
            
            # Iterate to find best helper
            # This is O(N) per row, which is acceptable for small N
            for a, a_wire in val_to_wires.items():
                diff = target ^ a
                pc = popcount(diff)
                if pc < min_pc:
                    min_pc = pc
                    best_a = a
                    best_diff = diff
                    # If diff is 0, we would have found it in the two-sum loop (a=target)
                    # If diff is an input, it's in val_to_wires, so two-sum would have found it.
                    # So we only reach here if diff is NOT in val_to_wires.
            
            if best_a is not None and best_diff is not None:
                a_wire = val_to_wires[best_a]
                # Build diff
                diff_wire = build_chain(best_diff)
                
                # If diff_wire is a new wire created by build_chain, it's in val_to_wires
                # If diff was 0, diff_wire is n
                # If diff was an input, diff_wire is that input (but this case is handled by two-sum)
                
                # XOR a_wire and diff_wire
                # Check if this result is already in val_to_wires? 
                # It should be 'target'. 
                # If target was in val_to_wires, we would have skipped this block.
                # So we create a new gate.
                gates.append([a_wire, diff_wire])
                row_to_wire[idx] = next_wire
                val_to_wires[target] = next_wire
                next_wire += 1
            else:
                # Absolute fallback (should not happen if n>0)
                row_to_wire[idx] = build_chain(target)

    # Build outputs in original order
    for i in range(m):
        if row_to_wire[i] is None:
            # Shouldn't happen, but fallback
            row_to_wire[i] = n
            
    outputs = row_to_wire
    
    return {'gates': gates, 'outputs': outputs}
