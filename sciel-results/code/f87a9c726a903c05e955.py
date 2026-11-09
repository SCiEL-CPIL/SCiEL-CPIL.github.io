"""Exact XOR-circuit synthesis via greedy CSE with shared sub-expression reuse."""
import time
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START

def _row_hash(row):
    """Convert a numpy boolean/int array row to a hashable int."""
    if len(row) == 0:
        return 0
    # Use a fast bit-packing approach
    # Convert to a tuple of bytes for hashing
    b = np.packbits(row.astype(np.uint8))
    return int.from_bytes(b.tobytes(), 'big')

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Normalize input
    if isinstance(matrix, list):
        matrix = np.array(matrix, dtype=np.int8)
    else:
        matrix = matrix.copy().astype(np.int8)
    
    m, n = matrix.shape
    # Input wires: 0..n-1, constant-zero wire: n
    # Gate k produces wire n+1+k
    
    # Represent each vector as a frozenset of wire indices (sparse representation)
    # For dense matrices, we use a different approach: track linear combinations
    
    # Strategy: Greedy CSE
    # 1. Start with target rows as vectors over GF(2)
    # 2. Repeatedly find pairs of vectors whose XOR has lower "cost" (Hamming weight)
    #    and replaces them, creating a new intermediate gate
    # 3. Continue until no beneficial pair exists or depth limit reached
    
    # We'll work with integer representations for speed
    # Each vector is a Python int where bit i represents wire i
    
    def vec_to_int(vec):
        """Convert a boolean array to an int representation."""
        result = 0
        for i, v in enumerate(vec):
            if v:
                result |= (1 << i)
        return result
    
    # Initialize vectors
    # Input wires are identity vectors
    # Target rows are the matrix rows
    
    # For efficiency, represent vectors as Python ints
    targets = []
    for i in range(m):
        targets.append(vec_to_int(matrix[i]))
    
    # Wire registry: maps wire_id -> vector (int representation)
    # Wires 0..n-1 are inputs, wire n is zero
    # Gates will be n+1, n+2, ...
    
    # Current set of vectors we need to produce (with their current wire assignments)
    # We'll use a list of (vector_int, wire_id) where wire_id is the output we need
    # Initially, each target row needs to be produced
    
    # Actually, let's think differently:
    # We have m output requirements. Each is a vector over the n inputs.
    # We can create intermediate gates that compute XORs of available wires.
    # Goal: minimize total number of gates.
    
    # Greedy CSE approach:
    # 1. Maintain a pool of available wires (initially: inputs 0..n-1 and zero n)
    # 2. Maintain a list of target expressions (each is a set of wire indices to XOR)
    # 3. Find common sub-expressions and factor them out
    
    # Better approach: work with the target rows directly
    # Each target row is a subset of input wires
    # We can create intermediate gates that are XORs of subsets
    
    # Let's use a different strategy:
    # Represent each target as a frozenset of input indices
    # Greedily find the most common pair of inputs across targets
    # Create a gate for that pair, update all targets that use both
    
    # Convert to frozenset representation
    target_sets = []
    for i in range(m):
        s = frozenset(j for j in range(n) if matrix[i, j])
        target_sets.append(s)
    
    # Available wires: 0..n-1 (inputs), n (zero)
    # Gates will be assigned IDs starting from n+1
    
    # Count how many targets use each input
    input_counts = defaultdict(int)
    for s in target_sets:
        for x in s:
            input_counts[x] += 1
    
    # Greedy: repeatedly find the pair (a, b) that appears together in the most targets
    # Create a gate for a XOR b, then replace {a, b} with {gate} in those targets
    
    gates = []  # List of [wire_a, wire_b]
    gate_id = n + 1  # Next gate wire ID
    
    # Track which targets have been simplified
    # We'll work with a list of sets (mutable copies)
    current_sets = [set(s) for s in target_sets]
    
    # For efficiency, maintain pair counts
    # pair_count[(a,b)] = number of targets containing both a and b
    # Only consider pairs where both are "simple" (input wires or single gates)
    
    # Actually, for this problem size (up to 16x16), let's do a simpler greedy:
    # At each step, find the pair of wires that appear together in the most targets
    # where both wires are currently "available" (inputs or previously created gates)
    
    # Let's track available wires
    available_wires = set(range(n))  # Input wires
    # Zero wire n is also available but XOR with zero is identity, so skip it
    
    # We'll iteratively:
    # 1. Count co-occurrences of wire pairs in current_sets
    # 2. Pick the pair with highest count (tie-break: lower sum of individual frequencies)
    # 3. Create a gate, update all sets that contain both wires
    
    max_iterations = m * n * 2  # Upper bound on useful gates
    iterations = 0
    
    while iterations < max_iterations:
        if time.time() > deadline:
            break
        
        # Count co-occurrences
        pair_counts = defaultdict(int)
        for s in current_sets:
            if len(s) <= 1:
                continue
            elements = list(s)
            for i in range(len(elements)):
                for j in range(i+1, len(elements)):
                    a, b = elements[i], elements[j]
                    if a > b:
                        a, b = b, a
                    pair_counts[(a, b)] += 1
        
        if not pair_counts:
            break
        
        # Find best pair: highest count, then lowest total frequency
        best_pair = None
        best_count = 0
        best_freq_sum = float('inf')
        
        for (a, b), count in pair_counts.items():
            if count > best_count or (count == best_count):
                # Frequency of a and b in current sets
                freq_a = sum(1 for s in current_sets if a in s)
                freq_b = sum(1 for s in current_sets if b in s)
                freq_sum = freq_a + freq_b
                if count > best_count or (count == best_count and freq_sum < best_freq_sum):
                    best_pair = (a, b)
                    best_count = count
                    best_freq_sum = freq_sum
        
        if best_pair is None or best_count <= 1:
            break
        
        a, b = best_pair
        gate_wire = gate_id
        gates.append([a, b])
        gate_id += 1
        
        # Update all sets that contain both a and b
        for s in current_sets:
            if a in s and b in s:
                s.remove(a)
                s.remove(b)
                s.add(gate_wire)
        
        available_wires.add(gate_wire)
        iterations += 1
    
    # Now each target set contains wires that need to be XORed together
    # We need to produce each target from these wires using additional gates
    
    # For each target, if it has more than 1 wire, we need gates to combine them
    # We can share gates across targets if they use the same sub-expressions
    
    # Simple approach: for each target, create a chain of gates to combine its wires
    # But this doesn't share. Let's do a second pass of CSE on the remaining structure.
    
    # Actually, let's just create gates for each target's remaining wires
    # Wire k in a target means we need to XOR all wires in that set
    
    # For a set {w1, w2, ..., wk}, we need k-1 gates to compute the XOR
    # We can do: gate1 = w1 XOR w2, gate2 = gate1 XOR w3, ..., final = gate_{k-1} XOR wk
    
    # But we can share: if two targets both need (w1 XOR w2), reuse the gate
    
    # Let's build a DAG of needed computations
    # For each target, decompose its set into a tree of XORs
    # Use memoization to share sub-trees
    
    # Simpler: just emit gates for each target independently
    # This is the "independent-row baseline" which we've already improved upon
    
    # To do better, let's do another round of CSE on the remaining sets
    # But for now, let's just compute the remaining gates
    
    # For each target set, if it has size > 1, create gates to combine
    output_wires = []
    
    for s in current_sets:
        if len(s) == 0:
            # XOR of nothing is 0
            output_wires.append(n)  # Zero wire
        elif len(s) == 1:
            # Single wire, just use it
            output_wires.append(list(s)[0])
        else:
            # Need to XOR all wires in the set
            # Create a chain
            wires = list(s)
            current = wires[0]
            for w in wires[1:]:
                new_gate = gate_id
                gates.append([current, w])
                gate_id += 1
                current = new_gate
            output_wires.append(current)
    
    return {'gates': gates, 'outputs': output_wires}

# EVOLVE-BLOCK-END
