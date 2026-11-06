"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def synthesize(matrix, depth_limit):
    """
    Synthesize an XOR circuit for the given matrix.
    
    Args:
        matrix: Binary matrix (m x n) where m is number of outputs, n is number of inputs
        depth_limit: Maximum depth of the circuit
    
    Returns:
        dict with 'gates' (list of [wire1, wire2]) and 'outputs' (list of wire numbers)
    """
    start_time = time.time()
    deadline = start_time + 55  # 55 second deadline for safety
    
    # Convert to numpy if needed
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)
    
    m, n = matrix.shape
    
    # Strategy: Greedy pair sharing
    # 1. Identify frequent pairs of inputs across rows
    # 2. Create gates for these pairs
    # 3. Express each row using these shared gates and remaining inputs
    
    # Step 1: Count occurrences of each pair in rows
    pair_counts = {}
    for i in range(m):
        row = matrix[i]
        support = np.where(row == 1)[0]
        for a in range(len(support)):
            for b in range(a+1, len(support)):
                pair = (support[a], support[b])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
    
    # Step 2: Keep pairs with count >= 2 (potential sharing)
    shared_pairs = [p for p, c in pair_counts.items() if c >= 2]
    shared_pairs.sort(key=lambda p: pair_counts[p], reverse=True)
    
    # Step 3: Build the gate list
    # Input wires: 0..n-1, constant-zero: n
    # Gate k produces wire n+1+k
    
    gates = []  # list of [wire1, wire2]
    wire_map = {}  # maps (pair) -> wire number
    next_wire = n + 1  # first gate wire
    
    # Add gates for shared pairs
    for pair in shared_pairs:
        i, j = pair
        wire = next_wire
        next_wire += 1
        gates.append([i, j])
        wire_map[pair] = wire
    
    # Step 4: For each row, compute the output using available shared gates
    outputs = []
    for i in range(m):
        row = matrix[i]
        support = np.where(row == 1)[0]
        
        if len(support) == 0:
            outputs.append(n)  # constant zero wire
            continue
        
        if len(support) == 1:
            outputs.append(support[0])
            continue
            
        # Try to use shared gates to reduce XOR count
        # We need to find a matching of pairs in the support that are in wire_map
        # Greedy matching: pick pairs that are in wire_map and don't conflict
        
        used_in_matching = set()
        chosen_pairs = []
        
        # Sort available pairs by frequency (or just iterate)
        # For efficiency, we can just check all pairs in wire_map
        # But wire_map can be large. Let's check pairs formed by support elements.
        
        # Actually, let's just iterate through shared_pairs that are subsets of support
        # This is O(|shared_pairs| * |support|^2) worst case, but support is small.
        
        # Better: For each pair in support, check if it's in wire_map
        # We need a maximum matching in the graph of support nodes where edges are shared pairs
        # Since support size is small (<= n), we can do a simple greedy matching
        
        # Build adjacency list for support nodes
        support_set = set(support)
        adj = {s: [] for s in support}
        
        for pair in shared_pairs:
            a, b = pair
            if a in support_set and b in support_set:
                adj[a].append(b)
                adj[b].append(a)
        
        # Greedy matching: pick edges that don't share vertices
        # Sort edges by some heuristic? For now, just pick arbitrarily
        # To be deterministic, we can sort by the wire number (which is by frequency)
        
        # Let's collect all valid edges
        valid_edges = []
        for a in support:
            for b in adj[a]:
                if a < b: # avoid duplicates
                    valid_edges.append((a, b))
        
        # Sort by frequency (wire_map value is assigned by frequency)
        # Lower wire number = higher frequency
        valid_edges.sort(key=lambda e: wire_map[(e[0], e[1])])
        
        for a, b in valid_edges:
            if a not in used_in_matching and b not in used_in_matching:
                chosen_pairs.append((a, b))
                used_in_matching.add(a)
                used_in_matching.add(b)
        
        # Now compute the output
        # Elements to XOR:
        # 1. Wires for chosen pairs
        # 2. Remaining support elements (not in any chosen pair)
        
        elements = []
        for pair in chosen_pairs:
            elements.append(wire_map[pair])
        
        for elem in support:
            if elem not in used_in_matching:
                elements.append(elem)
        
        if not elements:
            outputs.append(n)
        elif len(elements) == 1:
            outputs.append(elements[0])
        else:
            # XOR all elements together
            # Chain XORs: ((e1 XOR e2) XOR e3) ...
            current = elements[0]
            for elem in elements[1:]:
                wire = next_wire
                next_wire += 1
                gates.append([current, elem])
                current = wire
            outputs.append(current)
    
    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
