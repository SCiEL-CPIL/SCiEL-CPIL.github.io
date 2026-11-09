"""XOR circuit synthesis with Dijkstra-based shortest path for optimal gate sharing."""
import time
import heapq

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    
    if m == 0 or n == 0:
        return {'gates': [], 'outputs': []}
    
    # Convert matrix rows to integer bitmasks
    rows = []
    for i in range(m):
        val = 0
        for j in range(n):
            if matrix[i][j]:
                val |= (1 << j)
        rows.append(val)
    
    # Use Dijkstra's algorithm to find minimum-cost XOR circuit
    # State space: all possible XOR combinations of input wires
    # Start from zero (cost 0), expand by XOR-ing with available wires
    
    # Available wires: input wires 0..n-1 (each has cost 0 to "use")
    # New gates produce new wires with cost 1 each
    
    # We want to minimize total number of gates
    # Dijkstra over the space of XOR values
    
    # dist[val] = minimum number of gates to produce val
    # parent[val] = (prev_val, wire_used) to reconstruct the circuit
    
    # Initialize: input wires have cost 0
    dist = {}
    parent = {}
    
    # Zero wire is always available
    dist[0] = 0
    parent[0] = (None, None)  # base case
    
    # Input wires are free (they're given)
    for j in range(n):
        val = 1 << j
        dist[val] = 0
        parent[val] = (0, j)  # XOR with input j
    
    # Priority queue: (cost, value)
    pq = [(0, 0)]  # Start from zero
    for j in range(n):
        pq.append((0, 1 << j))
    
    # Target values we need to compute
    targets = set(rows)
    targets.discard(0)  # Zero is already available
    
    # Dijkstra's algorithm
    while pq and len(dist) < (1 << n) * 2:  # Safety bound
        if time.time() > deadline:
            break
        
        cost, val = heapq.heappop(pq)
        
        # Skip if we've already found a better path
        if val in dist and dist[val] < cost:
            continue
        
        # Check if we've found all targets
        if targets and all(t in dist for t in targets):
            break
        
        # Expand: XOR current value with each input wire
        for j in range(n):
            new_val = val ^ (1 << j)
            new_cost = cost + (1 if val != 0 else 0)  # Using input wire is free if we're at zero
            
            # Actually, let me reconsider the cost model
            # Each gate produces a new wire
            # If we XOR two existing wires, that's 1 gate
            # Input wires are "free" in the sense they don't require gates
            # But we need to track which wires are available
            
            # Let me use a different approach: 
            # dist[val] = minimum gates to produce val
            # When we produce new_val = val ^ (1<<j), we need 1 gate if val != 0
            # (XOR val with input j)
            # If val == 0, new_val = 1<<j which is already an input, cost 0
            
            if new_val in dist:
                if dist[new_val] <= new_cost:
                    continue
            
            dist[new_val] = new_cost
            parent[new_val] = (val, j)  # new_val = val XOR input_j
            
            heapq.heappush(pq, (new_cost, new_val))
    
    # Reconstruct the circuit
    gates = []
    wire_names = {}  # value -> wire index
    next_wire = n + 1
    
    # Assign wire names to all values we need
    # We need to topologically sort the parent chain
    
    # First, collect all values that are parents of targets
    all_values = set()
    for t in targets:
        if t not in dist:
            # Fallback: should not happen, but handle gracefully
            continue
        v = t
        while v != 0 and v not in wire_names:
            all_values.add(v)
            if v in parent:
                prev_v, _ = parent[v]
                v = prev_v
            else:
                break
        # Also add the zero wire
        all_values.add(0)
    
    # Process values in order of increasing cost (topological order)
    values_by_cost = sorted(all_values, key=lambda x: dist.get(x, float('inf')))
    
    for val in values_by_cost:
        if val in wire_names:
            continue
        
        if val == 0:
            wire_names[val] = n
        elif val in parent:
            prev_val, input_j = parent[val]
            if prev_val == 0:
                # val = 0 XOR input_j = input_j, already available
                wire_names[val] = input_j
            else:
                # val = prev_val XOR input_j
                # Need to create a gate
                wire_names[val] = next_wire
                gates.append([wire_names[prev_val], input_j])
                next_wire += 1
        else:
            # Should be an input wire
            # Find which input
            if val != 0:
                j = (val & (-val)).bit_length() - 1
                wire_names[val] = j
    
    # Build outputs
    outputs = []
    for i in range(m):
        row = rows[i]
        if row == 0:
            outputs.append(n)
        elif row in wire_names:
            outputs.append(wire_names[row])
        else:
            # Fallback: shouldn't happen, but use constant zero
            outputs.append(n)
    
    return {'gates': gates, 'outputs': outputs}
