import itertools
import collections

def construct(n):
    if n <= 1:
        return "1"
    
    # Nodes are permutations of length n-1 from symbols 1..n
    # There are n! such nodes.
    # We represent them as tuples of ints.
    
    # Generate all nodes
    nodes = []
    node_to_idx = {}
    for perm in itertools.permutations(range(1, n + 1)):
        # A node is a tuple of length n-1
        # It must contain n-1 distinct symbols from 1..n
        # So we choose n-1 symbols and permute them?
        # No, the node is just a sequence of n-1 distinct symbols from 1..n.
        # Count: P(n, n-1) = n!
        pass
    
    # Better generation:
    # Iterate over all permutations of 1..n.
    # Each permutation P defines a node u = P[:n-1].
    # But u is not unique to P.
    # We just need the set of all valid nodes.
    
    all_nodes = set()
    for perm in itertools.permutations(range(1, n + 1)):
        all_nodes.add(perm[:n-1])
        
    nodes = list(all_nodes)
    n_nodes = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # Build adjacency for Good and Bad edges
    # Good: u -> v where v = u[1:] + [x] and x is the missing symbol in u
    # Bad: u -> v where v = u[1:] + [u[0]]
    
    next_good = [0] * n_nodes
    next_bad = [0] * n_nodes
    
    for i, u in enumerate(nodes):
        # Find missing symbol
        present = set(u)
        all_syms = set(range(1, n + 1))
        missing = list(all_syms - present)[0]
        
        v_good = u[1:] + (missing,)
        v_bad = u[1:] + (u[0],)
        
        next_good[i] = node_to_idx[v_good]
        next_bad[i] = node_to_idx[v_bad]
        
    # Find cycles in the Good graph
    visited = [False] * n_nodes
    cycles = []
    
    for i in range(n_nodes):
        if not visited[i]:
            cycle = []
            cur = i
            while not visited[cur]:
                visited[cur] = True
                cycle.append(cur)
                cur = next_good[cur]
            cycles.append(cycle)
            
    # If there is only one cycle, we are done
    if len(cycles) == 1:
        cycle = cycles[0]
        # Start at cycle[0]
        word = list(nodes[cycle[0]])
        # Traverse all edges in the cycle
        for idx in cycle:
            u = nodes[idx]
            # The edge is u -> next_good[u]
            # The symbol appended is the last element of next_good[u]
            v = nodes[next_good[idx]]
            word.append(v[-1])
        return "".join(map(str, word))
        
    # Multiple cycles: We need to chain them.
    # We will greedily pick a cycle, traverse it, and jump to the next.
    # To jump, we use Bad edges.
    # We want to minimize the number of Bad edges.
    # Ideally, we can jump from the end of one cycle to the start of another with 1 Bad edge?
    # From node u, we can go to next_bad[u].
    # We want next_bad[u] to be in an unvisited cycle.
    
    unvisited_cycles = [i for i in range(len(cycles))]
    current_node = cycles[0][0]
    word = list(nodes[current_node])
    
    # Mark first cycle as visited
    visited_cycle_idx = 0
    unvisited_cycles.remove(0)
    
    # We are at current_node. We traverse the cycle.
    # We need to decide WHERE to leave the cycle.
    # We should leave at a node u such that next_bad[u] is in an unvisited cycle.
    # Or, we can traverse the whole cycle and then jump?
    # If we traverse the whole cycle, we end at the start node.
    # Let's try to find a node in the current cycle that has a Bad edge to an unvisited cycle.
    
    # Precompute for each node, which cycle its Bad edge leads to
    node_to_cycle = [0] * n_nodes
    for c_idx, cycle in enumerate(cycles):
        for node_idx in cycle:
            node_to_cycle[node_idx] = c_idx
            
    bad_edge_target_cycle = [0] * n_nodes
    for i in range(n_nodes):
        target_node = next_bad[i]
        bad_edge_target_cycle[i] = node_to_cycle[target_node]
        
    while unvisited_cycles:
        # Find a node in the current cycle that can jump to an unvisited cycle
        # Current cycle is cycles[visited_cycle_idx]
        current_cycle = cycles[visited_cycle_idx]
        
        jump_node = -1
        target_cycle_idx = -1
        
        for node_idx in current_cycle:
            target_c = bad_edge_target_cycle[node_idx]
            if target_c in unvisited_cycles:
                jump_node = node_idx
                target_cycle_idx = target_c
                break
                
        if jump_node == -1:
            # No direct jump. We might need to traverse the cycle and try again?
            # Or we might be stuck. For now, just traverse the whole cycle and hope for the best.
            # This might result in a very long string or failure.
            # Given the structure, direct jumps should exist.
            pass
            
        # Traverse from current_node to jump_node
        # current_node is the start of the cycle.
        # We need to follow Good edges until we reach jump_node.
        cur = current_node
        while cur != jump_node:
            # Append the symbol for the Good edge from cur
            v = nodes[next_good[cur]]
            word.append(v[-1])
            cur = next_good[cur]
            
        # Now we are at jump_node.
        # Take the Bad edge to jump to the next cycle.
        v_bad = nodes[next_bad[jump_node]]
        word.append(v_bad[-1])
        
        # Update state
        visited_cycle_idx = target_cycle_idx
        unvisited_cycles.remove(target_cycle_idx)
        current_node = v_bad
        
        # Note: We are now at the start of the new cycle?
        # v_bad is a node in the new cycle. We will start traversing from there.
        
    # Traverse the last cycle fully
    current_cycle = cycles[visited_cycle_idx]
    cur = current_node
    # We need to traverse all nodes in this cycle starting from current_node
    # Since it's a cycle, we can just go around once.
    # But we must ensure we cover all Good edges.
    # If we start at current_node and follow Good edges, we will cover the cycle.
    # We stop when we return to current_node?
    # No, we just need to append the symbols.
    
    # To be safe, we traverse all nodes in the cycle in order, starting from current_node.
    # Find the index of current_node in the cycle
    start_idx = current_cycle.index(current_node)
    
    # Traverse the cycle
    for i in range(len(current_cycle)):
        node_idx = current_cycle[(start_idx + i) % len(current_cycle)]
        v = nodes[next_good[node_idx]]
        word.append(v[-1])
        
    return "".join(map(str, word))
