import time
import math
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    if k != 3:
        # Fallback for other k, but task is specific to k=3
        # We can return a trivial graph or raise error, but let's try to handle k=3 only for now
        # If k != 3, we might fail the task, but the task specifies k=3, g=16.
        pass

    if g != 16:
        # This construction is specific to g=16 (depth 8)
        # For other g, we might need to adjust depth
        # Depth r = g/2
        # If g is odd, r = (g-1)/2?
        # Let's stick to g=16 as per task.
        pass

    # Tree depth r = g // 2 = 8
    r = g // 2
    if r < 1:
        return 0, []
        
    # Build the 3-regular tree of depth r
    # Node 0 is root.
    # Children of i:
    # We need to assign IDs.
    # Level 0: 0
    # Level 1: 1, 2, 3
    # Level 2: 4..12
    # ...
    
    nodes = []
    edges = []
    leaves = []
    
    # BFS to build tree
    # We need to track parent to avoid adding reverse edges twice?
    # We just add edges (u, v) for u parent, v child.
    
    queue = [(0, 0)] # (node, depth)
    nodes.append(0)
    # Map from (parent, depth) or just node id to depth?
    # We need to know depth to identify leaves.
    depth_map = {0: 0}
    
    next_id = 1
    # To ensure deterministic ordering for leaves, we can use a fixed order.
    # But we need to group by main branch.
    # Main branch is the child of the root.
    
    # Let's store for each node: parent, depth, main_branch
    # main_branch for root is None.
    # For children of root, main_branch is their ID (1, 2, 3).
    # For deeper nodes, main_branch is inherited from parent.
    
    node_info = {0: {'parent': -1, 'depth': 0, 'branch': -1}}
    
    while queue:
        u, d = queue.pop(0)
        if d == r:
            leaves.append(u)
            continue
            
        # Generate 3 children
        # For root, 3 children.
        # For others, 2 children (since 1 is parent).
        # Wait, in a 3-regular tree, root has 3 children.
        # Non-root nodes have 2 children (1 parent, 2 children to maintain degree 3 in the tree part? No, degree 2 in tree, will be 3 in graph).
        
        if d == 0:
            num_children = 3
        else:
            num_children = 2
            
        for i in range(num_children):
            v = next_id
            next_id += 1
            nodes.append(v)
            edges.append((u, v))
            
            if d == 0:
                branch = v
            else:
                branch = node_info[u]['branch']
                
            node_info[v] = {'parent': u, 'depth': d + 1, 'branch': branch}
            queue.append((v, d + 1))
            
    n = next_id
    
    # Group leaves by branch
    branches = {1: [], 2: [], 3: []}
    for leaf in leaves:
        b = node_info[leaf]['branch']
        branches[b].append(leaf)
        
    # Verify counts
    # Each branch should have 2^(r-1) leaves
    expected_leaves_per_branch = 2 ** (r - 1)
    for b in branches:
        if len(branches[b]) != expected_leaves_per_branch:
            # Error in tree construction
            return 0, []
            
    # Create matching
    # L1, L2, L3
    L1 = branches[1]
    L2 = branches[2]
    L3 = branches[3]
    
    # Shuffle to avoid regularity? Not strictly necessary for girth,
    # but randomizing might help if we want to explore other structures later.
    # For now, deterministic is safer.
    random.seed(42)
    random.shuffle(L1)
    random.shuffle(L2)
    random.shuffle(L3)
    
    half = expected_leaves_per_branch // 2
    # 128 leaves, half = 64
    
    # Pairs:
    # L1[0:half] with L2[0:half]
    for i in range(half):
        edges.append((L1[i], L2[i]))
        
    # L1[half:] with L3[0:half]
    for i in range(half):
        edges.append((L1[half + i], L3[i]))
        
    # L2[half:] with L3[half:]
    for i in range(half):
        edges.append((L2[half + i], L3[half + i]))
        
    # Check time
    if time.time() > deadline:
        return 0, []
        
    return n, edges
