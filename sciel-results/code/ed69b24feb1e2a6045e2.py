"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import itertools

def _generate_initial_order(n):
    """Generate the standard recursive order of (n-1)-permutations."""
    if n == 1:
        return [(1,)]
    if n == 2:
        return [(1,), (2,)]
    
    # Get the standard order of (n-1)-permutations
    prev_order = _generate_initial_order(n-1)
    # In the standard construction, we insert the new symbol m at each position
    # and also at the end, creating a specific ordering
    # The standard superpermutation construction uses a specific traversal
    # For n=3: 121312123121312132132123121312132132123121312132132123121312132132123
    # Let's use the known recursive structure
    
    # Actually, let's just generate permutations in lexicographic order for simplicity
    # and rely on VNS to improve it
    return list(itertools.permutations(range(1, n)))

def _compute_length(order, n):
    """Compute the total length of the superpermutation given an order of (n-1)-permutations."""
    if not order:
        return 0
    
    # Each (n-1)-permutation p becomes: p, n, p
    # We need to compute overlaps between consecutive blocks
    blocks = []
    for p in order:
        block = list(p) + [n] + list(p)
        blocks.append(block)
    
    total_length = len(blocks[0])
    for i in range(1, len(blocks)):
        prev_block = blocks[i-1]
        curr_block = blocks[i]
        
        # Find maximum overlap: suffix of prev_block that matches prefix of curr_block
        max_overlap = 0
        max_possible = min(len(prev_block), len(curr_block))
        for o in range(max_possible, 0, -1):
            if prev_block[-o:] == curr_block[:o]:
                max_overlap = o
                break
        
        total_length += len(curr_block) - max_overlap
    
    return total_length

def _apply_2opt(order, i, j):
    """Apply 2-opt: reverse the segment from index i to j."""
    new_order = order[:i] + list(reversed(order[i:j+1])) + order[j+1:]
    return new_order

def _apply_oropt(order, i, length, new_pos):
    """Apply Or-opt: move a segment of 'length' elements starting at i to new_pos."""
    segment = order[i:i+length]
    remaining = order[:i] + order[i+length:]
    # Adjust new_pos if it's after the removed segment
    if new_pos > i + length:
        new_pos -= length
    elif new_pos > i:
        new_pos -= 0  # No adjustment needed if new_pos is within the removed segment's original range
    # Insert segment at new_pos
    new_order = remaining[:new_pos] + segment + remaining[new_pos:]
    return new_order

def _vns_improve(order, n, max_iter=50000):
    """Apply Variable Neighborhood Search to improve the order."""
    current_order = list(order)
    current_length = _compute_length(current_order, n)
    
    best_order = list(current_order)
    best_length = current_length
    
    no_improve_count = 0
    max_no_improve = 5000
    
    for iteration in range(max_iter):
        if no_improve_count > max_no_improve:
            break
        
        improved = False
        m = len(current_order)
        
        # Try 2-opt moves
        for _ in range(100):
            if m < 2:
                break
            i = random.randint(0, m - 2)
            j = random.randint(i + 1, m - 1)
            
            new_order = _apply_2opt(current_order, i, j)
            new_length = _compute_length(new_order, n)
            
            if new_length < current_length:
                current_order = new_order
                current_length = new_length
                improved = True
                if current_length < best_length:
                    best_length = current_length
                    best_order = list(current_order)
                no_improve_count = 0
                break
        
        if improved:
            continue
        
        # Try Or-opt moves
        for _ in range(100):
            if m < 3:
                break
            length = random.choice([1, 2, 3])
            if length >= m:
                length = max(1, m - 1)
            i = random.randint(0, m - length - 1)
            new_pos = random.randint(0, m - length)
            
            if new_pos == i:
                continue
            
            new_order = _apply_oropt(current_order, i, length, new_pos)
            new_length = _compute_length(new_order, n)
            
            if new_length < current_length:
                current_order = new_order
                current_length = new_length
                improved = True
                if current_length < best_length:
                    best_length = current_length
                    best_order = list(current_order)
                no_improve_count = 0
                break
        
        if improved:
            continue
        
        no_improve_count += 1
    
    return best_order, best_length

def _build_superpermutation(order, n):
    """Build the actual superpermutation string from the order of (n-1)-permutations."""
    if n == 1:
        return "1"
    
    blocks = []
    for p in order:
        block = list(p) + [n] + list(p)
        blocks.append(block)
    
    if not blocks:
        return ""
    
    result = list(blocks[0])
    for i in range(1, len(blocks)):
        prev_block = blocks[i-1]
        curr_block = blocks[i]
        
        # Find maximum overlap
        max_overlap = 0
        max_possible = min(len(prev_block), len(curr_block))
        for o in range(max_possible, 0, -1):
            if prev_block[-o:] == curr_block[:o]:
                max_overlap = o
                break
        
        result.extend(curr_block[max_overlap:])
    
    return "".join(map(str, result))

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "1212"
    
    # Generate initial order using standard recursive construction
    # Use lexicographic order as starting point
    initial_order = list(itertools.permutations(range(1, n)))
    
    # Apply VNS to improve
    improved_order, improved_length = _vns_improve(initial_order, n, max_iter=30000)
    
    # Build the superpermutation
    word = _build_superpermutation(improved_order, n)
    
    return word
# EVOLVE-BLOCK-END
