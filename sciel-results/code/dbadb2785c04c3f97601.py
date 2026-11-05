"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from itertools import permutations
import random

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    
    all_syms = set(range(1, n+1))
    
    def get_missing(node):
        node_set = set(node)
        for s in all_syms:
            if s not in node_set:
                return s
        return None

    def get_next(node):
        s = get_missing(node)
        return node[1:] + (s,)
        
    def node_to_perm(node):
        s = get_missing(node)
        return node + (s,)

    # Find all cycles
    all_nodes = list(permutations(range(1, n+1), n-1))
    visited = set()
    cycles = []
    
    for node in all_nodes:
        if node not in visited:
            cycle = []
            cur = node
            while cur not in visited:
                visited.add(cur)
                cycle.append(cur)
                cur = get_next(cur)
            cycles.append(cycle)
            
    m = len(cycles) # n-1 cycles
    
    # Precompute overlap function
    def calc_overlap(u, v):
        # u and v are tuples of length n-1
        # Returns max k such that u[-k:] == v[:k]
        max_ovl = 0
        for k in range(1, n):
            if u[n-1-k:] == v[:k]:
                max_ovl = k
        return max_ovl

    # Local Search to find best order and node selection
    # State: list of (cycle_index, node_index_in_cycle)
    # We want to maximize sum of overlaps between consecutive nodes in the sequence
    
    def evaluate(state):
        # state is list of (ci, ni)
        total_ovl = 0
        for i in range(len(state) - 1):
            c1, n1 = state[i]
            c2, n2 = state[i+1]
            node1 = cycles[c1][n1]
            node2 = cycles[c2][n2]
            total_ovl += calc_overlap(node1, node2)
        return total_ovl

    best_state = None
    best_score = -1
    
    # Run multiple random restarts
    for restart in range(20):
        # Initialize state
        state = []
        for ci in range(m):
            # Pick a random node from cycle ci
            ni = random.randint(0, len(cycles[ci])-1)
            state.append((ci, ni))
        
        # Shuffle order
        random.shuffle(state)
        
        current_score = evaluate(state)
        
        improved = True
        while improved:
            improved = False
            
            # 1. Try changing node for each cycle in the current order
            for i in range(m):
                ci, ni = state[i]
                best_ni = ni
                best_local_score = current_score
                
                # Calculate current contribution of this node
                # It affects overlap with previous and next
                prev_node = None
                if i > 0:
                    prev_node = cycles[state[i-1][0]][state[i-1][1]]
                next_node = None
                if i < m - 1:
                    next_node = cycles[state[i+1][0]][state[i+1][1]]
                
                current_node = cycles[ci][ni]
                
                # Base score excluding overlaps involving current node
                # This is slightly complex, easier to just re-evaluate or delta
                # Let's just try all nodes and keep the best delta
                
                for new_ni in range(len(cycles[ci])):
                    if new_ni == ni:
                        continue
                    new_node = cycles[ci][new_ni]
                    
                    delta = 0
                    if prev_node:
                        delta -= calc_overlap(prev_node, current_node)
                        delta += calc_overlap(prev_node, new_node)
                    if next_node:
                        delta -= calc_overlap(current_node, next_node)
                        delta += calc_overlap(new_node, next_node)
                    
                    if current_score + delta > best_local_score:
                        best_local_score = current_score + delta
                        best_ni = new_ni
                
                if best_ni != ni:
                    state[i] = (ci, best_ni)
                    current_score = best_local_score
                    improved = True
            
            # 2. Try swapping order of two cycles
            for i in range(m):
                for j in range(i+1, m):
                    # Swap state[i] and state[j]
                    state[i], state[j] = state[j], state[i]
                    new_score = evaluate(state)
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                    else:
                        # Swap back
                        state[i], state[j] = state[j], state[i]
                        
            if current_score > best_score:
                best_score = current_score
                best_state = state[:]
                
    # Use best_state to build the word
    if not best_state:
        # Fallback to simple greedy if local search failed
        best_state = [(i, 0) for i in range(m)]
        
    # Order the cycles according to best_state
    # best_state is a list of (cycle_idx, node_idx)
    # The order in the list is the order of concatenation
    
    word = []
    for idx, (ci, ni) in enumerate(best_state):
        cycle = cycles[ci]
        start_node = cycle[ni]
        
        # Generate the string for this cycle starting at start_node
        # We need to traverse the cycle starting from start_node
        # The string is: start_node + missing(start_node) + ...
        
        # Since it's a cycle, we can just iterate
        cur = start_node
        for step in range(len(cycle)):
            if not word:
                word.extend(list(cur))
            else:
                # Overlap with current end of word
                # The word currently ends with the previous node's tuple?
                # No, the word is a flat list of symbols.
                # When we append a cycle, we overlap the end of the previous word with the start of this cycle.
                # The start of this cycle is `start_node`.
                # The end of the previous word is the `end_node` of the previous cycle.
                # But we are building sequentially.
                
                # Let's just build the full string for the cycle first, then merge?
                # No, merging is easier on the fly.
                
                # Actually, the "overlap" logic in the previous program was:
                # Append the new permutation, overlapping with the end of the current word.
                
                # Here, we are appending a whole cycle.
                # The cycle string starts with `start_node`.
                # We should overlap the end of `word` with `start_node`.
                
                max_ovl = 0
                for k in range(1, n):
                    if k <= len(word):
                        if tuple(word[-k:]) == start_node[:k]:
                            max_ovl = k
                
                if max_ovl == 0:
                    # No overlap, just append
                    # But wait, if max_ovl is 0, we should append the whole start_node?
                    # The cycle string is start_node + rest.
                    # If no overlap, we append start_node.
                    word.extend(list(start_node))
                else:
                    # We have overlapped k symbols.
                    # We need to append the rest of start_node?
                    # start_node is length n-1.
                    # We overlapped k.
                    # So we append start_node[k:]?
                    # Yes.
                    word.extend(list(start_node[max_ovl:]))
            
            # Move to next node in cycle
            cur = get_next(cur)
            if step < len(cycle) - 1:
                # Append the missing symbol of cur?
                # The next symbol in the string is the missing symbol of the current node.
                # Wait.
                # String for cycle:
                # Node u. Edge u->v.
                # Symbols: u (n-1 symbols) + missing(u) (1 symbol).
                # Next node v.
                # So after appending start_node (partially), we need to append the rest of the cycle.
                
                # Let's refine the loop.
                # It's safer to generate the full string for the cycle and then merge.
                pass

    # Let's restart the word construction part to be cleaner
    word = []
    for idx, (ci, ni) in enumerate(best_state):
        cycle = cycles[ci]
        start_node = cycle[ni]
        
        # Generate full string for this cycle
        cycle_str = []
        cur = start_node
        for step in range(len(cycle)):
            if not cycle_str:
                cycle_str.extend(list(cur))
            else:
                # Append missing symbol
                m_sym = get_missing(cur)
                cycle_str.append(m_sym)
            cur = get_next(cur)
            
        # Merge cycle_str into word
        if not word:
            word = cycle_str
        else:
            # Find max overlap between end of word and start of cycle_str
            max_ovl = 0
            # cycle_str starts with start_node
            # word ends with end_node of previous cycle
            # Max possible overlap is n-1
            for k in range(1, min(n, len(word), len(cycle_str))):
                if tuple(word[-k:]) == tuple(cycle_str[:k]):
                    max_ovl = k
            
            word.extend(cycle_str[max_ovl:])
            
    return "".join(map(str, word))

# EVOLVE-BLOCK-END
