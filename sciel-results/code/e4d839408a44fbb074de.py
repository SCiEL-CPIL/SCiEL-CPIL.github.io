"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict
import random
from functools import lru_cache
import sys

sys.setrecursionlimit(100000)

def synthesize(matrix, depth_limit):
    """
    Synthesize an XOR circuit for the given binary matrix.
    
    Args:
        matrix: 2D binary array (n_rows x n_cols) where each row represents an output function
                of the input variables (0..n_cols-1).
        depth_limit: Maximum allowed depth of the circuit.
                     
    Returns:
        dict with keys:
            'gates': list of [wire1, wire2] pairs, where each gate computes XOR of wire1 and wire2.
            'outputs': list of wire indices corresponding to the outputs of the circuit.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    A = np.array(matrix, dtype=np.int8)
    if A.ndim == 1:
        A = A.reshape(1, -1)
    
    n_rows, n_cols = A.shape
    n_inputs = n_cols
    zero_wire = n_inputs
    
    if n_rows == 0 or n_cols == 0:
        return {'gates': [], 'outputs': [zero_wire] * n_rows}
    
    # Check for zero rows
    row_popcounts = A.sum(axis=1)
    zero_rows = set(np.where(row_popcounts == 0)[0])
    
    # Represent each row as a frozenset of input indices
    row_fsets = []
    for i in range(n_rows):
        if i in zero_rows:
            row_fsets.append(frozenset())
        else:
            row_fsets.append(frozenset(np.where(A[i] == 1)[0].tolist()))
    
    def elapsed_ok():
        return time.time() < deadline
    
    def count_gates_in_tree(tree):
        """Count total gates in an XOR tree representation."""
        if tree is None:
            return 0
        if isinstance(tree, tuple) and len(tree) == 3:
            _, left, right = tree
            return 1 + count_gates_in_tree(left) + count_gates_in_tree(right)
        return 0
    
    def tree_to_gates(tree, next_wire_ref, gates, wire_depth, depth_limit):
        """
        Convert an XOR tree to a flat gate list.
        tree is either an int (input wire) or (op, left, right) where op=0 means XOR.
        Returns the wire index for the result.
        """
        if isinstance(tree, int):
            return tree
        # tree = (0, left, right)
        _, left, right = tree
        left_wire = tree_to_gates(left, next_wire_ref, gates, wire_depth, depth_limit)
        right_wire = tree_to_gates(right, next_wire_ref, gates, wire_depth, depth_limit)
        new_wire = next_wire_ref[0]
        next_wire_ref[0] += 1
        gates.append([left_wire, right_wire])
        wire_depth[new_wire] = max(wire_depth.get(left_wire, 0), wire_depth.get(right_wire, 0)) + 1
        return new_wire
    
    def try_branch_and_bound():
        """
        Branch and bound search for optimal XOR tree decomposition.
        For each target set, find the minimum-gate XOR tree.
        Uses memoization and pruning.
        """
        # First, get the unique non-empty sets we need to compute
        unique_sets = set()
        for fs in row_fsets:
            if len(fs) > 0:
                unique_sets.add(fs)
        
        if not unique_sets:
            return {'gates': [], 'outputs': [zero_wire] * n_rows}
        
        # For each set, we want to find the best XOR tree.
        # An XOR tree for set S is either:
        #   - A single input (if |S| == 1)
        #   - XOR of two subtrees whose sets XOR to S
        # 
        # We use DP: best[S] = (gate_count, tree)
        # where tree is a representation we can later flatten.
        
        # To make this tractable, we limit the search depth/complexity.
        # Key insight: for a set of size k, the optimal tree has at most k-1 gates
        # if we don't share subexpressions. With sharing, it can be less.
        
        # Strategy: For each target set, try all possible splits into two subsets
        # whose XOR equals the target. Memoize results.
        
        # To keep this bounded, we'll use a heuristic: only consider "balanced" splits
        # and limit the number of candidates explored.
        
        max_set_size = max(len(s) for s in unique_sets)
        
        # If sets are small, do exact search. Otherwise, use greedy.
        if max_set_size <= 6:
            # Exact DP for small sets
            all_relevant_sets = set()
            # Generate all subsets that could appear as intermediate results
            # This is exponential, so we limit it.
            
            # Better approach: iterative deepening on gate count
            # For each target set, try to find a tree with g gates, starting from g=0
            
            def min_gates_for_set(target_set, max_depth_allowed, memo):
                """Find minimum gates to compute target_set."""
                if target_set in memo:
                    return memo[target_set]
                
                if len(target_set) == 0:
                    memo[target_set] = (0, None)  # zero wire
                    return memo[target_set]
                
                if len(target_set) == 1:
                    # Single input
                    wire = list(target_set)[0]
                    memo[target_set] = (0, wire)  # 0 gates, just the input
                    return memo[target_set]
                
                # Try all splits: target = A XOR B, where A and B are non-empty
                # and A != B (to avoid redundancy), and A XOR B == target
                # This means B = A XOR target, so we need A and (A XOR target) to be "valid"
                
                best_count = float('inf')
                best_tree = None
                
                # Enumerate subsets A of target (or related sets)
                # Key: A XOR B = target => B = A XOR target
                # We need both A and B to be "simple" (small or already computed)
                
                # Heuristic: try splits where |A| is roughly half of |target|
                elems = list(target_set)
                n = len(elems)
                
                # Try all subsets of size 1 to n-1 (but limit for performance)
                # For n <= 6, this is at most 2^6 - 2 = 62 splits
                for mask in range(1, (1 << n)):
                    if mask == (1 << n) - 1:
                        continue  # A = target, then B = empty
                    a_set = frozenset(elems[i] for i in range(n) if mask & (1 << i))
                    b_set = frozenset(target_set ^ a_set)  # XOR of sets
                    
                    if len(a_set) == 0 or len(b_set) == 0:
                        continue
                    
                    # Check depth constraint
                    # We'll handle depth in the tree construction
                    a_count, a_tree = min_gates_for_set(a_set, max_depth_allowed, memo)
                    b_count, b_tree = min_gates_for_set(b_set, max_depth_allowed, memo)
                    
                    total = a_count + b_count + 1
                    if total < best_count:
                        best_count = total
                        best_tree = (0, a_tree, b_tree)
                
                if best_tree is None:
                    # Fallback: linear chain
                    best_count = n - 1
                    # Build linear tree
                    if n == 2:
                        best_tree = (0, elems[0], elems[1])
                    else:
                        left = (0, elems[0], elems[1])
                        for i in range(2, n):
                            left = (0, left, elems[i])
                        best_tree = left
                
                memo[target_set] = (best_count, best_tree)
                return memo[target_set]
            
            memo = {}
            trees = {}
            for fs in unique_sets:
                count, tree = min_gates_for_set(fs, depth_limit, memo)
                trees[fs] = (count, tree)
            
            # Now flatten all trees into a shared gate list
            # We need to share subexpressions across different target sets
            # This is complex; for now, just flatten each independently
            # and deduplicate gates by their (left_set, right_set) pair
            
            gates = []
            next_wire = zero_wire + 1
            wire_depth = {}
            for i in range(n_inputs):
                wire_depth[i] = 0
            wire_depth[zero_wire] = 0
            
            # Map from (frozen_set) to wire index, to share subexpressions
            set_to_wire = {}
            for i in range(n_inputs):
                set_to_wire[frozenset([i])] = i
            set_to_wire[frozenset()] = zero_wire
            
            def flatten_tree(tree, set_to_wire, gates, next_wire_ref, wire_depth, depth_limit):
                """Flatten a tree, reusing existing wires where possible."""
                if tree is None:
                    return zero_wire
                if isinstance(tree, int):
                    return tree
                
                # tree = (0, left, right)
                _, left, right = tree
                left_wire = flatten_tree(left, set_to_wire, gates, next_wire_ref, wire_depth, depth_limit)
                right_wire = flatten_tree(right, set_to_wire, gates, next_wire_ref, wire_depth, depth_limit)
                
                # Check if this XOR is already computed
                # We need to track what set each wire computes
                # For now, just create a new gate
                new_wire = next_wire_ref[0]
                next_wire_ref[0] += 1
                gates.append([left_wire, right_wire])
                wire_depth[new_wire] = max(wire_depth.get(left_wire, 0), wire_depth.get(right_wire, 0)) + 1
                return new_wire
            
            outputs = []
            for i in range(n_rows):
                fs = row_fsets[i]
                if len(fs) == 0:
                    outputs.append(zero_wire)
                else:
                    count, tree = trees[fs]
                    next_wire_ref = [next_wire]
                    out_wire = flatten_tree(tree, set_to_wire, gates, next_wire_ref, wire_depth, depth_limit)
                    next_wire = next_wire_ref[0]
                    outputs.append(out_wire)
            
            return {'gates': gates, 'outputs': outputs}
        else:
            # For larger sets, use the greedy approach
            return None
    
    def try_greedy_improved():
        """Improved greedy with wire merging and better tie-breaking."""
        gates = []
        next_wire = zero_wire + 1
        
        # Represent rows as sets
        rw = [set(fs) for fs in row_fsets]
        
        # Map from frozenset to wire index (for merging)
        set_to_wire = {}
        for i in range(n_inputs):
            set_to_wire[frozenset([i])] = i
        set_to_wire[frozenset()] = zero_wire
        
        wire_depth = {}
        for i in range(n_inputs):
            wire_depth[i] = 0
        wire_depth[zero_wire] = 0
        
        max_iter = 50000
        for _ in range(max_iter):
            if not elapsed_ok():
                break
            
            # Count co-occurrence of pairs
            pair_count = defaultdict(int)
            for i in range(n_rows):
                rs = rw[i]
                if len(rs) < 2:
                    continue
                elems = list(rs)
                for x in range(len(elems)):
                    for y in range(x + 1, len(elems)):
                        a, b = elems[x], elems[y]
                        if a > b:
                            a, b = b, a
                        if a == zero_wire or b == zero_wire:
                            continue
                        # Get the sets for these wires
                        # We need to track what set each wire represents
                        # For inputs, it's single element
                        # For computed wires, we need to track
                        pair_count[(a, b)] += 1
            
            if not pair_count:
                break
            
            # Find best pair
            best_pair = None
            best_benefit = 0
            for (a, b), cnt in pair_count.items():
                if cnt < 2:
                    continue
                benefit = cnt - 1
                if benefit > best_benefit:
                    best_benefit = benefit
                    best_pair = (a, b)
            
            if best_pair is None:
                break
            
            a, b = best_pair
            new_wire = next_wire
            next_wire += 1
            gates.append([a, b])
            wire_depth[new_wire] = max(wire_depth.get(a, 0), wire_depth.get(b, 0)) + 1
            
            # Update rows
            for i in range(n_rows):
                rs = rw[i]
                if a in rs and b in rs:
                    rs.discard(a)
                    rs.discard(b)
                    rs.add(new_wire)
        
        # Build outputs
        outputs = []
        for i in range(n_rows):
            rs = rw[i]
            if len(rs) == 0:
                outputs.append(zero_wire)
            elif len(rs) == 1:
                outputs.append(list(rs)[0])
            else:
                elems = list(rs)
                elems.sort(key=lambda w: wire_depth.get(w, 0))
                current = elems[0]
                for e in elems[1:]:
                    new_depth = max(wire_depth.get(current, 0), wire_depth.get(e, 0)) + 1
                    if new_depth > depth_limit:
                        break
                    gates.append([current, e])
                    current = next_wire
                    next_wire += 1
                    wire_depth[current] = new_depth
                outputs.append(current)
        
        return {'gates': gates, 'outputs': outputs}
    
    # Try strategies and keep the best
    best_result = None
    best_gate_count = float('inf')
    
    # Try branch and bound for small instances
    if n_cols <= 8 or max(len(fs) for fs in row_fsets) <= 6:
        if elapsed_ok():
            result = try_branch_and_bound()
            if result is not None:
                gc = len(result['gates'])
                if gc < best_gate_count:
                    best_gate_count = gc
                    best_result = result
    
    # Try improved greedy
    if elapsed_ok():
        result = try_greedy_improved()
        if result is not None:
            gc = len(result['gates'])
            if gc < best_gate_count:
                best_gate_count = gc
                best_result = result
    
    if best_result is None:
        # Fallback: independent row synthesis
        gates = []
        next_wire = zero_wire + 1
        outputs = []
        for i in range(n_rows):
            rs = row_fsets[i]
            if len(rs) == 0:
                outputs.append(zero_wire)
            elif len(rs) == 1:
                outputs.append(list(rs)[0])
            else:
                elems = list(rs)
                current = elems[0]
                for e in elems[1:]:
                    gates.append([current, e])
                    current = next_wire
                    next_wire += 1
                outputs.append(current)
        return {'gates': gates, 'outputs': outputs}
    
    return best_result

# EVOLVE-BLOCK-END
