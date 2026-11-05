"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import heapq
from collections import defaultdict

_T0 = time.time()
_DEADLINE = 90.0
_MARGIN = 10.0

def _remaining():
    return _DEADLINE - _MARGIN - (time.time() - _T0)

def _row_to_int(row):
    v = 0
    for i, b in enumerate(row):
        if b:
            v |= (1 << i)
    return v

def _int_to_row(v, n):
    return [1 if (v >> i) & 1 else 0 for i in range(n)]

def _baseline_gate_count(matrix):
    rows = np.asarray(matrix, dtype=np.int64)
    if rows.size == 0:
        return 0
    return int(rows.sum())

def _build_balanced_xor(terms, wire_id, depth_of_wires, gates):
    terms = list(terms)
    if len(terms) == 1:
        return terms[0]
    level = list(terms)
    next_wire = wire_id
    while len(level) > 1:
        pairs = len(level) // 2
        carry = len(level) % 2
        new_level = []
        for p in range(pairs):
            a = level[2 * p]
            b = level[2 * p + 1]
            w = next_wire
            next_wire += 1
            gates.append([a, b])
            depth_of_wires[w] = 1 + max(depth_of_wires[a], depth_of_wires[b])
            new_level.append(w)
        for c in range(carry):
            new_level.append(level[2 * pairs + c])
        level = new_level
    return level[0]

def synthesize(matrix, depth_limit):
    """
    Exact XOR-circuit synthesis using a Minimum Spanning Tree (MST) approach
    on the linear span of the target rows to maximize shared intermediate wires.
    """
    start = time.time()
    rows = np.asarray(matrix, dtype=np.int64)
    if rows.size == 0:
        return {'gates': [], 'outputs': []}
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    m, n = rows.shape
    
    # 1. Encode rows as integers
    row_ints = [_row_to_int(rows[i]) for i in range(m)]
    
    # Identify unique non-zero rows and their frequencies (targets)
    # We want to synthesize a circuit that computes all row_ints.
    # Let S be the set of all unique non-zero vectors in the linear span of row_ints 
    # that are "useful" (i.e., they appear in the MST construction).
    
    # For small n (<=16), the hypercube is 2^16 = 65536 nodes.
    # We can perform a BFS/DFS on the hypercube starting from 0, 
    # but we only care about the subgraph induced by the "required" vectors 
    # and the vectors that help connect them efficiently.
    
    # Strategy:
    # 1. Compute the rank and a basis of the row space.
    # 2. The "ideal" circuit is a tree in the hypercube connecting 0 to all target rows,
    #    minimizing the sum of "edge costs". 
    #    However, standard MST on the full hypercube is too big.
    #    Instead, we use a heuristic: 
    #    - Start with the set of target rows T.
    #    - We build a graph where nodes are vectors.
    #    - We add edges between vectors u and v if u^v is a "simple" vector 
    #      (e.g., has low weight) or if u^v is in T.
    #    - Actually, a better approach for XOR circuits is to find a basis 
    #      and then use the "Steiner Tree" or "MST" on the basis vectors.
    
    # Let's stick to the MST on the set of "Candidate Nodes".
    # Candidate Nodes = {0} U {all target rows} U {all pairwise XORs of target rows} U ...
    # This can explode.
    
    # Alternative Structural Strategy:
    # Use the "Basis Expansion" method but optimize the basis choice.
    # Current program uses Gaussian Elimination which gives a specific basis.
    # Different bases can lead to different circuit sizes.
    # We can try to find a basis that minimizes the sum of weights of the basis vectors 
    # plus the cost of combining them.
    
    # However, the prompt suggests "MST on the hypercube graph... prioritizing frequency".
    # Let's implement a localized MST on the set of "Frequent" vectors.
    
    # Step 1: Count frequency of each vector in the target set.
    freq = defaultdict(int)
    for r in row_ints:
        freq[r] += 1
        
    # Step 2: Generate a set of "Candidate Wires".
    # These are 0, the targets, and potentially some combinations.
    # To keep it fast, we limit the candidate set to:
    # 1. All targets.
    # 2. 0.
    # 3. For each pair of targets, their XOR (if it reduces complexity? No, XOR increases weight usually, but might be a common intermediate).
    # Actually, the "intermediate" in an XOR circuit is a linear combination.
    
    # Let's try a different structural change: 
    # Instead of Basis -> Tree, use **Greedy Set Cover** on the hypercube.
    # We want to cover all target vectors with a set of "gates" (edges in the hypercube).
    # A gate is an edge (u, v) where v = u ^ e (e is a unit vector? No, e is any vector).
    # In an XOR circuit, a gate computes w = a ^ b.
    # This is equivalent to adding an edge between a and b with weight 1, and the new node is w.
    
    # Let's use the **Minimum Spanning Tree** of the graph formed by the Target Rows and 0,
    # where the edge weight between u and v is the "cost" of computing u^v.
    # The cost of computing u^v is 1 gate if we have wires for u and v.
    # But we don't have wires for all u, v.
    
    # Correct Model:
    # We are building a tree rooted at 0 (constant zero).
    # Each node in the tree is a wire.
    # The children of a node are the inputs to the XOR gate.
    # Wait, XOR is binary. So the circuit is a DAG.
    # The "MST" analogy is tricky because XOR is not a tree operation in the same way as addition (where you sum children).
    # Here, each gate has exactly 2 inputs.
    # So the circuit is a binary tree for each output, but they can share subtrees.
    # This is exactly a **Shared Binary Tree** problem.
    
    # The current program does: Basis -> Independent Trees -> Combine.
    # The new strategy: **Global Greedy Sharing**.
    # 1. Start with the set of target rows.
    # 2. Identify "common sub-expressions" (linear combinations that appear in multiple targets).
    # 3. Replace them with a single wire.
    # 4. Repeat until no more sharing is found.
    
    # This is the "Iterative Global CSE" mentioned in the cell findings, which improved score to 1.097.
    # The current program (1.03) does NOT do this. It only shares the basis vectors.
    # So, implementing Global CSE is the structural change.
    
    # Implementation of Global CSE:
    # 1. Represent each target row as a set of "atoms" (input wires).
    # 2. A "wire" is a frozenset of atoms (or an integer mask).
    # 3. We maintain a dictionary: wire_mask -> wire_id.
    # 4. We start with input wires (single bits) and 0.
    # 5. We iteratively look for pairs of existing wires (a, b) such that a^b is a "useful" wire 
    #    (i.e., it is a target, or it is a sub-part of a target).
    #    To be efficient, we only consider "useful" wires as those that are targets or 
    #    can be formed by XORing two existing wires.
    
    # Better: **BFS on the Hypercube from 0, stopping at Targets**.
    # We perform a BFS in the hypercube graph where the "distance" is the number of gates.
    # But we want to MINIMIZE GATES, not depth.
    # This is the "Shortest Path" in a graph where nodes are vectors and edges are XOR operations?
    # No, that minimizes depth.
    
    # Let's stick to the **Greedy Sharing** algorithm:
    # 1. Collect all unique masks in the targets.
    # 2. Also collect all masks that are XORs of two targets (potential intermediates).
    # 3. Sort all these candidate masks by "popularity" (how many targets contain them? 
    #    Actually, how many targets can be expressed using this mask as a sub-expression?).
    #    A mask M is a sub-expression of target T if M is a subset of T's atoms? No, in GF(2), 
    #    "sub-expression" means T = M ^ K for some K.
    #    So M is useful if it appears in the linear combination of T.
    
    # Simplified Greedy CSE:
    # 1. Let Candidates = {0} U {targets} U {x^y for x,y in targets}.
    # 2. For each candidate C, calculate its "Value":
    #    Value(C) = (Number of targets that can be simplified by introducing C) * (Cost of C)
    #    This is hard to calculate exactly.
    
    # Let's use the **Basis Re-optimization** approach:
    # The current program picks a basis via Gaussian Elimination (pivot on highest bit).
    # This basis might not be optimal for sharing.
    # We can try multiple bases (e.g., random permutations of columns before elimination) 
    # and pick the one that yields the smallest circuit.
    
    # Given the time limit, we can try a few random bases.
    
    # 1. Get the row space.
    # 2. Generate K random bases for the row space.
    # 3. For each basis, synthesize using the current method (Basis -> Trees -> Combine).
    # 4. Return the best one.
    
    # This is a "different optimization method" (Randomized Search over Bases) 
    # compared to the deterministic Gaussian Elimination basis.
    
    if n > 16:
        # Fallback for large n: just use the current deterministic method
        return _synthesize_deterministic(rows, depth_limit)
        
    # Randomized Basis Search
    best_gates = None
    best_outputs = None
    best_size = float('inf')
    
    # Number of random bases to try
    # Budget: 90s. Synthesis is fast for n<=16.
    # Let's try 50 random bases.
    
    # First, get a standard basis to ensure we have a valid set of vectors spanning the space.
    std_basis = _get_basis(row_ints)
    if not std_basis:
        return {'gates': [], 'outputs': [n] * m}
        
    # We will perturb the basis.
    # A basis is a set of linearly independent vectors.
    # We can generate random vectors in the span and try to form a basis.
    
    # Actually, simpler: Permute the input columns (wires) before doing Gaussian Elimination.
    # This changes which vectors become the "basis" vectors.
    
    for trial in range(50):
        if _remaining() < 5.0:
            break
            
        # Random permutation of columns
        perm = np.random.permutation(n)
        
        # Re-encode rows with permuted columns
        permuted_rows = []
        for r in row_ints:
            # Map bit i in original to bit perm[i] in new? 
            # Or just permute the list of bits.
            bits = [1 if (r >> i) & 1 else 0 for i in range(n)]
            # Apply permutation: new_bit[j] = old_bit[perm[j]]
            new_bits = [bits[perm[j]] for j in range(n)]
            new_r = 0
            for j, b in enumerate(new_bits):
                if b:
                    new_r |= (1 << j)
            permuted_rows.append(new_r)
            
        # Synthesize with this permuted view
        res = _synthesize_with_basis(permuted_rows, n, depth_limit)
        
        if res is None:
            continue
            
        gates, outputs = res
        size = len(gates)
        
        if size < best_size:
            best_size = size
            best_gates = gates
            best_outputs = outputs
            # Map outputs back to original wire indices?
            # Wait, the outputs are wire IDs in the synthesized circuit.
            # The circuit structure depends on the permutation.
            # We need to output the circuit in the ORIGINAL wire order.
            # The gates are [wire_a, wire_b].
            # If we permuted the inputs, the input wires in the gates are permuted.
            # We need to un-permute the input wire indices in the gates and outputs.
            
            # Un-permute:
            # In the permuted circuit, input wire j corresponds to original input perm[j].
            # So if a gate uses input j, it should use input perm[j] in the original circuit.
            # Wait, perm was defined as: new_bits[j] = old_bits[perm[j]].
            # So new wire j is old wire perm[j].
            # So in the output circuit, any reference to input wire j (0..n-1) 
            # should be replaced by perm[j].
            
            unperm_gates = []
            for g in gates:
                a, b = g
                if a < n:
                    a = perm[a]
                if b < n:
                    b = perm[b]
                unperm_gates.append([a, b])
                
            unperm_outputs = []
            for o in outputs:
                if o < n:
                    o = perm[o]
                unperm_outputs.append(o)
                
            best_gates = unperm_gates
            best_outputs = unperm_outputs

    if best_gates is None:
        # Fallback to deterministic
        return _synthesize_deterministic(rows, depth_limit)
        
    return {'gates': best_gates, 'outputs': best_outputs}

def _get_basis(row_ints):
    basis = []
    for r in row_ints:
        cur = r
        for (pb, mask) in basis:
            if (cur >> pb) & 1:
                cur ^= mask
        if cur == 0:
            continue
        pb = cur.bit_length() - 1
        basis.append((pb, cur))
        new_basis = []
        for (qpb, qmask) in basis:
            if (qpb, qmask) == (pb, cur):
                new_basis.append((pb, cur))
                continue
            if (qmask >> pb) & 1:
                qmask ^= cur
            new_basis.append((qpb, qmask))
        basis = new_basis
    return basis

def _synthesize_with_basis(row_ints, n, depth_limit):
    """
    Synthesize using the standard basis method, but on the permuted rows.
    """
    m = len(row_ints)
    
    # 1. Find Basis
    basis = _get_basis(row_ints)
    
    # 2. Express each row as combo of basis
    row_basis_combo = []
    for r in row_ints:
        cur = r
        combo = [0] * len(basis)
        for i, (pb, mask) in enumerate(basis):
            if (cur >> pb) & 1:
                cur ^= mask
                combo[i] = 1
        row_basis_combo.append(combo)
        
    # 3. Synthesize Basis Vectors
    # Estimate max depth and size
    max_terms = max((mask.bit_count() for pb, mask in basis), default=0)
    max_combo = max((c.count(1) for c in row_basis_combo), default=0)
    
    # Conservative size estimate for depth_of_wires array
    # Each basis vector needs (weight-1) gates.
    # Each output needs (combo_weight-1) gates.
    total_gates_est = sum(max(0, mask.bit_count() - 1) for pb, mask in basis) + \
                      sum(max(0, c.count(1) - 1) for c in row_basis_combo)
                      
    depth_of_wires = [0] * (n + 1 + total_gates_est + 10)
    for i in range(n + 1):
        depth_of_wires[i] = 0
        
    gates = []
    next_wire = n + 1
    
    basis_wire = []
    for i, (pb, mask) in enumerate(basis):
        terms = [j for j in range(n) if (mask >> j) & 1]
        if not terms:
            basis_wire.append(n)
            continue
        start_wire = next_wire
        final_wire = _build_balanced_xor(terms, start_wire, depth_of_wires, gates)
        next_wire = start_wire + max(0, len(terms) - 1)
        basis_wire.append(final_wire)
        
    outputs = []
    for combo in row_basis_combo:
        sel = [basis_wire[i] for i in range(len(basis)) if combo[i]]
        if not sel:
            outputs.append(n)
            continue
        if len(sel) == 1:
            outputs.append(sel[0])
            continue
        start_wire = next_wire
        final_wire = _build_balanced_xor(sel, start_wire, depth_of_wires, gates)
        next_wire = start_wire + max(0, len(sel) - 1)
        outputs.append(final_wire)
        
    # Check Depth
    if outputs:
        max_depth = max(depth_of_wires[v] for v in outputs)
        if max_depth > depth_limit:
            return None
            
    return gates, outputs

def _synthesize_deterministic(rows, depth_limit):
    m, n = rows.shape
    row_ints = [_row_to_int(rows[i]) for i in range(m)]
    res = _synthesize_with_basis(row_ints, n, depth_limit)
    if res is None:
        # Fallback to independent
        return _fallback_independent(rows, depth_limit)
    gates, outputs = res
    return {'gates': gates, 'outputs': outputs}

def _fallback_independent(rows, depth_limit):
    m, n = rows.shape
    gates = []
    outputs = []
    depth_of_wires = [0] * (n + 1 + m * (n + 1))
    for i in range(n + 1):
        depth_of_wires[i] = 0
    next_wire = n + 1
    for i in range(m):
        terms = [j for j in range(n) if rows[i, j]]
        if not terms:
            outputs.append(n)
            continue
        if len(terms) == 1:
            outputs.append(terms[0])
            continue
        start_wire = next_wire
        final = _build_balanced_xor(terms, start_wire, depth_of_wires, gates)
        next_wire = start_wire + max(0, len(terms) - 1)
        outputs.append(final)
    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
