"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random
import sys
import time

N = 16

# EVOLVE-BLOCK-START

class BDD:
    """
    Minimal implementation of Binary Decision Diagrams for sets of bit-vectors of fixed length.
    Variables are indexed 0..N-1.
    A BDD represents a set of assignments.
    - Terminal 0: empty set
    - Terminal 1: universal set
    - Node (var, lo, hi): if var=0 go to lo, if var=1 go to hi.
    """
    def __init__(self):
        self.nodes = {}  # (var, lo, hi) -> node_id
        self.node_vars = []  # node_id -> var
        self.node_lo = []  # node_id -> lo
        self.node_hi = []  # node_id -> hi
        self.node_counter = 0
        self.unique = {}  # (var, lo, hi) -> node_id
        self.cache = {}  # (func_id, var_id, val) -> node_id
        self.counter = 0
        
        # Terminals: 0 is false, 1 is true
        # We'll use negative integers or specific IDs for terminals?
        # Let's use 0 for False, 1 for True.
        # Nodes will have ID >= 2.
        
    def make_node(self, var, lo, hi):
        if lo == hi:
            return lo
        key = (var, lo, hi)
        if key in self.unique:
            return self.unique[key]
        
        # Canonicalize: if lo > hi, swap? No, BDDs are ordered.
        # But we must ensure var < var(lo) and var < var(hi)
        # This is handled by the apply functions maintaining order.
        
        node_id = self.node_counter
        self.node_counter += 1
        self.unique[key] = node_id
        self.node_vars.append(var)
        self.node_lo.append(lo)
        self.node_hi.append(hi)
        return node_id

    def get_var(self, node):
        if node < 2:
            return -1
        return self.node_vars[node]

    def get_lo(self, node):
        if node < 2:
            return node
        return self.node_lo[node]

    def get_hi(self, node):
        if node < 2:
            return node
        return self.node_hi[node]

    def and_(self, a, b):
        if a == 0 or b == 0:
            return 0
        if a == 1:
            return b
        if b == 1:
            return a
        
        key = (a, b)
        if key in self.cache:
            return self.cache[key]
            
        var_a = self.get_var(a)
        var_b = self.get_var(b)
        
        if var_a == var_b:
            var = var_a
            res = self.make_node(var, self.and_(self.get_lo(a), self.get_lo(b)), self.and_(self.get_hi(a), self.get_hi(b)))
        elif var_a < var_b:
            var = var_a
            res = self.make_node(var, self.and_(self.get_lo(a), b), self.and_(self.get_hi(a), b))
        else:
            var = var_b
            res = self.make_node(var, self.and_(a, self.get_lo(b)), self.and_(a, self.get_hi(b)))
            
        self.cache[key] = res
        return res

    def or_(self, a, b):
        if a == 1 or b == 1:
            return 1
        if a == 0:
            return b
        if b == 0:
            return a
            
        key = (a, b)
        if key in self.cache:
            return self.cache[key]
            
        var_a = self.get_var(a)
        var_b = self.get_var(b)
        
        if var_a == var_b:
            var = var_a
            res = self.make_node(var, self.or_(self.get_lo(a), self.get_lo(b)), self.or_(self.get_hi(a), self.get_hi(b)))
        elif var_a < var_b:
            var = var_a
            res = self.make_node(var, self.or_(self.get_lo(a), b), self.or_(self.get_hi(a), b))
        else:
            var = var_b
            res = self.make_node(var, self.or_(a, self.get_lo(b)), self.or_(a, self.get_hi(b)))
            
        self.cache[key] = res
        return res

    def not_(self, a):
        if a == 0:
            return 1
        if a == 1:
            return 0
        # We don't support negation directly in reduced ordered BDDs without co-factors easily?
        # Actually, we can implement it, but for this task we might not need it.
        # We will construct the "unsorted" set directly.
        raise NotImplementedError("Negation not implemented")

    def restrict(self, node, var, val):
        if node < 2:
            return node
        if self.get_var(node) < var:
            return self.restrict(self.get_lo(node), var, val)
        elif self.get_var(node) > var:
            return self.restrict(self.get_hi(node), var, val)
        else:
            if val == 0:
                return self.get_lo(node)
            else:
                return self.get_hi(node)

    def count(self, node):
        if node == 0:
            return 0
        if node == 1:
            return 2 ** N
        # Counting nodes in BDD
        # We need to know the depth or just sum up?
        # For large N, 2^N is huge. But we only care if it's 0 or not.
        # So we can just check if node == 0.
        return 1 if node != 0 else 0

    def is_empty(self, node):
        return node == 0

    def is_full(self, node):
        return node == 1

def construct():
    """
    Construct a size-efficient sorting network for 16 wires using BDD-based verification.
    """
    
    # 1. Define the BDD for "sorted" outputs.
    # Sorted binary outputs are of the form 0...01...1.
    # Bit i is 0 if i < k, 1 if i >= k.
    # We want the set of all such outputs.
    # Let's build the BDD for the set of valid outputs.
    
    bdd = BDD()
    
    # The set of valid outputs is the union over k=0..N of the assignment where bits 0..k-1 are 0 and k..N-1 are 1.
    # We can build this iteratively or construct a BDD that accepts these.
    
    # A simpler way:
    # The condition for a vector x to be sorted is: for all i < j, if x[i]=1 then x[j]=1.
    # Equivalently, there is no i < j such that x[i]=1 and x[j]=0.
    # The "unsorted" set is the union over i<j of (x[i]=1 AND x[j]=0).
    
    # Let's build the BDD for the "unsorted" set directly.
    # Unsorted = OR_{i<j} (x_i AND NOT x_j)
    
    # We need to represent x_i and NOT x_j.
    # x_i is the BDD that is 1 iff bit i is 1.
    # NOT x_j is the BDD that is 1 iff bit j is 0.
    
    # Let's build BDDs for each variable and its negation.
    # var_bdd[i]: BDD representing x_i == 1
    # not_var_bdd[i]: BDD representing x_i == 0
    
    var_bdd = [0] * N
    not_var_bdd = [0] * N
    
    # We need a helper to build a BDD for a single variable constraint.
    # BDD for x_i == 1:
    # Path: 0 -> 0 -> ... -> 0 -> (at var i) -> 1 (if x_i=1) else 0.
    # Actually, standard BDD reduction:
    # Node for var 0: lo=0, hi=Node for var 1...
    
    def build_var_bdd(i, val):
        # Build BDD for x_i == val
        # If val == 1:
        #   For vars < i, both branches go down.
        #   At var i, lo=0, hi=1.
        #   For vars > i, doesn't matter (reduced).
        
        node = 1  # Start from top? No, we build bottom-up or top-down?
        # Let's build recursively from high var to low var.
        
        def build(var):
            if var > i:
                # Variable doesn't matter, so lo=hi.
                # If var > i, the decision is further down.
                # Actually, if we are at a variable > i, the result depends on lower vars.
                # So we return the node for var-1.
                return build(var - 1)
            elif var == i:
                if val == 1:
                    return bdd.make_node(var, 0, 1)
                else:
                    return bdd.make_node(var, 1, 0)
            else: # var < i
                # Both branches lead to the same sub-BDD (build(var-1))
                child = build(var - 1)
                return bdd.make_node(var, child, child)
        
        return build(N - 1)

    for i in range(N):
        var_bdd[i] = build_var_bdd(i, 1)
        not_var_bdd[i] = build_var_bdd(i, 0)
        
    # Build Unsorted BDD
    unsorted = 0
    for i in range(N):
        for j in range(i + 1, N):
            # x_i AND NOT x_j
            # We need AND operation.
            # term = var_bdd[i] AND not_var_bdd[j]
            term = bdd.and_(var_bdd[i], not_var_bdd[j])
            unsorted = bdd.or_(unsorted, term)
            
    # Now, we have a BDD representing the set of unsorted inputs.
    # A network is correct if, for all inputs x, the output y = Network(x) is NOT in the Unsorted set.
    # Equivalently, the set of inputs x such that Network(x) is unsorted is empty.
    
    # How to check this?
    # We can simulate the network on the BDD of all inputs?
    # The BDD of all inputs is 1.
    # We need to apply the transformation of the network to the set of inputs.
    # This is equivalent to computing the pre-image of the Unsorted set under the network function.
    
    # Let F be the function implemented by the network.
    # We want to check if {x | F(x) in Unsorted} is empty.
    # This is equivalent to checking if the composition of the network with the Unsorted predicate is always 0.
    
    # We can represent the predicate "Output is unsorted" as a BDD over the output variables.
    # Then we need to "pull back" this BDD through the network comparators.
    
    # Let's define a function `apply_comparator(bdd_state, i, j)` that takes a BDD representing a set of output states (or a predicate on output states) and returns the BDD representing the set of input states to this comparator that would lead to that output state.
    # Wait, the network maps Input -> Output.
    # We have a predicate P(Output) which is 1 if Output is unsorted.
    # We want to compute Q(Input) = P(Network(Input)).
    # We can do this by starting with P as a BDD over variables 0..N-1 (representing the current state of the wires).
    # Then we iterate through the comparators in REVERSE order.
    # For a comparator (i, j) at the end of the network, it transforms state S_in to S_out.
    # S_out is determined by S_in.
    # We have a BDD B representing the set of S_out states that are "bad" (unsorted).
    # We want to find the set of S_in states that map to a "bad" S_out.
    # Let's call this `preimage(B, i, j)`.
    
    # How does comparator (i, j) work?
    # If S_in[i] <= S_in[j], S_out = S_in.
    # If S_in[i] > S_in[j], S_out is S_in with bits i and j swapped.
    # Since we are dealing with binary values, S_in[i]=1, S_in[j]=0 is the only case where they differ and i<j implies 1>0.
    # So:
    # Case 1: S_in[i]=0, S_in[j]=0 -> S_out = S_in
    # Case 2: S_in[i]=0, S_in[j]=1 -> S_out = S_in
    # Case 3: S_in[i]=1, S_in[j]=0 -> S_out = S_in with i,j swapped (so i becomes 0, j becomes 1)
    # Case 4: S_in[i]=1, S_in[j]=1 -> S_out = S_in
    
    # So, S_out is "bad" if:
    # (S_in is in Case 1 or 2 and S_in is in B)
    # OR
    # (S_in is in Case 3 and swap(S_in) is in B)
    # OR
    # (S_in is in Case 4 and S_in is in B)
    
    # Note that in Case 3, swap(S_in) has 0 at i and 1 at j.
    # Let B be the BDD for bad outputs.
    # We want to construct B_in such that B_in(S_in) = 1 iff S_out is bad.
    
    # Let's decompose B based on the values of i and j in the OUTPUT state.
    # But the comparator acts on the INPUT state.
    # It's easier to think in terms of the mapping.
    
    # Let's define the preimage operation.
    # Given a BDD B over variables 0..N-1 (representing the state AFTER the comparator),
    # we want a BDD B' over variables 0..N-1 (representing the state BEFORE the comparator)
    # such that B'(x) = 1 iff B(C(x)) = 1, where C is the comparator function.
    
    # C(x) is:
    # if x[i]=0, x[j]=0: C(x) = x
    # if x[i]=0, x[j]=1: C(x) = x
    # if x[i]=1, x[j]=0: C(x) = x with i,j swapped
    # if x[i]=1, x[j]=1: C(x) = x
    
    # So B'(x) = 1 iff:
    # (x[i]=0 and x[j]=0 and B(x)=1)
    # OR
    # (x[i]=0 and x[j]=1 and B(x)=1)
    # OR
    # (x[i]=1 and x[j]=0 and B(swap(x))=1)
    # OR
    # (x[i]=1 and x[j]=1 and B(x)=1)
    
    # We can compute this using BDD operations.
    # We need to restrict B to specific values of i and j?
    # No, B is a BDD over the same variables.
    
    # Let's break B down into 4 parts based on the values of i and j in the argument to B.
    # Let B00 be the restriction of B where var i=0, var j=0.
    # Let B01 be the restriction of B where var i=0, var j=1.
    # Let B10 be the restriction of B where var i=1, var j=0.
    # Let B11 be the restriction of B where var i=1, var j=1.
    
    # Then:
    # B'(x) = 1 iff:
    # (x[i]=0 and x[j]=0 and B00(x)=1)  -- Note: B00 is a BDD over vars excluding i,j? No, restrictions reduce the BDD.
    # Actually, if we restrict B to i=0, j=0, we get a BDD B00 over the remaining variables.
    # But we need to combine it back with the conditions on i and j.
    
    # Let's use the `restrict` method.
    # B00 = restrict(restrict(B, i, 0), j, 0)
    # This BDD B00 is over variables < i, < j (excluding i, j).
    # To use it in B', we need to "lift" it back to the full variable set.
    # Lifting means making a BDD that is 1 iff the projection of x onto the remaining vars satisfies B00.
    # This is equivalent to making a BDD that doesn't depend on i and j.
    # We can achieve this by building a BDD from B00 that ignores i and j.
    # But our BDD class doesn't have a "lift" or "existential quantification" easily?
    # Actually, we can just build the BDD for the condition.
    
    # Alternative approach:
    # Instead of restricting, we can use AND with variable BDDs.
    # B'(x) = (NOT x[i] AND NOT x[j] AND B(x)) OR (NOT x[i] AND x[j] AND B(x)) OR (x[i] AND NOT x[j] AND B_swap(x)) OR (x[i] AND x[j] AND B(x))
    
    # Where B_swap(x) is B evaluated at swap(x).
    # swap(x) is x with bits i and j flipped.
    # How to compute B_swap as a BDD?
    # B_swap(x) = 1 iff B(x with i,j swapped) = 1.
    # This is equivalent to taking the BDD B and swapping the variables i and j in its structure?
    # Or simply, B_swap(x) = B(x') where x' is x with i,j swapped.
    # We can compute this by restricting B? No.
    # We can compute B_swap by noting that B_swap(x) depends on x.
    # If we swap the labels of variables i and j in the BDD B, we get the BDD for B_swap?
    # Yes, if we have a BDD representing f(x), and we swap the names of variables i and j, we get a BDD representing f(x with i,j swapped).
    # But our BDD class doesn't support variable renaming.
    
    # However, we can observe that for binary comparators, the swap only affects bits i and j.
    # B(x with i,j swapped) can be computed if we know the value of B for all combinations of other bits.
    
    # Let's try a different perspective.
    # We are checking if the set of inputs leading to unsorted outputs is empty.
    # This is equivalent to checking if the network sorts all inputs.
    # We can just simulate the network on the BDD of all inputs?
    # No, BDDs represent sets, not functions.
    
    # Let's stick to the preimage calculation.
    # We need to compute B' = Preimage(B, i, j).
    # B'(x) = 1 iff C(x) is in B.
    
    # Let's implement `preimage(B, i, j)`.
    # We can construct B' by combining 4 terms.
    # Term 1: x[i]=0, x[j]=0, B(x)=1
    #   This is equivalent to: (NOT x[i]) AND (NOT x[j]) AND B(x)
    #   We can compute this as: bdd.and_(bdd.and_(not_var_bdd[i], not_var_bdd[j]), B)
    #   Wait, B is a BDD over the same variables. This should work directly!
    #   The AND operation handles the variable dependencies correctly.
    
    # Term 2: x[i]=0, x[j]=1, B(x)=1
    #   This is: (NOT x[i]) AND (x[j]) AND B(x)
    #   bdd.and_(bdd.and_(not_var_bdd[i], var_bdd[j]), B)
    
    # Term 3: x[i]=1, x[j]=0, B(swap(x))=1
    #   Here we need B(swap(x)).
    #   Let's call this B_swapped.
    #   B_swapped(x) = 1 iff B(x with i,j swapped) = 1.
    #   How to get B_swapped?
    #   Note that swap(x) has 0 at i and 1 at j.
    #   Let y = swap(x). Then y[i]=0, y[j]=1.
    #   And x = swap(y).
    #   So B(y) = 1.
    #   We want to express this in terms of x.
    #   This is tricky without variable renaming.
    
    #   However, we can notice that Term 3 is the only one where the output state is different from the input state.
    #   In Term 3, the output state y has y[i]=0, y[j]=1.
    #   And y is obtained by swapping x's i and j.
    #   So y[k] = x[k] for k != i,j.
    #   y[i] = x[j] = 0.
    #   y[j] = x[i] = 1.
    
    #   So B(y) = 1 iff B evaluated at (x with i=0, j=1, and other bits same as x) is 1.
    #   This is exactly the restriction of B to i=0, j=1, evaluated at the other bits of x.
    #   Let B01 = restrict(restrict(B, i, 0), j, 1).
    #   B01 is a BDD over variables excluding i and j.
    #   We need to lift B01 back to the full variable set, ignoring i and j.
    #   Let's call this Lift(B01, i, j).
    #   Lift(B01, i, j) is the BDD that is 1 iff the projection of x onto vars \ {i,j} satisfies B01.
    
    #   How to implement Lift?
    #   We can build it recursively.
    #   Or, we can observe that Lift(B01, i, j) is equivalent to the BDD obtained by taking B01 and adding nodes for i and j that don't change the value.
    #   i.e., for var > max(i,j), pass through.
    #   For var == max(i,j), lo=hi=child.
    #   For var == min(i,j), lo=hi=child.
    #   For var < min(i,j), pass through.
    
    #   Let's implement a helper `lift(bdd_node, vars_to_ignore)`.
    #   Actually, we only need to ignore i and j.
    
    #   Let's refine the plan for Term 3:
    #   1. Compute B01 = restrict(restrict(B, i, 0), j, 1).
    #   2. Compute Lifted_B01 = lift(B01, i, j).
    #   3. Term 3 = (x[i]=1) AND (x[j]=0) AND Lifted_B01.
    #      bdd.and_(bdd.and_(var_bdd[i], not_var_bdd[j]), Lifted_B01)
    
    #   Term 4: x[i]=1, x[j]=1, B(x)=1
    #      bdd.and_(bdd.and_(var_bdd[i], var_bdd[j]), B)
    
    #   B' = OR(Term 1, Term 2, Term 3, Term 4)
    
    #   We need to implement `lift`.
    #   `lift(node, i, j)`: Returns a BDD equivalent to node but with variables i and j added as "don't care" (both branches go to the same child).
    #   We must ensure the variable ordering is respected.
    #   Assume i < j.
    
    #   Let's implement `lift`.
    
    def lift(node, i, j):
        # i < j
        # We need to add nodes for j and i.
        # Start with node.
        # If node's var > j, we need to add nodes for j and i below it?
        # No, BDDs are top-down.
        # If the root of `node` has var v.
        # If v > j, we need to create a new node for j, and then i.
        # This is complex.
        
        # Alternative: We can just build the BDD for the condition "projection satisfies node" from scratch?
        # No, that's expensive.
        
        # Let's try a different approach for Term 3.
        # Instead of lifting, can we compute B_swapped directly?
        # B_swapped(x) = B(x with i,j swapped).
        # We can compute this by applying the swap to the variables in B.
        # If we had a way to rename variables in a BDD, it would be easy.
        
        # Let's look at the structure of B.
        # B is a BDD over 0..N-1.
        # B_swapped is the BDD where every time we see variable i, we treat it as j, and vice versa?
        # No, that's not quite right because the ordering matters.
        
        # Let's go back to lifting.
        # We can implement `lift` by recursively rebuilding the BDD.
        # `lift(node, ignore_vars)`:
        #   If node is terminal, return node.
        #   Let v = var(node).
        #   If v in ignore_vars:
        #       This shouldn't happen if node is properly reduced and we are adding vars.
        #       But wait, `node` comes from `restrict`, so it doesn't contain i or j.
        #   So v is not i or j.
        #   If v < min(i,j):
        #       return make_node(v, lift(lo, i, j), lift(hi, i, j))
        #   If v > max(i,j):
        #       # We need to insert i and j between v and the children?
        #       # No, the order is fixed.
        #       # If v > j, then the next variable in the path could be anything < v.
        #       # We need to ensure that i and j are decided before v?
        #       # No, i and j are specific variables.
        #       # If v > j, and we haven't decided i and j yet, we should decide them now?
        #       # But the BDD structure of `node` assumes a specific order.
        #       # If `node` has a node with var v > j, and the next node has var u < i, that's fine.
        #       # But we want to insert "don't care" nodes for i and j.
        #       # This means that at the point where we would normally go to the next variable in `node`,
        #       # we first go through i and j (which don't change the value).
        #       
        #       # This is getting complicated.
        
        # Let's try a simpler verification method first.
        # The BDD approach for preimage is tricky.
        # Maybe we can just use the bit-parallel simulation for the initial pruning,
        # and only use BDDs if we find a way to make it fast.
        
        # Given the time constraints, I will revert to the bit-parallel simulation
        # but optimize it with NumPy or just efficient Python loops.
        # The previous solution scored 1.0, which means it likely found the 60-comparator network or close to it.
        # The task says "A verified 60-comparator construction is the reference".
        # If the current program scores 1.0, it might already be finding 60.
        # Let's check the score definition: "seed/value".
        # If seed is 120 (odd-even transposition) and value is 60, score is 2.0.
        # If score is 1.0, it means value == seed.
        # So the current program is returning 120 comparators?
        # Or maybe the seed program is the 60-comparator one?
        # "The initial odd-even transposition network uses 120 comparators."
        # "A verified 60-comparator construction is the reference".
        # "Score: comparator count, minimized... 1.0 equals the seed program".
        # If the seed program is the 120-comparator one, and we get 1.0, we are at 120.
        # If we get 60, the score should be 120/60 = 2.0.
        
        # So the current program is failing to prune below 120?
        # Or maybe the `verify` function is too slow and it times out, returning the initial network?
        # "Each evaluation must finish within 60 s."
        # If it times out, it scores 0.
        # But it scores 1.0. So it finishes and returns 120.
        # This implies the greedy removal is not working or is too slow to make progress.
        
        # Let's optimize the bit-parallel verifier.
        # We can use NumPy to simulate all 65536 inputs in parallel?
        # 65536 inputs * 16 bits = 1,048,576 bits = 128 KB.
        # We can store the states as a NumPy array of uint16 or uint32.
        # For each comparator, we update all 65536 states.
        
        # Let's try this.
        
        pass # Placeholder for the actual implementation below

    # --- Optimized Bit-Parallel Verification ---
    
    def verify_fast(comparators):
        # Use NumPy to simulate all 2^16 inputs in parallel.
        # Inputs are 0 to 65535.
        # We represent the 16 wires as 16 separate arrays of booleans?
        # Or one array of uint16?
        # uint16 is efficient.
        
        states = np.arange(1 << N, dtype=np.uint16)
        
        for i, j in comparators:
            # Extract bits i and j
            # bit_i = (states >> i) & 1
            # bit_j = (states >> j) & 1
            # Condition: bit_i == 1 and bit_j == 0
            # mask = bit_i & (~bit_j)
            # If mask is set, swap bits.
            # swap: states ^= (1 << i) | (1 << j) where mask is set.
            
            bit_i = (states >> i) & 1
            bit_j = (states >> j) & 1
            mask = bit_i & (~bit_j) # bit_i is 1, bit_j is 0
            
            # Apply swap
            # We need to XOR (1<<i) | (1<<j) only where mask is 1.
            # Since mask is 0 or 1, we can do:
            # swap_val = mask * ((1 << i) | (1 << j))
            # states ^= swap_val
            # But mask is an array of 0s and 1s.
            # We can broadcast.
            
            swap_val = mask.astype(np.uint16) * ((1 << i) | (1 << j))
            states ^= swap_val
            
        # Check if all states are sorted.
        # Sorted states are 0, 1, 3, 7, ..., 65535.
        # i.e., 2^k - 1.
        # We can check if states are in this set.
        # Or check if (states & (states + 1)) == 0?
        # If x is of form 0...01...1, then x+1 is 1...10...0.
        # x & (x+1) == 0.
        # Let's test:
        # x=0 (00), x+1=1 (01), 0&1=0.
        # x=1 (01), x+1=2 (10), 1&2=0.
        # x=3 (11), x+1=4 (100), 3&4=0.
        # x=2 (10), x+1=3 (11), 2&3=2 != 0.
        # So yes, (x & (x+1)) == 0 is the condition for sorted.
        
        # Check if any state fails this condition.
        # If np.any(states & (states + 1) != 0): return False
        
        # Note: states + 1 might overflow uint16?
        # Max state is 65535. 65535 + 1 = 65536 = 0 in uint16.
        # 65535 & 0 = 0. So it works.
        
        if np.any(states & (states + 1) != 0):
            return False
        return True

    # Start with a known network.
    # Let's use the 60-comparator network if we can find one quickly?
    # Or just start with 120 and prune.
    # With fast verification, we can do many more passes.
    
    network = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    # Greedy reduction with fast verification.
    # We can also try to start with a better initial network.
    # Batcher odd-even merge sort for N=16 has 120 comparators?
    # Actually, Batcher's network is often cited as having C(N) = N(N-1)/2 + ...
    # For N=16, Batcher has 120 comparators?
    # Let's check:
    # B(N) = B(N/2) + B(N/2) + C(N/2)
    # C(N) = N(N-1)/2
    # B(2) = 1
    # B(4) = 2*1 + 3 = 5
    # B(8) = 2*5 + 14 = 24
    # B(16) = 2*24 + 56 = 104
    # So Batcher has 104 comparators.
    # This is better than 120.
    
    # Let's implement Batcher's network.
    
    def batcher_network(n):
        # Returns a list of comparators for n wires.
        # n must be a power of 2.
        if n == 1:
            return []
        
        half = n // 2
        left = batcher_network(half)
        right = batcher_network(half)
        
        # Shift right network
        right = [(i + half, j + half) for i, j in right]
        
        # Merge
        merge = []
        # Odd-even merge
        # This is complex to implement from scratch.
        # Let's stick to the 120 comparator network and prune it.
        # With fast verification, we should be able to prune it down to 60.
        
        return left + right

    # Let's just use the 120 comparator network and prune.
    
    # To speed up pruning, we can process comparators in reverse order?
    # Or random order.
    
    # Let's do multiple passes.
    # With fast verification, we can afford more passes.
    
    start_time = time.time()
    max_time = 55  # Leave 5 seconds for overhead
    
    removed_any = True
    pass_count = 0
    while removed_any and time.time() - start_time < max_time:
        removed_any = False
        pass_count += 1
        indices = list(range(len(network)))
        random.shuffle(indices)
        for idx in indices:
            if time.time() - start_time > max_time:
                break
            if idx >= len(network):
                continue
            # Try removing network[idx]
            candidate = network[:idx] + network[idx+1:]
            if verify_fast(candidate):
                network = candidate
                removed_any = True
                # Update indices? We removed one, so indices are shifted.
                # It's easier to just break and restart the pass?
                # Or adjust indices.
                # Let's just break and restart to keep it simple and safe.
                # Actually, we can just continue, but the indices for the rest of the pass are invalid.
                # So we break the inner loop.
                break
    
    return network

# EVOLVE-BLOCK-END
