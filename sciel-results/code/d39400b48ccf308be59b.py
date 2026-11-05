def _nj_to_newick(tree):
    if isinstance(tree, int):
        return f"t{tree}"
    stack = [tree]
    result = []
    # We need to know when to close parentheses.
    # A post-order traversal with a stack of (node, state) where state is 0 (enter), 1 (exit)
    # But we need to output "(" on enter and ")" on exit for internal nodes.
    
    # Let's use a standard stack for post-order.
    # Actually, Newick is pre-order with closing parens.
    # (A,B) -> "(", A, ",", B, ")"
    
    # Iterative Pre-order with closing parens:
    # Stack holds (node, children_index) or similar.
    
    # Simpler:
    # Use a stack to simulate recursion.
    # Stack items: (node, is_first_child)
    # When pushing children, push right then left (so left is processed first).
    # When popping an internal node, if it's the first child, push a marker?
    
    # Let's just use the recursive one but with a higher limit?
    # The error says it exceeded.
    # Maybe the limit 10000 wasn't enough? Unlikely for N=32.
    # Maybe the tree is malformed?
    
    # Let's try to make `_nni_refine` safer.
