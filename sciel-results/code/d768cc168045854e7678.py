<<<<<<< SEARCH
    # Try greedy: remove one column at a time if possible
    for attempt in range(current_rank):
        if time.time() - start_time > max_time:
            break
        for r in range(current_rank):
            if time.time() - start_time > max_time:
                break
            # Try removing column r
            mask = [i for i in range(current_rank) if i != r]
            U_test = U[:, mask]
            V_test = V[:, mask]
            W_test = W[:, mask]
            if _verify(U_test, V_test, W_test, T):
                U, V, W = U_test, V_test, W_test
                current_rank -= 1
                break
    
    return U, V, W
=======
    # Try greedy: remove one column at a time if possible
    # Precompute for speed
    T_flat = T.flatten()
    for attempt in range(current_rank):
        if time.time() - start_time > max_time:
            break
        found = False
        for r in range(current_rank):
            if time.time() - start_time > max_time:
                break
            mask = [i for i in range(current_rank) if i != r]
            U_test = U[:, mask]
            V_test = V[:, mask]
            W_test = W[:, mask]
            T_test = np.einsum('ar,br,cr->abc', U_test, V_test, W_test)
            if np.allclose(T_flat, T_test.flatten(), atol=1e-9):
                U, V, W = U_test, V_test, W_test
                current_rank -= 1
                found = True
                break
        if not found:
            break
    
    return U, V, W
>>>>>>> REPLACE

This is still essentially the same. Let me just submit the current program with minor optimizations and see if the rank reduction can find rank 38.

Actually, I realize the issue: the greedy removal checks if removing term r still gives T. But if the tensor has rank 38 and we have 39 terms, exactly one term is in the span of the others. Greedy removal should find it. The problem is just speed.

Let me make the verification much faster by precomputing the tensor as a flat array.

