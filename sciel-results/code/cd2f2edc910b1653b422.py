<<<<<<< SEARCH
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
=======
    if time.time() > _DEADLINE:
        # Emergency fallback
        U, V, W = schoolbook_decomposition()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
>>>>>>> REPLACE

Wait, I need to make sure the `schoolbook_decomposition` function is defined before it's used in the deadline check. Let me restructure.

Actually, looking at my edits, `schoolbook_decomposition` is defined inside `construct()` before it's used, so it should be fine. But the deadline check at the end might not have access to it if the code path is different. Let me restructure to be safe.

Let me reconsider my approach. The key issue is that I'm not sure the Strassen terms are correct. Let me verify them numerically in the code and fall back gracefully.

Let me simplify: just make sure the verification is correct and the fallback works.

