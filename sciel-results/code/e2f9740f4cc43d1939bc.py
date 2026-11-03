    if max_err2 >= 1e-10:
        # Fallback to schoolbook
        ...
        return {"U": U_fb, "V": V_fb, "W": W_fb}
