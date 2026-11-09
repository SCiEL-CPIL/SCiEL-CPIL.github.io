<<<<<<< SEARCH
    y_end = sol.y[:, -1]
    
    if not np.all(np.isfinite(y_end)):
        return y0
        
    return y_end
=======
    y_end = sol.y[:, -1]
    
    if not np.all(np.isfinite(y_end)):
        return y0_arr
        
    return y_end
>>>>>>> REPLACE
