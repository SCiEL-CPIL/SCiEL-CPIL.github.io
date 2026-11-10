<<<<<<< SEARCH
    for (ma, mb), key, pts in slope_lines.items():
        lines.append(('slope', (ma, mb, key), pts))
=======
    for key, pts in slope_lines.items():
        slope_tuple, intercept = key
        ma, mb = slope_tuple
        lines.append(('slope', (ma, mb, intercept), pts))
>>>>>>> REPLACE
