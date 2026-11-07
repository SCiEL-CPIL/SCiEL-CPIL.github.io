import time
import sys
import __main__

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 90s limit, 5s safety margin
    
    # Robustly locate the data dictionary P
    P = None
    
    # 1. Check direct global 'P' first
    if 'P' in globals():
        candidate = globals()['P']
        if isinstance(candidate, dict) and 'edge_order' in candidate:
            P = candidate
            
    # 2. Scan current module globals for any dict with 'edge_order'
    if P is None:
        for name, val in globals().items():
            if isinstance(val, dict) and 'edge_order' in val:
                P = val
                break
            
    # 3. Scan __main__ globals for any dict with 'edge_order'
    if P is None:
        for name, val in vars(__main__).items():
            if isinstance(val, dict) and 'edge_order' in val:
                P = val
                break
                
    # 4. Search all loaded modules for any dict with 'edge_order'
    if P is None:
        for mod in sys.modules.values():
            try:
                mod_vars = vars(mod)
                for name, val in mod_vars.items():
                    if isinstance(val, dict) and 'edge_order' in val:
                        P = val
                        break
            except Exception:
                continue
            if P is not None:
                break

    if P is None:
        raise NameError("Could not find data dictionary P")
