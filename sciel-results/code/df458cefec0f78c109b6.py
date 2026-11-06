import importlib.util
import inspect
import os
import sys
import time

# Try to locate the module that contains pebble_baseline
def find_module(name="autoresearch_suite.science_tcs_expansion"):
    try:
        return importlib.import_module(name)
    except Exception as e:
        print("import error:", e)
        return None

mod = find_module()
if mod is not None:
    print("HAS pebble_baseline:", hasattr(mod, "pebble_baseline"))
    try:
        print(inspect.getsource(mod.pebble_baseline))
    except Exception as e:
        print("no source:", e)
    # list module attributes
    print("attrs:", [a for a in dir(mod) if not a.startswith("__")])
