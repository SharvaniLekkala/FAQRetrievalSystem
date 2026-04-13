import os
import subprocess
import json

def call_c_preprocessor(exe_path, text):
    """Bridge to the C preprocessor executable."""
    if not os.path.exists(exe_path):
        print(f"Error: C preprocessor executable not found at '{exe_path}'")
        return {"tokens": [text], "pos": ["O"], "ner": ["O"]} # Fallback to original text

    try:
        # Wrap text in quotes to handle shell correctly
        result = subprocess.run([exe_path, text], capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error calling C preprocessor: C executable failed with return code {e.returncode}")
        return {"tokens": [text], "pos": ["O"], "ner": ["O"]}
    except Exception as e:
        print(f"Error calling C preprocessor: {e}")
        return {"tokens": [text], "pos": ["O"], "ner": ["O"]}
