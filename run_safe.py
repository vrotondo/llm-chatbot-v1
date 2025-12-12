import os
import sys
import subprocess

# Safe runner: runs a Python script in a subprocess with CUDA disabled and a timeout.
# Usage: python run_safe.py [path/to/script.py] [timeout_seconds]

def main():
    target = sys.argv[1] if len(sys.argv) > 1 else r"tutorials\coding-attention-mechanisms\3.5.py"
    timeout = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0

    py = sys.executable
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '-1'

    print(f"Running {target} with timeout={timeout}s using {py}")
    try:
        result = subprocess.run([py, '-u', target], capture_output=True, text=True, env=env, timeout=timeout)
        print('--- STDOUT ---')
        print(result.stdout)
        print('--- STDERR ---')
        print(result.stderr)
        print('Return code:', result.returncode)
    except subprocess.TimeoutExpired as e:
        print(f"ERROR: Script timed out after {timeout} seconds. It may be hanging on import or initialization.")
        if e.stdout:
            print('--- PARTIAL STDOUT ---')
            print(e.stdout)
        if e.stderr:
            print('--- PARTIAL STDERR ---')
            print(e.stderr)
        print('You can increase the timeout or run the script without torch (edit the file to guard torch imports).')
    except Exception as ex:
        print('ERROR: Failed to run script:', ex)

if __name__ == '__main__':
    main()
