import subprocess
import sys
import json

def get_installed_packages():
    # Run pip list command and capture the output
    result = subprocess.run([sys.executable, '-m', 'pip', 'list', '--format=json'], 
                          capture_output=True, text=True)
    
    # Parse the JSON output
    packages = json.loads(result.stdout)
    
    # Create a dictionary of package names and versions
    return {package['name']: package['version'] for package in packages}

# Get the dictionary of installed packages
installed_packages = get_installed_packages()