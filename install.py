import os
import subprocess
import tempfile

def create_startup_shortcut():
    """Creates a Windows shortcut in the Startup folder for the clap launcher."""
    
    script_path = os.path.abspath("clap_launcher.pyw")
    if not os.path.exists(script_path):
        print(f"ERROR: Could not find {script_path}.")
        print("Please run this script from the same directory as clap_launcher.pyw")
        return

    startup_dir = os.path.join(os.environ["APPDATA"], "Microsoft", "Windows", "Start Menu", "Programs", "Startup")
    shortcut_path = os.path.join(startup_dir, "ClapLauncher.lnk")
    working_dir = os.path.dirname(script_path)
    
    # PowerShell script to create the shortcut using Windows Script Host (WSH)
    ps_script = f"""
$WshShell = New-Object -comObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut('{shortcut_path}')
# Target Pythonw.exe so no console window appears
$Shortcut.TargetPath = 'pythonw.exe'
$Shortcut.Arguments = '"{script_path}"'
$Shortcut.WorkingDirectory = '{working_dir}'
$Shortcut.WindowStyle = 7  # Minimized
$Shortcut.Save()
"""
    
    # Write powershell script to temporary file and execute it
    with tempfile.NamedTemporaryFile("w", suffix=".ps1", delete=False) as f:
        f.write(ps_script)
        temp_name = f.name
        
    try:
        print("Creating shortcut...")
        subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", temp_name], check=True)
        print("---------------------------------------------------------")
        print("SUCCESS! Startup shortcut created.")
        print(f"Location: {shortcut_path}")
        print("---------------------------------------------------------")
        print("The script will now start automatically when Windows boots.")
        print("Note: If you move this folder to a different location, you will need to run install.py again.")
    except Exception as e:
        print(f"Failed to create shortcut: {e}")
    finally:
        os.remove(temp_name)

if __name__ == "__main__":
    create_startup_shortcut()
