import subprocess
import os
import sys
import tempfile
import platform
import webbrowser
import datetime # Added for timestamp filename
from pathlib import Path

# --- Configuration ---
PLANTUML_JAR_NAME = "plantuml.jar"
JAVA_EXE = "java"  # Assume 'java' is in the system PATH
OUTPUT_SUBDIR = "static/qgen_images" # Relative path for saving images

# --- Helper Function to find plantuml.jar ---
def find_plantuml_jar(script_path):
    """Looks for plantuml.jar in the same directory as the script."""
    script_dir = script_path.parent.resolve()
    local_jar_path = script_dir / PLANTUML_JAR_NAME
    if local_jar_path.is_file():
        return str(local_jar_path)
    return None

# --- Helper Function to display the image ---
def display_image(image_path_str):
    """Tries to display the image using Pillow or fallback to webbrowser."""
    image_path = Path(image_path_str).resolve()
    print(f"\nAttempting to display: {image_path}")

    try:
        # Try Pillow first (more reliable)
        from PIL import Image
        img = Image.open(image_path)
        img.show()
        print("Image display command sent (using Pillow/default system viewer).")
        return True
    except ImportError:
        print("INFO: Pillow library not found (pip install Pillow). Falling back to webbrowser.")
    except Exception as e_pil:
        print(f"Error displaying image with Pillow: {e_pil}")
        # Continue to try webbrowser

    # Fallback to webbrowser
    try:
        # Create a file URI (handles spaces and different OS formats)
        file_uri = image_path.as_uri()
        print(f"Opening with webbrowser: {file_uri}")
        opened = webbrowser.open(file_uri)
        if opened:
            print("Image display command sent to webbrowser.")
            return True
        else:
            print("webbrowser.open() returned False. Display may have failed.")
            return False
    except Exception as e_web:
        print(f"Error using webbrowser: {e_web}")
        return False

# --- Main Script Logic ---
def main():
    script_path = Path(__file__) # Gets the path of the currently running script
    script_dir = script_path.parent.resolve()
    plantuml_jar_path = find_plantuml_jar(script_path)

    if not plantuml_jar_path:
        print(f"Error: '{PLANTUML_JAR_NAME}' not found in the script directory: {script_dir}")
        print(f"Please place {PLANTUML_JAR_NAME} in the same directory as this script.")
        sys.exit(1)

    print(f"Using PlantUML JAR: {plantuml_jar_path}")

    # --- Define and Create Output Directory ---
    output_dir = script_dir / OUTPUT_SUBDIR
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output images will be saved in: {output_dir.resolve()}")
    except OSError as e:
        print(f"Error: Could not create output directory '{output_dir.resolve()}': {e}")
        sys.exit(1)

    # --- Get PlantUML Code from User ---
    print("\nEnter your PlantUML code below.")
    print("End input with Ctrl+D (Linux/macOS) or Ctrl+Z then Enter (Windows).")
    print("You can also type 'EOF' on a new line to finish.")
    print("-" * 30)
    puml_code_lines = []
    try:
        while True:
            line = input()
            # Simple EOF marker for easier pasting in some terminals
            if line.strip().upper() == 'EOF':
                break
            puml_code_lines.append(line)
    except EOFError:
        pass  # Standard way to end multi-line input

    puml_code = "\n".join(puml_code_lines).strip()

    if not puml_code:
        print("\nError: No PlantUML code entered.")
        sys.exit(1)

    # Basic check: Add @startuml and @enduml if they seem missing
    if not puml_code.startswith("@start"):
        puml_code = "@startuml\n" + puml_code
        print("INFO: Added missing '@startuml'")
    if not puml_code.endswith("@enduml"):
        puml_code = puml_code + "\n@enduml"
        print("INFO: Added missing '@enduml'")

    print("\n--- Processing PlantUML Code ---")

    # --- Run PlantUML using the pipe mode ---
    command = [
        JAVA_EXE,
        "-jar",
        plantuml_jar_path,
        "-pipe",        # Read PlantUML source from stdin
        "-tpng",        # Output format: PNG
        "-charset", "UTF-8", # Ensure correct character encoding
        # Add '-verbose' for more detailed PlantUML output if needed for debugging
        # "-verbose",
    ]

    generated_image_path = None # Renamed for clarity
    try:
        print(f"Executing: {' '.join(command)}")
        process = subprocess.run(
            command,
            input=puml_code.encode('utf-8'), # Pass code to stdin, encoded as UTF-8
            capture_output=True,             # Capture stdout (image data) and stderr
            check=False                      # Don't raise exception on non-zero exit code
        )

        # --- Check for Java/PlantUML Errors ---
        if process.returncode != 0 or not process.stdout or not process.stdout.startswith(b'\x89PNG'):
            print("\n--- ERROR: PlantUML Execution Failed ---")
            print(f"Return Code: {process.returncode}")
            stderr_output = process.stderr.decode('utf-8', errors='ignore').strip()
            stdout_output = process.stdout.decode('utf-8', errors='ignore').strip()

            if stderr_output:
                print("\n--- Stderr ---")
                print(stderr_output)
                print("-" * 14)
            # Only show stdout if it's not empty and doesn't look like PNG data
            if stdout_output and not process.stdout.startswith(b'\x89PNG'):
                 print("\n--- Stdout ---")
                 print(stdout_output)
                 print("-" * 14)

            # Provide hints based on common errors
            if "java" in stderr_output.lower() and ("not recognized" in stderr_output.lower() or "command not found" in stderr_output.lower()):
                print("\nHint: Could not execute 'java'. Is Java installed and added to your system's PATH environment variable?")
            elif "Unsupported major.minor version" in stderr_output:
                print("\nHint: Java version mismatch. Your installed Java version might be too old or too new for this plantuml.jar.")
                print("      Check PlantUML documentation for required Java version.")
            elif "java.lang.OutOfMemoryError" in stderr_output:
                 print("\nHint: Java ran out of memory. Try increasing heap size, e.g.:")
                 print(f"      java -Xmx1024m -jar {PLANTUML_JAR_NAME} ...")
            elif not process.stdout and process.returncode == 0:
                 print("\nHint: PlantUML ran but produced no output image. Check your PlantUML syntax for errors.")
            elif not process.stdout.startswith(b'\x89PNG') and process.stdout:
                print("\nHint: PlantUML produced text output instead of a PNG image. Check your PlantUML syntax.")
            else:
                print("\nHint: Review the error messages above. Check PlantUML syntax or Java environment.")
            sys.exit(1)

        # --- Save the Output Image to the specified directory ---
        # Generate a unique filename using a timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_filename = f"plantuml_diagram_{timestamp}.png"
        output_file_path = output_dir / output_filename

        try:
            with open(output_file_path, 'wb') as f_out:
                f_out.write(process.stdout)
            generated_image_path = str(output_file_path) # Store the path as string
            print(f"\nSuccessfully generated diagram image.")
            print(f"Image saved to: {output_file_path.resolve()}")
        except IOError as e:
            print(f"\n--- ERROR: Failed to save image to {output_file_path.resolve()} ---")
            print(f"IOError: {e}")
            sys.exit(1) # Exit if we cannot save the file

        # --- Display Image ---
        if not display_image(generated_image_path):
            print("\nAutomatic image display failed or was not possible.")
            print(f"You can open the generated image manually: {Path(generated_image_path).resolve()}")
        else:
            # Optional: Keep the file path printed for reference even if displayed
            print(f"(Image file location: {Path(generated_image_path).resolve()})")


    except FileNotFoundError:
         # This happens if JAVA_EXE is not found
         print(f"\nError: Command '{JAVA_EXE}' not found.")
         print("Please ensure Java is installed and its 'bin' directory is in your system's PATH environment variable.")
         sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected script error occurred: {e}")
        if generated_image_path: # If image was created before error
             print(f"The generated image (if created) might be at: {Path(generated_image_path).resolve()}")
        sys.exit(1)

    # The image file is now saved permanently in ./static/qgen_images/
    print("\nScript finished.")


if __name__ == "__main__":
    main()