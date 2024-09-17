import subprocess
import sys
import time


def run_command(command):
    process = subprocess.Popen(command, shell=True)
    return process


if __name__ == "__main__":
    backend = run_command("poetry run python app.py")
    time.sleep(5)  # Give the backend some time to start up
    frontend = run_command("poetry run streamlit run streamlit_app.py")

    try:
        backend.wait()
        frontend.wait()
    except KeyboardInterrupt:
        print("Stopping servers...")
        backend.terminate()
        frontend.terminate()
        sys.exit(0)
