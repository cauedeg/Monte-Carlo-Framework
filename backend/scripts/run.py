import subprocess
import time
import sys
import os
import webbrowser

def start_frontend():
    try:
        start_path = "C:/Users/cauedeg/OneDrive/work/git/catalog/famework_mc/frontend/"
        npm_path = "C:/Program Files/nodejs/npm.cmd"
        subprocess.Popen([npm_path, "start"], cwd=start_path)
        time.sleep(5)  # Aguarde alguns segundos para garantir que o servidor esteja pronto
        webbrowser.open("http://localhost:5173")
    except Exception as e:
        with open("error.log", "a") as f:
            f.write(f"Failed to start frontend: {e}\n")

def main():
    start_frontend()

if __name__ == "__main__":
    main()