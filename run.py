import os
import sys
import time
import socket
import threading
import subprocess
import re

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
except:
    pass

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

def run_tunnel():
    """Background thread to establish public tunnel and print link in terminal"""
    time.sleep(1.5)  # Wait for Flask to bind port 5000
    cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-R", "80:localhost:5000",
        "nokey@localhost.run"
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        for line in iter(proc.stdout.readline, ''):
            match = re.search(r'https://[a-zA-Z0-9_\-\.]+\.lhr\.life', line)
            if match:
                tunnel_url = match.group(0)
                print("\n" + "="*70, flush=True)
                print(f" [+] PUBLIC INTERNET LINK READY FOR FRIENDS: {tunnel_url}", flush=True)
                print(" [*] Copy and send this link to your friends on WhatsApp/Mobile!", flush=True)
                print(" [*] Works from anywhere in the world on phone/laptop.", flush=True)
                print("="*70 + "\n", flush=True)
                break

        proc.wait()
    except Exception as e:
        print(f"\n[!] Tunnel notice: {e}", flush=True)

if __name__ == "__main__":
    from app import app, get_or_load_model
    get_or_load_model()
    
    local_ip = get_local_ip()

    print("\n" + "="*70, flush=True)
    print(" [*] SPAMGUARD AI - EMAIL CLASSIFIER IS RUNNING!", flush=True)
    print(f" [*] [1] Local Link (Your PC):        http://127.0.0.1:5000", flush=True)
    print(f" [*] [2] Friends Link (Same Wi-Fi):   http://{local_ip}:5000", flush=True)
    print(f" [*] [3] Public Internet Link:       Generating tunnel in terminal...", flush=True)
    print("="*70 + "\n", flush=True)

    # Launch tunnel in background thread
    t = threading.Thread(target=run_tunnel, daemon=True)
    t.start()

    # Start Flask server
    app.run(host='0.0.0.0', port=5000, debug=False)
