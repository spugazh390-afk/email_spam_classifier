import subprocess
import sys
import re
import time

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
except:
    pass

def start_public_tunnel():
    print("\n=======================================================", flush=True)
    print(" [*] Generating a Public Shareable Link for your friends...", flush=True)
    print(" [*] Connecting to secure tunnel (takes 3-5 seconds)...", flush=True)
    print("=======================================================\n", flush=True)

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

        tunnel_url = None
        for line in iter(proc.stdout.readline, ''):
            # Look for https link in output
            match = re.search(r'https://[a-zA-Z0-9_\-\.]+\.lhr\.life', line)
            if match:
                tunnel_url = match.group(0)
                print("\n" + "="*65, flush=True)
                print(" [+] YOUR LIVE PUBLIC SHAREABLE LINK IS READY!", flush=True)
                print(f" [>] LINK: {tunnel_url}", flush=True)
                print("="*65, flush=True)
                print("\n [*] Send this link to your friends on WhatsApp or Mobile!", flush=True)
                print(" [*] Anyone in the world can open and test your app now.", flush=True)
                print(" [*] Press Ctrl + C anytime in this window to stop sharing.\n", flush=True)
                break
        
        # Keep process alive
        proc.wait()

    except KeyboardInterrupt:
        print("\n[*] Stopping public tunnel... Link closed.", flush=True)
    except Exception as e:
        print(f"\n[!] Tunnel Error: {e}", flush=True)
        print("[!] Note: Make sure 'py app.py' is running in another terminal first!", flush=True)

if __name__ == "__main__":
    start_public_tunnel()
