import sys, os, platform, json
from pathlib import Path

def main():
    root = Path(__file__).resolve().parents[1]
    summary = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "project_root": str(root),
        "folders": [p.name for p in root.iterdir() if p.is_dir()],
        "sample_files": [str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()][:30]
    }
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
