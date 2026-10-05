import argparse
import json
import sys
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()

    with urllib.request.urlopen(f"{args.url}/health", timeout=10) as response:
        health = json.load(response)

    assert health["status"] == "ok"
    print("Health check: PASS")

    payload = json.dumps({
        "prompt": "Industrial engineering improves",
        "max_new_tokens": 8,
        "temperature": 0,
    }).encode()

    req = urllib.request.Request(
        f"{args.url}/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=60) as response:
        result = json.load(response)

    assert result["output_tokens"] >= 1
    assert isinstance(result["text"], str)

    print("Generation: PASS")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
