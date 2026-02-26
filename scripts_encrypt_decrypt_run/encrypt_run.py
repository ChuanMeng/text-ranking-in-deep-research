import argparse
import base64
import hashlib
import json
import os
from tqdm import tqdm


def derive_key(password: str, length: int) -> bytes:
    """Derive a fixed-length XOR key from the password using SHA256, repeated to `length`."""
    key32 = hashlib.sha256(password.encode("utf-8")).digest()  # 32 bytes
    return key32 * (length // len(key32)) + key32[: length % len(key32)]


def encode(plaintext: str, canary: str) -> str:
    """
    Encrypt `plaintext` using XOR with a repeated SHA256(canary) key, then base64-encode.
    Returns base64 ciphertext (the 'secret').
    """
    pt = plaintext.encode("utf-8")
    key = derive_key(canary, len(pt))
    ct = bytes(a ^ b for a, b in zip(pt, key))
    return base64.b64encode(ct).decode("ascii")


def decode(secret_b64: str, canary: str) -> str:
    """
    Decrypt base64 ciphertext using the same XOR key derivation from `canary`.
    Returns plaintext.
    """
    ct = base64.b64decode(secret_b64.encode("ascii"))
    key = derive_key(canary, len(ct))
    pt = bytes(a ^ b for a, b in zip(ct, key))
    return pt.decode("utf-8")


CANARY = "text-ranking-in-deep-research"


def process_file_enc(data: dict, canary: str) -> None:
    """Encode all item['output'] in data['result'] in place."""
    for item in data["result"]:
        if item['type'] == 'tool_call' and 'arguments' in item:
            if isinstance(item['arguments'], str):
                item['arguments'] = encode(item['arguments'], canary)
            else:
                item['arguments'] = encode(json.dumps(item['arguments']), canary)
        if 'output' not in item:
            continue
        if isinstance(item["output"], str):
            item["output"] = encode(item["output"], canary)
        else:
            item["output"] = encode(json.dumps(item["output"]), canary)


def process_file_dec(data: dict, canary: str) -> None:
    """Decode all item['output'] in data['result'] in place. If decoded content can be json, parse it, else retain as string."""
    for item in data["result"]:
        if item['type'] == 'tool_call' and 'arguments' in item:
            decoded_arg = decode(item['arguments'], canary)
            try:
                item['arguments'] = json.loads(decoded_arg)
            except (json.JSONDecodeError, TypeError):
                item['arguments'] = decoded_arg
        if 'output' not in item:
            continue
        decoded_out = decode(item["output"], canary)
        try:
            item["output"] = json.loads(decoded_out)
        except (json.JSONDecodeError, TypeError):
            item["output"] = decoded_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode or decode run JSON files.")
    parser.add_argument("--input-dir", type=str, required=True, help="Folder path containing JSON files to process")
    parser.add_argument("--mode", type=str, required=True, choices=["enc", "dec"], help="'enc' to encode, 'dec' to decode")
    parser.add_argument("--output-dir", type=str, required=True, help="Output folder for encoded/decoded JSON files")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    if not os.path.isdir(input_dir):
        raise SystemExit(f"Input path is not a directory: {input_dir}")

    json_files = sorted(f for f in os.listdir(input_dir) if f.endswith(".json"))
    if not json_files:
        raise SystemExit(f"No .json files found in {input_dir}")

    # Process all files in memory; on any error, nothing is saved
    results: list[tuple[str, dict]] = []
    for filename in tqdm(json_files):
        filepath = os.path.join(input_dir, filename)
        # print('Processing file: ', filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if args.mode == "enc":
            process_file_enc(data, CANARY)
        else:
            process_file_dec(data, CANARY)
        results.append((filename, data))

    os.makedirs(output_dir, exist_ok=True)
    for filename, data in results:
        outpath = os.path.join(output_dir, filename)
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

