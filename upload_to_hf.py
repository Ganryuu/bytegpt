"""Upload the prepared byte dataset to the Hugging Face Hub."""

import argparse

from huggingface_hub import HfApi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="e.g. username/fineweb-edu-bytes-10BT")
    ap.add_argument("--dir", default="data/fineweb_edu_bytes")
    args = ap.parse_args()

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        repo_id=args.repo,
        repo_type="dataset",
        folder_path=args.dir,
    )
    print(f"uploaded: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
