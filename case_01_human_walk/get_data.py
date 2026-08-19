"""
Fetch the motion capture dataset for the human walk case.

The trajectories are CMU Motion Capture subject 35 (walking), preprocessed and released
by Huang et al. with Graph Mechanics Networks. This script downloads that preprocessed
file so the two methods are trained and evaluated on identical data.

    python case_01_human_walk/get_data.py

Writes case_01_human_walk/data/motion.pkl and verifies its checksum.

The train/validation/test splits are not downloaded. They ship with this repository as
split_n1.pkl, split_n2.pkl and split_n3.pkl, one per rollout horizon. Each fixes which
frames of which trials are drawn, so deleting them changes the sample of frames and the
reported numbers along with it.
"""

# (c) All rights reserved. ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Laboratory of Intelligent Maintenance and Operations Systems (IMOS), 2025.
# Authors: Vinay Sharma and Olga Fink
# Released under the Non-Commercial License Agreement in LICENSE.txt.

import hashlib
import os
import sys
import urllib.error
import urllib.request

URL = ("https://raw.githubusercontent.com/hanjq17/GMN/main/"
       "spatial_graph/motion/motion.pkl")
SHA256 = "3ad06450285a4668aa053a0de3aef87a377d07968c82277418e7bda7e7d7f084"
EXPECTED_BYTES = 7137116

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, "data", "motion.pkl")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def report(count, block, total):
    if total <= 0:
        return
    done = min(count * block, total)
    pct = 100.0 * done / total
    sys.stdout.write(f"\r  {done / 1e6:6.2f} / {total / 1e6:6.2f} MB  ({pct:5.1f}%)")
    sys.stdout.flush()


def main():
    os.makedirs(os.path.dirname(DEST), exist_ok=True)

    if os.path.exists(DEST) and sha256(DEST) == SHA256:
        print(f"Already present and verified: {DEST}")
        return

    print(f"Downloading motion capture data from {URL}")
    try:
        urllib.request.urlretrieve(URL, DEST + ".part", reporthook=report)
    except urllib.error.URLError as exc:
        print(f"\nDownload failed: {exc}")
        print("The file is available in the GMN repository under spatial_graph/motion/.")
        sys.exit(1)
    print()

    size = os.path.getsize(DEST + ".part")
    digest = sha256(DEST + ".part")
    if size != EXPECTED_BYTES or digest != SHA256:
        print(f"Checksum mismatch. Expected {EXPECTED_BYTES} bytes / {SHA256},")
        print(f"got {size} bytes / {digest}. Not installing.")
        os.remove(DEST + ".part")
        sys.exit(1)

    os.replace(DEST + ".part", DEST)
    print(f"Saved {DEST}")
    print(f"  {size:,} bytes")
    print(f"  sha256 {sha256(DEST)}")

    splits = [f"split_n{n}.pkl" for n in (1, 2, 3)]
    missing = [s for s in splits if not os.path.exists(os.path.join(HERE, "data", s))]
    if missing:
        print(f"\nWarning: split files missing ({', '.join(missing)}).")
        print("They ship with the repository. Without them the loader resamples the")
        print("frame selection, which changes the reported errors.")


if __name__ == "__main__":
    main()
