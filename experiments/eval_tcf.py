import os
import sys
import argparse
import subprocess
import glob
import numpy as np
from tqdm import tqdm
import tempfile

def main():
    parser = argparse.ArgumentParser(description="Batch evaluate TCF on generated matches using C++ Batch Mode")
    parser.add_argument("--matches_dir", type=str, required=True, help="Directory containing .txt match files")
    parser.add_argument("--tcf_bin", type=str, required=True, help="Path to TCF executable (demo)")
    parser.add_argument("--re_thresh", type=float, default=15.0, help="Rotation error threshold for recall (deg)")
    parser.add_argument("--te_thresh", type=float, default=0.3, help="Translation error threshold for recall (m)")
    args = parser.parse_args()

    # 1. Find all match files
    match_files = glob.glob(os.path.join(args.matches_dir, "*.txt"))
    # Filter out .gt.txt files
    match_files = [f for f in match_files if not f.endswith(".gt.txt")]
    
    if not match_files:
        print(f"No .txt files found in {args.matches_dir}")
        return

    print(f"Found {len(match_files)} match files. Preparing file list...")

    # 2. Generate temporary file list
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp_file:
        file_list_path = tmp_file.name
        for match_file in match_files:
            # Construct expected GT file path
            gt_file = match_file.replace(".txt", ".gt.txt")
            if not os.path.exists(gt_file):
                gt_file = "" # Empty string if no GT
            
            # Write: matches_path gt_path
            tmp_file.write(f"{match_file} {gt_file}\n")
    
    print(f"File list generated at: {file_list_path}")
    print("Starting TCF batch execution...")

    re_list = []
    te_list = []
    time_list = []
    success_count = 0

    try:
        # 3. Run TCF in batch mode
        # Set OMP_NUM_THREADS to avoid oversubscription if TCF uses OpenMP internally
        # Since we run one process, we can let it use all cores or limit it.
        # Usually letting it use all cores (default) is fine for single process batch.
        # But if you want to be safe:
        # env = os.environ.copy()
        # env["OMP_NUM_THREADS"] = "8" 

        cmd = [args.tcf_bin, "--batch", file_list_path]
        
        # We need to capture stdout in real-time or wait for finish
        # For progress bar, we might want to read line by line
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        pbar = tqdm(total=len(match_files))
        
        for line in process.stdout:
            if line.startswith("CSV_RESULT"):
                _, re, te, time_ms = line.strip().split(",")
                re = float(re)
                te = float(te)
                time_ms = float(time_ms)
                
                re_list.append(re)
                te_list.append(te)
                time_list.append(time_ms)
                
                if re < args.re_thresh and te < args.te_thresh:
                    success_count += 1
                
                pbar.update(1)
        
        pbar.close()
        process.wait()
        
        if process.returncode != 0:
            print(f"TCF process exited with error code {process.returncode}")
            print("STDERR:", process.stderr.read())

    except Exception as e:
        print(f"Error during execution: {e}")
    finally:
        # Cleanup
        if os.path.exists(file_list_path):
            os.remove(file_list_path)

    # 4. Summary
    num_samples = len(re_list)
    if num_samples == 0:
        print("No valid results collected.")
        return

    rr = success_count / num_samples
    mean_re = np.mean(re_list)
    mean_te = np.mean(te_list)
    mean_time = np.mean(time_list)

    print("\n" + "="*40)
    print("TCF BATCH EVALUATION RESULTS (C++ Batch Mode)")
    print("="*40)
    print(f"Num Samples: {num_samples}")
    print(f"Registration Recall (RR): {rr:.4f} (@ RE<{args.re_thresh}deg, TE<{args.te_thresh}m)")
    print(f"Mean RRE: {mean_re:.4f} deg")
    print(f"Mean RTE: {mean_te:.4f} m")
    print(f"Mean Time: {mean_time:.2f} ms")
    print("="*40)

if __name__ == "__main__":
    main()
