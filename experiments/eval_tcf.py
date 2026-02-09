import os
import sys
import argparse
import subprocess
import glob
import numpy as np
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Batch evaluate TCF on generated matches")
    parser.add_argument("--matches_dir", type=str, required=True, help="Directory containing .txt match files")
    parser.add_argument("--tcf_bin", type=str, required=True, help="Path to TCF executable (demo)")
    parser.add_argument("--re_thresh", type=float, default=15.0, help="Rotation error threshold for recall (deg)")
    parser.add_argument("--te_thresh", type=float, default=0.3, help="Translation error threshold for recall (m)")
    args = parser.parse_args()

    match_files = glob.glob(os.path.join(args.matches_dir, "*.txt"))
    if not match_files:
        print(f"No .txt files found in {args.matches_dir}")
        return

    print(f"Found {len(match_files)} match files. Starting evaluation...")

    re_list = []
    te_list = []
    time_list = []
    success_count = 0

    for match_file in tqdm(match_files):
        try:
            # Construct expected GT file path
            # Assuming format: {seq_id}_{src}_{ref}.txt -> {seq_id}_{src}_{ref}.gt.txt
            gt_file = match_file.replace(".txt", ".gt.txt")
            if not os.path.exists(gt_file):
                # Fallback or skip
                gt_file = "" # Demo will skip error computation

            # Call TCF executable
            cmd = [args.tcf_bin, match_file]
            if gt_file:
                cmd.append(gt_file)
            
            # Pass resolution if needed (optional, using default)
            # cmd.append("0.3") 

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse output
            for line in result.stdout.splitlines():
                if line.startswith("CSV_RESULT"):
                    _, re, te, time_ms = line.split(",")
                    re = float(re)
                    te = float(te)
                    time_ms = float(time_ms)
                    
                    re_list.append(re)
                    te_list.append(te)
                    time_list.append(time_ms)
                    
                    if re < args.re_thresh and te < args.te_thresh:
                        success_count += 1
                    break
                    
        except subprocess.CalledProcessError as e:
            print(f"Error processing {match_file}: {e}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
        except Exception as e:
            print(f"Unexpected error on {match_file}: {e}")

    # Summary
    num_samples = len(re_list)
    if num_samples == 0:
        print("No valid results collected.")
        return

    rr = success_count / num_samples
    mean_re = np.mean(re_list)
    mean_te = np.mean(te_list)
    mean_time = np.mean(time_list)

    print("\n" + "="*40)
    print("TCF BATCH EVALUATION RESULTS")
    print("="*40)
    print(f"Num Samples: {num_samples}")
    print(f"Registration Recall (RR): {rr:.4f} (@ RE<{args.re_thresh}deg, TE<{args.te_thresh}m)")
    print(f"Mean RRE: {mean_re:.4f} deg")
    print(f"Mean RTE: {mean_te:.4f} m")
    print(f"Mean Time: {mean_time:.2f} ms")
    print("="*40)

if __name__ == "__main__":
    main()
