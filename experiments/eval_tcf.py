import os
import sys
import argparse
import subprocess
import glob
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_single_file(match_file, tcf_bin, re_thresh, te_thresh):
    """
    Process a single match file using TCF binary.
    Returns: (re, te, time_ms, is_success) or None if failed
    """
    try:
        # Construct expected GT file path
        # Assuming format: {seq_id}_{src}_{ref}.txt -> {seq_id}_{src}_{ref}.gt.txt
        gt_file = match_file.replace(".txt", ".gt.txt")
        if not os.path.exists(gt_file):
            # Fallback or skip
            gt_file = "" # Demo will skip error computation

        # Call TCF executable
        cmd = [tcf_bin, match_file]
        if gt_file:
            cmd.append(gt_file)
        
        # Set environment variable to limit OpenMP threads per process
        # This prevents CPU oversubscription when running in parallel
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            env=env
        )
        
        # Parse output
        for line in result.stdout.splitlines():
            if line.startswith("CSV_RESULT"):
                _, re, te, time_ms = line.split(",")
                re = float(re)
                te = float(te)
                time_ms = float(time_ms)
                
                is_success = (re < re_thresh and te < te_thresh)
                return (re, te, time_ms, is_success)
                
    except subprocess.CalledProcessError:
        pass 
    except Exception as e:
        print(f"Unexpected error on {match_file}: {e}")
    
    return None

def main():
    parser = argparse.ArgumentParser(description="Batch evaluate TCF on generated matches")
    parser.add_argument("--matches_dir", type=str, required=True, help="Directory containing .txt match files")
    parser.add_argument("--tcf_bin", type=str, required=True, help="Path to TCF executable (demo)")
    parser.add_argument("--re_thresh", type=float, default=15.0, help="Rotation error threshold for recall (deg)")
    parser.add_argument("--te_thresh", type=float, default=0.3, help="Translation error threshold for recall (m)")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(), help="Number of parallel workers")
    args = parser.parse_args()

    match_files = glob.glob(os.path.join(args.matches_dir, "*.txt"))
    # Filter out .gt.txt files
    match_files = [f for f in match_files if not f.endswith(".gt.txt")]
    
    if not match_files:
        print(f"No .txt files found in {args.matches_dir}")
        return

    print(f"Found {len(match_files)} match files. Starting evaluation with {args.num_workers} workers...")

    re_list = []
    te_list = []
    time_list = []
    success_count = 0

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_single_file, f, args.tcf_bin, args.re_thresh, args.te_thresh): f 
            for f in match_files
        }
        
        # Collect results as they complete
        for future in tqdm(as_completed(futures), total=len(match_files)):
            result = future.result()
            if result is not None:
                re, te, time_ms, is_success = result
                re_list.append(re)
                te_list.append(te)
                time_list.append(time_ms)
                if is_success:
                    success_count += 1

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
