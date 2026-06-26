import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SCRIPT_PATH = BASE_DIR / "colorectal_autokeras_raytune.py"
OUTPUT_BASE = BASE_DIR / "output"
BATCH_OUTPUT_BASE = OUTPUT_BASE / "batch_runs"

TASKS = ["binary", "multiclass"]
IMAGE_SIZES = [128, 160, 224, 384]
PATCH_SIZES = [16, 32]
VIT_DEPTHS = ["tiny", "small"]

NUM_SAMPLES = 4
TUNE_EPOCHS = 6
FINAL_EPOCHS = 30
USE_CLASS_WEIGHTS = True
CONTINUE_ON_ERROR = True
SKIP_IF_RESULT_EXISTS = True
SHOW_LIVE_OUTPUT = True


def describe_runtime():
    return {
        "python_executable": sys.executable,
        "launcher_cwd": str(Path.cwd()),
        "base_dir": str(BASE_DIR),
        "script_path": str(SCRIPT_PATH),
        "script_exists": SCRIPT_PATH.exists(),
        "is_wsl": "microsoft" in os.uname().release.lower() if hasattr(os, "uname") else False,
        "os_name": os.name,
    }


def prompt_single_choice(label: str, choices):
    print(f"\nSelect {label} (press Enter for all):")
    for index, choice in enumerate(choices, start=1):
        print(f"  {index}. {choice}")

    by_index = {str(i): value for i, value in enumerate(choices, start=1)}
    by_value = {str(value).lower(): value for value in choices}

    while True:
        raw = input(f"Enter {label}: ").strip().lower()
        if not raw:
            return list(choices)
        if raw in by_index:
            return [by_index[raw]]
        if raw in by_value:
            return [by_value[raw]]
        print("Invalid input, please try again.")


def resolve_user_selection():
    tasks = prompt_single_choice("task", TASKS)
    image_sizes = prompt_single_choice("image_size", IMAGE_SIZES)
    patch_sizes = prompt_single_choice("patch_size", PATCH_SIZES)
    vit_depths = prompt_single_choice("vit_depth", VIT_DEPTHS)
    return {
        "tasks": tasks,
        "image_sizes": image_sizes,
        "patch_sizes": patch_sizes,
        "vit_depths": vit_depths,
    }


def build_configs(selection):
    configs = []
    for task in selection["tasks"]:
        for image_size in selection["image_sizes"]:
            for patch_size in selection["patch_sizes"]:
                if image_size % patch_size != 0:
                    continue
                tokens_per_side = image_size // patch_size
                num_tokens = tokens_per_side * tokens_per_side
                for vit_depth in selection["vit_depths"]:
                    if patch_size == 32 and vit_depth == "tiny":
                        continue
                    configs.append({
                        "task": task,
                        "image_size": image_size,
                        "patch_size": patch_size,
                        "vit_depth": vit_depth,
                        "tokens_per_side": tokens_per_side,
                        "num_tokens": num_tokens,
                    })
    return configs


def get_batch_output_root(selection):
    task_tag = "_and_".join(selection["tasks"])
    image_tag = "all" if len(selection["image_sizes"]) == len(IMAGE_SIZES) else "-".join(map(str, selection["image_sizes"]))
    patch_tag = "all" if len(selection["patch_sizes"]) == len(PATCH_SIZES) else "-".join(map(str, selection["patch_sizes"]))
    depth_tag = "all" if len(selection["vit_depths"]) == len(VIT_DEPTHS) else "-".join(selection["vit_depths"])
    run_tag = f"tasks_{task_tag}__img_{image_tag}__patch_{patch_tag}__depth_{depth_tag}"
    return BATCH_OUTPUT_BASE / run_tag


def get_run_output_dir(cfg, output_root: Path):
    run_name = (
        f"{cfg['task']}_image{cfg['image_size']}_patch{cfg['patch_size']}_"
        f"{cfg['vit_depth']}_tokens{cfg['num_tokens']}"
    )
    return output_root / run_name


def get_expected_result_json(cfg):
    return OUTPUT_BASE / "colorectal_exp_auto" / cfg["task"] / (
        f"image{cfg['image_size']}_patch{cfg['patch_size']}_{cfg['vit_depth']}"
    ) / "results_raytuned_vit_final.json"


def stream_pipe(pipe, sink, prefix: str, is_err: bool):
    terminal = sys.stderr if is_err else sys.stdout
    try:
        for line in iter(pipe.readline, ""):
            sink.write(line)
            sink.flush()
            if SHOW_LIVE_OUTPUT:
                terminal.write(prefix + line)
                terminal.flush()
    finally:
        pipe.close()


def run_one(cfg, index, total, output_root: Path):
    run_output_dir = get_run_output_dir(cfg, output_root)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    expected_result = get_expected_result_json(cfg)

    metadata = {
        "index": index,
        "total": total,
        "config": cfg,
        "command": None,
        "cwd": str(BASE_DIR),
        "status": "pending",
        "started_at": None,
        "finished_at": None,
        "duration_seconds": None,
        "returncode": None,
        "expected_result_json": str(expected_result),
    }

    if SKIP_IF_RESULT_EXISTS and expected_result.exists():
        metadata["status"] = "skipped_existing"
        print(f"[{index}/{total}] skipped existing result: {expected_result}")
        return metadata

    command = [
        sys.executable,
        str(SCRIPT_PATH),
        "--mode", "auto_train",
        "--task", cfg["task"],
        "--image-size", str(cfg["image_size"]),
        "--patch-size", str(cfg["patch_size"]),
        "--vit-depth", cfg["vit_depth"],
        "--num-samples", str(NUM_SAMPLES),
        "--tune-epochs", str(TUNE_EPOCHS),
        "--final-epochs", str(FINAL_EPOCHS),
    ]
    if USE_CLASS_WEIGHTS and cfg["task"] == "binary":
        command.append("--use-class-weights")

    metadata["command"] = command
    metadata["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.time()
    child_env = os.environ.copy()
    child_env.setdefault("PYTHONIOENCODING", "utf-8")

    log_path = run_output_dir / "run.log"
    err_path = run_output_dir / "run.err.log"
    prefix = f"[{index}/{total}] "
    print(
        f"[{index}/{total}] task={cfg['task']} image={cfg['image_size']} "
        f"patch={cfg['patch_size']} depth={cfg['vit_depth']} tokens={cfg['num_tokens']}"
    )
    print("Command:", " ".join(command))

    with log_path.open("w", encoding="utf-8") as stdout_f, err_path.open("w", encoding="utf-8") as stderr_f:
        process = subprocess.Popen(
            command,
            cwd=str(BASE_DIR),
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        stdout_thread = threading.Thread(
            target=stream_pipe,
            args=(process.stdout, stdout_f, prefix, False),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=stream_pipe,
            args=(process.stderr, stderr_f, prefix, True),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        returncode = process.wait()
        stdout_thread.join()
        stderr_thread.join()

    metadata["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    metadata["duration_seconds"] = round(time.time() - t0, 2)
    metadata["returncode"] = returncode
    metadata["status"] = "ok" if returncode == 0 else "failed"
    print(
        f"[{index}/{total}] completed status={metadata['status']} "
        f"returncode={returncode} duration={metadata['duration_seconds']}s"
    )
    return metadata


def main():
    selection = resolve_user_selection()
    output_root = get_batch_output_root(selection)
    output_root.mkdir(parents=True, exist_ok=True)

    runtime = describe_runtime()
    if not runtime["script_exists"]:
        raise FileNotFoundError(f"Training script not found: {SCRIPT_PATH}")

    configs = build_configs(selection)
    summary = {
        "script_path": str(SCRIPT_PATH),
        "runtime": runtime,
        "selection": selection,
        "num_samples": NUM_SAMPLES,
        "tune_epochs": TUNE_EPOCHS,
        "final_epochs": FINAL_EPOCHS,
        "use_class_weights": USE_CLASS_WEIGHTS,
        "total_runs": len(configs),
        "configs": configs,
        "results": [],
    }

    summary_path = output_root / "batch_summary.json"
    print("\nRuntime:", json.dumps(runtime, indent=2, ensure_ascii=False))
    print("Selection:", json.dumps(selection, indent=2, ensure_ascii=False))
    print(f"Planned runs: {len(configs)}")
    for idx, cfg in enumerate(configs, start=1):
        print(
            f"  {idx:02d}. task={cfg['task']} image={cfg['image_size']} "
            f"patch={cfg['patch_size']} depth={cfg['vit_depth']} tokens={cfg['num_tokens']}"
        )

    for idx, cfg in enumerate(configs, start=1):
        result = run_one(cfg, idx, len(configs), output_root)
        summary["results"].append(result)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        if result["status"] == "failed" and not CONTINUE_ON_ERROR:
            raise SystemExit(f"Run failed at config {idx}: {cfg}")

    ok = sum(1 for item in summary["results"] if item["status"] == "ok")
    skipped = sum(1 for item in summary["results"] if item["status"] == "skipped_existing")
    failed = sum(1 for item in summary["results"] if item["status"] == "failed")
    print(f"Finished. ok={ok} skipped={skipped} failed={failed}")
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
