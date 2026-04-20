from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(description="Upload a compiled model to the board, run hrt_model_exec infer, and download dumps.")
    parser.add_argument("--host", default="192.168.31.110")
    parser.add_argument("--user", default="sunrise")
    parser.add_argument("--password", default="sunrise")
    parser.add_argument("--model-file", required=True, help="Local compiled model path, for example .bin or .hbm.")
    parser.add_argument("--input-bin", required=True, help="Local input tensor .bin path")
    parser.add_argument("--remote-dir", default="/home/sunrise/tinynav_segment_validation_20260310")
    parser.add_argument("--dump-subdir", default="qat_board_dump")
    parser.add_argument("--output-dir", required=True, help="Local output directory")
    return parser


def run_remote(client, command):
    stdin, stdout, stderr = client.exec_command(command)
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8", errors="ignore")
    err = stderr.read().decode("utf-8", errors="ignore")
    return exit_code, out, err


def main(argv=None):
    import paramiko

    args = build_parser().parse_args(argv)
    model_file = Path(args.model_file).resolve()
    input_bin = Path(args.input_bin).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    remote_model = f"{args.remote_dir}/{model_file.name}"
    remote_input = f"{args.remote_dir}/{input_bin.name}"
    remote_dump = f"{args.remote_dir}/{args.dump_subdir}"

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(args.host, username=args.user, password=args.password, timeout=10)
    sftp = client.open_sftp()
    try:
        sftp.put(str(model_file), remote_model)
        sftp.put(str(input_bin), remote_input)

        run_remote(client, f"mkdir -p {remote_dump}")
        model_info_code, model_info_out, model_info_err = run_remote(
            client, f"hrt_model_exec model_info --model_file {remote_model}"
        )
        infer_code, infer_out, infer_err = run_remote(
            client,
            f"hrt_model_exec infer --model_file {remote_model} "
            f"--input_file {remote_input} --frame_count 1 "
            f"--enable_dump true --dump_path {remote_dump} --dump_format txt",
        )

        local_dump = output_dir / args.dump_subdir
        local_dump.mkdir(parents=True, exist_ok=True)
        for entry in sftp.listdir_attr(remote_dump):
            remote_path = f"{remote_dump}/{entry.filename}"
            local_path = local_dump / entry.filename
            sftp.get(remote_path, str(local_path))

        report = {
            "host": args.host,
            "remote_model": remote_model,
            "remote_input": remote_input,
            "remote_dump": remote_dump,
            "model_info_exit_code": model_info_code,
            "infer_exit_code": infer_code,
            "model_info_stdout": model_info_out,
            "model_info_stderr": model_info_err,
            "infer_stdout": infer_out,
            "infer_stderr": infer_err,
            "local_dump_dir": str(local_dump),
        }
        report_path = output_dir / "board_verify_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        if model_info_code != 0 or infer_code != 0:
            raise SystemExit(1)
    finally:
        sftp.close()
        client.close()


if __name__ == "__main__":
    main()
