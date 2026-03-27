from __future__ import annotations

from pathlib import Path


def build_config_yaml(
    onnx_model,
    cal_data_dir,
    working_dir,
    output_model_file_prefix,
    jobs=8,
    optimize_level="O3",
    quantized="int8",
):
    int16_suffix = ",set_all_nodes_int16" if quantized == "int16" else ""
    return f"""model_parameters:
  onnx_model: '{onnx_model}'
  march: "bayes-e"
  layer_out_dump: False
  working_dir: '{working_dir}'
  output_model_file_prefix: '{output_model_file_prefix}'
input_parameters:
  input_name: ""
  input_type_rt: 'nv12'
  input_type_train: 'rgb'
  input_layout_train: 'NCHW'
  norm_type: 'data_scale'
  scale_value: 0.003921568627451
calibration_parameters:
  cal_data_dir: '{cal_data_dir}'
  cal_data_type: 'float32'
  calibration_type: 'default'
  optimization: set_Softmax_input_int8,set_Softmax_output_int8{int16_suffix}
compiler_parameters:
  jobs: {jobs}
  compile_mode: 'latency'
  debug: true
  optimize_level: '{optimize_level}'
"""


def write_config(path, **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = build_config_yaml(**kwargs)
    path.write_text(content, encoding="utf-8")
    return path
