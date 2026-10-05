// RUN: torch-mlir-opt <%s -split-input-file -verify-diagnostics -convert-torch-onnx-to-torch

// The expansion runs every sequence to seq_length.
func.func @test_gru_partial_sequence_lens(%arg0: !torch.vtensor<[4,2,3],f32>, %arg1: !torch.vtensor<[1,15,3],f32>, %arg2: !torch.vtensor<[1,15,5],f32>) -> !torch.vtensor<[1,2,5],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
  %none = torch.constant.none
  %lens = torch.operator "onnx.Constant"() {torch.onnx.value = dense<[4, 2]> : tensor<2xsi32>} : () -> !torch.vtensor<[2],si32>
  // expected-error@+1 {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:2 = torch.operator "onnx.GRU"(%arg0, %arg1, %arg2, %none, %lens) {torch.onnx.hidden_size = 5 : si64} : (!torch.vtensor<[4,2,3],f32>, !torch.vtensor<[1,15,3],f32>, !torch.vtensor<[1,15,5],f32>, !torch.none, !torch.vtensor<[2],si32>) -> (!torch.none, !torch.vtensor<[1,2,5],f32>)
  return %0#1 : !torch.vtensor<[1,2,5],f32>
}

// -----

// sequence_lens must have one entry for each batch entry.
func.func @test_gru_sequence_lens_not_per_batch(%arg0: !torch.vtensor<[4,2,3],f32>, %arg1: !torch.vtensor<[1,15,3],f32>, %arg2: !torch.vtensor<[1,15,5],f32>) -> !torch.vtensor<[1,2,5],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
  %none = torch.constant.none
  %lens = torch.operator "onnx.Constant"() {torch.onnx.value = dense<4> : tensor<1xsi32>} : () -> !torch.vtensor<[1],si32>
  // expected-error@+1 {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:2 = torch.operator "onnx.GRU"(%arg0, %arg1, %arg2, %none, %lens) {torch.onnx.hidden_size = 5 : si64} : (!torch.vtensor<[4,2,3],f32>, !torch.vtensor<[1,15,3],f32>, !torch.vtensor<[1,15,5],f32>, !torch.none, !torch.vtensor<[1],si32>) -> (!torch.none, !torch.vtensor<[1,2,5],f32>)
  return %0#1 : !torch.vtensor<[1,2,5],f32>
}

// -----

// R must be [num_directions, 3*hidden_size, hidden_size].
func.func @test_gru_wrong_r_shape(%arg0: !torch.vtensor<[4,2,3],f32>, %arg1: !torch.vtensor<[1,15,3],f32>, %arg2: !torch.vtensor<[1,9,5],f32>) -> !torch.vtensor<[1,2,5],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
  // expected-error@+1 {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:2 = torch.operator "onnx.GRU"(%arg0, %arg1, %arg2) {torch.onnx.hidden_size = 5 : si64} : (!torch.vtensor<[4,2,3],f32>, !torch.vtensor<[1,15,3],f32>, !torch.vtensor<[1,9,5],f32>) -> (!torch.none, !torch.vtensor<[1,2,5],f32>)
  return %0#1 : !torch.vtensor<[1,2,5],f32>
}

// -----

// layout must be 0 or 1.
func.func @test_gru_layout_2(%arg0: !torch.vtensor<[4,2,3],f32>, %arg1: !torch.vtensor<[1,15,3],f32>, %arg2: !torch.vtensor<[1,15,5],f32>) -> !torch.vtensor<[1,2,5],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
  // expected-error@+1 {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:2 = torch.operator "onnx.GRU"(%arg0, %arg1, %arg2) {torch.onnx.hidden_size = 5 : si64, torch.onnx.layout = 2 : si64} : (!torch.vtensor<[4,2,3],f32>, !torch.vtensor<[1,15,3],f32>, !torch.vtensor<[1,15,5],f32>) -> (!torch.none, !torch.vtensor<[1,2,5],f32>)
  return %0#1 : !torch.vtensor<[1,2,5],f32>
}

// -----

// Only Sigmoid, Tanh and Relu are supported.
func.func @test_gru_unsupported_activation(%arg0: !torch.vtensor<[4,2,3],f32>, %arg1: !torch.vtensor<[1,15,3],f32>, %arg2: !torch.vtensor<[1,15,5],f32>) -> !torch.vtensor<[1,2,5],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
  // expected-error@+1 {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:2 = torch.operator "onnx.GRU"(%arg0, %arg1, %arg2) {torch.onnx.activations = ["Sigmoid", "Elu"], torch.onnx.hidden_size = 5 : si64} : (!torch.vtensor<[4,2,3],f32>, !torch.vtensor<[1,15,3],f32>, !torch.vtensor<[1,15,5],f32>) -> (!torch.none, !torch.vtensor<[1,2,5],f32>)
  return %0#1 : !torch.vtensor<[1,2,5],f32>
}

// -----

// clip is not supported.
func.func @test_gru_clip(%arg0: !torch.vtensor<[4,2,3],f32>, %arg1: !torch.vtensor<[1,15,3],f32>, %arg2: !torch.vtensor<[1,15,5],f32>) -> !torch.vtensor<[1,2,5],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
  // expected-error@+1 {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:2 = torch.operator "onnx.GRU"(%arg0, %arg1, %arg2) {torch.onnx.clip = 1.000000e+00 : f32, torch.onnx.hidden_size = 5 : si64} : (!torch.vtensor<[4,2,3],f32>, !torch.vtensor<[1,15,3],f32>, !torch.vtensor<[1,15,5],f32>) -> (!torch.none, !torch.vtensor<[1,2,5],f32>)
  return %0#1 : !torch.vtensor<[1,2,5],f32>
}
