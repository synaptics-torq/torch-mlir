// RUN: torch-mlir-opt <%s --split-input-file -convert-torch-onnx-to-torch | FileCheck %s
// RUN: torch-mlir-opt <%s --split-input-file -convert-torch-onnx-to-torch="gru-split-gates-min-elements=2000" | FileCheck %s --check-prefix=SPLIT

// CHECK-LABEL:   func.func @test_gru_forward(
// The input projection of all three gates runs once, before the loop.
// CHECK:           torch.aten.linear {{.*}} -> !torch.vtensor<[8,15],f32>
// CHECK:           torch.prim.Loop
// CHECK:             torch.aten.select.int {{.*}} -> !torch.vtensor<[2,15],f32>
// With linear_before_reset, one matmul gives H.R^T of all three gates.
// CHECK:             torch.aten.linear {{.*}} -> !torch.vtensor<[2,15],f32>
// z and r share one sigmoid.
// CHECK:             torch.aten.sigmoid {{.*}} -> !torch.vtensor<[2,10],f32>
// CHECK-NOT:         torch.aten.linear
// CHECK-NOT:         torch.aten.sigmoid
// CHECK:             torch.aten.tanh
// CHECK:             torch.prim.Loop.condition

func.func @test_gru_forward(%arg0: !torch.vtensor<[4,2,3],f32>, %arg1: !torch.vtensor<[1,15,3],f32>, %arg2: !torch.vtensor<[1,15,5],f32>, %arg3: !torch.vtensor<[1,30],f32>) -> (!torch.vtensor<[4,1,2,5],f32>, !torch.vtensor<[1,2,5],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
  %0:2 = torch.operator "onnx.GRU"(%arg0, %arg1, %arg2, %arg3) {torch.onnx.hidden_size = 5 : si64, torch.onnx.linear_before_reset = 1 : si64} : (!torch.vtensor<[4,2,3],f32>, !torch.vtensor<[1,15,3],f32>, !torch.vtensor<[1,15,5],f32>, !torch.vtensor<[1,30],f32>) -> (!torch.vtensor<[4,1,2,5],f32>, !torch.vtensor<[1,2,5],f32>)
  return %0#0, %0#1 : !torch.vtensor<[4,1,2,5],f32>, !torch.vtensor<[1,2,5],f32>
}

// -----

// CHECK-LABEL:   func.func @test_gru_reverse(
// CHECK:           torch.aten.linear {{.*}} -> !torch.vtensor<[8,15],f32>
// CHECK:           torch.prim.Loop
// CHECK:           ^bb0(%[[I:.*]]: !torch.int,
// The reverse layer goes from the last timestep to the first.
// CHECK:             %[[T:.*]] = torch.aten.sub.int %{{.*}}, %[[I]]
// CHECK:             torch.aten.select.int %{{.*}}, %{{.*}}, %[[T]]
// Without linear_before_reset, the packed matmul holds z and r only.
// CHECK:             torch.aten.linear {{.*}} -> !torch.vtensor<[2,10],f32>
// CHECK:             torch.aten.sigmoid {{.*}} -> !torch.vtensor<[2,10],f32>
// CHECK:             torch.aten.linear {{.*}} -> !torch.vtensor<[2,5],f32>
// CHECK:             torch.aten.tanh
// CHECK:             torch.aten.slice_scatter %{{.*}}, %{{.*}}, %{{.*}}, %[[T]],

func.func @test_gru_reverse(%arg0: !torch.vtensor<[4,2,3],f32>, %arg1: !torch.vtensor<[1,15,3],f32>, %arg2: !torch.vtensor<[1,15,5],f32>) -> !torch.vtensor<[1,2,5],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
  %0:2 = torch.operator "onnx.GRU"(%arg0, %arg1, %arg2) {torch.onnx.direction = "reverse", torch.onnx.hidden_size = 5 : si64} : (!torch.vtensor<[4,2,3],f32>, !torch.vtensor<[1,15,3],f32>, !torch.vtensor<[1,15,5],f32>) -> (!torch.none, !torch.vtensor<[1,2,5],f32>)
  return %0#1 : !torch.vtensor<[1,2,5],f32>
}

// -----

// CHECK-LABEL:   func.func @test_gru_bidirectional(
// CHECK-SAME:      %[[X:[a-z0-9]+]]: !torch.vtensor<[4,2,3],f32>, %[[W:[a-z0-9]+]]: !torch.vtensor<[2,15,3],f32>, %[[R:[a-z0-9]+]]: !torch.vtensor<[2,15,5],f32>
// Direction 0 is forward and uses index 0 of W and R.
// CHECK:           torch.aten.select.int %[[W]], %{{.*}}, %int0{{(_[0-9]+)?}} :
// CHECK:           torch.aten.select.int %[[R]], %{{.*}}, %int0{{(_[0-9]+)?}} :
// CHECK:           torch.prim.Loop
// CHECK-NOT:         torch.aten.sub.int
// CHECK:             torch.prim.Loop.condition
// Direction 1 is reverse and uses index 1 of W and R.
// CHECK:           torch.aten.select.int %[[W]], %{{.*}}, %int1{{(_[0-9]+)?}} :
// CHECK:           torch.aten.select.int %[[R]], %{{.*}}, %int1{{(_[0-9]+)?}} :
// CHECK:           torch.prim.Loop
// CHECK:             torch.aten.sub.int
// CHECK:             torch.prim.Loop.condition
// CHECK:           torch.aten.cat {{.*}} -> !torch.vtensor<[4,2,2,5],f32>
// CHECK:           torch.aten.cat {{.*}} -> !torch.vtensor<[2,2,5],f32>

func.func @test_gru_bidirectional(%arg0: !torch.vtensor<[4,2,3],f32>, %arg1: !torch.vtensor<[2,15,3],f32>, %arg2: !torch.vtensor<[2,15,5],f32>, %arg3: !torch.vtensor<[2,30],f32>) -> (!torch.vtensor<[4,2,2,5],f32>, !torch.vtensor<[2,2,5],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
  %0:2 = torch.operator "onnx.GRU"(%arg0, %arg1, %arg2, %arg3) {torch.onnx.direction = "bidirectional", torch.onnx.hidden_size = 5 : si64, torch.onnx.linear_before_reset = 1 : si64} : (!torch.vtensor<[4,2,3],f32>, !torch.vtensor<[2,15,3],f32>, !torch.vtensor<[2,15,5],f32>, !torch.vtensor<[2,30],f32>) -> (!torch.vtensor<[4,2,2,5],f32>, !torch.vtensor<[2,2,5],f32>)
  return %0#0, %0#1 : !torch.vtensor<[4,2,2,5],f32>, !torch.vtensor<[2,2,5],f32>
}

// -----

// CHECK-LABEL:   func.func @test_gru_large_batch(
// CHECK:           torch.aten.linear {{.*}} -> !torch.vtensor<[128,48],f32>
// CHECK:           torch.prim.Loop
// CHECK:             torch.aten.linear {{.*}} -> !torch.vtensor<[64,48],f32>
// CHECK:             torch.aten.sigmoid {{.*}} -> !torch.vtensor<[64,32],f32>

// The packed input projection has 2 * 64 * 48 elements and the packed recurrent
// matmul 64 * 48. Both reach 2000, so each gate gets its own matmul.
// SPLIT-LABEL:   func.func @test_gru_large_batch(
// SPLIT-COUNT-3:   torch.aten.linear {{.*}} -> !torch.vtensor<[128,16],f32>
// SPLIT:           torch.prim.Loop
// SPLIT:             torch.aten.linear {{.*}} -> !torch.vtensor<[64,16],f32>
// SPLIT:             torch.aten.sigmoid {{.*}} -> !torch.vtensor<[64,16],f32>
// SPLIT:             torch.aten.linear {{.*}} -> !torch.vtensor<[64,16],f32>
// SPLIT:             torch.aten.sigmoid {{.*}} -> !torch.vtensor<[64,16],f32>
// SPLIT:             torch.aten.linear {{.*}} -> !torch.vtensor<[64,16],f32>
// SPLIT:             torch.aten.tanh

func.func @test_gru_large_batch(%arg0: !torch.vtensor<[2,64,8],f32>, %arg1: !torch.vtensor<[1,48,8],f32>, %arg2: !torch.vtensor<[1,48,16],f32>, %arg3: !torch.vtensor<[1,96],f32>) -> (!torch.vtensor<[2,1,64,16],f32>, !torch.vtensor<[1,64,16],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
  %0:2 = torch.operator "onnx.GRU"(%arg0, %arg1, %arg2, %arg3) {torch.onnx.hidden_size = 16 : si64, torch.onnx.linear_before_reset = 1 : si64} : (!torch.vtensor<[2,64,8],f32>, !torch.vtensor<[1,48,8],f32>, !torch.vtensor<[1,48,16],f32>, !torch.vtensor<[1,96],f32>) -> (!torch.vtensor<[2,1,64,16],f32>, !torch.vtensor<[1,64,16],f32>)
  return %0#0, %0#1 : !torch.vtensor<[2,1,64,16],f32>, !torch.vtensor<[1,64,16],f32>
}

// -----

// CHECK-LABEL:   func.func @test_gru_layout_1(
// CHECK-SAME:      %[[X:[a-z0-9]+]]: !torch.vtensor<[2,4,3],f32>
// Layout 1 is [batch, seq, ...]. The expansion transposes X to [seq, batch,
// input] before it reshapes X for the input projection.
// CHECK:           %[[XT:.*]] = torch.aten.transpose.int %[[X]], {{.*}} -> !torch.vtensor<[4,2,3],f32>
// CHECK:           torch.aten.reshape %[[XT]], {{.*}} -> !torch.vtensor<[8,3],f32>
// CHECK:           torch.prim.Loop
// CHECK:           torch.aten.transpose.int {{.*}} -> !torch.vtensor<[2,4,1,5],f32>
// CHECK:           torch.aten.transpose.int {{.*}} -> !torch.vtensor<[2,1,5],f32>

func.func @test_gru_layout_1(%arg0: !torch.vtensor<[2,4,3],f32>, %arg1: !torch.vtensor<[1,15,3],f32>, %arg2: !torch.vtensor<[1,15,5],f32>, %arg3: !torch.vtensor<[1,30],f32>, %arg4: !torch.vtensor<[2,1,5],f32>) -> (!torch.vtensor<[2,4,1,5],f32>, !torch.vtensor<[2,1,5],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
  %none = torch.constant.none
  %0:2 = torch.operator "onnx.GRU"(%arg0, %arg1, %arg2, %arg3, %none, %arg4) {torch.onnx.hidden_size = 5 : si64, torch.onnx.layout = 1 : si64} : (!torch.vtensor<[2,4,3],f32>, !torch.vtensor<[1,15,3],f32>, !torch.vtensor<[1,15,5],f32>, !torch.vtensor<[1,30],f32>, !torch.none, !torch.vtensor<[2,1,5],f32>) -> (!torch.vtensor<[2,4,1,5],f32>, !torch.vtensor<[2,1,5],f32>)
  return %0#0, %0#1 : !torch.vtensor<[2,4,1,5],f32>, !torch.vtensor<[2,1,5],f32>
}

// -----

// A sequence_lens that holds seq_length for every batch entry gives the same
// result as no sequence_lens.
// CHECK-LABEL:   func.func @test_gru_full_sequence_lens(
// CHECK:           torch.prim.Loop
// CHECK-NOT:       torch.operator "onnx.GRU"

func.func @test_gru_full_sequence_lens(%arg0: !torch.vtensor<[4,2,3],f32>, %arg1: !torch.vtensor<[1,15,3],f32>, %arg2: !torch.vtensor<[1,15,5],f32>) -> !torch.vtensor<[1,2,5],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
  %none = torch.constant.none
  %lens = torch.operator "onnx.Constant"() {torch.onnx.value = dense<4> : tensor<2xsi32>} : () -> !torch.vtensor<[2],si32>
  %0:2 = torch.operator "onnx.GRU"(%arg0, %arg1, %arg2, %none, %lens) {torch.onnx.hidden_size = 5 : si64} : (!torch.vtensor<[4,2,3],f32>, !torch.vtensor<[1,15,3],f32>, !torch.vtensor<[1,15,5],f32>, !torch.none, !torch.vtensor<[2],si32>) -> (!torch.none, !torch.vtensor<[1,2,5],f32>)
  return %0#1 : !torch.vtensor<[1,2,5],f32>
}
