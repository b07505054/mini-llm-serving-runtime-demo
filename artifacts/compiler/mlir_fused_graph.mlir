module {
  func.func @main(%arg0: tensor<1x128xf32>, %arg1: tensor<128x64xf32>, %arg2: tensor<1x64xf32>) -> tensor<1x64xf32> {
    %0 = tensor.empty() : tensor<1x64xf32>
    %1 = linalg.matmul {fusion.candidate = "matmul_bias_relu", fusion.group = "matmul_bias_relu_0", fusion.role = "producer"} ins(%arg0, %arg1 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%0 : tensor<1x64xf32>) -> tensor<1x64xf32>
    %mapped = linalg.map { arith.addf } ins(%1, %arg2 : tensor<1x64xf32>, tensor<1x64xf32>) outs(%0 : tensor<1x64xf32>) {fusion.group = "matmul_bias_relu_0", fusion.role = "bias_add"}
    %cst = arith.constant 0.000000e+00 : f32
    %mapped_0 = linalg.map ins(%mapped : tensor<1x64xf32>) outs(%0 : tensor<1x64xf32>) {fusion.group = "matmul_bias_relu_0", fusion.role = "activation"}
      (%in: f32) {
        %2 = arith.maximumf %in, %cst : f32
        linalg.yield %2 : f32
      }
    return %mapped_0 : tensor<1x64xf32>
  }
}

