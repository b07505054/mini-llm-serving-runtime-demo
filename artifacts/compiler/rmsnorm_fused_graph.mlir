module {
  func.func @main(%arg0: tensor<16x768xf16>) -> tensor<16x768xf16> {
    %0 = "hir.fused_rmsnorm"(%arg0) {fusion.candidate = "rmsnorm", fusion.group = "rmsnorm_0", kernel.selection = "runtime_profile", lowering.source = "llm.rmsnorm"} : (tensor<16x768xf16>) -> tensor<16x768xf16>
    return %0 : tensor<16x768xf16>
  }
}

