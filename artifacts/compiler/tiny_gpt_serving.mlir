tiny_gpt_serving.mlirmodule attributes {
  llm.model = "tiny-gpt",
  llm.num_layers = 12 : i64,
  llm.hidden_size = 768 : i64,
  llm.num_heads = 12 : i64,
  llm.intermediate_size = 3072 : i64,
  llm.vocab_size = 50257 : i64
} {
  func.func @tiny_gpt_prefill(%tokens: tensor<?xi32>) -> tensor<?x768xf16> {
    %0 = "llm.embed"(%tokens) : (tensor<?xi32>) -> tensor<?x768xf16>
    %1 = "llm.rmsnorm"(%0) : (tensor<?x768xf16>) -> tensor<?x768xf16>
    %q, %k, %v = "llm.qkv_projection"(%1) : (tensor<?x768xf16>) -> (tensor<?x768xf16>, tensor<?x768xf16>, tensor<?x768xf16>)
    %2 = "llm.attention_prefill"(%q, %k, %v) { kv_cache.role = "producer", serving.phase = "prefill" } : (tensor<?x768xf16>, tensor<?x768xf16>, tensor<?x768xf16>) -> tensor<?x768xf16>
    %3 = "llm.mlp"(%2) : (tensor<?x768xf16>) -> tensor<?x768xf16>
    return %3 : tensor<?x768xf16>
  }

  func.func @tiny_gpt_decode(%token: tensor<1xi32>) -> tensor<1x768xf16> {
    %0 = "llm.embed"(%token) : (tensor<1xi32>) -> tensor<1x768xf16>
    %1 = "llm.rmsnorm"(%0) : (tensor<1x768xf16>) -> tensor<1x768xf16>
    %q, %k, %v = "llm.qkv_projection"(%1) : (tensor<1x768xf16>) -> (tensor<1x768xf16>, tensor<1x768xf16>, tensor<1x768xf16>)
    %2 = "llm.attention_decode"(%q, %k, %v) { kv_cache.role = "consumer", serving.phase = "decode" } : (tensor<1x768xf16>, tensor<1x768xf16>, tensor<1x768xf16>) -> tensor<1x768xf16>
    %3 = "llm.mlp"(%2) : (tensor<1x768xf16>) -> tensor<1x768xf16>
    return %3 : tensor<1x768xf16>
  }
}