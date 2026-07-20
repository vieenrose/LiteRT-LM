// Copyright 2026 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_EMBEDDING_ENGINE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_EMBEDDING_ENGINE_H_

#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/engine/io_types.h"

namespace litert::lm {

struct EmbeddingOptions {
  bool normalize = false;
};

struct EmbeddingResponse {
  std::vector<float> embedding;
};

// Interface for embedding models.
class EmbeddingEngine {
 public:
  virtual ~EmbeddingEngine() = default;

  // Computes embedding response for the given single request.
  virtual absl::StatusOr<EmbeddingResponse> ComputeEmbedding(
      const std::vector<InputData>& contents,
      const EmbeddingOptions& options) = 0;

  // Computes a batch of embedding responses for the given batch of requests.
  virtual absl::StatusOr<std::vector<EmbeddingResponse>> ComputeEmbeddingBatch(
      const std::vector<std::vector<InputData>>& contents,
      const EmbeddingOptions& options) = 0;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_EMBEDDING_ENGINE_H_
