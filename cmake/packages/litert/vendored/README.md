# Vendored LiteRT headers

The whole `support/` header set LiteRT-LM includes (14 files), copied UNMODIFIED
from google-ai-edge/LiteRT `main` (Apache-2.0, same licence as this repo).

LiteRT-LM's own sources include them — e.g. `runtime/components/model_resources.h`,
`model_resources_task.h` and `model_resources_streaming.h` all carry
`#include "support/tokenizer/tokenizer.h"  // from @litert` — but the LiteRT revision this
project pins (`fb16353a648922cb6c67a8e9a7a9ebc946360ad2`, 2026-03-24, see
`cmake/packages/litert/litert.cmake`) predates that subtree entirely: the fetched tree has no
`support/` directory at all. A native build therefore dies with

    fatal error: support/tokenizer/tokenizer.h: No such file or directory

`litert_patcher.cmake` copies this directory into the fetched LiteRT source root, which is on
`LITERT_INCLUDE_PATHS`. The set was collected by scanning every `#include "support/..."` in this repo and following
those headers' own `support/` includes to a fixed point, so it is complete rather than
discovered one build failure at a time. Everything else they reference (`litert/cc/*`,
`tflite/types/half.h`, absl) already exists at the pinned revision.

Delete this directory and the copy step once the LiteRT pin advances past the commit that
added `support/`.
