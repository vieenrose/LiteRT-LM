# Local reconstruction of the preprocessor CMakeLists, which is missing from the public
# LiteRT-LM export (v0.14.0 and main) even though runtime/components/CMakeLists.txt does
# add_subdirectory(preprocessor) and other modules link its targets. The component is
# header-only here — only .h files ship — so every target is INTERFACE.
#
# Aliases below are exactly the ones referenced elsewhere in the tree:
#   Components::Preprocessor, ::Audio, ::AudioMiniAudio, ::Image, ::StbImage

set(PKG_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

# Umbrella target: the include path for all preprocessor headers.
add_litertlm_library(runtime_components_preprocessor INTERFACE)
add_library(LiteRTLM::Runtime::Components::Preprocessor ALIAS runtime_components_preprocessor)
target_include_directories(runtime_components_preprocessor
  INTERFACE
    ${LITERTLM_INCLUDE_PATHS}
    ${LITERT_INCLUDE_PATHS}
)

# audio_preprocessor.h — the abstract audio interface.
add_litertlm_library(runtime_components_preprocessor_audio INTERFACE)
add_library(LiteRTLM::Runtime::Components::Preprocessor::Audio ALIAS runtime_components_preprocessor_audio)
target_include_directories(runtime_components_preprocessor_audio
  INTERFACE
    ${LITERTLM_INCLUDE_PATHS}
    ${LITERT_INCLUDE_PATHS}
)
target_link_libraries(runtime_components_preprocessor_audio INTERFACE LITERTLM_DEPS)

# audio_preprocessor_miniaudio.h — the miniaudio-backed implementation.
add_litertlm_library(runtime_components_preprocessor_audio_miniaudio INTERFACE)
add_library(LiteRTLM::Runtime::Components::Preprocessor::AudioMiniAudio ALIAS runtime_components_preprocessor_audio_miniaudio)
target_include_directories(runtime_components_preprocessor_audio_miniaudio
  INTERFACE
    ${LITERTLM_INCLUDE_PATHS}
    ${LITERT_INCLUDE_PATHS}
    ${THIRD_PARTY_DIR}/miniaudio
)
target_link_libraries(runtime_components_preprocessor_audio_miniaudio
  INTERFACE
    LITERTLM_DEPS
    LiteRTLM::Runtime::Components::Preprocessor::Audio
)

# image_preprocessor.h — the abstract image interface.
add_litertlm_library(runtime_components_preprocessor_image INTERFACE)
add_library(LiteRTLM::Runtime::Components::Preprocessor::Image ALIAS runtime_components_preprocessor_image)
target_include_directories(runtime_components_preprocessor_image
  INTERFACE
    ${LITERTLM_INCLUDE_PATHS}
    ${LITERT_INCLUDE_PATHS}
)
target_link_libraries(runtime_components_preprocessor_image INTERFACE LITERTLM_DEPS)

# stb_image_preprocessor.h — the stb_image-backed implementation.
add_litertlm_library(runtime_components_preprocessor_stb_image INTERFACE)
add_library(LiteRTLM::Runtime::Components::Preprocessor::StbImage ALIAS runtime_components_preprocessor_stb_image)
target_include_directories(runtime_components_preprocessor_stb_image
  INTERFACE
    ${LITERTLM_INCLUDE_PATHS}
    ${LITERT_INCLUDE_PATHS}
    ${THIRD_PARTY_DIR}/stb
)
target_link_libraries(runtime_components_preprocessor_stb_image
  INTERFACE
    LITERTLM_DEPS
    LiteRTLM::Runtime::Components::Preprocessor::Image
)
