#!/bin/bash
# Copyright 2026 The ODML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script builds the litert-lm CLI Python package into a PyPI-ready
# wheel using Bazelisk, and verifies the built wheel in an isolated environment.

set -ex

# Get the workspace root
WORKSPACE_ROOT=$(bazelisk info workspace)
cd "${WORKSPACE_ROOT}"

# Get the bazel-bin path
BAZEL_BIN=$(bazelisk info bazel-bin)
WHEEL_DIR="${BAZEL_BIN}/python/litert_lm_cli"
rm -rf "${WHEEL_DIR}"

# Authenticate gcloud using the Kokoro keystore if available
if [[ -n "${KOKORO_KEYSTORE_DIR:-}" && -f "${KOKORO_KEYSTORE_DIR}/75272_litert_lm_api_gcs_service_account.json" ]]; then
  echo "Authenticating gcloud with service account..."
  export GOOGLE_APPLICATION_CREDENTIALS="${KOKORO_KEYSTORE_DIR}/75272_litert_lm_api_gcs_service_account.json"
  gcloud auth activate-service-account --key-file="${GOOGLE_APPLICATION_CREDENTIALS}" || true
fi

# Determine which API wheels to fetch from GCS based on the release mode
API_WHEELS_GCS_DIR="gs://litert-lm-api/macos/nightly_wheels"
API_PREFIX="litert_lm_api_nightly"
if [[ "${PUBLISH_STABLE_RELEASE}" == "1" ]]; then
  API_WHEELS_GCS_DIR="gs://litert-lm-api/macos/stable_wheels"
  API_PREFIX="litert_lm_api"
fi

API_WHEELS_DIR="${WORKSPACE_ROOT}/api_wheels"
mkdir -p "${API_WHEELS_DIR}"
EXTRA_BAZEL_ARGS=""

if [[ "${PUBLISH_STABLE_RELEASE}" == "1" ]]; then
  echo "Stable release mode: Will build the CLI wheel first and fetch exact matching API wheels."
else
  echo "Continuous/Nightly mode: Fetching the latest available API wheels from GCS to sync versions..."
  
  # Temporarily disable exit-on-error for the ls command
  set +e
  LATEST_WHEEL_PATH=$(gcloud storage ls --sort-by="~updated" --limit=1 "${API_WHEELS_GCS_DIR}/${API_PREFIX}-*.whl" 2>/dev/null | head -n 1 || true)
  set -e
  
  if [[ -n "${LATEST_WHEEL_PATH}" ]]; then
    # Parse the exact version string (e.g., 0.15.0.dev20260720)
    LATEST_VERSION=$(basename "$LATEST_WHEEL_PATH" | cut -d'-' -f2)
    # Extract just the date part (e.g., 20260720). Add || true to prevent set -e crash if dev isn't present
    LATEST_DATE=$(echo "$LATEST_VERSION" | grep -o 'dev[0-9]*' | sed 's/dev//' || true)
    
    echo "Latest available API version in GCS is: ${LATEST_VERSION} (Date: ${LATEST_DATE})"
    
    if [[ -n "${LATEST_DATE}" ]]; then
      # Override the DEV_VERSION in Bazel so the CLI wheel's requires.txt matches the available API wheel
      EXTRA_BAZEL_ARGS="--define=DEV_VERSION=${LATEST_DATE}"
    fi
  else
    echo "⚠️ Failed to find any API wheels in GCS! Proceeding with default version."
  fi
fi

echo "Building wheel using Bazelisk..."
bazelisk build //python/litert_lm_cli:wheel "$@" ${EXTRA_BAZEL_ARGS}

# 1. Read the EXACT version string from the freshly built CLI wheel!
CLI_WHEEL_PATH=$(ls "${WHEEL_DIR}"/*.whl 2>/dev/null | head -n 1 || true)
CLI_VERSION=$(basename "${CLI_WHEEL_PATH}" | cut -d'-' -f2 || true)

if [[ -z "${CLI_VERSION}" ]]; then
  echo "❌ Failed to parse CLI version from wheel: ${CLI_WHEEL_PATH}"
  exit 1
fi
echo "Detected CLI Version: ${CLI_VERSION}"

# 3. Download the exact matching API wheels
echo "Downloading API wheels for version ${CLI_VERSION}..."
gcloud storage cp "${API_WHEELS_GCS_DIR}/${API_PREFIX}-${CLI_VERSION}*.whl" "${API_WHEELS_DIR}/" || true

if ! ls "${API_WHEELS_DIR}"/*.whl > /dev/null 2>&1; then
  echo "❌ Failed to find matching API wheels in GCS for version ${CLI_VERSION}!"
  exit 1
fi

TEST_VENV="${WORKSPACE_ROOT}/python/litert_lm_cli/test_venv"

for PY_VER in "3.10" "3.11" "3.12" "3.13" "3.14"; do
  echo "------------------------------------------------"
  echo "Setting up temporary virtual environment for Python ${PY_VER}..."
  echo "------------------------------------------------"
  rm -rf "${TEST_VENV}"

  # Force uv to use or download the specific target Python version
  uv venv --python="${PY_VER}" "${TEST_VENV}"

  # Universal Cross-Platform venv Activation & Python binary selection
  if [[ -d "${TEST_VENV}/Scripts" ]]; then
    source "${TEST_VENV}/Scripts/activate"
    PY_EXE="python"
  else
    source "${TEST_VENV}/bin/activate"
    PY_EXE="python3"
  fi

  echo "Installing the freshly built CLI wheel and the GCS API wheels..."
  uv pip install --index-url https://pypi.org/simple --find-links "${API_WHEELS_DIR}" "${WHEEL_DIR}"/*.whl

  cd "${TEST_VENV}"

  # Run E2E CLI tests
  bash "${WORKSPACE_ROOT}/python/litert_lm_cli/e2e_tests/cli_tests.sh"

  cd "${WORKSPACE_ROOT}"
  deactivate
done

rm -rf "${TEST_VENV}"
echo "✨ Verification completed successfully for all Python versions!"
