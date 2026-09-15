# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# docs/licenses/SAM2-Apache-2.0.txt file of this repository.

from hydra import initialize_config_module

initialize_config_module("sam2_train", version_base="1.2")
