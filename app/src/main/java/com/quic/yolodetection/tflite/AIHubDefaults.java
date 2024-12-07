// ---------------------------------------------------------------------
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.quic.yolodetection.tflite;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

public class AIHubDefaults {
    // Delegates enabled to replicate AI Hub's defaults on Qualcomm devices.
    public static final Set<com.quic.yolodetection.tflite.TFLiteHelpers.DelegateType> enabledDelegates = new HashSet<>(Arrays.asList(
            com.quic.yolodetection.tflite.TFLiteHelpers.DelegateType.QNN_NPU,
            com.quic.yolodetection.tflite.TFLiteHelpers.DelegateType.GPUv2
    ));


    // Number of threads AI Hub uses by default for layers running on CPU.
    // https://app.aihub.qualcomm.com/docs/hub/api.html#profile-inference-options
    public static final int numCPUThreads = Runtime.getRuntime().availableProcessors() / 2;

    // The default delegate registry order for AI Hub.
    // For more details, see the JavaDoc for TFLiteHelpers::CreateInterpreterAndDelegatesFromOptions.
    public static final com.quic.yolodetection.tflite.TFLiteHelpers.DelegateType[][] delegatePriorityOrder = new com.quic.yolodetection.tflite.TFLiteHelpers.DelegateType[][] {
            // 1. QNN_NPU + GPUv2 + XNNPack
            // https://app.aihub.qualcomm.com/docs/hub/api.html#profile-inference-options
            // Similar to AI Hub "compute_unit=all", or "compute_unit=npu,gpu,cpu" on QC devices that support QNN
            // AI Hub sets some GPUv2 settings that are not accessible via the Java API.
            { com.quic.yolodetection.tflite.TFLiteHelpers.DelegateType.QNN_NPU, com.quic.yolodetection.tflite.TFLiteHelpers.DelegateType.GPUv2 },

            // 2. GPUv2 + XNNPack
            // https://app.aihub.qualcomm.com/docs/hub/api.html#profile-inference-options
            // Similar to AI Hub "compute_unit=gpu" on all devices
            // AI Hub sets some GPU settings that are not accessible via the Java API.
            { com.quic.yolodetection.tflite.TFLiteHelpers.DelegateType.GPUv2 },

            // 3. XNNPack (final, CPU-only fallback)
            // https://app.aihub.qualcomm.com/docs/hub/api.html#profile-inference-options
            // Same as AI Hub "compute_unit=cpu" on all devices
            { }

            // ------
            //
            // Additional delegates (eg. NNAPI, or something targeting non-Qualcomm hardware) could be added here or above,
            // if you desire to define a custom delegate selection order.
            //
            // ------
    };


    public static final com.quic.yolodetection.tflite.TFLiteHelpers.DelegateType[][] delegateGPU = new com.quic.yolodetection.tflite.TFLiteHelpers.DelegateType[][] {
            { com.quic.yolodetection.tflite.TFLiteHelpers.DelegateType.GPUv2 },
            { }
    };


    // Create a version of the above delegate priority order that can only use the provided delegates.
    public static com.quic.yolodetection.tflite.TFLiteHelpers.DelegateType[][] delegatePriorityOrderForDelegates(Set<com.quic.yolodetection.tflite.TFLiteHelpers.DelegateType> enabledDelegates) {
        return Arrays.stream(delegatePriorityOrder).filter(x -> enabledDelegates.containsAll(Arrays.asList(x))).toArray(com.quic.yolodetection.tflite.TFLiteHelpers.DelegateType[][]::new);
    }
}
