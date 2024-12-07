package com.quic.yolodetection.tflite;

import android.content.Context;
import android.util.Log;
import android.util.Pair;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Delegate;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.support.image.TensorImage;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.security.NoSuchAlgorithmException;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class TFLiteInferenceEngine implements AutoCloseable {
    public class TensorInfo {
        private String name;
        private int[] shape;
        private DataType dataType;
        private int bufferIndex;
        private int ID;
        private float[] quantize_scale_offset;
        private float[] data = null;

        public TensorInfo(String name, int[] shape, DataType dataType, int bufferIndex, int ID) {
            this.name = name;
            this.shape = shape;
            this.dataType = dataType;
            this.bufferIndex = bufferIndex;
            this.ID = ID;
        }


        public String getName() {
            return name;
        }
        public int getID() {
            return ID;
        }

        public int[] getShape() {
            return shape;
        }

        public float[] getData() {
            return this.data;
        }


        public void setQuantizeScaleOffset(float[] quantize_scale_offset) {
            this.quantize_scale_offset = quantize_scale_offset;
        }

        public float[] getQuantizeScaleOffset() {
            return this.quantize_scale_offset;
        }

        public void setData(float[] data) {
            this.data = data;
        }

        public DataType getDataType() {
            return dataType;
        }

        public int getBufferIndex() {
            return bufferIndex;
        }
    }

    private String TAG = "model_name";
    private final Interpreter tfLiteInterpreter;
    private final Map<TFLiteHelpers.DelegateType, Delegate> tfLiteDelegateStore;
    public List<TensorInfo> inputList;
    public List<TensorInfo> outputList;

    public TFLiteInferenceEngine(Context context,
                                 String modelPath,
                                 TFLiteHelpers.DelegateType[][] delegatePriorityOrder) throws IOException, NoSuchAlgorithmException {
        // Load TF Lite model
        Pair<MappedByteBuffer, String> modelAndHash = TFLiteHelpers.loadModelFile(context.getAssets(), modelPath);
        Pair<Interpreter, Map<TFLiteHelpers.DelegateType, Delegate>> iResult = TFLiteHelpers.CreateInterpreterAndDelegatesFromOptions(
                modelAndHash.first,
                delegatePriorityOrder,
                AIHubDefaults.numCPUThreads,
                context.getApplicationInfo().nativeLibraryDir,
                context.getCacheDir().getAbsolutePath(),
                modelAndHash.second
        );
        tfLiteInterpreter = iResult.first;
        tfLiteDelegateStore = iResult.second;

        inputList = getInputDetails();
        outputList = getOutputDetails();
        TAG = modelPath;

    }

    /**
     * Free resources used by the classifier.
     */
    @Override
    public void close() {
        tfLiteInterpreter.close();
        for (Delegate delegate : tfLiteDelegateStore.values()) {
            delegate.close();
        }
    }

    public String getLastInferenceTime() {
        NumberFormat timeFormatter = new DecimalFormat("0.00");
        String inferenceTime = timeFormatter.format((double) tfLiteInterpreter.getLastNativeInferenceDurationNanoseconds() / 1000000);
        return inferenceTime;
    }

    public List<TensorInfo> getOutputDetails() {
        List<TensorInfo> tensorInfos = new ArrayList<>();
        int outputCount = tfLiteInterpreter.getOutputTensorCount();

        for (int i = 0; i < outputCount; i++) {
            Tensor outputTensor = tfLiteInterpreter.getOutputTensor(i);
            String tensorName = outputTensor.name();
            int[] tensorShape = outputTensor.shape();
            DataType tensorDataType = outputTensor.dataType();
            int ID = outputTensor.index();
            // Create a TensorInfo object and add it to the list
            TensorInfo output_tmp = new TensorInfo(tensorName, tensorShape, tensorDataType, i, ID);
            float[] scale_offset = {outputTensor.quantizationParams().getScale(), outputTensor.quantizationParams().getZeroPoint()};
            if (scale_offset[0]<0.0001 && scale_offset[1] < 0.0001){
                scale_offset[0]=1.0f;
                scale_offset[1]=0.0f;
            }
            output_tmp.setQuantizeScaleOffset(scale_offset);
            tensorInfos.add(output_tmp);
        }
        return tensorInfos;
    }

    public List<TensorInfo> getInputDetails() {
        List<TensorInfo> tensorInfos = new ArrayList<>();
        int inputCount = tfLiteInterpreter.getInputTensorCount();

        for (int i = 0; i < inputCount; i++) {
            Tensor inputTensor = tfLiteInterpreter.getInputTensor(i);
            String tensorName = inputTensor.name();
            int[] tensorShape = inputTensor.shape();
            int ID = inputTensor.index();
            DataType tensorDataType = inputTensor.dataType();
            // Create a TensorInfo object and add it to the list
            tensorInfos.add(new TensorInfo(tensorName, tensorShape, tensorDataType, i, ID));

        }

        return tensorInfos;
    }

    public List<TensorInfo> predict(TensorImage preprocessed_image) {
        ByteBuffer inputBuffer;
        inputBuffer = preprocessed_image.getTensorBuffer().getBuffer();
        ByteBuffer[] inputs = new ByteBuffer[]{inputBuffer};
        // Inference tfLiteInterpreter
        tfLiteInterpreter.runForMultipleInputsOutputs(inputs, new HashMap<>());
        Log.e("tfLiteInterpreter", TAG+" Inference time: " + getLastInferenceTime());
        for (int i = 0; i < outputList.size(); i++) {
            ByteBuffer outputBuffer = tfLiteInterpreter.getOutputTensor(outputList.get(i).getBufferIndex()).asReadOnlyBuffer();
            if (outputList.get(i).getDataType() == DataType.FLOAT32) {
                FloatBuffer floatBuf = outputBuffer.asFloatBuffer();
                float[] floatArray = new float[floatBuf.remaining()];
                floatBuf.get(floatArray);
                outputList.get(i).setData(floatArray);
            } else if (outputList.get(i).getDataType() == DataType.UINT8) {
                outputList.get(i).setData(dequantize(outputBuffer,
                        outputList.get(i).quantize_scale_offset[0],
                        outputList.get(i).quantize_scale_offset[1],
                        true));
            }
            else if (outputList.get(i).getDataType() == DataType.INT8) {
                outputList.get(i).setData(dequantize(outputBuffer,
                        outputList.get(i).quantize_scale_offset[0],
                        outputList.get(i).quantize_scale_offset[1],
                        false));
            }
        }
        return outputList;
    }

    public float[] dequantize(ByteBuffer byteBuffer, float scaleFactor, float offset, boolean unsinged) {
        // Ensure the ByteBuffer is ready for reading
        byteBuffer.rewind();
        // Calculate the number of integers in the ByteBuffer
        int intCount = byteBuffer.remaining(); // Integer.BYTES;
        float[] floatArray = new float[intCount];
        // Read integers and convert to floats with offset
        for (int i = 0; i < intCount; i++) {
            byte intValue = byteBuffer.get();
            if (unsinged == true)
            {
                int uintValue = intValue & 0xFF;
                floatArray[i] = (uintValue-offset) * scaleFactor;
            }
            else
            {
                floatArray[i] = (intValue-offset) * scaleFactor;
            }
        }

        return floatArray;
    }


}
