/*
 * Copyright (c) 2023 Oracle and/or its affiliates.
 *
 * The Universal Permissive License (UPL), Version 1.0
 *
 * Subject to the condition set forth below, permission is hereby granted to any
 * person obtaining a copy of this software, associated documentation and/or data
 * (collectively the "Software"), free of charge and under any and all copyright
 * rights in the Software, and any and all patent rights owned or freely
 * licensable by each licensor hereunder covering either (i) the unmodified
 * Software as contributed to or provided by such licensor, or (ii) the Larger
 * Works (as defined below), to deal in both
 *
 * (a) the Software, and
 * (b) any piece of software and/or hardware listed in the lrgrwrks.txt file if
 * one is included with the Software (each a "Larger Work" to which the Software
 * is contributed by such licensors),
 *
 * without restriction, including without limitation the rights to copy, create
 * derivative works of, display, perform, and distribute the Software and make,
 * use, sell, offer for sale, import, export, have made, and have sold the
 * Software and the Larger Work(s), and to sublicense the foregoing rights on
 * either these or other terms.
 *
 * This license is subject to the following condition:
 * The above copyright notice and either this complete permission notice or at
 * a minimum a reference to the UPL must be included in all copies or
 * substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package com.oracle.labs.mlrg.sd4j;

import ai.onnx.proto.OnnxMl;
import com.google.protobuf.ByteString;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Base64;

/**
 * Generates an onnx model which resizes a batch of image inputs from [batch_size, 3, x, y] to [batch_size, 3, 224, 224]
 * so it's suitable for the standard safety checker in stable diffusion class models.
 */
public final class ResizeModelGenerator {
    public static OnnxMl.TensorShapeProto getShapeProto(long[] dimensions, String[] dimensionOverrides) {
        OnnxMl.TensorShapeProto.Builder builder = OnnxMl.TensorShapeProto.newBuilder();
        for (int i = 0; i < dimensions.length; i++) {
            if (dimensions[i] == -1) {
                builder.addDim(OnnxMl.TensorShapeProto.Dimension.newBuilder().setDimParam(dimensionOverrides[i]).build());
            } else {
                builder.addDim(OnnxMl.TensorShapeProto.Dimension.newBuilder().setDimValue(dimensions[i]).build());
            }
        }
        return builder.build();
    }

    public static OnnxMl.TypeProto buildTensorTypeNode(long[] dimensions, String[] dimensionOverrides, OnnxMl.TensorProto.DataType type) {
        OnnxMl.TypeProto.Builder builder = OnnxMl.TypeProto.newBuilder();

        OnnxMl.TypeProto.Tensor.Builder tensorBuilder = OnnxMl.TypeProto.Tensor.newBuilder();
        tensorBuilder.setElemType(type.getNumber());
        tensorBuilder.setShape(getShapeProto(dimensions, dimensionOverrides));
        builder.setTensorType(tensorBuilder.build());

        return builder.build();
    }

    public static OnnxMl.TensorProto constant(String baseName, float value) {
        return OnnxMl.TensorProto.newBuilder()
                .setName(baseName)
                .setDataType(OnnxMl.TensorProto.DataType.FLOAT.getNumber())
                .addFloatData(value)
                .build();
    }

    public static OnnxMl.TensorProto constant(String baseName, long[] value) {
        return OnnxMl.TensorProto.newBuilder()
                .setName(baseName)
                .addDims(value.length)
                .setDataType(OnnxMl.TensorProto.DataType.INT64.getNumber())
                .addAllInt64Data(Arrays.stream(value).boxed().toList())
                .build();
    }

    /**
     * Generates the image resizing onnx model used by the safety checker.
     * <p>
     * Writes to "resize.onnx" in the working directory.
     * @param args Ignored
     * @throws IOException If it could not write the file.
     */
    public static void main(String[] args) throws IOException {
        OnnxMl.GraphProto.Builder graph = OnnxMl.GraphProto.newBuilder();
        graph.setName("CLIPImageProcessor");

        // Add placeholders
        OnnxMl.ValueInfoProto.Builder input = OnnxMl.ValueInfoProto.newBuilder();
        input.setName("input");
        OnnxMl.TypeProto inputType = buildTensorTypeNode(new long[]{-1, 3, -1, -1}, new String[]{"batch_size", null, "height", "width"}, OnnxMl.TensorProto.DataType.FLOAT);
        input.setType(inputType);
        graph.addInput(input);
        OnnxMl.ValueInfoProto.Builder output = OnnxMl.ValueInfoProto.newBuilder();
        output.setName("output");
        OnnxMl.TypeProto outputType = buildTensorTypeNode(new long[]{-1, 3, 224, 224}, new String[]{"batch_size", null, null, null}, OnnxMl.TensorProto.DataType.FLOAT);
        output.setType(outputType);
        graph.addOutput(output);

        // Add initializers
        // Image size [1,3,224,224]
        OnnxMl.TensorProto.Builder imageSize = OnnxMl.TensorProto.newBuilder();
        imageSize.addDims(3);
        imageSize.addInt64Data(3);
        imageSize.addInt64Data(224);
        imageSize.addInt64Data(224);
        imageSize.setDataType(OnnxMl.TensorProto.DataType.INT64.getNumber());
        imageSize.setName("image-size");
        graph.addInitializer(imageSize);

        // Constants 2, 0.5, 1 and 0.
        graph.addInitializer(constant("const-two",2f));
        graph.addInitializer(constant("const-half",0.5f));
        graph.addInitializer(constant("const-one",1f));
        graph.addInitializer(constant("const-zero",0f));
        graph.addInitializer(constant("const-tensor-zero", new long[] {0}));

        // Channel means
        OnnxMl.TensorProto.Builder channelMeans = OnnxMl.TensorProto.newBuilder();
        channelMeans.addDims(1);
        channelMeans.addDims(3);
        channelMeans.addDims(1);
        channelMeans.addDims(1);
        channelMeans.addFloatData(0.48145466f);
        channelMeans.addFloatData(0.4578275f);
        channelMeans.addFloatData(0.40821073f);
        channelMeans.setDataType(OnnxMl.TensorProto.DataType.FLOAT.getNumber());
        channelMeans.setName("channel-means");
        graph.addInitializer(channelMeans);

        // Channel std dev
        OnnxMl.TensorProto.Builder channelStdDevs = OnnxMl.TensorProto.newBuilder();
        channelStdDevs.addDims(1);
        channelStdDevs.addDims(3);
        channelStdDevs.addDims(1);
        channelStdDevs.addDims(1);
        channelStdDevs.addFloatData(0.26862954f);
        channelStdDevs.addFloatData(0.26130258f);
        channelStdDevs.addFloatData(0.27577711f);
        channelStdDevs.setDataType(OnnxMl.TensorProto.DataType.FLOAT.getNumber());
        channelStdDevs.setName("channel-stddevs");
        graph.addInitializer(channelStdDevs);

        // Add operations
        // Scale to 0-1 ((val / 2.0) + 0.5)
        OnnxMl.NodeProto.Builder scaleTwo = OnnxMl.NodeProto.newBuilder();
        scaleTwo.setName("divide-0");
        scaleTwo.setOpType("Div");
        scaleTwo.addInput("input");
        scaleTwo.addInput("const-two");
        scaleTwo.addOutput("divide-0-output");
        graph.addNode(scaleTwo);
        OnnxMl.NodeProto.Builder addHalf = OnnxMl.NodeProto.newBuilder();
        addHalf.setName("add-0");
        addHalf.setOpType("Add");
        addHalf.addInput("divide-0-output");
        addHalf.addInput("const-half");
        addHalf.addOutput("add-0-output");
        graph.addNode(addHalf);

        // Clamp to [0,1]
        OnnxMl.NodeProto.Builder lowerClamp = OnnxMl.NodeProto.newBuilder();
        lowerClamp.setName("max-0");
        lowerClamp.setOpType("Max");
        lowerClamp.addInput("add-0-output");
        lowerClamp.addInput("const-zero");
        lowerClamp.addOutput("max-0-output");
        graph.addNode(lowerClamp);
        OnnxMl.NodeProto.Builder upperClamp = OnnxMl.NodeProto.newBuilder();
        upperClamp.setName("min-0");
        upperClamp.setOpType("Min");
        upperClamp.addInput("max-0-output");
        upperClamp.addInput("const-one");
        upperClamp.addOutput("min-0-output");
        graph.addNode(upperClamp);

        // Make reshape size
        OnnxMl.NodeProto.Builder shape = OnnxMl.NodeProto.newBuilder();
        shape.setName("shape-0");
        shape.setOpType("Shape");
        shape.addInput("input");
        shape.addOutput("shape-0-output");
        graph.addNode(shape);

        OnnxMl.NodeProto.Builder gather = OnnxMl.NodeProto.newBuilder();
        gather.setName("gather-0");
        gather.setOpType("Gather");
        gather.addInput("shape-0-output");
        gather.addInput("const-tensor-zero");
        gather.addOutput("gather-0-output"); // batchSize
        graph.addNode(gather);

        OnnxMl.NodeProto.Builder concat = OnnxMl.NodeProto.newBuilder();
        concat.setName("concat-0");
        concat.setOpType("Concat");
        concat.addAttribute(OnnxMl.AttributeProto.newBuilder().setName("axis").setType(OnnxMl.AttributeProto.AttributeType.INT).setI(0).build());
        concat.addInput("gather-0-output");
        concat.addInput("image-size");
        concat.addOutput("concat-0-output");
        graph.addNode(concat);

        // Resize image
        OnnxMl.NodeProto.Builder resize = OnnxMl.NodeProto.newBuilder();
        resize.setName("resize-0");
        resize.setOpType("Resize");
        resize.addAttribute(OnnxMl.AttributeProto.newBuilder().setName("mode").setType(OnnxMl.AttributeProto.AttributeType.STRING).setS(ByteString.copyFrom("linear", StandardCharsets.UTF_8)).build());
        resize.addInput("min-0-output");
        resize.addInput("");//roi
        resize.addInput("");//scales
        resize.addInput("concat-0-output");
        resize.addOutput("resize-0-output");
        graph.addNode(resize);

        // Subtract mean
        OnnxMl.NodeProto.Builder subMean = OnnxMl.NodeProto.newBuilder();
        subMean.setName("subtract-0");
        subMean.setOpType("Sub");
        subMean.addInput("resize-0-output");
        subMean.addInput("channel-means");
        subMean.addOutput("subtract-0-output");
        graph.addNode(subMean);

        // Scale by std dev
        OnnxMl.NodeProto.Builder scaleStd = OnnxMl.NodeProto.newBuilder();
        scaleStd.setName("divide-1");
        scaleStd.setOpType("Div");
        scaleStd.addInput("subtract-0-output");
        scaleStd.addInput("channel-stddevs");
        scaleStd.addOutput("output");
        graph.addNode(scaleStd);

        // Build model
        OnnxMl.ModelProto.Builder model = OnnxMl.ModelProto.newBuilder();
        model.setGraph(graph);
        model.setDocString("CLIP Image Processor");
        model.setModelVersion(0);
        model.setIrVersion(8);
        model.setDomain("com.huggingface.transformers");
        model.addOpsetImport(OnnxMl.OperatorSetIdProto.newBuilder().setVersion(18).build());
        try (OutputStream os = Files.newOutputStream(Paths.get("resize.onnx"))) {
            model.build().writeTo(os);
        }
        byte[] outputArr = model.build().toByteArray();
        System.out.println(Base64.getEncoder().encodeToString(outputArr));
    }

}
