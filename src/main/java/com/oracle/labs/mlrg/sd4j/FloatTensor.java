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

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * A tensor containing primitive floats in a buffer.
 */
public final class FloatTensor extends Tensor<FloatBuffer> {

    /**
     * Creates a float tensor from the supplied buffer and shape.
     * @param buffer The buffer.
     * @param shape The shape.
     */
    public FloatTensor(FloatBuffer buffer, long[] shape) {
        super(buffer, shape);
    }

    /**
     * Creates an empty float tensor of the supplied shape backed by a direct byte buffer.
     * @param shape The shape.
     */
    public FloatTensor(long[] shape) {
        super(alloc(shape), shape);
    }

    @Override
    public FloatTensor copy() {
        FloatBuffer copy = alloc(shape);
        copy.put(buffer);
        copy.rewind();
        buffer.rewind();
        return new FloatTensor(copy, Arrays.copyOf(shape, shape.length));
    }

    /**
     * Splits this tensor into a list of new {@code FloatTensor}s.
     * <p>
     * The tensors are split in linear row major order, partitioned on the leading dimension.
     * @throws IllegalArgumentException If the supplied shape does not split this tensor in equal chunks.
     * @param newShape The new shape for the tensors.
     * @return A list containing the new tensors.
     */
    public List<FloatTensor> split(long[] newShape) {
        int newNumElements = computeNumElements(newShape);
        if (numElements % newNumElements != 0) {
            throw new IllegalArgumentException("Invalid shape for splitting, expected to split in into equal chunks.");
        }
        int numTensors = numElements / newNumElements;
        List<FloatTensor> output = new ArrayList<>();
        int position = 0;
        for (int i = 0; i < numTensors; i++) {
            FloatTensor tensor = new FloatTensor(newShape);
            tensor.buffer.put(0, buffer, position, tensor.numElements);
            position += tensor.numElements;
            output.add(tensor);
        }
        return output;
    }

    @Override
    public OnnxTensor wrapForORT(OrtEnvironment env) throws OrtException {
        return OnnxTensor.createTensor(env, this.buffer, this.shape);
    }

    /**
     * Adds the supplied tensor to this one.
     * @throws IllegalArgumentException If the other tensor is not the same shape as this one.
     * @param t The tensor to add.
     */
    public void add(FloatTensor t) {
        if (!Arrays.equals(t.shape,shape)) {
            throw new IllegalArgumentException("Invalid shape. Expected " + Arrays.toString(shape) + ", found " + Arrays.toString(t.shape));
        }
        for (int i = 0; i < numElements; i++) {
            buffer.put(i, buffer.get(i) + t.buffer.get(i));
        }
    }

    /**
     * Scales each element of the buffer by the supplied float.
     * <p>
     * Leaves the buffer position unchanged.
     * @param scalar The scalar.
     */
    public void scale(float scalar) {
        for (int i = 0; i < buffer.capacity(); i++) {
            buffer.put(i,buffer.get(i)*scalar);
        }
    }

    /**
     * Gets an element from this tensor.
     * @param idxArr The index to return.
     * @return The element at the index.
     */
    public float get(long... idxArr) {
        int idx = computeIdx(idxArr);
        return buffer.get(idx);
    }

    /**
     * Concatenates two tensors along their last dimension. All other dimensions must be equal.
     *
     * <p>For example a = [5, 10, 15], b = [5, 10, 3] gives concat(a,b) = [5, 10, 18].
     *
     * @param first The first tensor.
     * @param second The second tensor.
     * @return The row-wise concatenation of the two tensors.
     */
    public static FloatTensor concat(FloatTensor first, FloatTensor second) {
        // validate shapes
        if (first.shape.length != second.shape.length) {
            throw new IllegalArgumentException("Invalid shapes for concatenation, got " + Arrays.toString(first.shape) + " and " + Arrays.toString(second.shape));
        }
        long numRows = 1;
        for (int i = 0; i < first.shape.length - 1; i++) {
            if (first.shape[i] != second.shape[i]) {
                throw new IllegalArgumentException("Invalid shapes for concatenation, got " + Arrays.toString(first.shape) + " and " + Arrays.toString(second.shape));
            }
            numRows *= first.shape[i];
        }

        // create output
        var newShape = Arrays.copyOf(first.shape, first.shape.length);
        newShape[newShape.length-1] += second.shape[newShape.length-1];
        FloatTensor output = new FloatTensor(newShape);

        // concatenate row-wise, which is just reading from first then second until we run out
        int firstRowLength = (int) first.shape[first.shape.length-1];
        int secondRowLength = (int) second.shape[second.shape.length-1];
        int firstOffset = 0;
        int secondOffset = 0;
        int outputOffset = 0;
        for (int i = 0; i < numRows; i++) {
            output.buffer.put(outputOffset, first.buffer, firstOffset, firstRowLength);
            outputOffset += firstRowLength;
            firstOffset += firstRowLength;
            output.buffer.put(outputOffset, second.buffer, secondOffset, secondRowLength);
            outputOffset += secondRowLength;
            secondOffset += secondRowLength;
        }

        return output;
    }

    /**
     * Creates a direct {@link FloatBuffer} with capacity equal to the supplied shape.
     * @throws IllegalArgumentException if the shape is larger than the largest buffer.
     * @param shape The shape.
     * @return An int buffer.
     */
    private static FloatBuffer alloc(long[] shape) {
        int elements = computeNumElements(shape);
        if (elements < 0) {
            throw new IllegalArgumentException("Invalid shape for Java tensor, expected less than Integer.MAX_VALUE elements, found " + Arrays.toString(shape));
        }
        return ByteBuffer.allocateDirect(elements * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
    }
}
