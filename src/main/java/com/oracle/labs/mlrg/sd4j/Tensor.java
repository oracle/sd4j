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

import java.nio.Buffer;
import java.util.Arrays;

/**
 * Base class for tensors wrapped around {@link java.nio.Buffer} instances.
 * <p>
 * Contains a buffer and a shape. Most of the mutation methods live on the subclasses
 * to allow them to use primitive types. This class is mutable and exposes the buffer for
 * mutation. Remember to rewind it whenever you operate on the buffer using the implicit position methods.
 * @param <B> The buffer type.
 */
public sealed abstract class Tensor<B extends Buffer> permits FloatTensor, IntTensor, LongTensor {

    /**
     * The buffer holding the values.
     */
    protected final B buffer;
    /**
     * The shape of the tensor.
     */
    protected final long[] shape;
    /**
     * Stride values for indexing into the tensor.
     */
    protected final long[] strides;

    /**
     * The total number of elements in this tensor.
     */
    protected final int numElements;

    /**
     * Creates a Tensor from the supplied buffer and shape.
     * @param buffer The buffer containing the data.
     * @param shape The shape.
     */
    public Tensor(B buffer, long[] shape) {
        this.buffer = buffer;
        this.shape = Arrays.copyOf(shape, shape.length);
        this.strides = new long[this.shape.length];
        this.strides[strides.length-1] = 1;
        for (int i = strides.length-1; i > 0; i--) {
            this.strides[i-1] = strides[i] * this.shape[i];
        }
        this.numElements = computeNumElements(this.shape);
        if (this.buffer.capacity() != this.numElements) {
            throw new IllegalArgumentException("Buffer has different capacity than the shape expects. Buffer.capacity = " + this.buffer.capacity() + ", numElements = " + this.numElements);
        }
    }

    /**
     * Access the buffer directly.
     * @return The buffer.
     */
    public B buffer() {
        return buffer;
    }

    /**
     * The shape.
     * @return The shape.
     */
    public long[] shape() {
        return shape;
    }

    /**
     * Deep copy of this tensor.
     * @return A copy of the tensor.
     */
    public abstract Tensor<B> copy();

    /**
     * Wraps this tensor into an {@code OnnxTensor} for passing into ORT.
     * @param env The ORT environment.
     * @return An OnnxTensor.
     * @throws OrtException If the tensor could not be created.
     */
    public abstract OnnxTensor wrapForORT(OrtEnvironment env) throws OrtException;

    /**
     * Computes the linear index from the supplied index array.
     * @param idxArr The index array.
     * @return The linear index into the buffer.
     */
    protected int computeIdx(long[] idxArr) {
        int idx = 0;
        for (int i = 0; i < shape.length; i++) {
            idx += idxArr[i] * strides[i];
        }
        return idx;
    }

    /**
     * Computes the number of elements.
     * <p>
     * If we overflow the int it returns -1, and the tensor is invalid.
     * @param shape The shape.
     * @return The number of elements.
     */
    protected static int computeNumElements(long[] shape) {
        int total = 1;
        for (int i = 0; i < shape.length; i++) {
            long cur = shape[i];
            if ((((int) cur) != cur) || (cur < 0)){
                total = -1;
                break;
            } else {
                total *= cur;
                if (total <= 0) {
                    break;
                }
            }
        }
        return total;
    }
}
