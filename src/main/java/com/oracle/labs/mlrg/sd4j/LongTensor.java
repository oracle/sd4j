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
import java.nio.LongBuffer;
import java.util.Arrays;

/**
 * A tensor containing primitive longs in a buffer.
 */
public final class LongTensor extends Tensor<LongBuffer> {

    /**
     * Creates a long tensor from the supplied buffer and shape.
     * @param buffer The buffer.
     * @param shape The shape.
     */
    public LongTensor(LongBuffer buffer, long[] shape) {
        super(buffer, shape);
    }

    /**
     * Creates an empty long tensor of the supplied shape backed by a direct byte buffer.
     * @param shape The shape.
     */
    public LongTensor(long[] shape) {
        super(alloc(shape), shape);
    }

    @Override
    public LongTensor copy() {
        LongBuffer copy = alloc(shape);
        copy.put(buffer);
        copy.rewind();
        buffer.rewind();
        return new LongTensor(copy, Arrays.copyOf(shape, shape.length));
    }

    @Override
    public OnnxTensor wrapForORT(OrtEnvironment env) throws OrtException {
        return OnnxTensor.createTensor(env, this.buffer, this.shape);
    }

    /**
     * Scales each element of the buffer by the supplied long.
     * <p>
     * Leaves the buffer position unchanged.
     * @param scalar The scalar.
     */
    public void scale(long scalar) {
        for (int i = 0; i < buffer.capacity(); i++) {
            buffer.put(i,buffer.get(i)*scalar);
        }
    }

    /**
     * Gets an element from this tensor.
     * @param idxArr The index to return.
     * @return The element at the index.
     */
    public long get(long... idxArr) {
        int idx = computeIdx(idxArr);

        return buffer.get(idx);
    }

    /**
     * Creates a direct{@link LongBuffer} with capacity equal to the supplied shape.
     * @throws IllegalArgumentException if the shape is larger than the largest buffer.
     * @param shape The shape.
     * @return An int buffer.
     */
    private static LongBuffer alloc(long[] shape) {
        int elements = computeNumElements(shape);
        if (elements < 0) {
            throw new IllegalArgumentException("Invalid shape for Java tensor, expected less than Integer.MAX_VALUE elements, found " + Arrays.toString(shape));
        }
        return ByteBuffer.allocateDirect(elements * Long.BYTES).order(ByteOrder.LITTLE_ENDIAN).asLongBuffer();
    }
}
