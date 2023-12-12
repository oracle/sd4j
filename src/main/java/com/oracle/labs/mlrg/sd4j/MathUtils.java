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

import java.util.Arrays;

/**
 * Math utility functions for use in {@link Scheduler} implementations.
 */
public final class MathUtils {

    private MathUtils() {}

    /**
     * Interpolate the range for the given timesteps.
     * @param timesteps The timesteps to interpolate the value.
     * @param range The range of values.
     * @param sigmas The noise values.
     * @return The interpolated values.
     */
    public static float[] interpolate(float[] timesteps, float[] range, float[] sigmas) {
        // Create an output array with the same shape as timesteps
        var result = new float[timesteps.length+1];

        // Loop over each element of timesteps
        for (int i = 0; i < timesteps.length; i++) {
            // Find the index of the first element in range that is greater than or equal to timesteps[i]
            int index = Arrays.binarySearch(range, timesteps[i]);

            // If timesteps[i] is exactly equal to an element in range, use the corresponding value in sigma
            if (index >= 0) {
                result[i] = sigmas[index];
            } else if (index == -1) {
                // If timesteps[i] is less than the first element in range, use the first value in sigmas
                result[i] = sigmas[0];
            } else if (index == -range.length - 1) {
                // If timesteps[i] is greater than the last element in range, use the last value in sigmas
                result[i] = sigmas[sigmas.length-1];
            } else {
                // Otherwise, interpolate linearly between two adjacent values in sigmas
                index = ~index; // bitwise complement of j gives the insertion point of x[i]
                float t = (timesteps[i] - range[index - 1]) / (range[index] - range[index - 1]); // fractional distance between two points
                result[i] = sigmas[index - 1] + t * (sigmas[index] - sigmas[index - 1]); // linear interpolation formula
            }
        }

        return result;
    }

    /**
     * Mirrors numpy linspace and NumSharp linspace. Returns values evenly spaced between start and end.
     * @param start The start value.
     * @param end The end value.
     * @param numSteps The number of steps.
     * @param includeEnd Should end be the last value?
     * @return An array of values evenly spaced between start and end.
     */
    public static float[] linspace(float start, float end, int numSteps, boolean includeEnd) {
        if (end <= start) {
            throw new IllegalArgumentException("Invalid range, end must be strictly greater than start");
        }
        if (numSteps <= 0) {
            throw new IllegalArgumentException("Invalid number of steps, " + numSteps);
        }
        float stepSize = (end - start) / (includeEnd ? numSteps - 1 : numSteps);
        float[] output = new float[numSteps];
        for (int i = 0; i < numSteps; i++) {
            output[i] = start + (i * stepSize);
        }
        return output;
    }

    /**
     * Mirrors numpy arange and NumSharp arange. Returns values stepped by {@code stepSize}.
     * @param start The start value.
     * @param end The end value.
     * @param stepSize The step size.
     * @return An array of values stepped between start and end.
     */
    public static float[] arange(float start, float end, float stepSize) {
        if (end <= start) {
            throw new IllegalArgumentException("Invalid range, end must be strictly greater than start");
        }
        if (stepSize <= 0.00001f) {
            throw new IllegalArgumentException("Invalid stepSize, must be positive.");
        }
        int numSteps = Math.round((float)Math.ceil((end - start)/stepSize));
        float[] output = new float[numSteps];
        for (int i = 0; i < numSteps; i++) {
            output[i] = start + (i * stepSize);
        }
        return output;
    }

    /**
     * Linear probe for the specified value in the target array.
     * <p>
     * Used when the array order prevents {@link Arrays#binarySearch(int[], int)}.
     * @param array The array to search.
     * @param target The target value.
     * @return The index of the target, or -1 if not found.
     */
    public static int findIdx(int[] array, int target) {
        int idx = -1;
        for (int i = 0; i < array.length; i++) {
            if (array[i] == target) {
                idx = i;
            }
        }
        return idx;
    }
}
