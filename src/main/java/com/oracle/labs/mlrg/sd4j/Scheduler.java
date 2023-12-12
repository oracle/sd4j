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

/**
 * Interface for diffusion schedulers which control how noise is eliminated from the image.
 * <p>
 * Scheduler implementations are stateful and not thread-safe.
 */
public interface Scheduler {

    /**
     * The type of the schedule.
     */
    public enum ScheduleType {
        /**
         * Linear noise schedule.
         */
        LINEAR,
        /**
         * Scaled linear noise schedule.
         */
        SCALED_LINEAR
    }

    /**
     * The initial noise standard deviation.
     * @return The noise standard deviation.
     */
    public float getInitialNoiseSigma();

    /**
     * Sets the number of timesteps for this inference run.
     * <p>
     * This mutates the object and configures it for the next run.
     * @param numInferenceSteps The number of timesteps.
     * @return An array
     */
    public int[] setTimesteps(int numInferenceSteps);

    /**
     * Scales the input tensor for the appropriate timestep if necessary.
     * @param sample The input tensor.
     * @param timestep The current timestep.
     */
    public void scaleInPlace(FloatTensor sample, int timestep);

    /**
     * Takes a diffusion step, producing the next latent sample.
     * <p>
     * Sets the order to 4.
     * @param modelOutput The model output.
     * @param timestep The current timestep.
     * @param sample The noise sample.
     * @return The next latent sample.
     */
    default public FloatTensor step(FloatTensor modelOutput, int timestep, FloatTensor sample) {
        return step(modelOutput, timestep, sample, 4);
    }

    /**
     * Takes a diffusion step, producing the next latent sample.
     * @param modelOutput The model output.
     * @param timestep The current timestep.
     * @param sample The noise sample.
     * @param order The order of the scheduler.
     * @return The next latent sample.
     */
    public FloatTensor step(FloatTensor modelOutput, int timestep, FloatTensor sample, int order);
}
