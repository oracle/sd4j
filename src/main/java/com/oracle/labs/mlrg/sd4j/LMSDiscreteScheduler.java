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

import org.apache.commons.math4.legacy.analysis.UnivariateFunction;
import org.apache.commons.math4.legacy.analysis.integration.RombergIntegrator;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;
import java.util.stream.IntStream;

/**
 * A linear multi-step scheduler.
 * <p>
 * Scheduler implementations are stateful and not thread-safe.
 */
public final class LMSDiscreteScheduler implements Scheduler {
    private static final Logger logger = Logger.getLogger(LMSDiscreteScheduler.class.getName());

    private static final RombergIntegrator integrator = new RombergIntegrator();

    private final int numTrainTimesteps;
    private final float[] alphasCumulativeProducts;
    private final float[] initialVariance;
    private final float initNoiseSigma;
    private final List<FloatTensor> derivatives;

    private float[] sigmas = null;
    private int[] timesteps = null;

    /**
     * Creates a linear multi-step scheduler with the default parameters:
     * train timesteps = 1000, beta start = 0.00085f, beta end = 0.012f, Scaled Linear schedule.
     */
    public LMSDiscreteScheduler() {
        this(1000, 0.00085f, 0.012f, ScheduleType.SCALED_LINEAR);
    }

    /**
     * Creates a linear multi-step scheduler with the specified parameters.
     * @param numTrainTimesteps The number of training time diffusion steps.
     * @param betaStart The start value of the noise level.
     * @param betaEnd The end value of the noise level.
     * @param betaSchedule The noise schedule.
     */
    public LMSDiscreteScheduler(int numTrainTimesteps, float betaStart, float betaEnd, ScheduleType betaSchedule) {
        this.numTrainTimesteps = numTrainTimesteps;
        this.derivatives = new ArrayList<>();

        float[] betas = switch (betaSchedule) {
            case LINEAR -> MathUtils.linspace(betaStart, betaEnd, numTrainTimesteps, true);
            case SCALED_LINEAR -> {
                var start = (float) Math.sqrt(betaStart);
                var end = (float) Math.sqrt(betaEnd);
                var tmp = MathUtils.linspace(start, end, numTrainTimesteps, true);
                for (int i = 0; i < tmp.length; i++) {
                    tmp[i] = tmp[i] * tmp[i];
                }
                yield tmp;
            }
        };

        var alphas = new float[betas.length];
        this.alphasCumulativeProducts = new float[alphas.length];
        var cumProd = 1.0f;
        for (int i = 0; i < alphas.length; i++) {
            alphas[i] = 1 - betas[i];
            cumProd *= alphas[i];
            alphasCumulativeProducts[i] = cumProd;
        }

        // Create sigmas as a list and reverse it
        float curMax = Float.NEGATIVE_INFINITY;
        this.initialVariance = new float[alphasCumulativeProducts.length];
        for (int i = 0; i < alphasCumulativeProducts.length; i++) {
            float curVal = alphasCumulativeProducts[(alphasCumulativeProducts.length-1) - i];
            float newVal = (float) Math.sqrt((1-curVal) / curVal);
            initialVariance[i] = newVal;
            if (newVal > curMax) {
                curMax = newVal;
            }
        }

        // standard deviation of the initial noise distribution
        this.initNoiseSigma = curMax;
    }

    @Override
    public float getInitialNoiseSigma() {
        return initNoiseSigma;
    }

    /**
     * Reinitializes the scheduler with the specified number of inference steps.
     * @param numInferenceSteps The number of inference steps.
     * @return The new timesteps.
     */
    @Override
    public int[] setTimesteps(int numInferenceSteps) {
        this.derivatives.clear();
        float start = 0;
        float stop = numTrainTimesteps - 1;
        float[] timesteps = MathUtils.linspace(start, stop, numInferenceSteps, true);

        this.timesteps = new int[timesteps.length];
        for (int i = 0; i < timesteps.length; i++) {
            this.timesteps[i] = (int) timesteps[(timesteps.length-1)-i];
        }

        var range = MathUtils.arange(0, initialVariance.length, 1.0f);
        this.sigmas = MathUtils.interpolate(timesteps, range, initialVariance);
        return this.timesteps;
    }

    @Override
    public void scaleInPlace(FloatTensor sample, int timestep) {
        // Get step index of timestep from timesteps
        int stepIndex = MathUtils.findIdx(this.timesteps,timestep);
        // Get sigma at stepIndex
        var sigma = this.sigmas[stepIndex];
        sigma = (float)Math.sqrt((sigma*sigma) + 1);

        sample.scale(1/sigma);
    }

    /**
     * Computes the LMS coefficient for the current position by integrating over the noise computation.
     * @param order The order of the solver.
     * @param t The timestep.
     * @param currentOrder The current order.
     * @return The LMS coefficient.
     */
    private double getLmsCoefficient(int order, int t, int currentOrder) {
        // Compute a linear multistep coefficient.

        UnivariateFunction lmsDerivative = (double tau) -> {
            double prod = 1.0;
            for (int k = 0; k < order; k++) {
                if (currentOrder == k) {
                    continue;
                }
                prod *= (tau - this.sigmas[t - k]) / (this.sigmas[t - currentOrder] - this.sigmas[t - k]);
            }
            return prod;
        };
        // Inverted as the commons math integrator only goes one way.
        double integratedCoeff = -integrator.integrate(50, lmsDerivative, this.sigmas[t+1], this.sigmas[t]);
        //System.out.println("sigma[t] = "+this.sigmas[t]+", sigma[t+1] = "+this.sigmas[t+1]+", integratedCoeff = "+integratedCoeff);

        return integratedCoeff;
    }

    @Override
    public FloatTensor step(FloatTensor modelOutput, int timestep, FloatTensor sample, int order) {
        int stepIndex = MathUtils.findIdx(this.timesteps,timestep);
        var sigma = this.sigmas[stepIndex];

        // 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        FloatTensor predOriginalSample = new FloatTensor(modelOutput.shape);
        for (int i = 0; i < modelOutput.numElements; i++) {
            predOriginalSample.buffer.put(i, sample.buffer.get(i) - (sigma * modelOutput.buffer.get(i)));
        }

        // 2. Convert to an ODE derivative
        var derivativeItems = new FloatTensor(sample.shape);
        for (int i = 0; i < modelOutput.numElements; i++) {
            derivativeItems.buffer.put(i, (sample.buffer.get(i) - predOriginalSample.buffer.get(i)) / sigma);
        }

        this.derivatives.add(derivativeItems);
        if (this.derivatives.size() > order) {
            // remove first element
            this.derivatives.remove(0);
        }

        // 3. compute linear multistep coefficients
        var orderLim = Math.min(stepIndex + 1, order);
        var lmsCoeffs = IntStream.range(0, orderLim).mapToObj(currOrder -> getLmsCoefficient(orderLim, stepIndex, currOrder)).toList();

        // 4. compute previous sample based on the derivative path
        // Reverse list of tensors this.derivatives
        // Much easier with JEP 431.
        List<FloatTensor> revDerivatives = new ArrayList<>(this.derivatives.size());
        for (int i = 0; i < this.derivatives.size(); i++) {
            revDerivatives.add(this.derivatives.get((this.derivatives.size() - 1) - i));
        }

        // Create tensor for product of lmscoeffs and derivatives
        var lmsDerProduct = new FloatTensor(revDerivatives.get(0).shape);

        for(int m = 0; m < revDerivatives.size(); m++) {
            var curDeriv = revDerivatives.get(m);
            var curCoeff = lmsCoeffs.get(m);
            // Multiply to coeff by each derivative to create the new tensors
            var tmp = curDeriv.copy();
            tmp.scale(curCoeff.floatValue());
            lmsDerProduct.add(tmp);
        }

        // Add the summed tensor to the sample
        var prevSample = sample.copy();
        prevSample.add(lmsDerProduct);

        return prevSample;
    }
}