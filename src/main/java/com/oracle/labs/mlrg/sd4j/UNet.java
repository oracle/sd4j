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
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.Consumer;
import java.util.logging.Logger;

/**
 * A UNet model which performs the inverse diffusion, predicting the noise which should be subtracted from the image.
 */
public final class UNet implements AutoCloseable {
    private static final Logger logger = Logger.getLogger(UNet.class.getName());

    private final OrtEnvironment env;
    private final OrtSession.SessionOptions unetOpts;
    private final OrtSession unet;

    private final TensorInfo.OnnxTensorType timestepType;

    /**
     * Creates a UNet model with the default session options.
     *
     * @param unetPath The path to the model ONNX file.
     * @throws OrtException If the model could not be loaded.
     */
    public UNet(Path unetPath) throws OrtException {
        this(unetPath, new OrtSession.SessionOptions());
    }

    /**
     * Creates a UNet model with the supplied session options.
     *
     * @param unetPath The path to the model ONNX file.
     * @param opts     The ONNX Runtime session options.
     * @throws OrtException If the model could not be loaded.
     */
    public UNet(Path unetPath, OrtSession.SessionOptions opts) throws OrtException {
        this.env = OrtEnvironment.getEnvironment();
        this.unetOpts = opts;
        this.unet = env.createSession(unetPath.toString(), unetOpts);
        var infos = this.unet.getInputInfo();
        var timestepNode = infos.get("timestep");
        if (timestepNode != null) {
            var timestepTensorInfo = (TensorInfo) timestepNode.getInfo();
            this.timestepType = timestepTensorInfo.onnxType;
        } else {
            throw new IllegalArgumentException("Invalid unet, does not accept a timestep");
        }
    }

    /**
     * Packages the inputs into the supplied map suitable for inference in ONNX Runtime.
     *
     * @param map                 The input map, will be cleared.
     * @param encoderHiddenStates The encoder hidden states (i.e. the text input).
     * @param sample              The current image sample.
     * @param timestep            The timestep number.
     * @throws OrtException If the OnnxTensor construction failed.
     */
    private void createUnetModelInput(Map<String, OnnxTensor> map, FloatTensor encoderHiddenStates, FloatTensor sample, long timestep) throws OrtException {
        map.clear();

        map.put("encoder_hidden_states", encoderHiddenStates.wrapForORT(env));
        map.put("sample", sample.wrapForORT(env));
        OnnxTensor timestepTensor = switch (this.timestepType) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 -> OnnxTensor.createTensor(env, new int[]{(int) timestep});
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 -> OnnxTensor.createTensor(env, new long[]{timestep});
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT -> OnnxTensor.createTensor(env, new float[]{timestep});
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE -> OnnxTensor.createTensor(env, new double[]{timestep});
            default -> throw new IllegalStateException("Invalid tensor type for timestep tensor.");
        };
        map.put("timestep", timestepTensor);
    }

    /**
     * Packages the inputs into the supplied map suitable for SDXL inference in ONNX Runtime.
     *
     * @param map                 The input map, will be cleared.
     * @param encoderHiddenStates The encoder hidden states (i.e. the text input).
     * @param sample              The current image sample.
     * @param timestep            The timestep number.
     * @param textEmbeds            The pooled text embeddings.
     * @param additionalImageInputs The timeids input (source & target image size, along with crop position)
     * @throws OrtException If the OnnxTensor construction failed.
     */
    private void createUnetXLModelInput(Map<String, OnnxTensor> map, FloatTensor encoderHiddenStates, FloatTensor sample, long timestep, FloatTensor textEmbeds, FloatTensor additionalImageInputs) throws OrtException {
        map.clear();

        map.put("encoder_hidden_states", encoderHiddenStates.wrapForORT(env));
        map.put("sample", sample.wrapForORT(env));
        OnnxTensor timestepTensor = switch (this.timestepType) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 -> OnnxTensor.createTensor(env, new int[]{(int) timestep});
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 -> OnnxTensor.createTensor(env, new long[]{timestep});
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT -> OnnxTensor.createTensor(env, new float[]{timestep});
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE -> OnnxTensor.createTensor(env, new double[]{timestep});
            default -> throw new IllegalStateException("Invalid tensor type for timestep tensor.");
        };
        map.put("timestep", timestepTensor);
        map.put("text_embeds", textEmbeds.wrapForORT(env));
        map.put("time_ids", additionalImageInputs.wrapForORT(env));
    }

    /**
     * Samples an initial latent space tensor by sampling from a zero mean gaussian with the supplied noise level.
     *
     * @param batchSize          The number of images to create.
     * @param height             The image height.
     * @param width              The image width.
     * @param seed               The RNG seed.
     * @param initialNoiseStdDev The std dev of the gaussian noise.
     * @return A latent space sample.
     */
    private static FloatTensor sampleLatent(int batchSize, int height, int width, int seed, float initialNoiseStdDev) {
        var rng = new SplittableRandom(seed);
        var channels = 4;
        var latents = new FloatTensor(new long[]{batchSize, channels, height / 8, width / 8});

        for (int i = 0; i < latents.buffer.capacity(); i++) {
            double stdNormal = rng.nextGaussian(0, initialNoiseStdDev);
            latents.buffer.put(i, (float) stdNormal);
        }

        return latents;
    }

    /**
     * Applies the classifier-free guidance by subtracting the transformed predicted noise.
     * <p>
     * Mutates the {@code noisePred} input.
     *
     * @param noisePred     The noise prediction for the negative or unconditional image.
     * @param noisePredText The noise prediction for the positive text.
     * @param guidanceScale The guidance strength.
     */
    private static void performGuidance(FloatTensor noisePred, FloatTensor noisePredText, float guidanceScale) {
        for (int i = 0; i < noisePred.buffer.capacity(); i++) {
            float curNoise = noisePred.buffer.get(i);
            float curNoiseText = noisePredText.buffer.get(i);
            float update = curNoise + (guidanceScale * (curNoiseText - curNoise));
            noisePred.buffer.put(i, update);
        }
    }

    /**
     * Runs UNet inverse diffusion for the specified number of steps.
     *
     * @param numInferenceSteps The number of inference steps.
     * @param textEmbeddings    The text embedding vectors.
     * @param guidanceScale     The strength of the classifier-free guidance.
     * @param batchSize         The number of generated images.
     * @param height            The image height.
     * @param width             The image width.
     * @param seed              The RNG seed.
     * @param callback          The callback function, called with the step count after each step.
     * @param schedulerEnum     The scheduler to use.
     * @return A batch of images in latent space.
     * @throws OrtException If the inference call failed in ONNX Runtime.
     */
    public FloatTensor inference(int numInferenceSteps, FloatTensor textEmbeddings, float guidanceScale, int batchSize, int height, int width, int seed, Consumer<Integer> callback, Schedulers schedulerEnum) throws OrtException {
        return inference(numInferenceSteps, textEmbeddings, null, guidanceScale, batchSize, height, width, seed, callback, schedulerEnum);
    }

    /**
     * Runs UNet inverse diffusion for the specified number of steps.
     *
     * <p>When pooledTextEmbeddings is null this performs regular SD v1.5 or v2 inference, when it is non-null it
     * performs SDXL inference.
     *
     * @param numInferenceSteps The number of inference steps.
     * @param textEmbeddings    The text embedding vectors.
     * @param pooledTextEmbeddings The pooled text embedding vectors. When this is non-null it performs SDXL inference.
     * @param guidanceScale     The strength of the classifier-free guidance.
     * @param batchSize         The number of generated images.
     * @param height            The image height.
     * @param width             The image width.
     * @param seed              The RNG seed.
     * @param callback          The callback function, called with the step count after each step.
     * @param schedulerEnum     The scheduler to use.
     * @return A batch of images in latent space.
     * @throws OrtException If the inference call failed in ONNX Runtime.
     */
    public FloatTensor inference(int numInferenceSteps, FloatTensor textEmbeddings, FloatTensor pooledTextEmbeddings, float guidanceScale, int batchSize, int height, int width, int seed, Consumer<Integer> callback, Schedulers schedulerEnum) throws OrtException {
        var rng = new SplittableRandom(seed);
        var scheduler = schedulerEnum.create(rng.nextLong());
        var timesteps = scheduler.setTimesteps(numInferenceSteps);

        // create latent tensor
        var latents = sampleLatent(batchSize, height, width, seed, scheduler.getInitialNoiseSigma());

        boolean doClassifierFreeGuidance = guidanceScale >= 1.0;
        logger.info("Classifier free guidance = " + doClassifierFreeGuidance);

        boolean sdxl = pooledTextEmbeddings != null;
        FloatTensor additionalImageConditions;
        if (sdxl) {
            logger.info("SDXL inference");
            // original size [h,w], crop position [h,w], target size [h, w]
            float[] conditionsArr = new float[]{height, width, 0, 0, height, width};
            if (doClassifierFreeGuidance) {
                // guidance means we duplicate this vector
                // technically there are negative embeddings for those things which can change generation behaviour
                // but given we're fixing them anyway let's not bother about that
                additionalImageConditions = new FloatTensor(new long[]{2L*batchSize, 6L});
                for (int i = 0; i < batchSize; i++) {
                    additionalImageConditions.buffer.put(conditionsArr);
                    additionalImageConditions.buffer.put(conditionsArr);
                }
            } else {
                additionalImageConditions = new FloatTensor(new long[]{batchSize, 6L});
                for (int i = 0; i < batchSize; i++) {
                    additionalImageConditions.buffer.put(conditionsArr);
                }
            }
            additionalImageConditions.buffer.rewind();
        } else {
            logger.info("SD inference");
            additionalImageConditions = null;
        }

        long[] guidedLatentShape = new long[]{2L*batchSize, 4L, height / 8, width / 8};
        long[] unguidedLatentShape = new long[]{batchSize, 4L, height / 8, width / 8};

        var input = new HashMap<String, OnnxTensor>(6);
        for (int t = 0; t < timesteps.length; t++) {
            logger.info("Running inference step " + t);
            latents.buffer().rewind();
            FloatTensor latentModelInput;
            if (doClassifierFreeGuidance) {
                // guidance uses two states, a positive latent value to move towards and a
                // negative latent value to move away from
                latentModelInput = new FloatTensor(guidedLatentShape);
                latentModelInput.buffer.put(latents.buffer());
                latents.buffer().rewind();
            } else {
                latentModelInput = new FloatTensor(unguidedLatentShape);
            }
            latentModelInput.buffer.put(latents.buffer());
            latents.buffer().rewind();
            latentModelInput.buffer.rewind();

            scheduler.scaleInPlace(latentModelInput, timesteps[t]);

            try {
                if (sdxl) {
                    createUnetXLModelInput(input, textEmbeddings, latentModelInput, timesteps[t], pooledTextEmbeddings, additionalImageConditions);
                } else {
                    createUnetModelInput(input, textEmbeddings, latentModelInput, timesteps[t]);
                }

                // Run Inference
                try (var output = unet.run(input)) {
                    var outputTensor = (OnnxTensor) output.get(0);

                    var fb = outputTensor.getFloatBuffer();
                    if (!Arrays.equals(latentModelInput.shape, outputTensor.getInfo().getShape())) {
                        throw new IllegalStateException("Expected output shape " + Arrays.toString(latentModelInput.shape) + ", found " + Arrays.toString(outputTensor.getInfo().getShape()));
                    }
                    FloatTensor noisePred;
                    if (doClassifierFreeGuidance) {
                        // Split tensors from 2*batch_size,4,64,64 to 1*batch_size,4,64,64
                        var ft = new FloatTensor(fb, outputTensor.getInfo().getShape());
                        var splitTensors = ft.split(unguidedLatentShape);
                        noisePred = splitTensors.get(0);
                        var noisePredText = splitTensors.get(1);

                        // Perform guidance
                        performGuidance(noisePred, noisePredText, guidanceScale);
                    } else {
                        // If no guidance then wrap in FloatTensor
                        noisePred = new FloatTensor(fb, unguidedLatentShape);
                    }

                    // Scheduler Step
                    latents = scheduler.step(noisePred, timesteps[t], latents);
                }

                callback.accept(t + 1);
            } finally {
                OnnxValue.close(input);
                input.clear();
            }
        }

        return latents;
    }

    @Override
    public void close() throws OrtException {
        unet.close();
        unetOpts.close();
    }
}
