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
import ai.onnxruntime.OrtSession;

import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

/**
 * The Variational Auto-Encoder which decodes from the latent space of the UNet to pixel space, running in ONNX Runtime.
 */
public final class VAEDecoder implements AutoCloseable {
    private static final Logger logger = Logger.getLogger(VAEDecoder.class.getName());

    /**
     * Scaling coefficient for transforming SD models from latent to pixel space.
     */
    public static final float SD_LATENT_SCALAR = 1.0f / 0.18215f;

    /**
     * Scaling coefficient for transforming SDXL models from latent to pixel space.
     */
    public static final float SDXL_LATENT_SCALAR = 1.0f / 0.13025f;

    private final OrtEnvironment env;

    private final OrtSession.SessionOptions vaeOpts;
    private final OrtSession vae;

    /**
     * Constructs a VAEDecoder from the supplied model using the default session options.
     * @param vaeModelPath The path to the VAE decoder ONNX file.
     * @throws OrtException If the model could not be loaded.
     */
    public VAEDecoder(Path vaeModelPath) throws OrtException {
        this(vaeModelPath, new OrtSession.SessionOptions());
    }

    /**
     * Constructs a VAEDecoder from the supplied model path and session options.
     * @param vaeModelPath The path to the VAE decoder ONNX file.
     * @param opts The session options.
     * @throws OrtException If the model could not be loaded.
     */
    public VAEDecoder(Path vaeModelPath, OrtSession.SessionOptions opts) throws OrtException {
        this.env = OrtEnvironment.getEnvironment();
        this.vaeOpts = opts;
        this.vae = env.createSession(vaeModelPath.toString(), vaeOpts);
    }

    /**
     * Decodes the latents into the image. Mutates the input latents
     * @param latents The latent image variables.
     * @param scalar The scaling factor applied to the latents before decoding.
     * @return The image stored as a float tensor.
     * @throws OrtException If the model failed.
     */
    public FloatTensor decoder(FloatTensor latents, float scalar) throws OrtException {
        logger.info("Decoding latents");
        latents.scale(scalar);
        try (var inputTensor = OnnxTensor.createTensor(env, latents.buffer, latents.shape);
            var result = vae.run(Map.of("latent_sample", inputTensor))) {
            var outputTensor = (OnnxTensor) result.get(0);
            var fb = outputTensor.getFloatBuffer();
            return new FloatTensor(fb, outputTensor.getInfo().getShape());
        }
    }

    /**
     * Converts the pixel space tensor into a batch of {@link BufferedImage} instances.
     * @param imageTensor The pixel space tensor [batch_size, 3, height, width].
     * @return A list of images.
     */
    public static List<BufferedImage> convertToBufferedImage(FloatTensor imageTensor) {
        logger.info("Generating image from decoded tensor");
        var height = (int) imageTensor.shape[2];
        var width = (int) imageTensor.shape[3];
        var output = new ArrayList<BufferedImage>();
        for (int i = 0; i < imageTensor.shape[0]; i++) {
            BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
            WritableRaster raster = image.getRaster();

            for (var y = 0; y < height; y++) {
                for (var x = 0; x < width; x++) {
                    // output image is ARGB, but input floats are RGB
                    int red = convertValue(imageTensor.get(i,0,y,x));
                    int green = convertValue(imageTensor.get(i,1,y,x));
                    int blue = convertValue(imageTensor.get(i,2,y,x));

                    raster.setSample(x, y, 0, red);
                    raster.setSample(x, y, 1, green);
                    raster.setSample(x, y, 2, blue);
                }
            }
            output.add(image);
        }
        return output;
    }

    /**
     * Decodes a latent space tensor into a batch of images.
     * @param latents The latent space tensor [batch_size, 4, height/8, width/8].
     * @param scalar The scaling factor applied to the latents before decoding.
     * @return A list of generated images.
     * @throws OrtException If the decode operation failed.
     */
    public List<BufferedImage> decodeToBufferedImage(FloatTensor latents, float scalar) throws OrtException {
        var floatImage = decoder(latents, scalar);
        return convertToBufferedImage(floatImage);
    }

    /**
     * Converts a colour value from the range [-1, 1] to [0, 255].
     * @param colourValue The colour value to convert.
     * @return An unsigned colour byte.
     */
    public static int convertValue(float colourValue) {
        float scaled = (colourValue / 2.0f) + 0.5f;
        float clamped = Math.min(1.0f, Math.max(scaled, 0.0f));
        int round = Math.round(clamped * 255);
        return round;
    }

    @Override
    public void close() throws OrtException {
        vae.close();
        vaeOpts.close();
    }
}
